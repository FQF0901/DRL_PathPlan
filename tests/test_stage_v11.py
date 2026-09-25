"""v1.1 阶段契约回归：GL 路径守卫、未来窗口查表（stride）、WM detach、参数冻结。

对应 `.slim/deepwork/p2-contract.md` §8 前置 + user v1.1 阶段修订：

- spawn worker 的 GL 崩溃修复（``pipeline.gl_runtime``）必须幂等且可从仓库根计算；
- 阶段 A 未来目标 = ``(episode, step + k·stride)`` 查表 + t0 帧对齐（BC harvest 的
  ``step`` 是 env step，帧只在策略步边界记录 → stride=5）；
- 阶段 B rollout 轨迹辅助的因果链：``wm_detach=True`` 时 WM 参数不得收到梯度；
- primary→specific 分段训练依赖 ``apply_freeze_prefixes`` 的前缀语义。
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from net.model import DrivingModel
from pipeline.trainer import apply_freeze_prefixes
from tests.test_net_shapes import make_obs


# ---------------------------------------------------------------------- GL 守卫
def test_ensure_gl_library_path_idempotent() -> None:
    from pipeline.gl_runtime import ensure_gl_library_path, gl_library_dir

    gl_dir = gl_library_dir()
    if gl_dir is None:
        pytest.skip("venv 本地 glvnd 目录不存在（非本项目环境）")
    entry = str(gl_dir)
    before = os.environ.get("LD_LIBRARY_PATH")
    try:
        # 1) 缺失时必须前插（优先于其它路径）
        os.environ["LD_LIBRARY_PATH"] = "/tmp/other"
        assert ensure_gl_library_path() == entry
        parts = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
        assert parts[0] == entry, "缺失时必须前插，优先于其它路径"
        # 2) 幂等：重复调用不产生重复条目
        assert ensure_gl_library_path() == entry
        assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep).count(entry) == 1
        # 3) 已存在但不在首位（cv2 import 会前插自己的路径）时：不重复添加、也不报错
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(["/tmp/other", entry])
        assert ensure_gl_library_path() == entry
        assert os.environ["LD_LIBRARY_PATH"].split(os.pathsep).count(entry) == 1
    finally:
        if before is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = before


# ------------------------------------------------------------- 未来窗口查表/对齐
def _synthetic_windows():
    """1 个 episode、3 帧（step=0/5/10）、匀速直线 + 一个世界系 +2 m/s 的目标。

    世界系：ego x(i)=3i；目标 x(i)=20+i（每帧 i 间隔 0.5 s，即 +2 m/s）。
    - 未来帧原始 dx(i) = 20+i-3i = 20-2i（i=1→18，i=2→16）；
    - 对齐到 t0 后 dx(i) = 20+i（i=1→21，i=2→22）；
    - 相对速度 vx = 2-3 = -1（恒定）。
    """
    from env.obs.base import FrameAlignment
    from pipeline.stages import FrameWindows, _bc_channel_keys

    count, stride = 3, 5
    pose = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=np.float32)
    od = np.zeros((count, 16, 9), dtype=np.float32)
    od_mask = np.zeros((count, 16), dtype=np.float32)
    od_mask[:, 0] = 1.0
    for index in range(count):
        od[index, 0, 0] = 20.0 - 2.0 * index  # dx（未来自车系）
        od[index, 0, 2] = -1.0  # vx（相对速度）
        od[index, 0, 4] = 1.0  # cosθ
    ld = np.zeros((count, 16, 7), dtype=np.float32)
    ld_mask = np.zeros((count, 16), dtype=np.float32)
    arrays = {
        "episode_id": np.zeros(count, dtype=np.int64),
        "step": np.arange(count, dtype=np.int64) * stride,
        "pose": pose,
        "od": od,
        "od_mask": od_mask,
        "ld": ld,
        "ld_mask": ld_mask,
        "action": np.zeros((count, 6, 2), dtype=np.float32),
    }
    alignments = {
        "od": FrameAlignment(point_pairs=((0, 1),), vector_pairs=((2, 3), (4, 5))),
        "ld": FrameAlignment(point_pairs=((0, 1),)),
    }
    windows = FrameWindows(
        arrays,
        alignments,
        episode_key="episode_id",
        step_key="step",
        keys=_bc_channel_keys(),
        step_stride=stride,
    )
    return windows


def test_frame_windows_future_lookup_and_alignment() -> None:
    windows = _synthetic_windows()
    future = windows.build_future(np.array([0]), future=6)
    # k=1..2 有未来帧（step=5/10）；k>=3 缺失 → valid=0、mask=0
    assert future["valid"][0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert future["od_mask"][0, :, 0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # 世界系 +2 m/s 目标对齐到 t0 帧：k=1 → dx=21，k=2 → dx=22（原始未来帧为 18/16）
    assert future["od_fut"][0, 0, 0, 0] == pytest.approx(21.0, abs=1e-4)
    assert future["od_fut"][0, 1, 0, 0] == pytest.approx(22.0, abs=1e-4)


def test_frame_windows_stride_one_matches_dense_steps() -> None:
    """stride=1（PPO replay schema）时 k 逐 env step；这里只验证 valid 语义。"""
    windows = _synthetic_windows()
    windows.step_stride = 1
    future = windows.build_future(np.array([0]), future=6)
    # 查表 (0, 1..4) 不存在，只有 (0,5) → 第 5 个策略步有效（k=5 → step+5）
    assert future["valid"][0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_match_future_od_slots_reorders_by_identity() -> None:
    """未来帧槽位顺序变化（TTC 排序）时，匹配必须把目标换回当前槽位对应的对象。"""
    from pipeline.stages import match_future_od_slots

    future = {
        "od_fut": np.zeros((1, 1, 2, 9), dtype=np.float32),
        "od_mask": np.ones((1, 1, 2), dtype=np.float32),
        "valid": np.ones((1, 1), dtype=np.float32),
    }
    current = np.zeros((1, 2, 9), dtype=np.float32)
    current[0, 0, :2] = [10.0, 0.0]
    current[0, 1, :2] = [20.0, 0.0]
    # 未来帧槽位 0/1 与当前槽位顺序相反（同一两个对象）
    future["od_fut"][0, 0, 0, :2] = [20.1, 0.0]
    future["od_fut"][0, 0, 1, :2] = [9.9, 0.0]
    out = match_future_od_slots(future, current, np.ones((1, 2), dtype=np.float32), gate_m=8.0)
    assert out["od_fut"][0, 0, 0, 0] == pytest.approx(9.9, abs=1e-4)
    assert out["od_fut"][0, 0, 1, 0] == pytest.approx(20.1, abs=1e-4)
    # 两个未来槽位都远离当前槽位（对象离场）→ gate 过滤 → mask=0
    future["od_fut"][0, 0, 0, :2] = [100.0, 0.0]
    future["od_fut"][0, 0, 1, :2] = [100.0, 0.0]
    out = match_future_od_slots(future, current, np.ones((1, 2), dtype=np.float32), gate_m=8.0)
    assert out["od_mask"][0, 0].tolist() == [0.0, 0.0]


# ------------------------------------------------------------- WM detach / 冻结
def _model_with_unsaturated_policy() -> DrivingModel:
    """policy.mu 末层零初始化会在第一步挡住上游梯度 → 先给一个小的非零权重。"""
    torch.manual_seed(0)
    model = DrivingModel()
    with torch.no_grad():
        model.policy.mu.weight.normal_(0.0, 0.01)
    return model


def test_wm_detach_blocks_gradient_to_world_model() -> None:
    model = _model_with_unsaturated_policy()
    obs = make_obs(batch=2)
    out = model(obs, rollout=True, world_model=False, wm_detach=True)
    out["traj_xy"].pow(2).mean().backward()
    wm_grads = [p.grad for name, p in model.named_parameters() if name.startswith("world_model.")]
    assert wm_grads, "模型应包含 world_model 参数"
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in wm_grads), "detach 后 WM 不得有梯度"
    assert model.policy.mu.weight.grad is not None and float(model.policy.mu.weight.grad.abs().sum()) > 0.0
    enc_grads = [p.grad for name, p in model.named_parameters() if name.startswith("encoders.")]
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in enc_grads), "共享主干仍应可训练"


def test_wm_gradients_flow_without_detach() -> None:
    model = _model_with_unsaturated_policy()
    obs = make_obs(batch=2)
    out = model(obs, rollout=True, world_model=False, wm_detach=False)
    out["traj_xy"].pow(2).mean().backward()
    wm_grads = [p.grad for name, p in model.named_parameters() if name.startswith("world_model.")]
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in wm_grads), "不 detach 时 WM 应有梯度"


def test_apply_freeze_prefixes_scopes_and_restores() -> None:
    model = DrivingModel()
    frozen = apply_freeze_prefixes(model, ("world_model.", "value.", "moe.experts."))
    assert frozen and all(name.startswith(("world_model.", "value.", "moe.experts.")) for name in frozen)
    assert all(
        not p.requires_grad for name, p in model.named_parameters() if name.startswith(("world_model.", "value.", "moe.experts."))
    )
    assert all(p.requires_grad for name, p in model.named_parameters() if name.startswith("policy."))
    # 空前缀 = 全部解冻
    apply_freeze_prefixes(model, ())
    assert all(p.requires_grad for p in model.parameters())
