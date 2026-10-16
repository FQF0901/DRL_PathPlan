"""R41 第一批-E：**horizon parity h1–h6 小样本守卫**（fixture + 小模型；CPU 可跑，无研究工作区依赖）。

背景
----
R41 复审 §七.5：「不能只检查第一步。逐 horizon 记录：训练输入分布 / 评测 h1..h6 输入分布 /
residual 输出分布 / Mahalanobis distance / P50/P90/max 偏移；重点验证第 2–5 步 imagined
exposure shift」。本测试在 fixture 尺度上锁住 **horizon 语义与 h1 口径 parity**（大资源版
读数在 `.slim/deepwork/p1_v2/round41/e_parity.py` → `runs/round41/e/**`；大资源同源逐位守卫
在 ``tests/test_r40_train_eval_feature_parity.py``）。

锁定的契约（与 e_parity.py 一致）
--------------------------------
1. **h1..h6 = 残差头 6 次调用**（k=1..6；h1 用真实 t0 状态+动作，h2..h6 用自产想象状态+动作；
   调用 k 的动作 = ``plan[:, k-1]``）；
2. **h1 口径 parity**：同 obs、同模型、同 batching 下，评测链 k=1 的特征/ΔA_θ ≡ 训练侧
   ``pipeline.residual_features.transition_features`` 输出（逐位，max_abs_diff=0）；
3. **horizon 对齐**：逐 horizon 特征 ``prev_exog`` ≡ WorldState probe 同 horizon 状态视图
   ``long_exog``（h1..h6）；WorldState 记录步序 = 0..6（S6 末步不回头消费）；
4. **逐 horizon 可算漂移读数**（小样本 MD 有限；h6 ≠ h1），供 e 报告复用。

大资源缺失无关：本文件只用 ``tests/fixtures/residual_mini`` + 随机初始化小模型（CPU）。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/residual_mini"

import pipeline.residual_dataset as rd  # noqa: E402
import pipeline.residual_features as rf  # noqa: E402
from net.probes import ProbeRecorder  # noqa: E402
from net.world_residual import ReadoutResidualHead  # noqa: E402

#: h1..h6（= 残差头调用序号 1..6）
HORIZONS = tuple(f"h{i}" for i in range(1, 7))


def _fixture_args(**overrides) -> Namespace:
    base = dict(data=str(FIXTURE), shards=None, max_pairs=None, seed=0, val_frac=0.25, chunk=64)
    base.update(overrides)
    return Namespace(**base)


def _tiny_model():
    """确定性小模型（随机初始化；CPU；结构化头开；MoE 关）——与 B 自包含守卫同构。"""
    from net.model import DrivingModel

    torch.manual_seed(7)
    model = DrivingModel(
        hidden=32, num_experts=1, expert_hidden=16, router_hidden=16,
        primary_hidden=32, trunk_hidden=32, net_hidden=32, spatial_layers=1,
    )
    model.set_moe(enabled=False)
    model.set_path_head(True)
    return model.eval()


@pytest.fixture(scope="module")
def horizon_capture():
    """fixture 观测 → 想象链（analytic_learned + 小残差头）逐 horizon 捕获。"""
    data = rd.build_pairs(_fixture_args())
    rows = data["rows"]
    ids = np.asarray(data["main_ids"][:8], dtype=np.int64)
    device = torch.device("cpu")
    obs = rd.build_obs(rows, ids, device=device, base_dataset=None)

    model = _tiny_model()
    model.set_world_state_api("analytic_learned", speed_alpha=1.0)
    model.set_world_residual_scale(1.0)
    torch.manual_seed(1234)
    head = ReadoutResidualHead(hidden=16).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.1)
    model.set_world_residual_head(head)

    probes = ProbeRecorder(enabled=False)
    probes.world_state.enabled = True
    model.set_probes(probes)

    calls: list[dict] = []

    def _hook(module, hook_args, hook_kwargs, output):
        calls.append({
            "features": {key: value.detach().clone() for key, value in hook_kwargs.items()},
            "delta": output.detach().clone(),
        })

    handle = head.register_forward_hook(_hook, with_kwargs=True)
    with torch.no_grad():
        out = model(obs, forward_mode="world_model_teacher_forcing")
    handle.remove()

    with torch.no_grad():
        train_feat = rf.transition_features(model, obs)
    return {
        "model": model, "head": head, "obs": obs, "out": out, "calls": calls,
        "records": list(probes.world_state.records), "train_feat": train_feat,
        "batch": int(ids.size),
    }


def test_horizon_chain_has_six_residual_calls(horizon_capture) -> None:
    """h1..h6 = 残差头 6 次调用；每次键集 = FEATURE_KEYS；structured_ds/plan 列数 = 6。"""
    calls = horizon_capture["calls"]
    assert len(calls) == len(HORIZONS) == 6
    for call in calls:
        assert sorted(call["features"]) == sorted(rf.FEATURE_KEYS)
        assert tuple(call["delta"].shape) == (horizon_capture["batch"], 4)
    out = horizon_capture["out"]
    assert tuple(out["structured_ds"].shape) == (horizon_capture["batch"], 6)
    assert tuple(out["plan"].shape) == (horizon_capture["batch"], 6, 2)
    # ds 列 ≡ plan ds（h1..h6 与动作链同一出处；plan0 ≡ t0 动作）
    assert torch.equal(out["structured_ds"], out["plan"][:, :, 0])
    assert torch.equal(out["plan"][:, 0, :], out["action_mu"])


def test_h1_feature_and_delta_parity_with_training_caliber(horizon_capture) -> None:
    """h1（k=1）特征/ΔA_θ ≡ 训练口径 ``transition_features``（同 obs / 同模型；逐位）。"""
    first = horizon_capture["calls"][0]
    train_feat = horizon_capture["train_feat"]
    for key in rf.FEATURE_KEYS:
        a = first["features"][key].numpy()
        b = train_feat["features"][key].detach().numpy()
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"h1 parity 失败：{key}"
    with torch.no_grad():
        delta_train = rf.head_delta(horizon_capture["head"], train_feat["feature_tensor"])
    assert np.array_equal(first["delta"].numpy(), delta_train.numpy()), "h1 ΔA_θ parity 失败"
    # base 目标列/动作亦须同源（h1 = 现训练目标）
    assert np.array_equal(first["features"]["base_exog"].numpy(), train_feat["base_cols"].numpy())
    assert np.array_equal(first["features"]["action"].numpy(), train_feat["action"].numpy())


def test_horizon_action_matches_plan_column(horizon_capture) -> None:
    """调用 k 的动作 = ``plan[:, k-1]``（h1 = a0 真实 t0 动作；h6 = a5 末步动作）。"""
    calls = horizon_capture["calls"]
    plan = horizon_capture["out"]["plan"]
    for index, call in enumerate(calls):
        expected = plan[:, index]
        assert np.array_equal(call["features"]["action"].numpy(), expected.numpy()), f"h{index + 1} 动作错位"


def test_prev_exog_equals_world_state_view_per_horizon(horizon_capture) -> None:
    """逐 horizon：特征 ``prev_exog`` ≡ WorldState probe ``long_exog``（h1..h6）；步序 = 0..6。"""
    calls = horizon_capture["calls"]
    records = horizon_capture["records"]
    assert [int(record["step"]) for record in records] == list(range(7))
    for index, call in enumerate(calls):
        view = records[index]["long_exog"].numpy()
        prev = call["features"]["prev_exog"].numpy()
        assert np.array_equal(prev, view), f"h{index + 1}: prev_exog ≠ ws view（horizon 错位）"
    # S6（末步）视图存在且不回头消费（不参与任何调用）
    assert "long_exog" in records[6]


def test_horizon_drift_readout_is_computable(horizon_capture) -> None:
    """逐 horizon MD 读数（参照 = h1 一阶/二阶）可算：有限、非负、h6 与 h1 不同。"""
    calls = horizon_capture["calls"]
    feats = np.stack([
        np.concatenate([call["features"][key].numpy().reshape(horizon_capture["batch"], -1)
                        for key in ("prev_exog", "prev_present", "action", "base_exog", "base_present")],
                       axis=1)
        for call in calls
    ], axis=0)  # (6, B, 26)
    ref = feats[0]
    mu = ref.mean(axis=0)
    std = ref.std(axis=0)
    std = np.maximum(std, max(1e-3 * float(np.median(std[std > 0])) if (std > 0).any() else 1e-6, 1e-6))
    md = np.sqrt(np.mean(((feats - mu) / std) ** 2, axis=-1))  # (6, B) 对角 MD
    assert np.isfinite(md).all() and (md >= 0).all()
    assert md[0].mean() < md[-1].mean() + 1e-9  # 允许相等；大资源版见 e_parity.py 的单调性读数
    assert not np.array_equal(feats[0], feats[-1])


def test_fixture_probe_flags_path_head_structured(horizon_capture) -> None:
    """结构化头契约：probe 记录含 producer 断言通过（readout 单一写点）。"""
    for record in horizon_capture["records"]:
        assert record["producer_assert_ok"] is True
