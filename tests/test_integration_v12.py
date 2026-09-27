"""v1.2 集成契约回归（跨 lane 对齐）：

1. **阶段 B 冻结前缀**：primary → specific 两段的可训练前缀集合 + 被冻层无梯度
   （``pipeline.stages._PRIMARY_PHASE_FREEZE/_SPECIFIC_PHASE_FREEZE`` 必须用 v2 模块名）；
2. **presence/entry = id 轴**：未来帧同 track id 是否在盒内 / 新 id 是否出现
   （合成"离场→入场"样例；槽位轴旧逻辑会漏掉同槽新 id）；
3. **Stage A plan head 梯度**：``ego_next`` 监督让 plan head/MoE 在 A 阶段有梯度
   （design-v1.2 §2.3 方案①），policy/value 冻结；
4. **monitoring router 口径**：router_weights = softmax 全专家分布（≠ top-2 expert_weights）；
5. **v1 ``net.world_model`` 已删除且无 import 引用**；
6. **config/model.yaml + train.yaml v2 语义**（top-2 softmax / 软目标 CE/KL / ego_next_coef）。
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from net.model import DrivingModel
from pipeline.monitoring import MoERoutingStatistics
from pipeline.stages import (
    _PRIMARY_PHASE_FREEZE,
    _SPECIFIC_PHASE_FREEZE,
    _parse_args,
    presence_entry_targets,
    run_stage_a,
)
from pipeline.trainer import apply_freeze_prefixes
from tests.test_net_shapes import make_obs
from tests.v2_synthetic import TINY_MODEL_YAML, write_v2_dataset

ROOT = Path(__file__).resolve().parents[1]

#: specific 段允许训练的模块前缀（其余全部冻结；lane T 用户定稿：主干可继续训，
#: 只冻结 primary 的策略头/专家 = ``policy.`` + ``plan_head.moe.primary.``）
_SPECIFIC_TRAINABLE = (
    "encoders.",
    "mem_encoder.",
    "plan_head.fusion.",
    "plan_head.norm.",
    "plan_head.ego_next.",
    "plan_head.gate.",
    "plan_head.moe.experts.",
    "plan_head.moe.router.",
    "plan_head.moe.residual_scale",
)


def _tiny_model() -> DrivingModel:
    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=8, expert_hidden=16)


def _backward(model: DrivingModel) -> None:
    """backward 一个覆盖策略/轨迹/router/WM 输出的损失（用于梯度可见性检查）。"""
    obs = make_obs(batch=2)
    out = model(obs, rollout=True, world_model=True)
    loss = (
        out["action_mu"].pow(2).mean()
        + out["traj_xy"].pow(2).mean()
        + out["od_pred"].pow(2).mean()
        + out["router_logits"].pow(2).mean()
        + out["ego_next"].pow(2).mean()
        + out["hard_logit"].pow(2).mean()  # lane T 二值门控头
    )
    loss.backward()


# ---------------------------------------------------------------------- 1. 冻结
def test_primary_phase_freeze_prefixes_match_v2_modules() -> None:
    """primary 段：冻结 ST-GNN/value/specific experts；主干+plan head+router+policy 可训练。"""
    model = _tiny_model()
    apply_freeze_prefixes(model, _PRIMARY_PHASE_FREEZE)
    for name, parameter in model.named_parameters():
        frozen = name.startswith(_PRIMARY_PHASE_FREEZE)
        assert parameter.requires_grad is not frozen, f"{name} 冻结状态错误"
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith("encoders.") for n in trainable_names)
    assert any(n.startswith("mem_encoder.") for n in trainable_names)
    assert any(n.startswith("plan_head.moe.primary.") for n in trainable_names)
    assert any(n.startswith("plan_head.moe.router.") for n in trainable_names)
    assert any(n.startswith("policy.") for n in trainable_names)
    assert not any(n.startswith("st_gnn.") for n in trainable_names)
    # 被冻层确实无梯度（requires_grad=False → autograd 不填 .grad）
    _backward(model)
    for name, parameter in model.named_parameters():
        if name.startswith(_PRIMARY_PHASE_FREEZE):
            assert parameter.grad is None, f"冻结层 {name} 不应有梯度"
    # 可训练层中必须有梯度张量（零初始化/独立输出的个别参数如 policy.log_std 可为 None，
    # 因此按前缀检查组内至少一个参数有梯度）
    for prefix in ("encoders.", "mem_encoder.", "plan_head.", "plan_head.moe.", "policy."):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None for g in grads), f"{prefix} 组内无任何梯度张量"


def test_specific_phase_freezes_primary_policy_head_and_experts() -> None:
    """specific 段（lane T）：冻结 primary 策略头/专家（``policy.`` + ``plan_head.moe.primary.``），
    主干 + 8 specific experts + router + 二值门控可继续训。"""
    model = _tiny_model()
    apply_freeze_prefixes(model, _SPECIFIC_PHASE_FREEZE)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {name for name, _ in model.named_parameters() if name.startswith(_SPECIFIC_TRAINABLE)}
    assert trainable == expected and trainable, "specific 段可训练集合必须恰为 主干+experts+router+gate"
    assert any(n.startswith("encoders.") for n in trainable), "主干可继续训"
    assert any(n.startswith("plan_head.moe.experts.") for n in trainable)
    assert any(n.startswith("plan_head.moe.router.") for n in trainable)
    assert any(n.startswith("plan_head.gate.") for n in trainable)
    assert not any(
        n.startswith(("policy.", "plan_head.moe.primary.", "st_gnn.", "value.")) for n in trainable
    ), "primary 策略头/专家（+WM/value）必须冻结"
    _backward(model)
    # 冻结组一律无梯度；可训练组内至少一个参数有梯度张量（zero-init 个别参数可为 None）
    for name, parameter in model.named_parameters():
        if name not in trainable:
            assert parameter.grad is None, f"specific 段冻结层 {name} 不应有梯度"
    for prefix in _SPECIFIC_TRAINABLE:
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(prefix)]
        assert any(g is not None for g in grads), f"specific 段 {prefix} 组内无任何梯度张量"


# ---------------------------------------------------- 2. presence/entry id 轴
def _future_payload(
    presence_t0: np.ndarray,
    id_t0: np.ndarray,
    presence_fut: np.ndarray,
    id_fut: np.ndarray,
) -> dict:
    return {
        "od_presence_t0": presence_t0.astype(np.float32),
        "od_id_t0": id_t0.astype(np.int64),
        "od_presence_fut": presence_fut.astype(np.float32),
        "od_id_fut": id_fut.astype(np.int64),
        "od_mask": (presence_t0 > 0.5).astype(np.float32),
    }


def test_presence_entry_targets_id_axis_leave_then_enter() -> None:
    """合成"离场→入场"：id=12 离场后同槽出现新 id=13 → entry=1（槽位轴旧逻辑=0）。

    t0：slot0=id11(在盒) / slot1=id12(在盒) / slot2 空 / slot3=id14(出盒未释放)
    k0：slot0=id11(在盒) / slot1=id12(出盒) / slot2 空 / slot3=id14(仍出盒)
    k1：slot0=id11(在盒) / slot1=id13(新 id，入场) / slot2 空 / slot3=id14(回入)
    """
    presence_t0 = np.array([[1, 1, 0, 0]], dtype=np.float32)
    id_t0 = np.array([[11, 12, -1, 14]], dtype=np.int64)
    presence_fut = np.array([[[1, 0, 0, 0], [1, 1, 0, 1]]], dtype=np.float32)  # (1,2,4)
    id_fut = np.array([[[11, 12, -1, 14], [11, 13, -1, 14]]], dtype=np.int64)

    presence_target, entry_target = presence_entry_targets(
        _future_payload(presence_t0, id_t0, presence_fut, id_fut)
    )
    # presence：t0 已观测的 id 在目标帧是否仍在盒内（slot3 的 id14 在 t0 未观测 → 恒 0）
    assert presence_target.tolist() == [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    # entry：k0 无新观测；k1 slot1 出现 id13（t0 未观测）→ 1；slot3 id14 回入也属"t0 无、t+k 有" → 1
    assert entry_target.tolist() == [[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]
    # 本样例 presence/entry 无重叠（v2 固定槽位下 id 不迁移槽位）
    assert float((presence_target * entry_target).sum()) == 0.0

    # 槽位轴旧逻辑（目标帧 presence ∧ t0 无 presence）在 k1/slot1 给出 0（漏报新 id）——
    # 这正是修复点：id 轴能看到同槽换 id。
    old_entry = (presence_fut > 0.5) & (presence_t0[:, None, :] <= 0.5)
    assert old_entry[0, 1, 1] == 0.0 and entry_target[0, 1, 1] == 1.0


def test_presence_target_follows_id_across_slots_and_wm_valid_gate() -> None:
    """presence 按 id 匹配（不限槽位）；新 id 进入空槽 → entry=1、presence=0。"""
    presence_t0 = np.array([[1, 0, 0, 0]], dtype=np.float32)
    id_t0 = np.array([[21, -1, -1, -1]], dtype=np.int64)
    # k0：id=21 出现在 slot2（槽位漂移）；k1：slot1 出现全新 id=22
    presence_fut = np.array([[[0, 0, 1, 0], [0, 1, 0, 0]]], dtype=np.float32)
    id_fut = np.array([[[-1, -1, 21, -1], [-1, 22, -1, -1]]], dtype=np.int64)
    presence_target, entry_target = presence_entry_targets(
        _future_payload(presence_t0, id_t0, presence_fut, id_fut)
    )
    assert presence_target[0, 0, 0] == 1.0, "同 id 换槽仍算在盒"
    assert presence_target[0, 1, 0] == 0.0
    assert entry_target[0, 1, 1] == 1.0, "新 id 进入空槽 = entry"
    assert presence_target[0, 1, 1] == 0.0


def test_presence_entry_targets_v1_fallback_slot_axis() -> None:
    """无 id 伴随数组（v1 数据）时回退槽位轴旧语义。"""
    presence_t0 = np.array([[1, 0]], dtype=np.float32)
    presence_fut = np.array([[[1, 1]]], dtype=np.float32)
    payload = {
        "od_presence_t0": presence_t0,
        "od_presence_fut": presence_fut,
        "od_mask": presence_t0,
    }
    presence_target, entry_target = presence_entry_targets(payload)
    assert presence_target.tolist() == [[[1.0, 0.0]]]
    assert entry_target.tolist() == [[[0.0, 1.0]]]


# ---------------------------------------------------- 3. Stage A plan head 梯度
def test_stage_a_ego_next_gradient_and_id_axis_metrics(tmp_path: Path) -> None:
    """Stage A 小样本：ego_next 监督可用、plan head/MoE 有梯度、policy/value 冻结。"""
    dataset_dir = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(TINY_MODEL_YAML, encoding="utf-8")
    out_dir = tmp_path / "stage_a"
    args = _parse_args([
        "--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--model-config", str(model_cfg), "--wm-epochs", "1",
        "--batch-size", "8", "--eval-frames", "8", "--device", "cpu", "--seed", "0",
    ])
    metrics = run_stage_a(args, {})

    assert metrics["ego_next_available"] == 1.0
    assert metrics["presence_entry_axis"] == "id"
    assert metrics["presence_available"] == 1.0
    assert np.isfinite(metrics["wm_loss_ego_next"]) and metrics["wm_loss_ego_next"] > 0.0
    assert np.isfinite(metrics["ego_next_loss"])
    per_horizon = metrics["per_horizon"]
    assert all("ego_next_loss" in item for item in per_horizon.values())
    assert any(np.isfinite(item["ego_next_loss"]) for item in per_horizon.values())

    grads = metrics["grad_norms_first_batch"]
    # design-v1.2 §2.3 方案①：plan head/MoE 在 Stage A 必须有梯度
    assert grads["plan_head"] > 0.0, "plan head 无梯度 = Stage A 冻结失效回归"
    assert grads["plan_head.moe.experts"] > 0.0, "specific experts 无梯度"
    assert grads["st_gnn"] > 0.0 and grads["encoders"] > 0.0 and grads["mem_encoder"] > 0.0
    # policy/value 不参与 Stage A
    assert grads["policy"] == 0.0 and grads["value"] == 0.0
    # router：experts 零初始化时首 batch 梯度=0（混合权重不影响零专家输出）；
    # 训练若干步 experts 非零后 router 必须收到梯度（last batch probe）。
    assert metrics["grad_norms_last_batch"]["plan_head.moe.router"] > 0.0, "router 始终无梯度"
    assert metrics["grad_norms_last_batch"]["policy"] == 0.0
    assert metrics["grad_norms_last_batch"]["value"] == 0.0


# ---------------------------------------------------- 4. monitoring router 口径
def test_monitoring_router_softmax_vs_top2_mixture_semantics() -> None:
    """router softmax 分布：effective_n=Σsoftmax；top-2 混合（Σ≡1）语义不同。"""
    logits = torch.tensor([[2.0, 2.0, -10.0, -10.0]])
    softmax = torch.softmax(logits, dim=-1)
    stats = MoERoutingStatistics()
    stats.update(softmax)
    out = stats.flush()
    assert out["moe/effective_n"] == pytest.approx(float(softmax.sum()), abs=1e-6)
    assert out["moe/expert_0/weight"] == pytest.approx(float(softmax[0, 0]), abs=1e-6)

    # top-2 混合权重（恰 2 个非零、和为 1）→ effective_n 恒 1（不可当 router 分布用）
    mixture = torch.tensor([[0.65, 0.35, 0.0, 0.0]])
    stats.update(mixture)
    assert stats.flush()["moe/effective_n"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------- 5. world_model 删除
def test_world_model_module_removed_and_unreferenced() -> None:
    """v1 ``net.world_model`` 已删除；Python 源码不得再有 import 引用。"""
    assert importlib.util.find_spec("net.world_model") is None, "net/world_model.py 应已删除"
    pattern = re.compile(r"^\s*(from\s+net\.world_model\s+import|import\s+net\.world_model)\b")
    offenders = []
    for path in ROOT.rglob("*.py"):
        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if pattern.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}")
    assert not offenders, f"仍有 net.world_model import 引用：{offenders}"


# ---------------------------------------------------- 6. config v2 语义
def test_config_router_and_stage_a_v2_semantics() -> None:
    """config/model.yaml router = top-2 softmax + 硬标签 CE；train.yaml A 段有 ego_next_coef。"""
    import yaml

    model_cfg = yaml.safe_load((ROOT / "config/model.yaml").read_text(encoding="utf-8"))
    router = model_cfg["moe"]["router"]
    assert router["type"] == "top2_softmax"
    assert router["supervision"] == "hard_cluster_ce"  # lane B B3：软目标/温度路径已删除
    assert "temperature" not in router
    assert router["top_k"] == 2
    assert router.get("labels") != "per_step_observable"
    # 数据集标签顺序仍保留（数据契约/切片分析用，不再是 router 监督）
    assert tuple(router["supervised_labels"]) == (
        "cutin_active", "cutout_active", "crowded", "car_following",
        "on_curve", "merging", "roundabout_near", "near_intersection",
    )

    train_cfg = yaml.safe_load((ROOT / "config/train.yaml").read_text(encoding="utf-8"))
    stage_a = train_cfg["stages"]["A"]["world_model"]
    assert float(stage_a["ego_next_coef"]) > 0.0
    assert float(stage_a["presence_coef"]) > 0.0 and float(stage_a["entry_coef"]) > 0.0
    trainable = [str(item) for item in stage_a["trainable"]]
    assert "st_gnn" in trainable and "plan_head" in trainable and "moe" in trainable
    assert "world_model" not in trainable and "spatial" not in trainable
    assert bool(train_cfg["stages"]["B"]["bc"]["wm_detach"]) is True
