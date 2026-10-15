"""R38-P2a：K-anchor 减枝移除守卫（model / config / pipeline 面）。

锁定（`docs/deprecation_manifest.yaml` B 组；三阶段 S2 = 删引用）：

- ``DrivingModel`` 构造签名与 state_dict 无 K-anchor 参数/键；forward 输出无 ``anchor_*`` 键；
- ``build_model`` 不再读取 ``plan_anchor`` 配置块（旧配置残留该键 = 被忽略，无行为）；
- 正式配置 ``config/default.yaml`` 无 ``plan_anchor`` 键；
- 训练侧无 ``anchor_wta_loss`` / ``anchor_ce_weight`` / ``anchor_wta_weight`` / ``anchor_temperature``；
  ``phase3_effective_config`` 降级清单只剩 ``PHASE3_WM_LOSS_KEYS``；
- 冻结表（specific_only / trunk_only / recovery / recovery_experts）无锚头前缀；
- ``eval_runner.task_model_config`` 不再透传 ``plan_anchor``；
- 阶段 B phase3 CLI 无 ``--phase3-anchor-{ce,wta}-weight``（保留 5k 锚数据集 ``--phase3-anchor``）。
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _tiny_kwargs() -> dict:
    return {"hidden": 16, "num_experts": 2, "expert_hidden": 16}


def test_driving_model_has_no_anchor_api() -> None:
    from net.model import DrivingModel

    signature = inspect.signature(DrivingModel.__init__)
    for name in ("num_anchors", "anchor_path", "anchor_temperature", "anchor_hard"):
        assert name not in signature.parameters, f"K-anchor 构造参数 {name} 不得存在"
    model = DrivingModel(**_tiny_kwargs())
    assert not hasattr(model, "num_anchors")
    state_keys = list(model.state_dict())
    assert not any("anchor" in key for key in state_keys), [
        key for key in state_keys if "anchor" in key
    ]


def test_default_forward_has_no_anchor_output_keys() -> None:
    from tests.test_net_shapes import make_obs

    from net.model import DrivingModel

    model = DrivingModel(**_tiny_kwargs()).eval()
    with torch.no_grad():
        out = model(make_obs(batch=2))
    assert not any("anchor" in key for key in out), [key for key in out if "anchor" in key]
    assert torch.equal(out["plan"][:, 0], out["action_mu"])


def test_build_model_ignores_legacy_plan_anchor_config() -> None:
    """旧配置残留 ``plan_anchor`` 块（如历史臂文件）→ 被忽略，模型正常构造。"""
    from pipeline.stages import build_model

    config = {"hidden_dim": 16, "plan_anchor": {"enabled": True, "num_anchors": 6}}
    model = build_model(config)
    assert not hasattr(model, "num_anchors")
    assert not any("anchor" in key for key in model.state_dict())


def test_official_config_has_no_plan_anchor_block() -> None:
    import yaml

    payload = yaml.safe_load((ROOT / "config" / "model.yaml").read_text(encoding="utf-8"))
    assert "plan_anchor" not in payload
    train_payload = yaml.safe_load((ROOT / "config" / "train.yaml").read_text(encoding="utf-8"))
    bc = train_payload["stages"]["B"]["bc"]
    for key in ("anchor_ce_weight", "anchor_wta_weight", "anchor_temperature"):
        assert key not in bc, f"K-anchor 损失键 {key} 不得留在正式配置"
    losses = train_payload["stages"]["B"]["phase3"]["losses"]
    for key in ("anchor_ce", "anchor_wta", "anchor_temperature"):
        assert key not in losses, f"K-anchor 损失键 {key} 不得留在正式配置"
    assert not (ROOT / "config" / "plan_anchors_k6.json").exists()


def test_trainer_has_no_anchor_loss_plumbing() -> None:
    from pipeline import trainer

    assert not hasattr(trainer, "anchor_wta_loss")
    for key in ("anchor_ce_weight", "anchor_wta_weight", "anchor_temperature"):
        assert key not in trainer.BCConfig.__dataclass_fields__, key
        assert key not in trainer.Phase3Config.__dataclass_fields__, key
    config = trainer.Phase3Config(freeze_mode="recovery", od_weight=0.005)
    effective, zeroed = trainer.phase3_effective_config(config)
    assert tuple(zeroed) == trainer.PHASE3_WM_LOSS_KEYS
    assert not any(key.startswith("anchor") for key in zeroed)
    assert float(effective.od_weight) == 0.0


def test_freeze_tables_have_no_anchor_prefixes() -> None:
    from pipeline import stages

    assert not hasattr(stages, "_PHASE3_ANCHOR_HEAD_PREFIXES")
    for name in (
        "_SPECIFIC_PHASE_FREEZE",
        "_PHASE3_SAFE_FREEZE",
        "_RECOVERY_FREEZE",
        "_RECOVERY_EXPERTS_FREEZE",
    ):
        prefixes = getattr(stages, name)
        assert not any(
            "anchor" in prefix or "speed_head" in prefix or "residual_head" in prefix
            for prefix in prefixes
        ), (name, prefixes)


def test_eval_runner_task_model_config_drops_plan_anchor() -> None:
    from pipeline.eval_runner import task_model_config

    config = task_model_config({"hidden_dim": 128, "plan_anchor": {"enabled": False}})
    assert "plan_anchor" not in config


def test_phase3_cli_has_no_anchor_loss_flags() -> None:
    source = Path(stages_file()).read_text(encoding="utf-8")
    assert "--phase3-anchor-ce-weight" not in source
    assert "--phase3-anchor-wta-weight" not in source
    # 5k 锚数据集开关保留（与 K-anchor 计划头无关）
    assert "--phase3-anchor" in source
    assert "--phase3-anchor-bc-dir" in source


def stages_file() -> str:
    from pipeline import stages

    return inspect.getsourcefile(stages) or str(ROOT / "pipeline" / "stages.py")


def test_legacy_k_anchor_snapshot_importable() -> None:
    """退役实现仍可从 legacy 快照导入（迁移未丢实现）。"""
    module = importlib.import_module("tools.legacy.k_anchor.anchor")
    assert hasattr(module, "anchor_mixture")
    plan_head = importlib.import_module("tools.legacy.k_anchor.plan_head")
    assert hasattr(plan_head.PlanHead, "plan_anchors")
