"""R38-P2b-D：Policy/path_head 双实体收编（单实体 + 键面迁移 + 开关退役）守卫。

出处：GPT §七-D（六步顺序）+ `docs/deprecation_manifest.yaml` D 组 +
`docs/checkpoint_migration_contract.md`。断言面：

1. **收编归属**：结构化头 = ``model.policy.structured_action``（键面 ``policy.structured_action.*``）；
   ``path_head`` 只读 alias；装配写入（``model.path_head = ...``）⇒ AttributeError fail-fast；
2. **开关退役**：``enable_path_head`` / ``path_head_ablation`` 属性不存在；
   ``set_path_head_ablation`` 降为 legacy 诊断入口（``policy``/``structured`` 两档，oneshot fail-fast）；
3. **一个动作出口**：default/structured 输出的动作源唯一；structured 不产出
   ``policy_action_mu``/``action_logstd``；``set_policy_diag`` ⇒ RuntimeError fail-fast，
   离线工具 ``tools.diagnostics.legacy_policy_diag.diag_policy_head`` 复算与默认路径逐位一致；
4. **契约门不弱化**：structured+rollout 的 ``contract_*`` 全 1.0；
5. **键面迁移器**：``path_head.*`` ↔ ``policy.structured_action.*`` 前缀改名、``policy.log_std →
   policy.logstd``、``meta.path_head_ablation/mode`` 识别；旧结构化 ckpt 经 mapper 可载新装配，
   反向（新键入默认装配）落条件族 drop 并逐族计数；
6. **三子结构**：``output_groups`` 给出 structured_action / action_distribution / auxiliary_intent；
   ``action_distribution(request_log_std=True)`` ⇒ NotImplementedError（均值方差不得来自两个头）。
"""

from __future__ import annotations

import torch
import pytest
import yaml

from net.model import DrivingModel
from net.path_head import StructuredPolicyHead
from tools.legacy.legacy_key_mapper import (
    LEGACY_META_KEYS,
    classify_keys,
    classify_meta,
    renamed_key,
)
from tests.test_net_shapes import make_obs

BATCH = 2


def _obs(structured: bool = False) -> dict[str, torch.Tensor]:
    obs = make_obs(batch=BATCH, seed=0)
    if structured:
        s = torch.linspace(0.0, 40.0, 41)
        obs["lane_anchor_xy"] = (
            torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
        )
        obs["path_dynamic_distance"] = torch.full((BATCH,), 30.0)
        obs["path_dynamic_ttc"] = torch.full((BATCH,), 8.0)
    return obs


def _model(structured: bool = False) -> DrivingModel:
    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    if structured:
        model.set_path_head(True)
    return model.eval()


def test_structured_head_owned_by_policy_entity() -> None:
    model = _model(structured=True)
    assert model.path_head is model.policy.structured_action
    assert isinstance(model.policy.structured_action, StructuredPolicyHead)
    assert model.path_head_mode() == "structured"
    assert model.action_generator_names() == ("policy.structured_action",)
    keys = set(model.state_dict())
    new_keys = {key for key in keys if key.startswith("policy.structured_action.")}
    assert len(new_keys) == 18, sorted(new_keys)
    assert not any(key.startswith("path_head.") for key in keys)
    names = model.path_head_parameter_names()
    assert names and all(name.startswith("policy.structured_action.") for name in names)


def test_path_head_alias_is_read_only() -> None:
    model = _model(structured=True)
    with pytest.raises(AttributeError):
        model.path_head = None  # type: ignore[misc]


def test_mirror_switch_and_mode_enum_removed() -> None:
    model = _model()
    assert not hasattr(model, "enable_path_head")
    assert not hasattr(model, "path_head_ablation")
    # legacy 诊断入口（两档 + oneshot fail-fast）
    model.set_path_head_ablation("structured")
    assert model.path_head_mode() == "policy"  # 头未构造 ⇒ 派生 mode 仍是 policy
    model.set_path_head(True)
    assert model.path_head_mode() == "structured"
    model.set_path_head_ablation("policy")
    assert model.path_head_mode() == "policy"
    with pytest.raises(ValueError, match="R38-P1 减枝删除"):
        model.set_path_head_ablation("oneshot")
    with pytest.raises(ValueError, match="必须是"):
        model.set_path_head_ablation("bogus")


def test_single_action_exit_and_contracts() -> None:
    default = _model()
    out_default = default(_obs())
    assert "action_logstd" in out_default  # 默认关路径：旧头出口
    assert "policy_action_mu" not in out_default
    assert float(out_default["action_mu"].shape[-1]) == 2

    model = _model(structured=True)
    out = model(_obs(structured=True))
    assert "action_logstd" not in out  # structured：旧头不前向
    assert "policy_action_mu" not in out
    assert float(out["contract_runtime_ok"]) == 1.0
    assert float(out["contract_policy_call_count_is_six"]) == 1.0
    assert float(out["contract_world_model_call_count_is_five"]) == 1.0
    assert float(out["contract_deployment_does_not_use_plan_override"]) == 1.0
    assert bool(torch.equal(out["plan"][:, 0, :], out["action_mu"]))


def test_set_policy_diag_fail_fast_points_to_offline_tool() -> None:
    model = _model(structured=True)
    with pytest.raises(RuntimeError, match="legacy_policy_diag"):
        model.set_policy_diag(True)


def test_offline_policy_diag_matches_default_path() -> None:
    from tools.diagnostics.legacy_policy_diag import diag_policy_head

    model = _model()
    obs = _obs()
    offline = diag_policy_head(model, obs)
    assert torch.equal(offline["policy_action_mu"], offline["action_mu_forward"])
    assert offline["action_logstd"].shape == (BATCH, 2)


def test_output_groups_three_substructures() -> None:
    model = _model(structured=True)
    head = model.policy.structured_action
    s = torch.linspace(0.0, 40.0, 41)
    anchor = torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
    with torch.no_grad():
        out = head(torch.zeros(BATCH, 32), anchor_xy=anchor)
    groups = head.output_groups(out)
    assert set(groups) == {"structured_action", "action_distribution", "auxiliary_intent"}
    assert groups["structured_action"]["action_mu"] is out["action_mu"]
    assert set(head.action_distribution(out)) == {"action_mu"}
    assert "kappa_mid" in groups["auxiliary_intent"]
    with pytest.raises(NotImplementedError, match="同源"):
        head.action_distribution(out, request_log_std=True)


def test_key_mapper_dual_entity_surface() -> None:
    assert renamed_key("policy.log_std") == "policy.logstd"
    assert renamed_key("path_head.lateral.0.weight") == "policy.structured_action.lateral.0.weight"
    assert tuple(LEGACY_META_KEYS) == ("path_head_ablation", "path_head_mode")
    meta = classify_meta({"path_head_ablation": "oneshot", "path_head_mode": "structured"})
    assert meta["oneshot_declared"] is True
    assert set(meta["recognized"]) == {"path_head_ablation", "path_head_mode"}

    # 旧结构化 ckpt（path_head.*）→ 新结构化装配：18 键全部 rename、无 missing
    model = _model(structured=True)
    model_keys = tuple(model.state_dict().keys())
    old_style = tuple(
        "path_head." + key[len("policy.structured_action."):]
        if key.startswith("policy.structured_action.")
        else key
        for key in model_keys
    )
    plan = classify_keys(old_style, model_keys)
    assert len(plan.rename) == 18
    assert plan.dropped_by_family == {}
    assert not plan.reject and not plan.missing
    assert plan.renamed_by_family == {"path_head.->policy.structured_action.": 18}

    # 旧结构化 ckpt → 默认装配：条件族 drop（装配缺实体）+ 逐族计数（不静默、不 reject）
    plan_default = classify_keys(old_style, model.state_dict().keys() if False else tuple(_model().state_dict().keys()))
    assert plan_default.dropped_by_family == {"path_head.->policy.structured_action.(assembly_missing)": 18}
    assert not plan_default.reject


def test_structured_assembly_matches_config_build() -> None:
    from pipeline.stages import build_model

    cfg = yaml.safe_load(open("config/model.yaml", encoding="utf-8"))
    model = build_model(cfg)
    default_keys = set(model.state_dict())
    model.set_path_head(True)
    structured_keys = set(model.state_dict())
    delta = structured_keys - default_keys
    assert len(delta) == 18
    assert all(key.startswith("policy.structured_action.") for key in delta)
