"""R38-P2a：``tools.legacy.legacy_key_mapper`` + ``checkpoint_migration`` 单测（ckpt 兼容）。

覆盖（`docs/checkpoint_migration_contract.md` §4.2/§5/§6-1）：

- B 组 ignore 集 = 20 键（K-anchor 7 前缀 16 键 + ``encoders.lane/ttc`` 4 键），逐族计数；
- 干净 ckpt：零丢弃 / 零拒绝；
- 未登记 unexpected → ``reject``（默认 fail-fast 保持）；missing → 交调用方；
- ``migrate_state_dict`` 只丢已登记族且保持其余张量逐位不变；
- ``load_checkpoint_state`` 默认拒载（未登记键）、``on_partial="record"`` 显式逃生门；
- 真实 K-anchor ckpt 键面（若本机存在）：16 键逐族命中。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tools.legacy.checkpoint_migration import LoadError, load_checkpoint_state, migrate_state_dict
from tools.legacy.legacy_key_mapper import (
    IGNORE_PREFIXES,
    classify_keys,
    family_of,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
K_ANCHOR_CKPT = ROOT / "runs" / "BTC20261005-0856_v7struct_v5" / "stage_b" / "final.pt"
CLEAN_CKPT = ROOT / "runs" / "BTC20261009-0824_sw_pri512" / "stage_b" / "final.pt"

#: 代表 K-anchor ckpt 的带键族（16 键；c5_ckpt_keys.json 实测）
K_ANCHOR_KEYS = (
    "plan_head.anchors",
    "plan_head.anchor_ds",
    "plan_head.anchor_dth",
    "plan_head.anchor_embed",
    "plan_head.anchor_head.0.weight",
    "plan_head.anchor_head.0.bias",
    "plan_head.anchor_head.2.weight",
    "plan_head.anchor_head.2.bias",
    "plan_head.residual_head.0.weight",
    "plan_head.residual_head.0.bias",
    "plan_head.residual_head.2.weight",
    "plan_head.residual_head.2.bias",
    "plan_head.speed_head.0.weight",
    "plan_head.speed_head.0.bias",
    "plan_head.speed_head.2.weight",
    "plan_head.speed_head.2.bias",
)
LANE_TTC_KEYS = (
    "encoders.lane.weight",
    "encoders.lane.bias",
    "encoders.ttc.weight",
    "encoders.ttc.bias",
)

_MODEL_KEYS = (
    "policy.mu.weight",
    "policy.mu.bias",
    "plan_head.fusion.0.weight",
    "plan_head.moe.router.0.weight",
)


def _synthetic_state() -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key in K_ANCHOR_KEYS + LANE_TTC_KEYS:
        state[key] = torch.full((2, 2), 7.0)
    for index, key in enumerate(_MODEL_KEYS):
        state[key] = torch.full((2, 2), float(index))
    return state


def test_ignore_prefixes_is_b_group_set() -> None:
    assert len(IGNORE_PREFIXES) == 9
    assert IGNORE_PREFIXES[:4] == (
        "plan_head.anchors",
        "plan_head.anchor_ds",
        "plan_head.anchor_dth",
        "plan_head.anchor_embed",
    )
    assert "encoders.lane." in IGNORE_PREFIXES and "encoders.ttc." in IGNORE_PREFIXES


def test_classify_drops_b_group_keys_by_family() -> None:
    plan = classify_keys(_synthetic_state().keys(), _MODEL_KEYS)
    assert plan.ok_without_escape
    assert len(plan.drop) == 20
    assert plan.load == _MODEL_KEYS
    assert not plan.reject and not plan.missing
    assert plan.dropped_by_family == {
        "plan_head.anchors": 1,
        "plan_head.anchor_ds": 1,
        "plan_head.anchor_dth": 1,
        "plan_head.anchor_embed": 1,
        "plan_head.anchor_head.": 4,
        "plan_head.residual_head.": 4,
        "plan_head.speed_head.": 4,
        "encoders.lane.": 2,
        "encoders.ttc.": 2,
    }
    summary = summarize(plan)
    assert summary["n_drop"] == 20 and summary["ok_without_escape"] is True


def test_classify_clean_ckpt_has_zero_drop_and_reject() -> None:
    state = {key: torch.zeros(1) for key in _MODEL_KEYS}
    plan = classify_keys(state.keys(), _MODEL_KEYS)
    assert not plan.drop and not plan.reject and not plan.missing
    assert family_of("policy.mu.weight") is None
    assert family_of("plan_head.anchor_head.0.weight") == "plan_head.anchor_head."


def test_classify_unregistered_unexpected_is_rejected() -> None:
    state = {key: torch.zeros(1) for key in _MODEL_KEYS}
    state["mystery_head.weight"] = torch.zeros(1)
    plan = classify_keys(state.keys(), _MODEL_KEYS)
    assert plan.reject == ("mystery_head.weight",)
    assert not plan.ok_without_escape


def test_migrate_state_dict_preserves_kept_values() -> None:
    state = _synthetic_state()
    filtered, report = migrate_state_dict(state, _MODEL_KEYS)
    assert set(filtered) == set(_MODEL_KEYS)
    for key in _MODEL_KEYS:
        assert torch.equal(filtered[key], state[key]), key
    assert report.n_drop == 20 and report.n_reject == 0 and report.n_missing == 0
    assert report.dropped_by_family["plan_head.anchor_head."] == 4
    assert len(report.dropped_keys) == 20


def test_load_checkpoint_state_fail_fast_and_escape() -> None:
    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mu = nn.Linear(2, 2)

    model = _Toy()
    clean = {"mu.weight": torch.ones(2, 2), "mu.bias": torch.zeros(2)}
    report = load_checkpoint_state(model, clean)
    assert report.n_drop == 0 and report.n_reject == 0 and report.n_missing == 0
    assert torch.allclose(model.mu.weight, torch.ones(2, 2))

    # 未登记 unexpected → 默认 fail-fast（模型不得被部分加载）
    dirty = dict(clean)
    dirty["mystery.weight"] = torch.ones(2, 2)
    with pytest.raises(LoadError, match="fail-fast"):
        load_checkpoint_state(_Toy(), dirty)

    # 已登记族 → 默认放行且逐族计数（这正是 B 组旧 ckpt 的装载路径）
    legacy = dict(clean)
    legacy["plan_head.anchors"] = torch.ones(6, 6, 2)
    legacy["plan_head.speed_head.0.weight"] = torch.ones(2, 2)
    report = load_checkpoint_state(_Toy(), legacy)
    assert report.n_drop == 2 and not report.explicit_escape
    assert report.dropped_by_family == {"plan_head.anchors": 1, "plan_head.speed_head.": 1}

    # 显式逃生门：无登记键但缺模型键 → on_partial 放行并记录
    report = load_checkpoint_state(_Toy(), {"mu.weight": torch.ones(2, 2)}, on_partial="record")
    assert report.explicit_escape is True and report.n_missing == 1


@pytest.mark.skipif(not K_ANCHOR_CKPT.exists(), reason="代表 K-anchor ckpt 不在本机")
def test_real_k_anchor_ckpt_key_faces() -> None:
    payload = torch.load(K_ANCHOR_CKPT, map_location="cpu", weights_only=False)
    state = payload["model"] if "model" in payload else payload
    plan = classify_keys(state.keys(), ("plan_head.fusion.0.weight",))
    assert len(plan.drop) == 20, plan.dropped_by_family
    assert plan.dropped_by_family["plan_head.anchor_head."] == 4
    assert plan.dropped_by_family["plan_head.speed_head."] == 4
    assert plan.dropped_by_family["plan_head.residual_head."] == 4
    assert plan.dropped_by_family["encoders.lane."] == 2
    assert plan.dropped_by_family["encoders.ttc."] == 2


@pytest.mark.skipif(not CLEAN_CKPT.exists(), reason="正式基线 ckpt 不在本机")
def test_real_clean_ckpt_has_no_registered_drops() -> None:
    payload = torch.load(CLEAN_CKPT, map_location="cpu", weights_only=False)
    state = payload["model"] if "model" in payload else payload
    plan = classify_keys(state.keys(), state.keys())
    assert plan.drop == () and plan.reject == () and plan.missing == ()


# --------------------------------------------------------------------------- #
# R38-P2b（D 组双实体收编）：前缀改名 / meta 键 / 装载入口
# --------------------------------------------------------------------------- #
_STRUCT_OLD_KEYS = (
    "path_head.lateral.0.weight",
    "path_head.lateral.0.bias",
    "path_head.lateral_geom.normalization",
)
_STRUCT_NEW_KEYS = (
    "policy.structured_action.lateral.0.weight",
    "policy.structured_action.lateral.0.bias",
    "policy.structured_action.lateral_geom.normalization",
)


def test_dual_entity_prefix_rename_and_assembly_missing() -> None:
    from tools.legacy.legacy_key_mapper import PREFIX_RENAME_MAP, renamed_key

    assert PREFIX_RENAME_MAP == (("path_head.", "policy.structured_action."),)
    assert renamed_key("path_head.lateral.0.weight") == "policy.structured_action.lateral.0.weight"

    plan = classify_keys(_STRUCT_OLD_KEYS, _STRUCT_NEW_KEYS)
    assert plan.rename == dict(zip(_STRUCT_OLD_KEYS, _STRUCT_NEW_KEYS))
    assert plan.renamed_by_family == {"path_head.->policy.structured_action.": 3}
    assert not plan.reject and not plan.drop and not plan.missing
    assert plan.ok_without_escape

    # 目标装配缺席（默认模型无结构化头）⇒ 条件族 drop + 逐族计数（不静默、不 reject）
    plain_plan = classify_keys(_STRUCT_OLD_KEYS, ("policy.mu.weight",))
    assert len(plain_plan.drop) == 3 and not plain_plan.reject
    assert plain_plan.dropped_by_family == {
        "path_head.->policy.structured_action.(assembly_missing)": 3
    }
    assert plain_plan.reasons["path_head.lateral.0.weight"].startswith("rename_target_absent:")


def test_log_std_rename_and_meta_keys() -> None:
    from tools.legacy.legacy_key_mapper import LEGACY_META_KEYS, classify_meta

    plan = classify_keys(("policy.log_std", "policy.trunk.0.weight"), ("policy.logstd", "policy.trunk.0.weight"))
    assert plan.rename == {"policy.log_std": "policy.logstd"}
    assert plan.renamed_by_family == {"policy.log_std->policy.logstd": 1}
    assert not plan.missing  # 改名目标已计入，不计 missing

    # 目标缺席（当前模型为 policy.logstd.weight/bias，无裸键）⇒ 保持 P2a 口径 = reject（fail-fast）
    absent = classify_keys(("policy.log_std",), ("policy.logstd.weight", "policy.logstd.bias"))
    assert absent.reject == ("policy.log_std",)
    assert absent.reasons["policy.log_std"].startswith("rename_target_absent:")

    assert tuple(LEGACY_META_KEYS) == ("path_head_ablation", "path_head_mode")
    meta = classify_meta({"path_head_ablation": "oneshot", "path_head_mode": "oneshot", "other": 1})
    assert meta["oneshot_declared"] is True
    assert meta["recognized"] == {"path_head_ablation": "oneshot", "path_head_mode": "oneshot"}
    assert classify_meta({})["oneshot_declared"] is False


def test_migrate_state_dict_applies_rename_values() -> None:
    state = {
        "path_head.lateral.0.weight": torch.full((2, 2), 3.5),
        "policy.mu.weight": torch.full((2, 2), 1.0),
    }
    filtered, report = migrate_state_dict(state, _STRUCT_NEW_KEYS + ("policy.mu.weight",))
    assert set(filtered) == {"policy.structured_action.lateral.0.weight", "policy.mu.weight"}
    assert torch.equal(filtered["policy.structured_action.lateral.0.weight"], torch.full((2, 2), 3.5))
    assert report.n_rename == 1 and report.rename_map == {
        "path_head.lateral.0.weight": "policy.structured_action.lateral.0.weight"
    }


def test_load_checkpoint_migrated_round_trip(tmp_path: Path) -> None:
    from tools.legacy.checkpoint_migration import load_checkpoint_migrated

    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.policy = nn.Module()
            self.policy.structured_action = nn.Linear(2, 2)

    model = _Toy()
    new_state = model.state_dict()
    old_state = {
        ("path_head." + key[len("policy.structured_action."):]): value.clone()
        for key, value in new_state.items()
    }
    path = tmp_path / "old_struct.pt"
    torch.save({"model": old_state, "meta": {"path_head_ablation": "structured"}}, path)
    fresh = _Toy()
    info = load_checkpoint_migrated(str(path), fresh)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert info["migration"]["n_rename"] == 2
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key]), key
