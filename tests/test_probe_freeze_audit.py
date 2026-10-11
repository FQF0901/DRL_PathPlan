"""P1-0C 批次 3a 单测：freeze recipe 审计探针（stub 模型，CPU；不建 env/数据集）。

覆盖：
- 静态层 ``parse_freeze_constants``：``AnnAssign`` + ``Tuple + Name`` 拼接解析；
- 四计数判定：PASS（freeze 组全零 / train 组 a>0、d>0）；
- FAIL 案例：冻结参数被塞进 optimizer、可训参数漏出 optimizer、组内冻结/可训混合；
- ``recovery`` / ``recovery_experts`` 配方 → 四计数审计（P1-2 已实现；`RECOVERY_SUGGESTION` 为对照清单）；
- 真配置集成（``config/default.yaml``，specific_only）。
"""

from __future__ import annotations

import json

import torch
from torch import nn

from tools.controls.probe_freeze_audit import (
    audit_model,
    build_parser,
    main,
    parse_freeze_constants,
    run_probe,
)

#: 与 phase3 specific_only 同形的 stub 冻结清单（只训 experts/router/residual_scale）
FREEZE_STUB = (
    "encoders.",
    "mem_encoder.",
    "st_gnn.",
    "plan_head.fusion.",
    "plan_head.moe.primary.",
    "policy.",
    "value.",
)


class _StubModel(nn.Module):
    """与真实模型同名前缀的极小模型（覆盖 encoder/mem/st_gnn/router/experts/primary/
    fusion/policy/value + other(norm/ego_next)）。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.Linear(4, 4)
        self.mem_encoder = nn.Linear(4, 4)
        self.st_gnn = nn.Linear(4, 4)
        self.value = nn.Linear(4, 4)
        self.policy = nn.Linear(4, 4)
        self.plan_head = nn.Module()
        self.plan_head.fusion = nn.Linear(4, 4)
        self.plan_head.norm = nn.LayerNorm(4)
        self.plan_head.ego_next = nn.Linear(4, 4)
        self.plan_head.moe = nn.Module()
        self.plan_head.moe.primary = nn.Linear(4, 4)
        self.plan_head.moe.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.plan_head.moe.router = nn.Linear(4, 2)
        self.plan_head.moe.residual_scale = nn.Parameter(torch.ones(1))


def _stub_loss(model: nn.Module):
    """stub 合成损失：可训参数平方均值（对所有可训参数产生非零梯度）。"""
    return lambda: sum(p.pow(2).mean() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# 静态层
# --------------------------------------------------------------------------- #

def test_parse_freeze_constants_supports_annassign_and_concat(tmp_path) -> None:
    source = tmp_path / "stages_stub.py"
    source.write_text(
        "from typing import Tuple\n"
        '_PHASE3_ANCHOR_HEAD_PREFIXES: Tuple[str, ...] = ("a.",)\n'
        '_SPECIFIC_PHASE_FREEZE: Tuple[str, ...] = ("b.",) + _PHASE3_ANCHOR_HEAD_PREFIXES\n'
        '_PHASE3_SAFE_FREEZE = ("c.",) + _SPECIFIC_PHASE_FREEZE\n',
        encoding="utf-8",
    )
    constants = parse_freeze_constants(source)
    assert constants["_PHASE3_ANCHOR_HEAD_PREFIXES"] == ("a.",)
    assert constants["_SPECIFIC_PHASE_FREEZE"] == ("b.", "a.")
    assert constants["_PHASE3_SAFE_FREEZE"] == ("c.", "b.", "a.")


# --------------------------------------------------------------------------- #
# 四计数：PASS / FAIL
# --------------------------------------------------------------------------- #

def test_audit_stub_specific_only_pass() -> None:
    model = _StubModel()
    result = audit_model(model, FREEZE_STUB, loss_fn=_stub_loss(model), freeze_mode="specific_only")
    assert result["status"] == "PASS", result["failures"]
    groups, expected = result["groups"], result["expected"]
    assert expected["encoder"] == "freeze"
    assert expected["experts"] == "train"
    assert expected["router"] == "train"
    assert expected["anchor"] == "empty"
    for group in ("encoder", "mem", "st_gnn", "fusion", "plan_head.primary", "policy", "value"):
        stat = groups[group]
        assert stat["n_params"] > 0
        assert stat["requires_grad"] == 0
        assert stat["in_optimizer"] == 0
        assert stat["grad_nonzero"] == 0
        assert stat["changed"] == 0
    assert groups["experts"]["requires_grad"] == groups["experts"]["in_optimizer"] > 0
    assert groups["experts"]["changed"] > 0
    assert groups["router"]["requires_grad"] == groups["router"]["in_optimizer"] > 0
    assert groups["router"]["changed"] > 0


def test_audit_fail_frozen_param_forced_into_optimizer() -> None:
    from pipeline.trainer import apply_freeze_prefixes

    model = _StubModel()
    apply_freeze_prefixes(model, FREEZE_STUB)
    params = dict(model.named_parameters())
    trainable = [p for p in params.values() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable + [params["encoders.weight"]], lr=1e-3)
    result = audit_model(
        model, FREEZE_STUB, loss_fn=_stub_loss(model), optimizer=optimizer, freeze_mode="specific_only"
    )
    assert result["status"] == "FAIL"
    assert "encoders.weight" in result["optimizer_not_trainable"]
    assert any("optimizer 含不可训参数" in failure for failure in result["failures"])


def test_audit_fail_trainable_missing_from_optimizer() -> None:
    from pipeline.trainer import apply_freeze_prefixes

    model = _StubModel()
    apply_freeze_prefixes(model, FREEZE_STUB)
    params = dict(model.named_parameters())
    experts = [p for name, p in params.items() if name.startswith("plan_head.moe.experts.")]
    optimizer = torch.optim.Adam(experts, lr=1e-3)  # 故意漏掉 router/residual_scale
    result = audit_model(
        model, FREEZE_STUB, loss_fn=_stub_loss(model), optimizer=optimizer, freeze_mode="specific_only"
    )
    assert result["status"] == "FAIL"
    assert any(name.startswith("plan_head.moe.router.") for name in result["trainable_not_in_optimizer"])
    assert any("可训参数不在 optimizer" in failure for failure in result["failures"])


def test_audit_fail_mixed_group() -> None:
    model = _StubModel()
    prefixes = FREEZE_STUB + ("plan_head.moe.experts.0.",)  # 只冻一个专家 → 组内混合
    result = audit_model(model, prefixes, loss_fn=_stub_loss(model), freeze_mode="specific_only")
    assert result["expected"]["experts"] == "mixed"
    assert result["status"] == "FAIL"
    assert any("mixed" in failure for failure in result["failures"])


# --------------------------------------------------------------------------- #
# recovery / CLI / 真配置集成
# --------------------------------------------------------------------------- #

def test_run_probe_recovery_recipes_real_config(tmp_path) -> None:
    """recovery（R1）/ recovery_experts（R2）已实现（P1-2）→ 四计数 PASS（真配置）。"""
    for recipe in ("recovery", "recovery_experts"):
        payload = run_probe(recipe, out=tmp_path / f"{recipe}.json", quiet=True)
        assert payload["status"] == "PASS", (recipe, payload["failures"])
        assert payload["static_match_runtime"] is True
        groups = payload["groups"]
        for group in ("encoder", "mem", "st_gnn", "router", "policy", "value"):
            assert groups[group]["expected"] == "freeze"
            assert groups[group]["changed"] == 0
        assert groups["fusion"]["expected"] == "train"
        assert groups["fusion"]["changed"] > 0
        assert groups["plan_head.primary"]["expected"] == "train"
        assert groups["plan_head.primary"]["changed"] > 0
        if recipe == "recovery":
            assert groups["experts"]["expected"] == "freeze"
            assert groups["experts"]["changed"] == 0
        else:
            assert groups["experts"]["expected"] == "train"
            assert groups["experts"]["changed"] > 0
    # CLI 退出码：PASS = 0
    assert main(["--recipe", "recovery", "--out", str(tmp_path / "recovery_cli.json")]) == 0


def test_cli_parser_defaults() -> None:
    args = build_parser().parse_args(["--recipe", "specific_only"])
    assert args.recipe == "specific_only"
    assert args.config == "config/default.yaml"
    assert args.device == "cpu"
    assert args.batch_size == 4


def test_run_probe_specific_only_real_config(tmp_path) -> None:
    payload = run_probe("specific_only", out=tmp_path / "freeze_audit.json", quiet=True)
    assert payload["status"] == "PASS", payload["failures"]
    assert payload["static_match_runtime"] is True
    assert payload["freeze_prefixes"]  # 非空
    groups = payload["groups"]
    assert groups["encoder"]["expected"] == "freeze"
    assert groups["encoder"]["changed"] == 0
    assert groups["experts"]["expected"] == "train"
    assert groups["experts"]["changed"] > 0
