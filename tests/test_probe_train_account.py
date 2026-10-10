"""P1-0C 批次 3a 单测：单步训练账本探针（stub 模型，CPU；共享 freeze_audit helper）。

覆盖：
- PASS：trainable ⊆ optimizer（coverage=1.0）、逐组 grad-norm / updated 计数；
- FAIL：可训参数漏出 optimizer、冻结参数被塞进 optimizer；
- 零值可训参数不更新 → 仅信息项（非一致性违规）；
- CLI 默认（recipe=all）+ 真配置集成。
"""

from __future__ import annotations

import json

import torch
from torch import nn

from tools.controls.probe_freeze_audit import audit_model
from tools.controls.probe_train_account import account_ledger, build_parser, main

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
    """与真实模型同名前缀的极小模型（encoder/mem/st_gnn/router/experts/primary/fusion/
    policy/value + other）。"""

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
    return lambda: sum(p.pow(2).mean() for p in model.parameters() if p.requires_grad)


def _ledger(model: nn.Module, *, optimizer=None) -> dict:
    result = audit_model(
        model, FREEZE_STUB, loss_fn=_stub_loss(model), optimizer=optimizer, freeze_mode="specific_only"
    )
    return account_ledger(result)


def test_account_pass_membership_and_group_stats() -> None:
    ledger = _ledger(_StubModel())
    assert ledger["status"] == "PASS", ledger["failures"]
    assert ledger["trainable_not_in_optimizer"] == []
    assert ledger["optimizer_not_trainable"] == []
    assert ledger["optimizer_coverage"] == 1.0
    assert ledger["n_trainable"] == ledger["n_optimizer_params"] > 0
    assert ledger["frozen_grad_nonzero"] == []
    assert ledger["frozen_changed"] == []
    assert ledger["groups"]["experts"]["grad_norm"] > 0
    assert ledger["groups"]["experts"]["changed"] > 0
    assert ledger["groups"]["router"]["changed"] > 0
    assert ledger["groups"]["encoder"]["grad_norm"] == 0.0
    assert ledger["groups"]["encoder"]["changed"] == 0


def test_account_fail_trainable_missing_from_optimizer() -> None:
    from pipeline.trainer import apply_freeze_prefixes

    model = _StubModel()
    apply_freeze_prefixes(model, FREEZE_STUB)
    experts = [
        p for name, p in model.named_parameters() if name.startswith("plan_head.moe.experts.")
    ]
    optimizer = torch.optim.Adam(experts, lr=1e-3)
    ledger = _ledger(model, optimizer=optimizer)
    assert ledger["status"] == "FAIL"
    assert any(name.startswith("plan_head.moe.router.") for name in ledger["trainable_not_in_optimizer"])
    assert ledger["optimizer_coverage"] < 1.0


def test_account_fail_frozen_param_in_optimizer() -> None:
    from pipeline.trainer import apply_freeze_prefixes

    model = _StubModel()
    apply_freeze_prefixes(model, FREEZE_STUB)
    params = dict(model.named_parameters())
    trainable = [p for p in params.values() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable + [params["st_gnn.weight"]], lr=1e-3)
    ledger = _ledger(model, optimizer=optimizer)
    assert ledger["status"] == "FAIL"
    assert "st_gnn.weight" in ledger["optimizer_not_trainable"]


def test_account_zero_param_not_updated_is_informational() -> None:
    model = _StubModel()
    with torch.no_grad():
        model.plan_head.moe.router.bias.zero_()  # 零值可训参数：合成损失梯度=0 → 不更新（非违规）
    ledger = _ledger(model)
    assert ledger["status"] == "PASS", ledger["failures"]
    assert "plan_head.moe.router.bias" in ledger["trainable_not_updated"]
    assert "非一致性违规" in ledger["trainable_not_updated_note"]


def test_cli_default_recipe_and_real_config(tmp_path) -> None:
    args = build_parser().parse_args([])
    assert args.recipe == "all"
    out = tmp_path / "train_account.json"
    assert main(["--recipe", "all", "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["status"] == "PASS", saved["failures"]
    assert saved["optimizer_coverage"] == 1.0
    assert saved["trainable_not_in_optimizer"] == []
    assert saved["frozen_changed"] == []
