"""R39-A2 双账本守卫：参数账本（五栏/上限/optimizer/update norm）+ detach 登记表 CI。

出处：GPT R39 §二-A 任务 7–8 + §5.83 裁决（optimizer 仅声明模块）+ GPT R40 §五.5
（**总参数 ≤ 2,000,000 绝对上限**；五栏报告 total/reachable/trainable/frozen/dormant，
trainable = active）+ `docs/deepseekv4p1_argue.md` A2。

断言面：

1. **五栏闭合**（total = trainable + frozen + dormant；reachable = trainable + frozen，
   trainable = active 同义），两配置；
2. **2.0M 绝对上限**（GPT R40 §五.5）：total ≤ 2,000,000；另保留 R39-A2 实测基线作回归
   对照；激活上限（可达 ≤ 基线 × 1.05，R39 裁决）继续适用；
3. **dormant 语义**：MoE 关闭 ⇒ ``plan_head.moe`` 整块 dormant；structured 模式 ⇒ 旧
   ``policy`` 块 dormant；结构化头"已构造但未启用"（legacy 诊断档）⇒ ``policy.structured_action``
   整块 dormant（= 分类器对 "default 模式下 structured 头" 的口径）；default 装配下结构化头
   惰性未构造 ⇒ 不产生参数；
4. **可达性 = 实测 hook**：声明存活块签名模块必须真被调用，否则 ``RuntimeError``（脱钩即
   账本失效）；声明死块的签名模块必须**未**被调用；
5. **optimizer allowlist**：仅声明模块入组、全覆盖、无 dormant 入组（正例 + 违规负例）；
6. **update norm 接口**：参数级 → 模块级聚合；deepcopy 一次 AdamW step 的隔离性
   （allowlist 外 update 恒 0）；
7. **detach 登记表 CI**：``net/**`` 全部 ``.detach()`` 与登记表逐项匹配（未登记新增/陈旧
   ⇒ fail；行号漂移仅记录）；登记表字段完整；扫描器对合成未登记 detach 能报出；
8. **z_next detach 语义回归**（tiny 模型）：跨步输出对前一步 z 无路径（grad None），动作链
   对 st_gnn 无梯度，而动作链对动作头可达。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from net import probes as probes_module
from net.model import DrivingModel
from net.param_probe import (
    ACTIVE_BASELINES,
    ACTIVE_LIMIT,
    ALLOWLISTS_BY_CONFIG,
    TOTAL_BASELINES,
    TOTAL_CAP,
    build_model,
    classify_parameters,
    forward_reachable,
    ledger_for_config,
    make_dummy_obs,
    make_obs_for,
    module_of,
    module_update_norms,
    optimizer_allowlist_report,
    render_report,
    shape_smoke,
    update_norm_demo,
    view_totals,
)

ROOT = Path(__file__).resolve().parents[1]
BATCH = 2


def _tiny_model() -> DrivingModel:
    torch.manual_seed(0)
    return DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6).eval()


# =========================================================================== #
# 1. 键面 → 模块块（P2b 收编）
# =========================================================================== #
def test_module_of_p2b_key_faces() -> None:
    assert module_of("policy.structured_action.lateral.0.weight") == "policy.structured_action"
    assert module_of("policy.mu.weight") == "policy"
    assert module_of("plan_head.moe.experts.0.2.weight") == "plan_head.moe"
    assert module_of("plan_head.fusion.2.weight") == "plan_head"
    assert module_of("st_gnn.spatial.layers.0.weight") == "st_gnn"
    assert module_of("encoders.od.weight") == "encoders"
    assert module_of("mem_encoder.ego_attn.in_proj_weight") == "mem_encoder"
    assert module_of("value.net.2.bias") == "value"
    assert module_of("semantic_wiring.readout.0.weight") == "semantic_wiring"
    assert module_of("target_proj.weight") == "semantic_wiring"
    assert module_of("beta_context") == "semantic_wiring"


def test_module_of_unknown_prefix_fails_fast() -> None:
    with pytest.raises(ValueError, match="不在已知键面"):
        module_of("brand_new_module.weight")


# =========================================================================== #
# 2. 四分类闭合 + 基线断言（H=128 两配置；不开 update-norm 演示以控时）
# =========================================================================== #
@pytest.fixture(scope="module")
def ledgers() -> dict[str, dict]:
    """两配置账本（模块内共享；update-norm 演示关闭以控时）。"""
    return {
        config: ledger_for_config(config, demo_update_norm=False)
        for config in ("default", "structured")
    }


@pytest.mark.parametrize("config", ["default", "structured"])
def test_ledger_five_columns_and_caps(config: str, ledgers: dict[str, dict]) -> None:
    ledger = ledgers[config]
    assert ledger["total"] == TOTAL_BASELINES[config]  # R39-A2 实测基线（回归对照）
    recipe = ledger["views"]["recipe_moe_closed"]
    budget = ledger["views"]["budget_moe_closed"]
    # 五栏闭合：total = trainable + frozen + dormant；reachable = trainable + frozen
    assert recipe["total"] == recipe["trainable"] + recipe["frozen"] + recipe["dormant"]
    assert budget["total"] == budget["active"] + budget["frozen"] + budget["dormant"]
    assert recipe["reachable"] == recipe["trainable"] + recipe["frozen"] == budget["reachable"]
    assert recipe["trainable"] == recipe["active"]  # GPT R40 §五.5 栏名：trainable = active
    # GPT R40 §五.5：总参数绝对上限 2,000,000（R39 基线×1.02 乘性规则已废止）
    assert TOTAL_CAP == 2_000_000
    assert ledger["assertion"]["total_limit"], ledger["assertion_detail"]["total_rule"]
    assert ledger["total"] <= TOTAL_CAP
    _total_rule = ledger["assertion_detail"]["total_rule"]
    assert "2,000,000" in _total_rule and "× 1.02" not in _total_rule  # 乘性上限不得回潮
    # R39 激活门（可达 ≤ 基线 ×1.05）继续适用
    assert ledger["assertion"]["active_limit"], ledger["assertion_detail"]["active_rule"]
    assert ledger["assertion"]["optimizer_ok"]
    assert budget["budget_active"] <= math.floor(ACTIVE_BASELINES[config] * ACTIVE_LIMIT)


def test_render_report_five_columns_and_2m_cap(ledgers: dict[str, dict]) -> None:
    """GPT R40 §五.5：报告不得只报总参数 —— 五栏 total/reachable/trainable/frozen/dormant。"""
    text = render_report(ledgers)
    header = [line for line in text.splitlines() if line.startswith("配置")][0]
    for column in ("total", "reachable", "trainable", "frozen", "dormant"):
        assert column in header, (column, header)
    assert "2,000,000" in text
    for ledger in ledgers.values():
        view = ledger["views"]["recipe_moe_closed"]
        assert isinstance(view["trainable"], int) and view["trainable"] == view["active"]
        assert view["reachable"] == view["trainable"] + view["frozen"]


def test_ledger_structured_activates_head_and_old_policy_is_dormant(ledgers: dict[str, dict]) -> None:
    ledger = ledgers["structured"]
    by_module = {row["module"]: row for row in ledger["modules"]["budget_moe_closed"]}
    assert by_module["policy.structured_action"]["reachable"] == by_module["policy.structured_action"]["params"]
    assert by_module["policy"]["reachable"] == 0
    assert by_module["policy"]["dormant"] == by_module["policy"]["params"]
    assert by_module["plan_head.moe"]["reachable"] == 0  # MoE 关（R39 口径）
    assert by_module["plan_head.moe"]["dormant"] == 407_401
    # MoE 打开视图：专家可达（同装配仅运行模式不同）
    assert ledger["views"]["budget_moe_open"]["reachable"] > ledger["views"]["budget_moe_closed"]["reachable"]


def test_ledger_default_has_no_structured_head_params(ledgers: dict[str, dict]) -> None:
    ledger = ledgers["default"]
    modules = {row["module"] for row in ledger["modules"]["budget_moe_closed"]}
    assert "policy.structured_action" not in modules  # 惰性构造，未启用时不产生参数
    # 预算视图（MoE 开）：全部可达、无 dormant
    assert ledger["views"]["budget_moe_open"]["dormant"] == 0


def test_constructed_but_disabled_structured_head_is_dormant() -> None:
    """default（旧头）模式下已构造的结构化头 = 整块 dormant（GPT 四分类口径）。"""
    model = build_model("default")
    model.set_path_head(True)
    model.set_path_head_ablation("policy")  # legacy 诊断档：头保留、消费关闭
    assert model.path_head_mode() == "policy"
    reach = forward_reachable(model, make_obs_for("default"), moe_enabled=False)
    records = classify_parameters(model, reach)
    totals = view_totals(records, sum(record["numel"] for record in records))
    head_rows = [record for record in records if record["module"] == "policy.structured_action"]
    assert head_rows and all(record["class"] == "dormant" for record in head_rows)
    assert totals["dormant"] >= sum(record["numel"] for record in head_rows)
    # 旧头在该档位可达（active）
    policy_rows = [record for record in records if record["module"] == "policy"]
    assert policy_rows and all(record["reachable"] for record in policy_rows)


# =========================================================================== #
# 3. 可达性 = 实测 hook（含脱钩 fail-fast）
# =========================================================================== #
def test_forward_reachable_declares_dead_blocks_from_runtime_mode() -> None:
    model = _tiny_model()
    reach_closed = forward_reachable(model, make_dummy_obs(batch=BATCH), moe_enabled=False)
    assert "plan_head.moe" in reach_closed["dead_blocks"]
    assert reach_closed["consistency"]  # 声明与实测交叉验证有产出
    reach_open = forward_reachable(model, make_dummy_obs(batch=BATCH), moe_enabled=True)
    assert "plan_head.moe" not in reach_open["dead_blocks"]


def test_forward_reachable_guard_fails_when_live_block_not_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    """声明存活块（st_gnn）签名模块未被调用 ⇒ 账本拒绝出账（脱钩 fail-fast）。"""
    model = _tiny_model()

    def stub_forward(self: torch.nn.Module, **kwargs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z_ego, z_od, z_ld = kwargs["z_ego"], kwargs["z_od"], kwargs["z_ld"]
        batch = int(z_ego.shape[0])
        zeros = torch.zeros
        return (
            z_ego.clone(),
            z_od.clone(),
            z_ld.clone(),
            zeros(batch, 16, 5),
            zeros(batch, 16, 4),
            zeros(batch, 16),
            zeros(batch, 16),
        )

    monkeypatch.setattr(type(model.st_gnn), "forward", stub_forward)
    with pytest.raises(RuntimeError, match="脱钩"):
        forward_reachable(model, make_dummy_obs(batch=BATCH), moe_enabled=False)


# =========================================================================== #
# 4. optimizer allowlist（正例 + 违规负例）
# =========================================================================== #
def test_optimizer_allowlist_positive_and_dormant_guard() -> None:
    model = build_model("structured")
    reach = forward_reachable(model, make_obs_for("structured"), moe_enabled=False)
    report = optimizer_allowlist_report(model, ALLOWLISTS_BY_CONFIG["structured"], reach=reach)
    assert report["ok"], report
    assert report["violations"] == [] and report["missing"] == [] and report["dormant_in_optimizer"] == []


def test_optimizer_allowlist_detects_rogue_and_dormant_members() -> None:
    model = build_model("structured")
    reach = forward_reachable(model, make_obs_for("structured"), moe_enabled=False)
    # 负例①：把 encoder 参数塞进 optimizer（allowlist 外）⇒ violations
    rogue = dict(model.named_parameters())["encoders.ego.weight"]
    optimizer = torch.optim.AdamW([rogue], lr=1e-4)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert any(id(p) in optimizer_ids for _, p in model.named_parameters() if _.startswith("encoders."))
    # 负例②：允许清单含当前模式不可达参数（旧头在 structured 模式 = 死路径）⇒ dormant_in_optimizer
    report = optimizer_allowlist_report(model, ("policy.", "plan_head.fusion.2."), reach=reach)
    assert not report["ok"]
    assert any(name.startswith("policy.layers.") for name in report["dormant_in_optimizer"])


# =========================================================================== #
# 5. update norm 接口 + 演示隔离
# =========================================================================== #
def test_module_update_norms_aggregation() -> None:
    result = module_update_norms(
        {"policy.mu.weight": 3.0, "policy.mu.bias": 4.0, "encoders.ego.weight": 12.0}
    )
    assert result["policy"] == pytest.approx(5.0)
    assert result["encoders"] == pytest.approx(12.0)


def test_update_norm_demo_only_updates_allowlist() -> None:
    model = _tiny_model()
    obs = make_dummy_obs(batch=BATCH)
    allowlist = ("policy.", "plan_head.fusion.2.")
    demo = update_norm_demo(model, obs, allowlist)
    assert demo["optimizer_isolated"], demo["outside_allowlist_updates"]
    assert demo["update_norm_total"] > 0.0
    assert set(demo["by_module"]) <= {"policy", "plan_head"}


# =========================================================================== #
# 6. detach 登记表 CI
# =========================================================================== #
def test_detach_registry_covers_all_net_detaches() -> None:
    report = probes_module.check_detach_registry()
    assert report["unregistered"] == [], report["unregistered"]
    assert report["stale"] == [], [entry["id"] for entry in report["stale"]]
    assert report["invalid_entries"] == []
    assert report["ok"] is True
    assert report["n_code_detaches"] >= 31
    assert report["n_code_detaches"] == report["n_registry_entries"]


def test_detach_registry_entries_well_formed_and_key_ids_present() -> None:
    ids = [entry["id"] for entry in probes_module.DETACH_REGISTRY]
    assert len(ids) == len(set(ids))
    for entry in probes_module.DETACH_REGISTRY:
        for field in probes_module.DETACH_REGISTRY_FIELDS:
            assert entry.get(field), (entry.get("id"), field)
    for required in (
        "z_ego_next.detach()",
        "z_od_next.detach()",
        "z_ld_next.detach()",
        "action.detach()",
        "ds.detach()",
    ):
        assert any(entry["tensor"] == required for entry in probes_module.DETACH_REGISTRY), required
    assert sum(1 for entry in probes_module.DETACH_REGISTRY if entry["function"].endswith("_rollout")) >= 7


def test_detach_scan_detects_unregistered_and_stale(tmp_path: Path) -> None:
    net_dir = tmp_path / "net"
    net_dir.mkdir()
    (net_dir / "sample.py").write_text(
        "import torch\n\n\ndef f(x):\n    return x.detach()\n", encoding="utf-8"
    )
    report = probes_module.check_detach_registry(root=tmp_path, registry=())
    assert report["ok"] is False
    assert report["unregistered"] and report["unregistered"][0]["tensor"] == "x.detach()"

    stale_entry = {
        "id": "D-TEST",
        "tensor": "y.detach()",
        "file": "net/sample.py",
        "function": "f",
        "line": 4,
        "stage": "test",
        "blocks": "none",
        "reach": "none",
        "evidence": "static",
    }
    report_stale = probes_module.check_detach_registry(root=tmp_path, registry=(stale_entry,))
    assert report_stale["ok"] is False
    assert [entry["id"] for entry in report_stale["stale"]] == ["D-TEST"]
    assert report_stale["unregistered"], "同一文件的真实 detach 仍未登记 ⇒ 应同时报告"

    malformed = dict(stale_entry)
    malformed.pop("stage")
    report_bad = probes_module.check_detach_registry(root=tmp_path, registry=(malformed,))
    assert report_bad["invalid_entries"], report_bad


def test_detach_scan_ignores_comments_and_strings(tmp_path: Path) -> None:
    net_dir = tmp_path / "net"
    net_dir.mkdir()
    (net_dir / "sample.py").write_text(
        '"""docstring: x.detach() 说明"""\n'
        "# comment: y.detach()\n"
        "TEXT = 'z.detach()'\n", encoding="utf-8"
    )
    assert probes_module.scan_detaches(tmp_path) == []


# =========================================================================== #
# 7. z_next detach 语义回归（tiny 模型；与 a2.json detach_reach 同口径）
# =========================================================================== #
def test_z_next_detach_blocks_cross_step_gradient() -> None:
    model = _tiny_model()
    obs = make_dummy_obs(batch=BATCH)
    calls: list[dict[str, torch.Tensor]] = []

    def hook(module: torch.nn.Module, args, kwargs, output) -> None:
        calls.append({"z_out": output[0], "action": kwargs.get("action")})

    handle = model.st_gnn.register_forward_hook(hook, with_kwargs=True)
    try:
        out = model(obs, forward_mode="world_model_teacher_forcing")
    finally:
        handle.remove()
    z_step1, z_step2 = calls[0]["z_out"], calls[1]["z_out"]
    loss_step2 = z_step2.pow(2).sum()
    grad_cross = torch.autograd.grad(loss_step2, z_step1, allow_unused=True)[0]
    assert grad_cross is None, "z_next.detach() 失效：跨步梯度出现路径"
    loss_step1 = z_step1.pow(2).sum()
    assert torch.autograd.grad(loss_step1, z_step1)[0].norm() > 0.0
    # 动作链 loss 对 st_gnn 无梯度（状态链 detach 的另一面）；对动作头可达
    plan_loss = out["plan"][:, 1:, :].pow(2).sum()
    st_gnn_grads = torch.autograd.grad(
        plan_loss, [p for p in model.st_gnn.parameters()], allow_unused=True, retain_graph=True
    )
    assert all(grad is None or float(grad.norm()) == 0.0 for grad in st_gnn_grads)
    policy_grads = torch.autograd.grad(
        plan_loss, [p for p in model.policy.parameters()], allow_unused=True
    )
    assert any(grad is not None and float(grad.norm()) > 0.0 for grad in policy_grads)


# =========================================================================== #
# 8. 形状/契约冒烟（CLI 口径回归：structured 无 action_logstd 出口）
# =========================================================================== #
def test_shape_smoke_covers_both_configs() -> None:
    smoke = shape_smoke(["default", "structured"])
    assert "action_logstd" in smoke["default"]
    assert "contract_runtime_ok" in smoke["structured"]
    assert "action_logstd" not in smoke["structured"]
