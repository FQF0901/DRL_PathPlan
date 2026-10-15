"""R38-D1：§四 七类强制 Probes/Hooks（``net/probes.py``）单元测试。

覆盖：
- 默认关：键集逐位一致 / RNG 不消耗 / optimizer+ckpt 不变 / hooks 零采集；
- 开启后：①..⑤ 记录键与形状、④ Memory（mask/身份/权重/贡献/全 mask/顺序）；
- ⑤ 断言：合规构造下全通过；构造违例（rollout_steps=5 / 错误 z linkage）可触发；
- 统一出口 write（张量字典 + JSON 摘要）；
- ⑥ 首次越界/碰撞扫描（含 JSON 可序列化）；⑦ 每步单独反向梯度账本（含 update/ckpt）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from net.model import DrivingModel
from net.probes import (
    ChainHooks,
    MemoryHooks,
    ProbeRecorder,
    StateSourceHooks,
    first_boundary_breach,
    first_od_contact,
    grad_ledger,
    interpolate_plan_poses,
    state_dict_hash,
)
from tests.test_net_shapes import make_obs

BATCH = 2


def _structured_extra(obs: dict[str, torch.Tensor], batch: int = BATCH) -> dict[str, torch.Tensor]:
    s = torch.linspace(0.0, 40.0, 41)
    anchor = torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()
    obs["lane_anchor_xy"] = anchor
    obs["lane_anchor_mask"] = torch.ones(batch, 41)
    for key, value in (
        ("path_left_margin", 3.0),
        ("path_right_margin", 3.0),
        ("path_static_distance", 40.0),
        ("path_dynamic_distance", 60.0),
        ("path_dynamic_ttc", 8.0),
        ("path_nav_progress", 0.2),
    ):
        obs[key] = torch.full((batch, 1), float(value))
    return obs


def _model(seed: int = 0, *, structured: bool = False) -> DrivingModel:
    torch.manual_seed(seed)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    if structured:
        model.set_path_head(True)
        model.set_path_head_ablation("structured")
    return model.eval()


# ---------------------------------------------------------------------- ①④ 记忆/状态
def test_default_off_bitwise_rng_and_probe_zero_capture() -> None:
    """默认关：加载 disabled 容器后输出逐位一致、RNG 不消耗、hooks 零采集。"""
    model = _model()
    obs = make_obs(batch=BATCH, seed=0)
    baseline = model(obs)

    probes = ProbeRecorder(enabled=False)
    model.set_probes(probes)
    rng_before = torch.random.get_rng_state()
    out = model(obs)
    rng_after = torch.random.get_rng_state()

    assert sorted(baseline.keys()) == sorted(out.keys())
    for key in baseline:
        assert torch.equal(baseline[key], out[key]), f"默认关下 {key} 不逐位一致"
    assert torch.equal(rng_before, rng_after), "默认关路径消耗了 RNG"
    assert not probes.tensors()
    assert probes.chain.st_gnn_total_calls == 0
    assert probes.chain.policy_steps == []
    assert probes.memory.records == [] and probes.memory.attn == []


def test_default_off_optimizer_and_ckpt_unchanged() -> None:
    """默认关：前向+反向不改 ckpt（参数）与 optimizer 状态摘要。"""
    model = _model()
    obs = make_obs(batch=BATCH, seed=1)
    probes = ProbeRecorder(enabled=False)
    model.set_probes(probes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_before = state_dict_hash(model)
    opt_before = json.dumps(
        {key: sorted(value.keys()) for key, value in optimizer.state_dict()["state"].items()},
        sort_keys=True,
    )
    out = model(obs)
    out["action_mu"].sum().backward()
    ckpt_after = state_dict_hash(model)
    opt_after = json.dumps(
        {key: sorted(value.keys()) for key, value in optimizer.state_dict()["state"].items()},
        sort_keys=True,
    )
    assert ckpt_before == ckpt_after
    assert opt_before == opt_after
    assert not probes.tensors()


def test_hooks_on_structured_records_keys_and_shapes() -> None:
    """开启后（structured）：①..⑤ 记录 + ST-GNN 总调用单列 + z linkage 全通过。"""
    model = _model(structured=True)
    obs = _structured_extra(make_obs(batch=BATCH, seed=2))
    probes = ProbeRecorder(enabled=True)
    model.set_probes(probes)
    out = model(obs)

    summary = probes.summary()
    assert summary["state"]["sources"] == ["real"] + ["predicted"] * 6
    assert summary["policy"]["records"] == 6
    assert summary["wm"]["records"] == 6
    assert summary["chain"]["st_gnn_total_calls"] == 6
    assert summary["chain"]["wm_advances"] == 5
    assert summary["chain"]["checks"]["st_gnn_total_calls_matches_rollout_steps"] is True
    assert summary["chain"]["violations"] == []

    tensors = probes.tensors()
    assert "state/r00/z_ego" in tensors and "state/r06/nav" in tensors
    assert "state/r00/memory/ego_ctx" in tensors
    # ① 与 ③ 一致性：state/r{k} 的 z_k == wm/r{k-1} 的 z_out（k≥1）
    for k in (1, 3, 6):
        assert torch.equal(tensors[f"state/r{k:02d}/z_ego"], tensors[f"wm/r{k - 1:02d}/z_ego_out"])
        assert torch.equal(tensors[f"state/r{k:02d}/z_od"], tensors[f"wm/r{k - 1:02d}/z_od_out"])
    assert "policy/r00/latent" in tensors and "policy/r05/dtheta" in tensors
    assert tensors["policy/r00/kappa_near"].shape == (BATCH,)
    assert tensors["policy/r00/ds_aux"].shape == (BATCH, 5)
    assert "policy/r00/masks/origin" not in tensors
    assert "policy/r00/masks/anchor_mask" in tensors
    assert tensors["wm/r00/z_od_in"].shape == (BATCH, 16, 32)
    assert tensors["memory/encode00/od_id"].shape == (BATCH, 6, 16)
    assert tensors["memory/attn01/od/weights"].shape == (BATCH, 6, 16)
    assert tensors["memory/attn01/od/contribution"].shape == (BATCH, 6, 16, 32)
    assert tensors["memory/attn01/od/all_masked"].shape == (BATCH, 16)
    # 输出契约不被探针改写（与探针关闭的对象同前向逐位）
    out_off = _model(structured=True)(obs)  # 同种子重建模型（权重一致）
    assert sorted(out.keys()) == sorted(out_off.keys())
    for key in out:
        if torch.is_tensor(out[key]):
            assert torch.equal(out[key], out_off[key]), f"开启探针改变了输出 {key}"


def test_memory_hooks_capture_attention_full_mask() -> None:
    """④ Memory：逐帧 mask / slot identity / attention 权重 / 全 mask 输出 / 历史顺序。"""
    hooks = MemoryHooks(enabled=True)

    class _Mem:  # 最小 MemBank 鸭子类型（只读字段）
        frames = 6
        ego_mask = torch.ones(1, 6, 1)
        ego_valid = torch.ones(1, 6)
        od_mask = torch.ones(1, 6, 2)
        od_valid = torch.ones(1, 6)
        od_id = torch.tensor([[[7, 9]] * 6])
        od_presence = torch.ones(1, 6, 2)
        ld_mask = torch.ones(1, 6, 2)
        ld_valid = torch.ones(1, 6)
        others_mask = torch.ones(1, 6, 1)
        others_valid = torch.ones(1, 6)

    encode_record = hooks.capture_encode(_Mem())
    ok = torch.ones(1, 6, 2, dtype=torch.bool)
    ok[0, :, 1] = False  # 槽 1 整列无有效帧 ⇒ 全 mask 输出
    x = torch.randn(1, 6, 2, 8)
    weights = torch.softmax(torch.randn(1, 6, 2), dim=1)
    weights = weights * ok
    record = hooks.capture_attention(
        name="od",
        x=x,
        weights=weights,
        ok=ok,
        pooled=torch.randn(1, 2, 8),
        output=torch.randn(1, 2, 8),
        contribution=weights.unsqueeze(-1) * x,
        valid=torch.ones(1, 6),
    )
    assert record is not None
    assert int(record["all_masked"][0, 0]) == 0 and int(record["all_masked"][0, 1]) == 1
    assert encode_record is not None
    assert torch.equal(encode_record["age_order"], torch.tensor([5, 4, 3, 2, 1, 0]))
    tensors = hooks.tensors()
    assert tensors["memory/encode00/od_id"].shape == (1, 6, 2)
    assert tensors["memory/attn00/od/weights"].shape == (1, 6, 2)
    # 关闭：零采集
    hooks_off = MemoryHooks(enabled=False)
    assert hooks_off.capture_encode(_Mem()) is None
    assert hooks_off.capture_attention(
        name="od", x=x, weights=weights, ok=ok, pooled=x[:, 0], output=x[:, 0]
    ) is None
    assert hooks_off.tensors() == {}


def test_state_source_substituted_label() -> None:
    """① 真值替换实验标签：``substituted`` 可显式标注（默认 predicted）。"""
    hooks = StateSourceHooks(enabled=True)
    hooks.capture(step=1, source="substituted", z_ego=torch.zeros(1, 4))
    assert hooks.summary()["sources"] == ["substituted"]
    with pytest.raises(ValueError):
        hooks.capture(step=2, source="bogus")


# ---------------------------------------------------------------------- ⑤ 调用链
def test_chain_check_detects_constructed_violation() -> None:
    """构造违例（rollout_steps=5）：⑤ 断言必须触发（头调用 5 / WM 推进 4 / ST-GNN 5）。"""
    model = _model(structured=True)
    model.set_contract_asserts(False)  # 关模型内置契约门，验证 probe 侧独立断言
    obs = _structured_extra(make_obs(batch=BATCH, seed=3))
    probes = ProbeRecorder(enabled=True)
    model.set_probes(probes)
    model.rollout_steps = 5  # 构造违例：六步契约被破坏（期望仍为装载时的 6）
    model(obs)
    check = probes.chain.check()
    assert check["violations"] == [
        "policy_call_count_is_six",
        "st_gnn_total_calls_matches_rollout_steps",
        "world_model_advances_five",
    ]
    assert check["st_gnn_total_calls"] == 5 and check["wm_advances"] == 4 and check["policy_calls"] == 5


def test_chain_linkage_flags_wrong_z() -> None:
    """⑤ linkage：a_k 读到错误 z_k 时 ``each_action_reads_corresponding_z_k`` 必须 False。"""
    hooks = ChainHooks(enabled=True, expected_st_gnn_calls=6)
    for step in range(6):
        hooks.on_policy_call(step=step, instance=1, z={"z_ego": torch.full((1, 4), float(step))})
    z_t0 = {"z_ego": torch.zeros(1, 4), "z_od": torch.zeros(1, 1, 4), "z_ld": torch.zeros(1, 1, 4)}
    z_pred = {
        "z_ego": torch.stack([torch.full((1, 4), float(step)) for step in range(1, 6)], dim=1),
        "z_od": torch.zeros(1, 5, 1, 4),
        "z_ld": torch.zeros(1, 5, 1, 4),
    }
    ok = ChainHooks(enabled=True, expected_st_gnn_calls=6)
    for step in range(6):
        ok.on_policy_call(step=step, instance=1, z={"z_ego": torch.full((1, 4), float(step))})
    check_ok = ok.check(z_t0=z_t0, z_pred=z_pred)
    assert check_ok["checks"]["each_action_reads_corresponding_z_k"] is True
    # 篡改 step3 的 z（读错一步）⇒ 断言失败
    hooks.policy_z[3]["z_ego"] = torch.full((1, 4), 99.0)
    check_bad = hooks.check(z_t0=z_t0, z_pred=z_pred)
    assert check_bad["checks"]["each_action_reads_corresponding_z_k"] is False
    assert "each_action_reads_corresponding_z_k" in check_bad["violations"]


def test_probe_recorder_write_roundtrip(tmp_path: Path) -> None:
    """统一出口：张量字典（torch.save）+ JSON 摘要。"""
    model = _model(structured=True)
    obs = _structured_extra(make_obs(batch=BATCH, seed=4))
    probes = ProbeRecorder(enabled=True)
    model.set_probes(probes)
    model(obs)
    paths = probes.write(tmp_path, stem="probe")
    tensors = torch.load(paths["tensors"], map_location="cpu", weights_only=True)
    summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert isinstance(tensors, dict) and tensors and all(torch.is_tensor(v) for v in tensors.values())
    assert summary["probe_schema_version"] == 1
    assert summary["chain"]["st_gnn_total_calls"] == 6


# ---------------------------------------------------------------------- ⑥ 连续路径
def test_first_boundary_breach_detects_curve_and_clears_straight() -> None:
    """⑥ 首次越界：强弯 plan ⇒ 报告首个越界点；直行 plan ⇒ 无越界（margin>0）。"""
    anchor = torch.stack([torch.linspace(0.0, 40.0, 41), torch.zeros(41)], dim=-1).unsqueeze(0)
    margins = lambda: (torch.full((1, 1), 3.0), torch.full((1, 1), 3.0))
    left, right = margins()
    straight = torch.zeros(1, 6, 2)
    straight[..., 0] = 5.0
    result_straight = first_boundary_breach(
        plan=straight, anchor_xy=anchor, margin_left=left, margin_right=right,
        ego_length=4.515, ego_width=1.852,
    )
    assert result_straight["batch"][0]["first"] is None
    assert result_straight["batch"][0]["min_margin_m"] > 0.0

    curve = torch.zeros(1, 6, 2)
    curve[..., 0] = 5.0
    curve[..., 1] = 1.2
    result_curve = first_boundary_breach(
        plan=curve, anchor_xy=anchor, margin_left=left, margin_right=right,
        ego_length=4.515, ego_width=1.852,
    )
    first = result_curve["batch"][0]["first"]
    assert first is not None and first["margin_m"] < 0.0
    assert result_curve["batch"][0]["breach_n"] > 0
    assert json.dumps(result_curve)  # machine-readable


def test_first_od_contact_detects_box_ahead() -> None:
    """⑥ 首次碰撞：正前方 OD 盒 ⇒ 首个接触点；侧向盒不接触。"""
    plan = torch.zeros(1, 6, 2)
    plan[..., 0] = 5.0
    od = torch.zeros(1, 16, 9)
    od[..., 4] = 1.0  # cosθ
    od[0, 0, 0], od[0, 0, 1] = 8.0, 0.0
    od[0, 0, 6], od[0, 0, 7] = 4.5, 1.8
    live = torch.zeros(1, 16)
    live[0, 0] = 1.0
    contact = first_od_contact(plan=plan, od_now=od, od_live=live, ego_length=4.515, ego_width=1.852)
    assert contact["batch"][0]["first"] is not None
    assert contact["batch"][0]["first"]["slot"] == 0

    od_far = od.clone()
    od_far[0, 0, 1] = 20.0  # 侧向 20 m ⇒ 无接触
    contact_far = first_od_contact(plan=plan, od_now=od_far, od_live=live, ego_length=4.515, ego_width=1.852)
    assert contact_far["batch"][0]["first"] is None
    assert json.dumps(contact_far)


# ---------------------------------------------------------------------- ⑦ 梯度账本
def _unblocked_structured_train_model() -> DrivingModel:
    model = _model(structured=True)
    model.train()
    with torch.no_grad():  # 解阻零初始化末层（与 K1/J1 结构探针同口径）
        for branch in (model.path_head.lateral, model.path_head.longitudinal):
            branch[-1].weight.normal_(0.0, 0.05)
            branch[-1].bias.normal_(0.0, 0.05)
    return model


def _step_losses(model: DrivingModel, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = model(obs)
    plan = out["plan"]
    return {f"step{k}": plan[:, k - 1, 0].mean() + plan[:, k - 1, 1].mean() for k in (1, 3, 6)}


def test_grad_ledger_per_step_reachability() -> None:
    """⑦ 第 1/3/6 步 loss 单独反向：逐 loss 梯度可达（κ 支 / ds 支 / 上游 latent 链）。"""
    model = _unblocked_structured_train_model()
    obs = _structured_extra(make_obs(batch=BATCH, seed=5))
    ckpt_before = state_dict_hash(model)
    ledger = grad_ledger(model, _step_losses(model, obs), strict=True)
    for key in ("step1", "step3", "step6"):
        entry = ledger["steps"][key]
        assert entry["nonzero_grad_params"] > 0
        prefixes = entry["touched_prefixes"]
        grads = entry["grad_norms"]
        # depth=2 前缀 = policy.structured_action / plan_head / encoders / mem_encoder...
        assert any(p.startswith("policy.structured_action") for p in prefixes)
        assert any(p.startswith(("encoders", "mem_encoder", "plan_head")) for p in prefixes)
        # 分支级（全名）可达：κ 支 / ds 支
        assert any(name.startswith("policy.structured_action.lateral") for name in grads)
        assert any(name.startswith("policy.structured_action.longitudinal") for name in grads)
    assert ledger["update"] is None
    assert ledger["ckpt"]["changed"] is None
    assert state_dict_hash(model) == ckpt_before  # 只读账本不改参数
    assert ledger["params"] and ledger["params"][0]["module"]


def test_grad_ledger_step_update_and_ckpt_change() -> None:
    """⑦ optimizer step（调用方副本）：update norm 与 ckpt 变化读数。"""
    model = _unblocked_structured_train_model()
    obs = _structured_extra(make_obs(batch=BATCH, seed=6))
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    ledger = grad_ledger(model, _step_losses(model, obs), optimizer=optimizer, step=True, strict=True)
    assert ledger["update"] is not None
    assert ledger["update"]["update_norm_total"] > 0.0
    assert ledger["update"]["update_norms"]
    assert ledger["ckpt"]["changed"] is True
    assert any(entry["optimizer"] for entry in ledger["params"])


def test_grad_ledger_strict_rejects_foreign_losses() -> None:
    """⑦ fail-fast：detach 的 loss / 与 model 不同源的 loss 必须显式报错（禁止静默全 0）。"""
    model_a = _model(structured=True)
    model_b = _model(structured=True)
    obs = _structured_extra(make_obs(batch=BATCH, seed=7))
    losses_live = _step_losses(model_a, obs)  # 与 model_b 不同源的活图
    with pytest.raises(ValueError):
        grad_ledger(model_b, losses_live, strict=True)
    detached = {key: value.detach() for key, value in losses_live.items()}
    with pytest.raises(ValueError):
        grad_ledger(model_a, detached, strict=True)


def test_interpolate_plan_poses_matches_model_helper() -> None:
    """⑥ 插值口径与 ``net/model.py::interpolate_actions`` 同源（≥1 m 连续 footprint 基础）。"""
    from net.model import interpolate_actions

    plan = torch.zeros(2, 6, 2)
    plan[..., 0] = 3.0
    plan[..., 1] = 0.3
    a = interpolate_plan_poses(plan)
    b = interpolate_actions(plan)
    assert torch.equal(a, b)
    assert a.shape == (2, 30, 3)
