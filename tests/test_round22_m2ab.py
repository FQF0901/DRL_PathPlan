"""Round-22 Gate A 契约测试：M2a/M2b（``LdAggregate``）——默认关、baseline 逐位不变。

六项契约（round-22 §B）：
① ``alpha=0`` 与 baseline 前向逐位一致（≤1e-6）；
② attn 模式下 LD pooled summary **全部**走 ``aggregate_ld``（spy 锁 ``_masked_mean``：不得再对
   LD 槽做旧均值；per-slot token 路径豁免）；
③ 全灭 mask：输出精确 0 且 fwd/bwd 有限（禁 NaN/Inf）；
④ 训练中 attention 参数获得非零 update（**α 先行**：α=0 时 attention 梯度恰为 0，α≠0 后获得）；
⑤ ``no_chain``/``attn_no_chain`` 仅改 LD-LD 邻接（baseline 默认逐位不变）；
⑥ ckpt 严格完整加载（**先** ``set_ld_mode`` 再 ``load_state_dict``）。

另附 round-21 教训测试：**消费契约**——α≠0 后 PlanHead 输出必须变化（"替换 X"必须配消费断言）。
"""

from __future__ import annotations

import pytest
import torch

import net.model as model_mod
from net.spatial import LdAggregate, SpatialEncoder
from tests.test_recovery_wiring import _tiny_model

NUM_OD, NUM_LD, H = 16, 16, 16


def _obs(batch: int = 1, seed: int = 0, ld_valid: int = NUM_LD) -> dict:
    """固定随机 obs（LD 有效槽数可调；其余槽位全有效）。"""
    torch.manual_seed(seed)
    ld_mask = torch.zeros(batch, NUM_LD)
    ld_mask[:, : int(ld_valid)] = 1.0
    return {
        "ego": torch.randn(batch, 1, 8),
        "ego_mask": torch.ones(batch, 1),
        "ego_world": torch.randn(batch, 1, 3),
        "ego_world_mask": torch.ones(batch, 1),
        "od": torch.randn(batch, NUM_OD, 9),
        "od_mask": torch.ones(batch, NUM_OD),
        "od_presence": torch.ones(batch, NUM_OD),
        "ld": torch.randn(batch, NUM_LD, 7),
        "ld_mask": ld_mask,
        "nav": torch.randn(batch, 1, 11),
        "nav_mask": torch.ones(batch, 1),
        "signal": torch.randn(batch, 1, 4),
        "signal_mask": torch.ones(batch, 1),
        "others": torch.randn(batch, 1, 33),
        "others_mask": torch.ones(batch, 1),
        "route_world": torch.randn(batch, 64, 2),
        "route_world_mask": torch.ones(batch, 64),
        "hist_valid": torch.ones(batch, 6),
        "ego_hist": torch.randn(batch, 6, 1, 8),
        "ego_hist_mask": torch.ones(batch, 6, 1),
        "ld_hist": torch.randn(batch, 6, NUM_LD, 7),
        "ld_hist_mask": torch.ones(batch, 6, NUM_LD),
        "od_hist": torch.randn(batch, 6, NUM_OD, 9),
        "od_hist_mask": torch.ones(batch, 6, NUM_OD),
        "od_id_hist": torch.zeros(batch, 6, NUM_OD, dtype=torch.long),
        "od_presence_hist": torch.ones(batch, 6, NUM_OD),
        "others_hist": torch.randn(batch, 6, 1, 33),
        "others_hist_mask": torch.ones(batch, 6, 1),
    }


# --------------------------------------------------------------------------- ①
def test_alpha_zero_bitwise_equals_baseline() -> None:
    torch.manual_seed(0)
    base = _tiny_model()
    torch.manual_seed(0)
    attn = _tiny_model()
    attn.set_ld_mode("attn")
    assert attn.ld_aggregator is not None
    assert float(attn.ld_aggregator.alpha.detach()) == 0.0
    assert attn.stgnn_flags() == (True, False)  # attn 模式保留 LD-LD 链与 LD 节点 MP
    obs = _obs()
    with torch.no_grad():
        out_base = base(obs, forward_mode="policy_wm_rollout")
        out_attn = attn(obs, forward_mode="policy_wm_rollout")
        enc_base = base.encode(obs)
        enc_attn = attn.encode(obs)
    for key in ("action_mu", "plan", "latent"):
        delta = float((out_base[key] - out_attn[key]).abs().max())
        assert delta <= 1e-6, f"{key} 差异 {delta} > 1e-6（alpha=0 应逐位等价）"
    pool_delta = float((enc_base["ld_pool"] - enc_attn["ld_pool"]).abs().max())
    assert pool_delta <= 1e-6, f"ld_pool 差异 {pool_delta} > 1e-6"


def test_plan_head_consumes_aggregate_output() -> None:
    """消费契约（round-21 教训）：α≠0 后 **PlanHead 融合**（latent）与策略/计划输出必须变化。

    注：``net/policy.py`` 的 ``mu`` 末层零初始化（初始动作 = 界中点，对任何输入不敏感），
    故动作级消费断言前需先打散 ``mu`` 末层（测试内局部扰动，不改生产代码）。
    """
    torch.manual_seed(0)
    model = _tiny_model()
    model.set_ld_mode("attn")
    obs = _obs()
    with torch.no_grad():
        torch.nn.init.normal_(model.policy.mu.weight, std=0.1)
        torch.nn.init.normal_(model.policy.mu.bias, std=0.05)
        enc0 = model.encode(obs)
        out0 = model(obs, forward_mode="policy_wm_rollout")
        model.ld_aggregator.alpha.fill_(1.0)
        enc1 = model.encode(obs)
        out1 = model(obs, forward_mode="policy_wm_rollout")
    pool_delta = float((enc0["ld_pool"] - enc1["ld_pool"]).abs().max())
    latent_delta = float((enc0["latent"] - enc1["latent"]).abs().max())
    plan_delta = float((out0["plan"] - out1["plan"]).abs().max())
    mu_delta = float((out0["action_mu"] - out1["action_mu"]).abs().max())
    assert pool_delta > 1e-3, "α 未改变 LD pooled summary"
    assert latent_delta > 1e-6, "PlanHead 融合未消费 aggregate_ld 输出（latent 不变）"
    assert plan_delta > 1e-6, "rollout 计划未消费 aggregate_ld 输出（plan 不变）"
    assert mu_delta > 1e-9, "action_mu 未消费 aggregate_ld 输出"


# --------------------------------------------------------------------------- ②
def test_attn_mode_has_no_hidden_mean_path(monkeypatch) -> None:
    obs = _obs(ld_valid=12)  # LD 有效槽 = 12（与 OD 16 可区分）

    def _count(model) -> tuple[int, int]:
        calls: list[tuple[int, ...]] = []
        agg_calls: list[tuple[int, ...]] = []
        original = model_mod._masked_mean

        def spy(x, mask):
            calls.append(tuple(x.shape))
            return original(x, mask)

        original_agg = model.aggregate_ld

        def agg_spy(*args, **kwargs):
            agg_calls.append(tuple(args[0].shape))
            return original_agg(*args, **kwargs)

        monkeypatch.setattr(model_mod, "_masked_mean", spy)
        monkeypatch.setattr(model, "aggregate_ld", agg_spy)
        with torch.no_grad():
            model(obs, forward_mode="policy_wm_rollout")
        monkeypatch.undo()
        return len(calls), len(agg_calls)

    torch.manual_seed(0)
    base = _tiny_model()
    base_calls, base_agg = _count(base)
    torch.manual_seed(0)
    attn = _tiny_model()
    attn.set_ld_mode("attn")
    attn_calls, attn_agg = _count(attn)
    # 每个 plan_step/encode 恰少一次 _masked_mean 调用（LD 那一路）→ attn = baseline/2
    assert attn_calls * 2 == base_calls, f"attn={attn_calls} 应为 baseline={base_calls} 的一半"
    # 两种模式都走统一入口 aggregate_ld（t0 plan_step + encode 自身 + rollout 5 步 = 7）
    assert base_agg == 7 and attn_agg == 7, f"aggregate_ld 调用次数 base={base_agg} attn={attn_agg}"
    assert base.ld_aggregator is None, "baseline 不得构造聚合器"
    # attn：聚合器模块本体被消费 7 次（逐调用诊断计数）
    attn.ld_aggregator.collect_stats = True
    with torch.no_grad():
        attn(obs, forward_mode="policy_wm_rollout")
    calls = len(attn.ld_aggregator._stats["alpha"])
    assert calls == 7, f"LdAggregate 前向次数 {calls} != 7（LD summary 未全部走聚合器）"
    stats = attn.ld_aggregator.pop_stats()
    assert stats["alpha"] == 0.0 and stats["mean_norm"] > 0.0


# --------------------------------------------------------------------------- ③
def test_all_masked_ld_finite_fwd_bwd_and_exact_zero() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    model.set_ld_mode("attn")
    agg = model.ld_aggregator
    with torch.no_grad():
        agg.alpha.fill_(0.5)  # 打开门控（α=0 时 attention 分支不参与反向）
    z = torch.randn(2, NUM_LD, H)
    live = torch.zeros(2, NUM_LD)
    out = agg(z, live, torch.randn(2, 1, H), torch.randn(2, 1, H), torch.randn(2, 1, H))
    assert torch.equal(out, torch.zeros_like(out)), "全灭 mask 输出必须精确 0"

    obs = _obs(ld_valid=0)
    out = model(obs, forward_mode="policy_wm_rollout")
    assert torch.isfinite(out["action_mu"]).all()
    assert torch.isfinite(out["plan"]).all()
    loss = out["action_mu"].pow(2).sum() + out["plan"].pow(2).sum()
    loss.backward()
    for name, parameter in agg.named_parameters():
        assert parameter.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(parameter.grad).all(), f"{name} 梯度含 NaN/Inf"


def test_partial_mask_attention_weights_normalized() -> None:
    torch.manual_seed(0)
    agg = LdAggregate(H, heads=4, ctx_dim=3 * H)
    z = torch.randn(3, NUM_LD, H)
    live = torch.zeros(3, NUM_LD)
    live[0, :] = 1.0
    live[1, :5] = 1.0
    weights = agg.attention_weights(z, live, torch.randn(3, 1, H), torch.randn(3, 1, H))
    assert weights.shape == (3, NUM_LD)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights[0].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(weights[1].sum(), torch.tensor(1.0), atol=1e-5)
    assert float(weights[1, 5:].abs().sum()) < 1e-6, "无效槽权重应为 0"
    assert float(weights[2].abs().sum()) < 1e-9, "全灭行权重应为 0"


# --------------------------------------------------------------------------- ④
def test_alpha_first_then_attention_updates() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    model.set_ld_mode("attn")
    agg = model.ld_aggregator
    obs = _obs()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    attn_names = [name for name, _ in agg.named_parameters() if name != "alpha"]
    before = {name: parameter.detach().clone() for name, parameter in agg.named_parameters()}
    alpha_moved_at: int | None = None
    attn_moved_at: int | None = None
    grad_attn_step0: float | None = None
    grad_alpha_step0: float | None = None
    for step in range(6):
        out = model(obs, forward_mode="policy_wm_rollout")
        loss = out["action_mu"].pow(2).sum() + out["plan"].pow(2).sum() + out["latent"].pow(2).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step == 0:
            grad_attn_step0 = float(
                sum(
                    float(agg.get_parameter(name).grad.abs().sum())
                    for name in attn_names
                    if agg.get_parameter(name).grad is not None
                )
            )
            grad_alpha_step0 = float(agg.alpha.grad.abs().sum())
        optimizer.step()
        if alpha_moved_at is None and not torch.equal(agg.alpha.detach(), before["alpha"]):
            alpha_moved_at = step
        if attn_moved_at is None and any(
            not torch.equal(agg.get_parameter(name).detach(), before[name]) for name in attn_names
        ):
            attn_moved_at = step
    assert grad_alpha_step0 is not None and grad_alpha_step0 > 0.0, "α 第一步必须有非零梯度"
    assert grad_attn_step0 == 0.0, f"α=0 时 attention 参数梯度必须恰为 0（实测 {grad_attn_step0}）"
    assert alpha_moved_at is not None, "α 未获得更新"
    assert attn_moved_at is not None, "attention 参数未获得更新（α 打开后应有非零 update）"
    assert attn_moved_at >= alpha_moved_at, "α 必须先行"
    assert float(agg.alpha.detach()) != 0.0, "α 应离开 0"


# --------------------------------------------------------------------------- ⑤
def test_ld_chain_flags_only_touch_ld_ld_edges() -> None:
    spatial = SpatialEncoder(H, layers=1, num_od=NUM_OD, num_ld=NUM_LD)
    mask = torch.ones(2, 1 + NUM_OD + NUM_LD)
    mask[1, 1 + NUM_OD + 5 :] = 0.0
    pose = torch.zeros(2, 1 + NUM_OD + NUM_LD, 3)
    ld_slice = slice(1 + NUM_OD, 1 + NUM_OD + NUM_LD)
    base = spatial.build_adjacency(mask, pose)
    assert int(base[:, ld_slice, ld_slice].sum()) > 0
    others = {
        "ego_ld": int(base[:, 0, ld_slice].sum()),
        "ego_od": int(base[:, 0, 1 : 1 + NUM_OD].sum()),
        "od_od": int(base[:, 1 : 1 + NUM_OD, 1 : 1 + NUM_OD].sum()),
    }
    spatial.include_ld_chain = False
    cut = spatial.build_adjacency(mask, pose)
    assert int(cut[:, ld_slice, ld_slice].sum()) == 0
    for key, value in others.items():
        if key == "ego_ld":
            assert int(cut[:, 0, ld_slice].sum()) == value
        elif key == "ego_od":
            assert int(cut[:, 0, 1 : 1 + NUM_OD].sum()) == value
        else:
            assert int(cut[:, 1 : 1 + NUM_OD, 1 : 1 + NUM_OD].sum()) == value

    torch.manual_seed(0)
    m0 = _tiny_model()
    torch.manual_seed(0)
    m1 = _tiny_model()
    m1.set_ld_mode("no_chain")
    torch.manual_seed(0)
    m2b = _tiny_model()
    m2b.set_ld_mode("attn_no_chain")
    assert m0.stgnn_flags() == (True, False)
    assert m1.stgnn_flags() == (False, False)
    assert m2b.stgnn_flags() == (False, False)
    assert m2b.ld_aggregator is not None


def test_baseline_default_unchanged() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    assert model.ld_mode == "baseline"
    assert model.stgnn_flags() == (True, False)
    assert model.ld_aggregator is None and model.ld_set_attn is None
    keys = set(model.state_dict().keys())
    assert not any("ld_aggregator" in key for key in keys), "baseline 不得含聚合器参数"
    with pytest.raises(ValueError):
        model.set_ld_mode("bogus")


# --------------------------------------------------------------------------- ⑥
def test_strict_ckpt_roundtrip_requires_ld_mode_first(tmp_path) -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    model.set_ld_mode("attn")
    state = model.state_dict()
    path = tmp_path / "m2a.pt"
    torch.save(state, path)
    torch.manual_seed(1)
    fresh = _tiny_model()
    fresh.set_ld_mode("attn")
    fresh.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    assert set(fresh.state_dict().keys()) == set(state.keys())
    # 未先 set_ld_mode 的 baseline 模型必须 fail-fast（unexpected ld_aggregator.*）
    torch.manual_seed(1)
    baseline = _tiny_model()
    with pytest.raises(RuntimeError):
        baseline.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
