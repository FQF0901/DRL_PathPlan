"""R41 第二批执行-G：**训练循环守卫**（fixture + 小模型；CPU；零研究工作区依赖）。

锁定契约（与 ``tools/train_lateral.py`` 训练分支 docstring 逐条对应）：
1. **冻结面 = allowlist**（``net.path_head.TRAIN_ALLOWLIST_PREFIXES``）：除
   ``policy.structured_action.`` + ``plan_head.fusion.2.`` 外全冻；optimizer id 集合 == allowlist；
2. **arm 装配**：p0 ⇒ 横向项零图（``lateral_total == 0.0``、terms 空）；p1 ⇒ κ 项 > 0
   （= ``kappa_coef × 掩码加权 smooth_l1``，与数据侧目标同源）；
3. **L_action / L_plan 口径**：L_action = 首步 (ds,dθ) 平方误差逐样本均值 × frame_weight 归一；
   L_plan = ``trainer.weighted_action_chain_loss``（k=1..5）；
4. **训练循环端到端**（fixture + 合成 obs）：1 epoch 收敛为可跑通；ckpt 落盘（``stages.save_checkpoint``
   口径）+ meta 字段（arm/round/lane/geometry）+ 可回读；冻结面逐位不变；
5. **P3/P4 fail-fast**：``replan_coef>0`` / ``lqr_reweight`` 在训练模式下拒绝（本批未接线）。
"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/residual_mini"

import pipeline.residual_dataset as rd  # noqa: E402
from pipeline import lateral_losses  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("train_lateral_runner_g", ROOT / "tools/train_lateral.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model():
    """确定性小模型（随机初始化；CPU；结构化头开；MoE 关）——与 E 守卫同构。"""
    from net.model import DrivingModel

    torch.manual_seed(7)
    model = DrivingModel(
        hidden=32, num_experts=1, expert_hidden=16, router_hidden=16,
        primary_hidden=32, trunk_hidden=32, net_hidden=32, spatial_layers=1,
    )
    model.set_moe(enabled=False)
    model.set_path_head(True)
    return model.eval()


def _fixture_args(**overrides) -> Namespace:
    base = dict(
        data=str(FIXTURE),
        base_dataset="unused",
        shards=None,
        max_pairs=64,
        seed=0,
        val_frac=0.25,
        horizon_margin_m=0.0,
        anchor_min_ds=2.0,
        anchor_noise_sigma_m=0.0,
        device="cpu",
        init=str(FIXTURE / "manifest.json"),
        model_config=str(FIXTURE / "manifest.json"),
        ckpt="unused",
    )
    base.update(overrides)
    return Namespace(**base)


def _stub_base(data: dict) -> dict:
    """base 数据集 stub：lookup + train_weight/balance_weight（全 1）。"""
    rows = data["rows"]
    lookup = {
        (int(rows["spec_id"][i]), int(rows["episode_id"][i]), int(rows["step"][i])): i
        for i in range(int(rows["n"]))
    }
    n = int(rows["n"])
    arrays = {
        "train_weight": np.ones(n, dtype=np.float32),
        "balance_weight": np.ones(n, dtype=np.float32),
    }
    return {"dataset": SimpleNamespace(arrays=arrays), "lookup": lookup}


_REAL_BUILD_OBS = rd.build_obs


def _fake_build_obs(rows, idx, *, device, base_dataset, lane_width_default: float = 3.5):
    """合成 obs 路径（真 ``build_obs`` + ``base_dataset=None``：零模板 + 真实锚/外源列）。"""
    return _REAL_BUILD_OBS(
        rows, np.asarray(idx, dtype=np.int64), device=device, base_dataset=None,
        lane_width_default=lane_width_default,
    )


def _patch_training_environment(module, monkeypatch, model, stub_base: dict) -> None:
    monkeypatch.setattr(rd, "build_obs", _fake_build_obs)
    monkeypatch.setattr(rd, "load_base_dataset", lambda path: stub_base)
    monkeypatch.setattr(
        module, "train_model_assembly",
        lambda args: (model, {"ckpt": "unused", "ckpt_sha256": "0" * 64, "missing_keys": [], "unexpected_keys": []}),
    )


# --------------------------------------------------------------------------- #
# 1) 冻结面 / allowlist
# --------------------------------------------------------------------------- #
def test_freeze_to_allowlist_exact_and_optimizer_ids() -> None:
    from net.path_head import TRAIN_ALLOWLIST_PREFIXES

    module = _load_runner()
    model = _tiny_model()
    trainable, names = module.freeze_to_allowlist(model)
    assert names and all(name.startswith(tuple(TRAIN_ALLOWLIST_PREFIXES)) for name in names)
    assert len(names) == len(trainable) == 18  # 结构化头 16 + fusion.2 2（= J1 口径）
    frozen = [
        name for name, p in model.named_parameters() if p.requires_grad and name not in set(names)
    ]
    assert frozen == []
    optimizer = torch.optim.Adam([p for _, p in trainable], lr=3e-4)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert optimizer_ids == {id(p) for _, p in trainable}


# --------------------------------------------------------------------------- #
# 2) 损失装配：p0 零图 / p1 κ 项 = 手算
# --------------------------------------------------------------------------- #
def test_batch_loss_terms_p0_zero_touch_and_p1_matches_manual(monkeypatch) -> None:
    module = _load_runner()
    data = module.build_lateral_data(_fixture_args())
    view = {
        "index": np.asarray(data["main_ids"], dtype=np.int64),
        "action": np.asarray(data["action"], dtype=np.float32)[np.asarray(data["main_ids"])],
        "step_mask": np.asarray(data["step_mask"], dtype=np.float32),
        "station_target": np.asarray(data["station_target"], dtype=np.float32),
        "station_mask": np.asarray(data["station_mask"], dtype=np.float32),
        "speed_weight": np.asarray(data["speed_weight"], dtype=np.float32),
        "frame_weight": np.ones(data["main_ids"].size, dtype=np.float32),
    }
    index = np.arange(min(16, view["index"].size), dtype=np.int64)
    model = _tiny_model()
    obs = _fake_build_obs(data["rows"], view["index"][index], device=torch.device("cpu"), base_dataset=None)
    batch = module.batch_tensors(view, index, torch.device("cpu"))
    args0 = Namespace(action_weight=1.0, chain_weight=0.2, arm="p0", kappa_coef=None,
                      delta_kappa_coef=None, replan_coef=None, speed_weight_mode=None,
                      dk_thresh=lateral_losses.DK_THRESH_DEFAULT, lqr_clip_max=1.0, lqr_alpha=1.0,
                      lqr_reweight=False, forward_mode="t0")
    with torch.no_grad():
        terms0 = module.batch_loss_terms(model, obs, batch, lateral_losses.config_for_arm("p0"), args0)
    assert float(terms0["lateral_total"]) == 0.0 and terms0["lateral_terms"] == {}
    assert terms0["lateral_active"] == ()
    args1 = Namespace(**{**vars(args0), "arm": "p1", "kappa_coef": 0.5})
    with torch.no_grad():
        terms1 = module.batch_loss_terms(model, obs, batch, lateral_losses.config_for_arm("p1"), args1)
    assert terms1["lateral_active"] == ("kappa",)
    # 手算：0.5 × 掩码加权 smooth_l1（单一出处单测）
    act = model.encode(obs)["path"]
    kappa_pred = torch.stack([act["kappa_near"], act["kappa_mid"], act["kappa_far"]], dim=-1)
    manual = lateral_losses.kappa_supervision_loss(
        kappa_pred, batch["station_target"], batch["station_mask"]
    )["loss"]
    assert torch.allclose(
        terms1["lateral_terms"]["kappa"], 0.5 * manual, atol=1e-6
    )
    # L_action 手算：首步 (ds,dθ) 平方误差逐样本均值（frame_weight = 1 ⇒ 批均值）
    manual_action = ((act["action"] - batch["action"][:, 0]) ** 2).mean(dim=-1).mean()
    assert torch.allclose(terms1["l_action"], manual_action, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3) 端到端：--smoke-steps（无 ckpt）/ --train（ckpt + 冻结守卫 + meta）
# --------------------------------------------------------------------------- #
def test_train_loop_smoke_steps_end_to_end(tmp_path, monkeypatch) -> None:
    module = _load_runner()
    args = module.parse_args(
        [
            "--smoke-steps", "2", "--arm", "p1", "--device", "cpu", "--max-pairs", "64",
            "--data", str(FIXTURE), "--train-out", str(tmp_path),
        ]
    )
    data = module.build_lateral_data(args)
    _patch_training_environment(module, monkeypatch, _tiny_model(), _stub_base(data))
    assert module.train_loop(args, data) == 0
    payload = json.loads((tmp_path / "smoke_steps.json").read_text(encoding="utf-8"))
    assert payload["stop"]["steps_done"] == 2
    assert payload["frozen_guard"]["bitwise_unchanged"] is True
    assert payload["grad_probe"] is not None
    assert payload["model"]["allowlist_tensors"] == 18
    assert payload["geometry"]["readout_lane_geometry"] == "default"
    assert not (tmp_path / "ckpt_epoch001.pt").exists()  # smoke 不写 ckpt


def test_train_loop_train_writes_ckpt_and_selection(tmp_path, monkeypatch) -> None:
    module = _load_runner()
    args = module.parse_args(
        [
            "--train", "--epochs", "1", "--arm", "p1", "--device", "cpu", "--max-pairs", "64",
            "--log-every", "1", "--data", str(FIXTURE), "--train-out", str(tmp_path),
        ]
    )
    data = module.build_lateral_data(args)
    model = _tiny_model()
    _patch_training_environment(module, monkeypatch, model, _stub_base(data))
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    assert module.train_loop(args, data) == 0
    ledger = json.loads((tmp_path / "train_log.json").read_text(encoding="utf-8"))
    assert ledger["selection"]["best"]["epoch"] == 1
    ckpt_path = Path(ledger["selection"]["best"]["path"])
    assert ckpt_path.is_file() and (tmp_path / "ckpt_selected.pt").is_file()
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert payload["meta"]["arm"] == "p1"
    assert payload["meta"]["round"] == "round41"
    assert payload["meta"]["lane"] == "g"
    assert "readout_lane_geometry" in payload["meta"]  # R41-D 保存链注入
    state = payload["model"]
    assert set(state) == set(model.state_dict())  # canary 装载（_extract_state）可直接消费
    # allowlist 内确有更新；冻结面逐位不变
    assert any(
        not torch.equal(before[name], p.detach())
        for name, p in model.named_parameters()
        if name.startswith("policy.structured_action.")
    )
    frozen_changed = [
        name for name, p in model.named_parameters()
        if not name.startswith(tuple(payload["meta"]["allowlist"])) and not torch.equal(before[name], p.detach())
    ]
    assert frozen_changed == []
    assert ledger["frozen_guard"]["max_abs_delta"] == 0.0
    assert ledger["hashes"]["ckpt_sha256"]  # 四 hash 之一


# --------------------------------------------------------------------------- #
# 4) P3/P4 训练期未接线 ⇒ fail-fast
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arm", ["p3", "p4"])
def test_train_mode_rejects_p3_p4(arm, tmp_path, monkeypatch) -> None:
    module = _load_runner()
    args = module.parse_args(
        ["--train", "--arm", arm, "--device", "cpu", "--max-pairs", "32", "--data", str(FIXTURE),
         "--train-out", str(tmp_path)]
    )
    with pytest.raises(SystemExit):
        module.train_loop(args, {"main_ids": np.zeros(0, dtype=np.int64)})
