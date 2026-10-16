"""R41-D：几何口径**模型实例配置**（corrected/legacy_aligned）五要件 + 预算常量处置回归。

依据：R41 复审 §二.3（几何模式改为模型实例配置；入 model config / checkpoint metadata /
protocol hash / 模型缓存键 / metrics；进程级全局与环境变量降级为内部默认）+ 任务书 + 台账。
CPU-only（不建 env/不 spawn/不训练）。

断言面：

1. **规范名与映射**：``fixed``→``corrected``、``production``→``legacy_aligned``；哨兵
   ``default``；未知取值 fail-fast；``step_geometry`` 接受规范名且与内部值逐位同结果。
2. **模型实例**：构造参数/属性/``set_readout_lane_geometry``；state_dict 键面/参数不受影响；
   哨兵 ⇒ 两个消费路径零触达（``_lane_geometry_override() is None`` + step_geometry 收
   ``geometry_mode=None``）；显式值 ⇒ 轴敏感（structured 前向 / readout margin 列变化）。
3. **model config**：``pipeline.stages.build_model`` 读取根键（缺席 = 哨兵逐位不变；显式档
   透传）；``task_model_config`` 携带键。
4. **checkpoint metadata**：``stages.save_checkpoint`` 注入 meta（DrivingModel 恒有；
   普通 module 零触达）；``load_checkpoint`` 读取判读（缺失/哨兵 ⇒ 默认；实例未配置 +
   ckpt 显式 ⇒ 采用；显式不一致 ⇒ 警告保持实例）。
5. **protocol hash / 缓存键**：默认档逐字节不变（无新键、旧键 ``(ckpt, device)``）；
   非默认档追加轴且别名同 hash/同键。
6. **eval_runner 解析链**：``_resolve_readout_lane_geometry`` 优先级（CLI/config > ckpt meta >
   config > 默认）与冲突记录；``_load_ckpt_model`` 解析落地 + summary/load_report 记录
   （纯 legacy 默认零记录）；CLI/tools.test 声明与转发；manifest 行仅在非默认时追加。
7. **预算常量**：``net.param_probe`` 不再导出 ``TOTAL_LIMIT``（正式语义移除；快照迁
   ``tools/legacy/budget_constants.py``）；``ACTIVE_LIMIT_WARN`` = 警告阈值（非断言硬门）。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from net.model import DrivingModel
from net.param_probe import ACTIVE_LIMIT_WARN, ACTIVE_LIMIT, TOTAL_CAP
from net.path_head import (
    READOUT_LANE_GEOMETRY_ALIASES,
    READOUT_LANE_GEOMETRY_DEFAULT,
    READOUT_LANE_GEOMETRY_INTERNAL,
    READOUT_LANE_GEOMETRY_MODES,
    normalize_readout_lane_geometry,
    readout_lane_geometry_internal,
    step_geometry,
)
from net.world_state import WorldState, analytic_update, readout, world_state_from_encode
from pipeline import eval_runner as ev
from pipeline import stages
from tests.test_r39_b1_world_state import _obs_structured, _tiny_structured

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _tiny_config(geometry: str | None = None) -> dict:
    config: dict = {
        "hidden_dim": 32,
        "moe": {"experts": {"count": 2, "hidden_dim": 32}},
        "world_model": {"rollout_steps": 6},
        "policy": {},
        "value": {},
        "spatial": {},
    }
    if geometry is not None:
        config["readout_lane_geometry"] = geometry
    return config


def _save_tiny_ckpt(path: Path, *, geometry: str | None = "default", raw_meta: dict | None = None) -> str:
    model = build = stages.build_model(_tiny_config())
    if geometry not in (None, "default"):
        build.set_readout_lane_geometry(geometry)
    if raw_meta is not None:
        # 旧格式（无几何键）/损坏 meta
        torch.save({"model": model.state_dict(), "meta": dict(raw_meta)}, path)
        return str(path)
    stages.save_checkpoint(str(path), build, meta={"stage": "T"})
    return str(path)


def _tiny_spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "specs_tiny.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "count": 2,
        "specs": [
            {"id": 1, "seed": 11, "split": "train", "difficulty": "easy"},
            {"id": 2, "seed": 12, "split": "train", "difficulty": "hard"},
        ],
    }), encoding="utf-8")
    return path


def _forward(model: DrivingModel, obs: dict | None = None) -> dict:
    obs = _obs_structured(batch=1, seed=0) if obs is None else obs
    with torch.no_grad():
        return model(obs, forward_mode="policy_wm_rollout")


def _first_path_state(model: DrivingModel, obs: dict) -> WorldState:
    with torch.no_grad():
        encoded = model.encode(obs)
        state = world_state_from_encode(model.st_gnn, encoded, dt=model.dt)
        analytic_update(state, ds=torch.tensor([2.0, 2.0])[: state.ego_speed.shape[0]],
                        dtheta=torch.tensor([0.02, 0.02])[: state.ego_speed.shape[0]],
                        pose=torch.stack([torch.full_like(state.ego_speed, 1.5),
                                          torch.zeros_like(state.ego_speed),
                                          torch.zeros_like(state.ego_speed)], dim=-1),
                        step=1)
    return state


# =========================================================================== #
# 1. 规范名与映射
# =========================================================================== #
def test_canonical_modes_and_alias_mapping() -> None:
    assert READOUT_LANE_GEOMETRY_MODES == ("corrected", "legacy_aligned")
    assert READOUT_LANE_GEOMETRY_ALIASES == {"fixed": "corrected", "production": "legacy_aligned"}
    assert READOUT_LANE_GEOMETRY_INTERNAL == {"corrected": "fixed", "legacy_aligned": "production"}
    assert normalize_readout_lane_geometry("fixed") == "corrected"
    assert normalize_readout_lane_geometry("production") == "legacy_aligned"
    assert normalize_readout_lane_geometry("CORRECTED") == "corrected"
    assert normalize_readout_lane_geometry(" legacy_aligned ") == "legacy_aligned"
    for sentinel in (None, "", "default", "DEFAULT"):
        assert normalize_readout_lane_geometry(sentinel) == READOUT_LANE_GEOMETRY_DEFAULT
    assert readout_lane_geometry_internal("fixed") == "fixed" and readout_lane_geometry_internal(
        "corrected") == "fixed"
    assert readout_lane_geometry_internal("production") == "production"
    assert readout_lane_geometry_internal(None) is None
    with pytest.raises(ValueError):
        normalize_readout_lane_geometry("bogus")


def test_step_geometry_accepts_canonical_names_bitwise_same_as_internal() -> None:
    anchor = torch.zeros((1, 4, 2)); anchor[0, 1] = torch.tensor([2.0, 0.5])
    anchor[0, 2] = torch.tensor([4.0, 1.0]); anchor[0, 3] = torch.tensor([6.0, 1.5])
    mask = torch.ones((1, 4))
    exog = torch.zeros((1, 8)); exog[0, 1] = 3.5; exog[0, 2] = 3.5
    present = torch.ones((1, 8))
    pose = torch.zeros((1, 3)); progress = torch.zeros((1,))

    fixed = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="fixed")
    corrected = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="corrected")
    prod = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="production")
    legacy = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="legacy_aligned")
    for key in ("anchor_xy", "long_exog", "long_present"):
        assert torch.equal(fixed[key], corrected[key]), key
        assert torch.equal(prod[key], legacy[key]), key
    with pytest.raises(ValueError):
        step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="bogus")


# =========================================================================== #
# 2. 模型实例配置
# =========================================================================== #
def test_model_instance_default_sentinel_and_setter() -> None:
    model = _tiny_structured(seed=0)
    assert model.readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT
    assert model._lane_geometry_override() is None
    assert model.set_readout_lane_geometry("fixed") is model
    assert model.readout_lane_geometry == "corrected" and model._lane_geometry_override() == "fixed"
    model.set_readout_lane_geometry("production")
    assert model.readout_lane_geometry == "legacy_aligned"
    assert model._lane_geometry_override() == "production"
    model.set_readout_lane_geometry(None)
    assert model.readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT
    with pytest.raises(ValueError):
        model.set_readout_lane_geometry("bogus")
    # 构造参数同名字面量（build_model 透传口）
    torch.manual_seed(0)
    explicit = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6,
                            readout_lane_geometry="fixed")
    assert explicit.readout_lane_geometry == "corrected"


def test_instance_config_is_plain_attribute_no_keyface_no_param_change() -> None:
    torch.manual_seed(0)
    base = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    torch.manual_seed(0)
    explicit = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6,
                            readout_lane_geometry="legacy_aligned")
    assert set(base.state_dict().keys()) == set(explicit.state_dict().keys())
    for key in base.state_dict():
        assert torch.equal(base.state_dict()[key], explicit.state_dict()[key]), key
    assert "readout_lane_geometry" not in set(base.state_dict().keys())


def test_default_sentinel_zero_touch_on_step_geometry_and_readout(monkeypatch) -> None:
    """哨兵 ⇒ 两消费路径显式收 ``None``（= 进程级历史默认；逐位不变的结构性证据）。"""
    model = _tiny_structured(seed=0)
    calls: list = []

    import net.model as model_module

    real_geometry = model_module.step_geometry

    def spy_step(*args, **kwargs):
        calls.append(kwargs.get("geometry_mode", "MISSING"))
        return real_geometry(*args, **kwargs)

    monkeypatch.setattr(model_module, "step_geometry", spy_step)
    _forward(model)
    assert calls and all(value is None for value in calls), calls


def test_geometry_axis_moves_structured_forward_and_readout_margins() -> None:
    # 侧向扰动观测：零填充锚（mask=0 段 ⇒ production foot 被吸到 padding、几何整段零化）
    obs = _obs_structured(batch=1, seed=0)
    anchor = torch.tensor([[5.0, 0.0], [10.0, 2.0], [16.0, -2.0], [0.0, 0.0]])
    obs["lane_anchor_xy"] = anchor.unsqueeze(0).clone()
    obs["lane_anchor_mask"] = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    base = _forward(_tiny_structured(seed=0), obs)
    corrected = _forward(_tiny_structured(seed=0).set_readout_lane_geometry("corrected"), obs)
    legacy = _forward(_tiny_structured(seed=0).set_readout_lane_geometry("legacy_aligned"), obs)
    # ws-off structured：默认（= production/legacy）与显式 legacy_aligned 逐位一致；
    # corrected 轴改变想象步几何 ⇒ plan 变化（轴敏感；row0 同、rows≥1 变化 = v3 机制口径）
    assert torch.equal(base["plan"], legacy["plan"])
    delta = (base["plan"] - corrected["plan"]).abs().max().item()
    assert delta > 0.0
    assert torch.equal(base["plan"][:, 0], corrected["plan"][:, 0])

    # readout（ws 路径）：lane_geometry 显式切换 margin 列（lat ≠ 0 ⇒ 符号口径交换，变化）
    obs_m = _obs_structured(batch=1, seed=0)
    anchor_m = torch.tensor([[-2.0, 1.0], [4.0, 1.5], [10.0, 2.0]])
    obs_m["lane_anchor_xy"] = anchor_m.unsqueeze(0).clone()
    obs_m["lane_anchor_mask"] = torch.ones(1, 3)
    model = _tiny_structured(seed=0)
    with torch.no_grad():
        encoded = model.encode(obs_m)
        state = world_state_from_encode(model.st_gnn, encoded, dt=model.dt)
    view_default = readout(state)
    view_legacy = readout(state, lane_geometry="legacy_aligned")
    view_corrected = readout(state, lane_geometry="corrected")
    assert torch.equal(view_corrected["long_exog"], view_default["long_exog"])  # corrected = 现默认
    left_default = view_default["fields"]["left_margin"]
    left_legacy = view_legacy["fields"]["left_margin"]
    right_default = view_default["fields"]["right_margin"]
    right_legacy = view_legacy["fields"]["right_margin"]
    assert not torch.equal(left_default, left_legacy)  # 符号口径交换 ⇒ 值互换
    assert torch.equal(left_default, right_legacy) and torch.equal(right_default, left_legacy)
    assert not torch.equal(left_default, right_default)  # lat ≠ 0 前提（非退化 fixture）


# =========================================================================== #
# 3. model config（config/*.yaml + 构造参数）
# =========================================================================== #
def test_build_model_reads_config_key(tmp_path: Path) -> None:
    assert stages.build_model(_tiny_config()).readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT
    assert stages.build_model(_tiny_config("default")).readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT
    assert stages.build_model(_tiny_config("corrected")).readout_lane_geometry == "corrected"
    assert stages.build_model(_tiny_config("fixed")).readout_lane_geometry == "corrected"
    with pytest.raises(ValueError):
        stages.build_model(_tiny_config("bogus"))
    # 仓库 config/model.yaml：键存在且为哨兵（默认逐位不变）
    merged = stages.load_config(str(ROOT / "config/default.yaml"))
    assert merged.get("readout_lane_geometry") == "default"


# =========================================================================== #
# 4. checkpoint metadata（保存注入 + 读取判读）
# =========================================================================== #
def test_save_checkpoint_injects_geometry_meta(tmp_path: Path) -> None:
    default_model = _tiny_structured(seed=0)
    path = tmp_path / "d.pt"
    stages.save_checkpoint(str(path), default_model, meta={"stage": "A"})
    meta = torch.load(path, map_location="cpu", weights_only=False)["meta"]
    assert meta["readout_lane_geometry"] == READOUT_LANE_GEOMETRY_DEFAULT  # 哨兵照实记录
    explicit = _tiny_structured(seed=0).set_readout_lane_geometry("corrected")
    path2 = tmp_path / "c.pt"
    stages.save_checkpoint(str(path2), explicit)
    assert torch.load(path2, map_location="cpu", weights_only=False)[
        "meta"]["readout_lane_geometry"] == "corrected"
    # 非 DrivingModel（无属性）零触达
    linear = torch.nn.Linear(2, 2)
    path3 = tmp_path / "l.pt"
    save_meta = stages._trainer_save_checkpoint  # 原语 + wrapper 对照
    stages.save_checkpoint(str(path3), linear)
    assert "readout_lane_geometry" not in torch.load(
        path3, map_location="cpu", weights_only=False)["meta"]
    assert stages.geometry_meta_of(linear) is None


def test_load_checkpoint_reads_geometry_meta(tmp_path: Path, capsys) -> None:
    # ① ckpt 哨兵 + 实例哨兵 ⇒ 不改
    sentinel_ckpt = _save_tiny_ckpt(tmp_path / "s.pt")
    model = stages.build_model(_tiny_config())
    meta = stages.load_checkpoint(sentinel_ckpt, model)
    assert meta["readout_lane_geometry"] == READOUT_LANE_GEOMETRY_DEFAULT
    assert model.readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT

    # ② 实例未配置 + ckpt 显式 ⇒ 采用
    explicit_ckpt = _save_tiny_ckpt(tmp_path / "c.pt", geometry="corrected")
    model2 = stages.build_model(_tiny_config())
    stages.load_checkpoint(explicit_ckpt, model2)
    assert model2.readout_lane_geometry == "corrected"
    assert "实例采用" in capsys.readouterr().out

    # ③ 实例显式不一致 ⇒ 警告 + 保持实例
    model3 = stages.build_model(_tiny_config("legacy_aligned"))
    stages.load_checkpoint(explicit_ckpt, model3)
    assert model3.readout_lane_geometry == "legacy_aligned"
    assert "不一致" in capsys.readouterr().out

    # ④ meta 缺失（旧格式）⇒ 按默认（不触达）
    legacy_ckpt = _save_tiny_ckpt(tmp_path / "old.pt", raw_meta={"stage": "X"})
    model4 = stages.build_model(_tiny_config())
    record = stages.readout_geometry_meta_check(model4, {"stage": "X"}, source=legacy_ckpt)
    assert record["value"] == READOUT_LANE_GEOMETRY_DEFAULT and record["source"] == "missing"
    stages.load_checkpoint(legacy_ckpt, model4)
    assert model4.readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT

    # ⑤ meta 非法 ⇒ 警告 + 按缺失
    bad = _save_tiny_ckpt(tmp_path / "bad.pt", raw_meta={"readout_lane_geometry": "nope"})
    model5 = stages.build_model(_tiny_config())
    stages.load_checkpoint(bad, model5)
    assert model5.readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT
    assert "非法" in capsys.readouterr().out


# =========================================================================== #
# 5. protocol hash / 模型缓存键
# =========================================================================== #
def _protocol_base(tmp_path: Path) -> dict:
    spec = _tiny_spec_file(tmp_path)
    return dict(spec_path=spec, policy="ckpt", tracker="exact", eval_reference="plan",
                moe_off=False, path_head="structured", workers=1, max_steps=10, device="cpu",
                net_tree_sha16="deadbeef")


def test_protocol_hash_and_cache_key_default_byte_identical(tmp_path: Path) -> None:
    base = _protocol_base(tmp_path)
    p_default = ev._formal_entry_protocol(**base)
    p_sentinel = ev._formal_entry_protocol(**base, readout_lane_geometry="default")
    p_none = ev._formal_entry_protocol(**base, readout_lane_geometry=None)
    assert p_default == p_sentinel == p_none
    assert "readout_lane_geometry" not in p_default["model"]
    # 别名/显式档：追加键 + hash 变化 + 同档同 hash
    p_corr = ev._formal_entry_protocol(**base, readout_lane_geometry="corrected")
    p_fixed = ev._formal_entry_protocol(**base, readout_lane_geometry="fixed")
    p_legacy = ev._formal_entry_protocol(**base, readout_lane_geometry="legacy_aligned")
    assert p_corr["model"]["readout_lane_geometry"] == "corrected"
    assert p_legacy["model"]["readout_lane_geometry"] == "legacy_aligned"
    assert p_corr["hashes"]["protocol_hash"] == p_fixed["hashes"]["protocol_hash"]
    assert p_corr["hashes"]["protocol_hash"] != p_default["hashes"]["protocol_hash"]
    assert p_legacy["hashes"]["protocol_hash"] not in {
        p_default["hashes"]["protocol_hash"], p_corr["hashes"]["protocol_hash"]}

    # 缓存键：默认 = 旧键；非默认追加段；别名同键；1.0-类哨兵不追加
    k0 = ev._model_cache_key("runs/x.pt", "cpu")
    assert k0 == ("runs/x.pt", "cpu")
    assert ev._model_cache_key("runs/x.pt", "cpu", readout_lane_geometry="default") == k0
    kc = ev._model_cache_key("runs/x.pt", "cpu", readout_lane_geometry="corrected")
    assert kc == ("runs/x.pt", "cpu", ("readout_lane_geometry", "corrected"))
    assert ev._model_cache_key("runs/x.pt", "cpu", readout_lane_geometry="fixed") == kc
    assert ev._model_cache_key("runs/x.pt", "cpu", readout_lane_geometry="legacy_aligned") != kc


# =========================================================================== #
# 6. eval_runner 解析链
# =========================================================================== #
def test_resolve_priority_and_conflict() -> None:
    r = ev._resolve_readout_lane_geometry(requested=None, ckpt_meta=None, config_value=None)
    assert r["value"] == "default" and r["source"] == "default" and not r["conflict"]
    r = ev._resolve_readout_lane_geometry(requested=None, ckpt_meta="corrected")
    assert r["value"] == "corrected" and r["source"] == "checkpoint_metadata"
    r = ev._resolve_readout_lane_geometry(requested=None, ckpt_meta=None, config_value="legacy_aligned")
    assert r["value"] == "legacy_aligned" and r["source"] == "model_config"
    r = ev._resolve_readout_lane_geometry(requested="legacy_aligned", requested_source="cli",
                                          ckpt_meta="corrected", config_value="legacy_aligned")
    assert r["value"] == "legacy_aligned" and r["source"] == "cli" and r["conflict"] is True
    r = ev._resolve_readout_lane_geometry(requested=None, ckpt_meta=None, config_value="corrected",
                                          ckpt_meta_raw="garbage")
    assert r["value"] == "corrected" and r["ckpt_meta"] == "garbage"


def test_load_ckpt_model_resolution_and_record(tmp_path: Path) -> None:
    ev._CKPT_MODEL_CACHE.clear(); ev._CKPT_LOAD_SUMMARY.clear()
    config = _tiny_config()  # 无 config 键 ⇒ 不携带
    # ① 旧格式无 meta（纯 legacy 默认）⇒ 解析记录不落（逐字节不变）+ 模型哨兵
    ckpt_old = _save_tiny_ckpt(tmp_path / "old.pt", raw_meta={"stage": "X"})
    model = ev._load_ckpt_model(ckpt_old, config, "cpu")
    assert model.readout_lane_geometry == READOUT_LANE_GEOMETRY_DEFAULT
    summary = ev._ckpt_load_summary(ckpt_old, "cpu")
    assert "geometry" not in summary
    assert "geometry" not in summary["load_report"]

    # ② 新格式哨兵 meta ⇒ 记录 source（meta 在场）且值 = 默认
    ckpt_new = _save_tiny_ckpt(tmp_path / "new.pt")
    model = ev._load_ckpt_model(ckpt_new, config, "cpu")
    summary = ev._ckpt_load_summary(ckpt_new, "cpu")
    assert summary["geometry"]["readout_lane_geometry"] == READOUT_LANE_GEOMETRY_DEFAULT
    assert summary["geometry"]["ckpt_meta"] == READOUT_LANE_GEOMETRY_DEFAULT

    # ③ CLI 显式（requested）⇒ 模型落地 + source=cli
    ckpt_corr = _save_tiny_ckpt(tmp_path / "c.pt", geometry="corrected")
    model = ev._load_ckpt_model(ckpt_corr, config, "cpu", readout_lane_geometry="legacy_aligned")
    assert model.readout_lane_geometry == "legacy_aligned"
    summary = ev._ckpt_load_summary(ckpt_corr, "cpu", readout_lane_geometry="legacy_aligned")
    assert summary["geometry"]["source"] == "cli"
    assert summary["geometry"]["conflict"] is True

    # ④ CLI 缺省 + ckpt 显式 ⇒ 采用 checkpoint_metadata
    model = ev._load_ckpt_model(ckpt_corr, config, "cpu")
    assert model.readout_lane_geometry == "corrected"
    summary = ev._ckpt_load_summary(ckpt_corr, "cpu")
    assert summary["geometry"]["source"] == "checkpoint_metadata"

    # ⑤ config 键（模型配置入口）⇒ 落地 + source=model_config
    config_corr = _tiny_config("corrected")
    ckpt_sentinel = _save_tiny_ckpt(tmp_path / "s.pt")
    model = ev._load_ckpt_model(ckpt_sentinel, config_corr, "cpu")
    assert model.readout_lane_geometry == "corrected"
    summary = ev._ckpt_load_summary(ckpt_sentinel, "cpu")
    assert summary["geometry"]["source"] == "model_config"


def test_load_ckpt_model_geometry_source_passthrough(tmp_path: Path) -> None:
    """装配侧来源标注（cli vs model_config）透传解析记录（run_meta/metrics 审计口径）。"""
    ev._CKPT_MODEL_CACHE.clear(); ev._CKPT_LOAD_SUMMARY.clear()
    ckpt = _save_tiny_ckpt(tmp_path / "c.pt")
    ev._load_ckpt_model(ckpt, _tiny_config(), "cpu", readout_lane_geometry="corrected",
                        readout_lane_geometry_source="model_config")
    summary = ev._ckpt_load_summary(ckpt, "cpu", readout_lane_geometry="corrected")
    assert summary["geometry"]["source"] == "model_config"
    assert summary["geometry"]["readout_lane_geometry"] == "corrected"


def test_eval_hashes_geometry_record_only_when_non_default(tmp_path: Path) -> None:
    spec = _tiny_spec_file(tmp_path)
    base = dict(spec_path=spec, policy="ckpt", ckpt=None, tracker="exact", eval_reference="plan",
                moe_off=False, path_head="structured", workers=1, max_steps=10, device="cpu")
    default = ev._evaluation_hashes(**base)
    assert "geometry" not in default
    axis = ev._evaluation_hashes(**base, readout_lane_geometry="corrected",
                                 readout_lane_geometry_source="cli")
    assert axis["geometry"] == {"readout_lane_geometry": "corrected", "source": "cli"}
    assert axis["evaluation_protocol_hash"] != default["evaluation_protocol_hash"]


def test_cli_declaration_and_tools_test_passthrough() -> None:
    parser = ev.build_parser()
    assert parser.parse_args([]).readout_lane_geometry == "default"
    for value in ("corrected", "legacy_aligned", "fixed", "production"):
        assert parser.parse_args(["--readout-lane-geometry", value]).readout_lane_geometry == value
    with pytest.raises(SystemExit):
        parser.parse_args(["--readout-lane-geometry", "bogus"])

    # tools/test.py 不重复声明评测轴（R41-D 写域纪律）：评测开关经其 ``_extra`` 透传机制转发
    # （parse_known_args → forward += _drop_name_arg(_extra)；探针端到端见 d_eval_smoke.py 走 tools/test.py）。
    import tools.test as test_entry

    _, extra = test_entry.build_parser().parse_known_args(
        ["--policy", "ckpt", "--ckpt", "x.pt", "--readout-lane-geometry", "fixed"]
    )
    assert "--readout-lane-geometry" in extra and "fixed" in extra


def test_manifest_geometry_line_only_when_non_default(tmp_path: Path) -> None:
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.txt").write_text("stage: eval\n", encoding="utf-8")
    payload = {"hashes": {"evaluation_protocol_hash": "h", "model_tree_hash": "t",
                          "schema_hash": "s", "checkpoint_hash": "c"}}
    ev._record_run_meta(out, dict(payload))
    assert "r41d_geometry:" not in (out / "manifest.txt").read_text(encoding="utf-8")
    payload_axis = dict(payload)
    payload_axis["geometry"] = {"readout_lane_geometry": "corrected", "source": "cli"}
    out2 = tmp_path / "run2"
    out2.mkdir()
    (out2 / "manifest.txt").write_text("stage: eval\n", encoding="utf-8")
    ev._record_run_meta(out2, payload_axis)
    text = (out2 / "manifest.txt").read_text(encoding="utf-8")
    assert "r41d_geometry:" in text and '"readout_lane_geometry": "corrected"' in text


def test_controller_passes_geometry_assembly(tmp_path: Path) -> None:
    ev._CKPT_MODEL_CACHE.clear(); ev._CKPT_LOAD_SUMMARY.clear()
    ckpt = _save_tiny_ckpt(tmp_path / "c.pt", geometry="corrected")
    task = {
        "ckpt": ckpt, "device": "cpu", "model_config": _tiny_config(),
        "tracker": "exact", "eval_reference": "plan",
        "model_assembly": {"path_head": "off", "readout_lane_geometry": "legacy_aligned"},
    }
    controller = ev._CkptController(None, SimpleNamespace(id=0, seed=0), task)
    assert controller.model.readout_lane_geometry == "legacy_aligned"


# =========================================================================== #
# 7. 预算常量处置
# =========================================================================== #
def test_param_probe_budget_constants_disposition() -> None:
    import net.param_probe as probe

    assert ACTIVE_LIMIT_WARN == 1.05
    assert ACTIVE_LIMIT == ACTIVE_LIMIT_WARN  # 兼容别名（语义 = 警告阈值）
    assert TOTAL_CAP == 2_000_000
    assert not hasattr(probe, "TOTAL_LIMIT")  # 正式语义移除

    from tools.legacy import budget_constants as legacy

    assert legacy.TOTAL_LIMIT == 1.02  # 兼容快照（tools/legacy）
    assert legacy.ACTIVE_LIMIT == 1.05
