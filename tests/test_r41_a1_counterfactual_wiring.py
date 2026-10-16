"""R41-A1：oracle/teacher-force/speed 反事实**正式接线**回归（CPU-only；不建 env/不 spawn）。

依据：``docs/chatgpt5p6sol_review.md``（`deb8ee9` R41 复审 §二.1/§二.2/§三 第一批）+
``docs/deepseekv4p1_argue.md``（R41 当期快照 §1.3/§2）+ 任务书 §1–§3。

断言面（每开关五要件 = 正式 CLI ∧ 模型执行路径 ∧ protocol hash ∧ 模型缓存键 ∧
metrics/manifest ∧ 默认关逐位一致）：

1. **CLI**：``--teacher-force {off,ego,od,ld,ego_od_ld,lateral,all}`` / ``--record-truth`` /
   ``--inject-truth`` / ``--speed-scale`` / ``--speed-oracle {off,expert}`` 声明于
   ``pipeline.eval_runner`` 与 ``tools/test.py``（默认 off/None ⇒ 不透传）；
   组合校验（record×inject 互斥；inject 需 teacher_force；teacher_force 需 ws 非 off；
   baseline 策略拒绝；truth 目录指纹）。
2. **protocol hash / 缓存键轴**：非默认档追加轴且默认档逐字节不变（与历史口径 byte-identical）。
3. **模型执行路径（net/model.py 原生；无 .slim patch）**：默认 off 逐位 + state_dict 键面不变；
   状态面替换落到 WorldState 物理字段；横向面 κ 替换语义（ds 不变、dθ=κ·ds、t0 首动作不动）；
   hook 事件五要件（模块/步/来源/替换前后 state hash/替换前后 Policy 输出）；
   dump 边界登记（D-TF01）+ 默认关不可达。
4. **两遍法 truth IO**：记录 round-trip（扩展 schema）+ V4B legacy schema 兼容；
   truth 目录指纹确定性。
5. **一步合法 transition 路线**：engine snapshot/clone 探针（route=two_pass_replay；
   MetaDrive ``BaseEngine`` 无全状态 API）+ 动作条件（命令面/实测面）纯函数 +
   帧级"同状态门" + 五类漂移/阈剔除（``oracle_eligible``）+ 裁决视图剔除非法组合。
6. **metrics/manifest**：反事实 run 的 manifest 行与 ``oracle_audit``；默认关不追加。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch

from net import probes as probes_module
from net.model import TEACHER_FORCE_MODES, TEACHER_FORCE_SURFACES, teacher_force_snapshot
from pipeline import eval_runner as ev
from tests.test_r39_b1_world_state import _obs_structured, _tiny_structured

_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
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


def _payload(k_max: int = 6, *, kappa: float = 0.05, valid: float = 1.0) -> Dict[int, Dict[str, Any]]:
    payload: Dict[int, Dict[str, Any]] = {}
    for k in range(1, int(k_max) + 1):
        payload[k] = {
            "pose": torch.tensor([0.2 * k, 0.05 * k, 0.01 * k]),
            "speed": torch.tensor([5.0 + 0.1 * k]),
            "od_pos": torch.zeros(16, 2), "od_vw": torch.zeros(16, 2),
            "od_cs": torch.ones(16, 2), "od_h": torch.zeros(16),
            "od_static": torch.ones(16, 3), "od_valid": torch.zeros(16),
            "ld_pos": torch.zeros(16, 2), "ld_h": torch.zeros(16),
            "ld_curv": torch.zeros(16), "ld_valid": torch.zeros(16),
            "kappa": torch.tensor([float(kappa)]), "kappa_valid": torch.tensor([float(valid)]),
            "meta": {"truth_file": "synthetic.npz", "recorded_frame": k},
        }
    return payload


def _tiny_ws_model(mode: str = "off") -> Any:
    model = _tiny_structured(seed=0)
    model.set_world_state_api("analytic")
    model.set_teacher_force(mode)
    return model


def _forward(model: Any, obs: Dict[str, torch.Tensor] | None = None) -> Dict[str, Any]:
    obs = _obs_structured(batch=1, seed=0) if obs is None else obs
    with torch.no_grad():
        return model(obs, forward_mode="policy_wm_rollout")


class _StubAgent:
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0, speed: float = 5.0) -> None:
        self.position = (float(x), float(y))
        self.heading_theta = float(theta)
        self.speed = float(speed)


def _stub_env(step: int = 0, rng: Any = None) -> Any:
    env = SimpleNamespace()
    env.agent = _StubAgent()
    env.episode_step = int(step)
    engine = SimpleNamespace()
    if rng is not None:
        engine.np_random = rng
    env.engine = engine
    return env


def _stub_obs(frames: int = 1) -> Dict[str, np.ndarray]:
    return {
        "ego": np.zeros((1, 8), dtype=np.float32),
        "od": np.zeros((16, 9), dtype=np.float32),
        "od_mask": np.zeros(16, dtype=np.float32),
        "od_presence": np.zeros(16, dtype=np.float32),
        "ld": np.zeros((16, 7), dtype=np.float32),
        "ld_mask": np.zeros(16, dtype=np.float32),
    }


# =========================================================================== #
# 1. 五要件之"正式 CLI"
# =========================================================================== #
def test_cli_declares_counterfactual_switches_and_defaults() -> None:
    parser = ev.build_parser()
    args = parser.parse_args([])
    assert args.teacher_force == "off"
    assert args.record_truth is None and args.inject_truth is None
    assert args.speed_scale is None and args.speed_oracle == "off"
    for mode in ev.LOAD_TEACHER_FORCE_MODES:
        assert parser.parse_args(["--teacher-force", mode]).teacher_force == mode
    for mode in ev.LOAD_SPEED_ORACLE_MODES:
        assert parser.parse_args(["--speed-oracle", mode]).speed_oracle == mode
    assert parser.parse_args(["--speed-scale", "0.75"]).speed_scale == pytest.approx(0.75)
    with pytest.raises(SystemExit):
        parser.parse_args(["--teacher-force", "bogus"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--speed-oracle", "bogus"])
    # 档位与 net 侧常量同源
    assert tuple(ev.LOAD_TEACHER_FORCE_MODES) == tuple(TEACHER_FORCE_MODES)


def test_tools_test_entry_declares_and_forwards_counterfactual_switches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tools.test as test_entry

    captured: Dict[str, Any] = {}

    def _fake_eval_main(argv):  # noqa: ANN001
        captured["argv"] = [str(item) for item in argv]
        return 0

    monkeypatch.setattr("pipeline.eval_runner.main", _fake_eval_main)
    monkeypatch.setattr("pipeline.gl_runtime.ensure_gl_library_path", lambda **kwargs: None)
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    rc = test_entry.main([
        "--config", str(_ROOT / "config/default.yaml"),
        "--policy", "ckpt",
        "--ckpt", str(_ROOT / "config/default.yaml"),
        "--out", str(tmp_path),
        "--teacher-force", "all",
        "--inject-truth", str(truth_dir),
        "--speed-scale", "0.5",
        "--speed-oracle", "expert",
    ])
    assert rc == 0
    argv = captured["argv"]
    assert argv[argv.index("--teacher-force") + 1] == "all"
    assert argv[argv.index("--inject-truth") + 1] == str(truth_dir)
    assert argv[argv.index("--speed-scale") + 1] == "0.5"
    assert argv[argv.index("--speed-oracle") + 1] == "expert"
    # 默认关 ⇒ 不透传（逐位不变）
    captured.clear()
    rc = test_entry.main([
        "--config", str(_ROOT / "config/default.yaml"),
        "--policy", "baseline", "--out", str(tmp_path),
    ])
    assert rc == 0
    assert "--teacher-force" not in captured["argv"]
    assert "--record-truth" not in captured["argv"]
    assert "--inject-truth" not in captured["argv"]
    assert "--speed-scale" not in captured["argv"]
    assert "--speed-oracle" not in captured["argv"]


def test_counterfactual_from_args_default_inactive_and_validation(tmp_path: Path) -> None:
    parser = ev.build_parser()
    cf, err = ev._counterfactual_from_args(parser.parse_args([]), {})
    assert err is None and cf["active"] is False
    assert cf["inject_truth_sha16"] is None and cf["speed_scale"] is None

    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    np.savez_compressed(str(truth_dir / "0001.npz"), a=np.zeros(1))

    def build(*argv: str):
        return ev._counterfactual_from_args(parser.parse_args(list(argv)), {})

    # baseline 拒绝
    _, err = build("--policy", "baseline", "--speed-scale", "0.5")
    assert err and "ckpt" in err
    # record × inject 互斥
    _, err = build("--policy", "ckpt", "--teacher-force", "ego", "--world-state-api", "analytic",
                   "--record-truth", str(truth_dir), "--inject-truth", str(truth_dir))
    assert err and "互斥" in err
    # inject 需 teacher_force
    _, err = build("--policy", "ckpt", "--inject-truth", str(truth_dir))
    assert err and "teacher-force" in err
    # teacher_force 需 ws 非 off
    _, err = build("--policy", "ckpt", "--teacher-force", "ego", "--inject-truth", str(truth_dir))
    assert err and "world-state-api" in err
    # teacher_force 需 inject-truth
    _, err = build("--policy", "ckpt", "--teacher-force", "ego", "--world-state-api", "analytic")
    assert err and "inject-truth" in err
    # 目录缺失 / 空目录
    _, err = build("--policy", "ckpt", "--teacher-force", "ego", "--world-state-api", "analytic",
                   "--inject-truth", str(tmp_path / "missing"))
    assert err and "不存在" in err
    empty = tmp_path / "empty"
    empty.mkdir()
    _, err = build("--policy", "ckpt", "--teacher-force", "ego", "--world-state-api", "analytic",
                   "--inject-truth", str(empty))
    assert err and "无 npz" in err
    # speed-scale 非法
    _, err = build("--policy", "ckpt", "--speed-scale", "0")
    assert err and "speed-scale" in err
    _, err = build("--policy", "ckpt", "--speed-scale", "nan")
    assert err and "speed-scale" in err
    # 合法：truth 指纹 + 激活
    cf, err = build("--policy", "ckpt", "--teacher-force", "ego", "--world-state-api", "analytic",
                    "--inject-truth", str(truth_dir), "--speed-scale", "1.0")
    assert err is None and cf["active"] is True
    assert cf["inject_truth_sha16"] and len(cf["inject_truth_sha16"]) == 16
    assert cf["inject_truth_digest"]["n_files"] == 1
    assert cf["speed_scale"] is None  # 显式 1.0 = 默认档（不触达）
    # 阈值覆盖（config 口径）
    cf2, err = ev._counterfactual_from_args(
        parser.parse_args(["--policy", "ckpt", "--speed-scale", "0.5"]),
        {"counterfactual": {"drift_thresholds": {"pose_p95_m": 0.123}}},
    )
    assert err is None and cf2["drift_thresholds"]["pose_p95_m"] == pytest.approx(0.123)


# =========================================================================== #
# 2. 五要件之 protocol hash + 模型缓存键（轴敏感 ∧ 默认逐字节不变）
# =========================================================================== #
def test_protocol_hash_axes_conditional_and_default_byte_identical(tmp_path: Path) -> None:
    spec = _tiny_spec_file(tmp_path)
    base = dict(
        spec_path=spec, policy="ckpt", tracker="exact", eval_reference="plan", moe_off=False,
        path_head="structured", workers=1, max_steps=10, device="cpu", net_tree_sha16="deadbeefdeadbeef",
        world_state_api="analytic",
    )
    protocol_default = ev._formal_entry_protocol(**base)
    explicit_default = ev._formal_entry_protocol(
        **base, teacher_force="off", record_truth=False, inject_truth_sha16=None,
        speed_scale=None, speed_oracle="off",
    )
    assert protocol_default == explicit_default, "默认档必须逐字节不变"
    assert "teacher_force" not in protocol_default["model"]
    # 非默认档逐轴入 hash（且互不相同）
    hashes = {}
    for name, kwargs in (
        ("tf", dict(teacher_force="ego", inject_truth_sha16="a" * 16)),
        ("record", dict(record_truth=True)),
        ("speed", dict(speed_scale=0.5)),
        ("oracle", dict(speed_oracle="expert")),
    ):
        protocol = ev._formal_entry_protocol(**base, **kwargs)
        hashes[name] = protocol["hashes"]["protocol_hash"]
        assert hashes[name] != protocol_default["hashes"]["protocol_hash"]
    assert len(set(hashes.values())) == len(hashes)
    protocol_tf = ev._formal_entry_protocol(**base, teacher_force="all", inject_truth_sha16="b" * 16)
    assert protocol_tf["model"]["teacher_force"] == "all"
    assert protocol_tf["model"]["inject_truth_sha16"] == "b" * 16
    protocol_record = ev._formal_entry_protocol(**base, record_truth=True)
    assert protocol_record["model"]["record_truth"] == "on"
    protocol_speed = ev._formal_entry_protocol(**base, speed_scale=0.5, speed_oracle="expert")
    assert protocol_speed["model"]["speed_scale"] == pytest.approx(0.5)
    assert protocol_speed["model"]["speed_oracle"] == "expert"
    # 缺省 protocol（off/tf=none）逐字节同历史口径：与未传轴的 dict 相等
    assert protocol_speed != protocol_default
    # speed_scale=1.0（默认语义）⇒ 不追加键
    assert ev._formal_entry_protocol(**base, speed_scale=1.0) == protocol_default


def test_model_cache_key_axes_stable_by_default() -> None:
    default = ev._model_cache_key("ckpt.pt", "cpu")
    explicit_default = ev._model_cache_key(
        "ckpt.pt", "cpu", teacher_force="off", record_truth=False, inject_truth_sha16=None,
        speed_scale=None, speed_oracle="off",
    )
    assert default == explicit_default == ("ckpt.pt", "cpu")
    one = ev._model_cache_key("ckpt.pt", "cpu", teacher_force="ego", inject_truth_sha16="x" * 16)
    two = ev._model_cache_key("ckpt.pt", "cpu", record_truth=True)
    three = ev._model_cache_key("ckpt.pt", "cpu", speed_scale=0.5)
    four = ev._model_cache_key("ckpt.pt", "cpu", speed_oracle="expert")
    assert len({default, one, two, three, four}) == 5
    assert all(entry[0][0] == "counterfactual" for entry in (one[2:], two[2:], three[2:], four[2:]))
    # 全部默认语义（1.0/约束）不改变旧键
    assert ev._model_cache_key("ckpt.pt", "cpu", speed_scale=1.0) == default


def test_load_ckpt_model_applies_teacher_force_and_load_summary(tmp_path: Path) -> None:
    """模型装载路径真实设置 teacher_force（非 .slim patch）；LoadReport 键面不受影响。"""
    from pipeline.stages import build_model, load_config

    config = load_config(str(_ROOT / "config" / "model.yaml"))
    model = build_model(config)
    model.set_path_head(True)
    model.set_path_head_ablation("structured")
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": {k: v.clone() for k, v in model.state_dict().items()}}, ckpt)
    ev._CKPT_MODEL_CACHE.clear()
    ev._CKPT_LOAD_SUMMARY.clear()
    try:
        loaded = ev._load_ckpt_model(
            str(ckpt), config, "cpu", path_head="structured", world_state_api="analytic",
            teacher_force="ld", inject_truth_sha16="c" * 16,
        )
        assert loaded.teacher_force_mode == "ld"
        summary = ev._ckpt_load_summary(
            str(ckpt), "cpu", path_head="structured", world_state_api="analytic",
            teacher_force="ld", inject_truth_sha16="c" * 16,
        )
        assert summary["counterfactual"]["teacher_force"] == "ld"
        # 默认档装载不设置 teacher_force（off）且摘要无 counterfactual 键
        ev._CKPT_MODEL_CACHE.clear()
        ev._CKPT_LOAD_SUMMARY.clear()
        default_loaded = ev._load_ckpt_model(str(ckpt), config, "cpu", path_head="structured")
        assert default_loaded.teacher_force_mode == "off"
        default_summary = ev._ckpt_load_summary(str(ckpt), "cpu", path_head="structured")
        assert "counterfactual" not in default_summary
    finally:
        ev._CKPT_MODEL_CACHE.clear()
        ev._CKPT_LOAD_SUMMARY.clear()


# =========================================================================== #
# 3. 五要件之"模型执行路径"（net/model.py 原生；默认关逐位）
# =========================================================================== #
def test_teacher_force_off_bitwise_and_keyface_unchanged() -> None:
    model_off = _tiny_ws_model("off")
    model_default = _tiny_structured(seed=0)
    assert model_default.teacher_force_mode == "off"
    # state_dict 键面/参数不因开关新增
    assert sorted(model_off.state_dict()) == sorted(model_default.state_dict())
    with torch.enable_grad():
        out_a = _forward(model_off)
    with torch.enable_grad():
        out_b = _forward(model_off)
    assert all(
        torch.equal(out_a[key], out_b[key])
        for key in out_a if torch.is_tensor(out_a[key])
    )
    # 开→关往返逐位一致
    model_off.set_teacher_force("all")
    model_off.set_teacher_force("off")
    out_c = _forward(model_off)
    assert all(torch.equal(out_a[key], out_c[key]) for key in out_a if torch.is_tensor(out_a[key]))
    with pytest.raises(ValueError):
        model_off.set_teacher_force("bogus")


def test_teacher_force_state_injection_replaces_world_state_fields() -> None:
    model = _tiny_ws_model("ego_od_ld")
    baseline = _forward(model)
    payload = _payload()
    model.begin_teacher_force_frame(payload, frame=3)
    injected = _forward(model)
    stats = model.end_teacher_force_frame()
    assert not torch.equal(baseline["plan"], injected["plan"])
    assert stats["n_ego_rows"] == 6 and stats["n_od_rows"] == 6 and stats["n_ld_rows"] == 6
    assert stats["n_lateral_rows"] == 0
    # 末步（k=6）状态替换后 WorldState 物理字段 = 载荷（pose/speed 逐位）
    ws = model._world_state
    assert torch.allclose(ws.ego_pose.reshape(3), payload[6]["pose"], atol=0, rtol=0)
    assert float(ws.ego_speed.reshape(-1)[0]) == pytest.approx(float(payload[6]["speed"][0]))
    assert ws.ego_pose.grad_fn is None and ws.od_anchor.grad_fn is None


def test_teacher_force_lateral_patch_semantics() -> None:
    model = _tiny_ws_model("lateral")
    baseline = _forward(model)
    payload = _payload(kappa=0.2)
    model.begin_teacher_force_frame(payload, frame=0)
    patched = _forward(model)
    stats = model.end_teacher_force_frame()
    assert stats["n_lateral_rows"] == 5  # k=1..5 头调用；t0（a_0）不动（口径 = V4B）
    assert stats["n_state_rows"] == 0
    # t0 首步（structured_kappa_near[:,0]）不变；想象步 κ 被替换为载荷 κ
    assert torch.equal(baseline["structured_kappa_near"][:, 0], patched["structured_kappa_near"][:, 0])
    for k in range(1, 6):
        assert float(patched["structured_kappa_near"][0, k]) == pytest.approx(0.2, abs=1e-6)
    ds = patched["structured_ds"][0, 1:6]
    dtheta = patched["plan"][0, 1:6, 1]
    assert torch.allclose(dtheta, 0.2 * ds, atol=1e-6, rtol=0), "dθ = κ·ds 语义"
    assert not torch.equal(baseline["plan"][:, 1:, :], patched["plan"][:, 1:, :])
    # κ 无效 ⇒ 跳过（计入统计，不替换）
    model.begin_teacher_force_frame(_payload(kappa=0.2, valid=0.0), frame=0)
    skipped = _forward(model)
    stats_skip = model.end_teacher_force_frame()
    assert stats_skip["n_lateral_rows"] == 0 and stats_skip["n_skip_kappa_invalid"] == 5
    assert torch.equal(skipped["plan"], baseline["plan"])


def test_teacher_force_hook_event_five_elements() -> None:
    events: list[dict] = []

    class _Hook:
        def on_injection(self, event: dict) -> None:
            events.append(dict(event))

    model = _tiny_ws_model("all")
    model.set_teacher_force_hooks(_Hook())
    model.begin_teacher_force_frame(_payload(), frame=7)
    _forward(model)
    model.end_teacher_force_frame()
    assert events, "hook 未收到注入事件"
    surfaces = {event["surface"] for event in events}
    assert surfaces == {"ego", "od", "ld", "lateral"}
    for event in events:
        # ① 注入模块 ② 注入步 ③ 真实来源 ④ 替换前后 state ⑤ 替换前后 Policy 输出
        assert event["module"].startswith("net.model._rollout")
        assert 1 <= int(event["step"]) <= 6
        assert event["frame"] == 7
        assert event["meta"]["recorded_frame"] == int(event["step"])
        assert event["meta"]["truth_file"] == "synthetic.npz"
        if event["surface"] == "lateral":
            assert event["before_state"] is None and event["after_state"] is None
            before = event["before_policy"]
            after = event["after_policy"]
        else:
            assert event["before_state"] and event["after_state"]
            before = event["before_policy"]
            after = event["after_policy"]
            if before is not None:
                assert sorted(before) == ["action", "ds_exec", "kappa_near"]
                assert sorted(after) == ["action", "ds_exec", "kappa_near"]
                assert ev._tensor_bundle_digest(before) != ev._tensor_bundle_digest(after)
        if event["surface"] == "lateral":
            kappa_after = float(after["kappa_near"].reshape(-1)[0])
            ds_after = float(after["ds_exec"].reshape(-1)[0])
            assert kappa_after == pytest.approx(0.05, abs=1e-6)
            assert float(after["dtheta"].reshape(-1)[0]) == pytest.approx(kappa_after * ds_after, abs=1e-6)
    # 状态事件 hash 口径：替换前后 state hash 不同（ego 面）
    ego = [event for event in events if event["surface"] == "ego"][0]
    assert ev._tensor_bundle_digest(ego["before_state"]) != ev._tensor_bundle_digest(ego["after_state"])
    # 末步（k=6，无头调用）事件 policy 输出显式 None（不伪填）
    last_ego = [event for event in events if event["surface"] == "ego" and event["step"] == 6][0]
    assert last_ego["before_policy"] is None and last_ego["after_policy"] is None


def test_teacher_force_requires_ws_and_batch_singleton() -> None:
    # 状态面需要 ws_on（world_state_api=off ⇒ fail-fast）
    model = _tiny_structured(seed=0)
    model.set_teacher_force("ego")
    model.begin_teacher_force_frame(_payload(), frame=0)
    with pytest.raises(ValueError, match="world_state_api"):
        _forward(model)
    # batch=1 约束
    model = _tiny_ws_model("ego")
    model.begin_teacher_force_frame(_payload(), frame=0)
    obs = _obs_structured(batch=2, seed=0)
    with pytest.raises(ValueError, match="batch=1"):
        with torch.no_grad():
            model(obs, forward_mode="policy_wm_rollout")


def test_teacher_force_snapshot_detach_reach() -> None:
    """D-TF01（teacher_force_snapshot）可达性：默认关不可达；快照 detached、主图不受污染。"""
    from net import model as model_mod

    calls = {"n": 0}
    original = model_mod.teacher_force_snapshot

    def counting(tensors):  # noqa: ANN001
        calls["n"] += 1
        return original(tensors)

    model_mod.teacher_force_snapshot = counting
    try:
        model = _tiny_ws_model("off")
        model.begin_teacher_force_frame(_payload(), frame=0)
        _forward(model)
        model.end_teacher_force_frame()
        assert calls["n"] == 0, "默认关（off）不得触达 detach 快照"
        events: list[dict] = []

        class _Hook:
            def on_injection(self, event: dict) -> None:  # noqa: D401
                events.append(dict(event))

        model = _tiny_ws_model("all")
        model.set_teacher_force_hooks(_Hook())
        model.begin_teacher_force_frame(_payload(), frame=0)
        with torch.enable_grad():
            out = model(_obs_structured(batch=1, seed=0), forward_mode="policy_wm_rollout")
        model.end_teacher_force_frame()
        assert calls["n"] > 0
        checked = []
        for event in events:
            for key in ("before_state", "after_state", "before_policy", "after_policy"):
                bundle = event.get(key)
                if isinstance(bundle, dict):
                    checked.extend(
                        bool(torch.is_tensor(v) and v.grad_fn is None and not v.requires_grad)
                        for v in bundle.values()
                    )
        assert checked and all(checked), "快照张量必须 detached（grad_fn=None）"
        plan_loss = out["plan"][:, 1:, :].pow(2).sum()
        head_grads = torch.autograd.grad(
            plan_loss, [p for p in model.policy.structured_action.parameters()],
            allow_unused=True, retain_graph=True,
        )
        assert sum(g.norm() for g in head_grads if g is not None) > 0.0, "主图梯度链必须存活"
        assert float(out["contract_runtime_ok"]) == 1.0, "探针头调用不得破坏 structured 契约"
    finally:
        model_mod.teacher_force_snapshot = original


def test_detach_registry_covers_teacher_force_snapshot() -> None:
    report = probes_module.check_detach_registry()
    assert report["ok"] is True, (report["unregistered"], [e["id"] for e in report["stale"]])
    entry = [e for e in probes_module.DETACH_REGISTRY if e["id"] == "D-TF01"]
    assert entry and entry[0]["tensor"] == "value.detach()"
    assert entry[0]["function"] == "teacher_force_snapshot"


# =========================================================================== #
# 4. 两遍法 truth IO
# =========================================================================== #
class _Rng:
    def get_state(self):  # noqa: ANN201
        return ("MT19937", np.arange(624, dtype=np.uint32), 3, 0, 0.0)


def test_truth_recorder_roundtrip_and_payload(tmp_path: Path) -> None:
    recorder = ev._TruthRecorder()
    env = _stub_env(step=5, rng=_Rng())
    obs = _stub_obs()
    obs["od"][0, :4] = [10.0, 1.0, -2.0, 0.0]
    obs["od_mask"][:4] = 1.0
    obs["od_presence"][:4] = 1.0
    labels = {"action": np.asarray([8.0, 0.1])}
    for frame in range(3):
        recorder.capture_step(env=env, step_index=frame, rng_digest=12345)
        recorder.capture_frame(
            env=env, obs=obs, label=labels, action_exec=np.asarray([1.0, 0.05]),
            plan_exec=np.tile(np.asarray([1.0, 0.05]), (6, 1)), rng_digest=12345,
        )
    path = recorder.save(tmp_path, 7, termination="arrive_dest", meta={"protocol_hash": "h"})
    assert path.name == "0007.npz"
    record = ev._TruthRecord(path)
    summary = record.summary()
    assert summary["n_frames"] == 3 and summary["n_steps"] == 3
    assert summary["extended_schema"] is True and summary["has_rng"] is True
    assert summary["has_step_trace"] is True
    assert summary["expert_valid_frac"] == pytest.approx(1.0)
    assert record.meta["protocol_hash"] == "h"
    assert record.termination_code == ev.TERMINATION_CODES["arrive_dest"]
    # 载荷：k=1 的 pose = 相对位姿（同帧 ⇒ 0）；od 重表达有限；κ 有效
    payload = record.build_payload(0, device="cpu")
    assert sorted(payload) == [1, 2]
    entry = payload[1]
    assert entry["pose"].shape == (3,)
    assert float(entry["pose"][0]) == pytest.approx(0.0, abs=1e-9)
    assert entry["od_pos"].shape == (16, 2) and entry["ld_valid"].shape == (16,)
    assert float(entry["kappa_valid"].reshape(-1)[0]) == 1.0
    assert entry["meta"]["recorded_frame"] == 1
    # 实测/命令动作证据
    realized = record.realized_action(0)
    assert realized is not None and float(realized[0]) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(record.commanded_action(0, 1), [1.0, 0.05])
    assert record.rng_digest(0) == 12345


def test_truth_record_legacy_v4b_schema_compat(tmp_path: Path) -> None:
    """V4B legacy 键面（无 action_exec/plan_exec/rng/step 轨迹）必须可读（能力显式降级）。"""
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        str(path),
        frame_env_step=np.arange(4, dtype=np.float64),
        ego_pose_world=np.arange(12, dtype=np.float64).reshape(4, 3),
        ego_speed=np.full(4, 5.0),
        od=np.zeros((4, 16, 9), dtype=np.float32),
        od_mask=np.zeros((4, 16), dtype=np.float32),
        od_presence=np.zeros((4, 16), dtype=np.float32),
        ld=np.zeros((4, 16, 7), dtype=np.float32),
        ld_mask=np.zeros((4, 16), dtype=np.float32),
        kappa_expert=np.full(4, 0.1), ds_expert=np.full(4, 8.0), expert_valid=np.ones(4),
    )
    record = ev._TruthRecord(path)
    summary = record.summary()
    assert summary["extended_schema"] is False
    assert summary["has_rng"] is False and summary["has_step_trace"] is False
    assert record.n_steps == 0 and record.termination_code == ev.TERMINATION_CODES["other"]
    assert record.commanded_action(0, 1) is None
    assert record.rng_digest(0) is None
    assert record.realized_action(0) is not None  # 实测口径（位姿派生）仍可用
    payload = record.build_payload(0)
    assert sorted(payload) == [1, 2, 3]  # 4 帧 ⇒ k=1..3


def test_truth_dir_digest_deterministic_and_detects_change(tmp_path: Path) -> None:
    for spec_id in (1, 2):
        np.savez_compressed(str(tmp_path / f"{spec_id:04d}.npz"), a=np.asarray([spec_id]))
    first = ev._truth_dir_digest(tmp_path)
    second = ev._truth_dir_digest(tmp_path)
    assert first == second and first["n_files"] == 2 and len(first["sha16"]) == 16
    np.savez_compressed(str(tmp_path / "0003.npz"), a=np.asarray([3]))
    assert ev._truth_dir_digest(tmp_path)["sha16"] != first["sha16"]


def test_rng_digest_stable_and_sensitive() -> None:
    rng_a = np.random.RandomState(42)
    rng_b = np.random.RandomState(42)
    rng_c = np.random.RandomState(43)
    assert ev._rng_digest(rng_a) == ev._rng_digest(rng_b)
    assert ev._rng_digest(rng_b) != ev._rng_digest(rng_c)
    assert ev._rng_digest(None) is None


# =========================================================================== #
# 5. 一步合法 transition（路线探针 + 动作条件 + 漂移/剔除）
# =========================================================================== #
def test_probe_env_snapshot_support_routes() -> None:
    plain = ev.probe_env_snapshot_support(_stub_env())
    assert plain["supported"] is False and plain["route"] == "two_pass_replay"
    faked = ev.probe_env_snapshot_support(SimpleNamespace(engine=SimpleNamespace(dump_state=lambda: None)))
    assert faked["supported"] is True and faked["route"] == "clone_env_branch"
    assert "engine.dump_state" in faked["present"]


def test_metadrive_engine_lacks_full_state_snapshot_api() -> None:
    """§三.① 事实核实：MetaDrive engine 无 dump_state/load_state/snapshot/clone（source 为准）。"""
    metadrive = pytest.importorskip("metadrive")
    from metadrive.engine.base_engine import BaseEngine

    del metadrive
    present = [
        name for name in ("dump_state", "load_state", "snapshot", "clone", "save_state", "restore_state")
        if callable(getattr(BaseEngine, name, None))
    ]
    assert present == [], f"MetaDrive engine 出现全状态 API {present} ⇒ 应改走 clone 路线"


def test_action_condition_and_drift_helpers() -> None:
    pose_a = np.asarray([0.0, 0.0, 0.0])
    pose_b = np.asarray([1.0, 0.0, math.pi / 2])
    action = ev._pose_realized_action(pose_a, pose_b)
    assert float(action[0]) == pytest.approx(1.0)
    assert float(action[1]) == pytest.approx(math.pi / 2)
    od = ev._od_divergence(
        np.asarray([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        np.asarray([1.0]), np.asarray([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        np.asarray([1.0]), np.asarray([1.0]),
    )
    assert od["available"] and od["mean_abs"] == pytest.approx(1.0 / 9.0, rel=1e-6)
    assert ev._p95([1.0, 2.0, 3.0, 4.0]) == pytest.approx(3.85, rel=1e-6)
    assert ev._mean_abs([-2.0, 2.0]) == pytest.approx(2.0)
    assert ev._max_abs([-3.0, 2.0]) == pytest.approx(3.0)
    assert ev._p95([]) is None
    # 速度反事实纯函数（plan 同缩 ⇒ κ 不变；oracle clip）
    plan = np.asarray([[2.0, 0.2], [4.0, -0.4]])
    scaled = ev._apply_plan_speed_scale(plan, 0.5)
    assert np.allclose(scaled, plan * 0.5)
    kappa_before = plan[:, 1] / plan[:, 0]
    kappa_after = scaled[:, 1] / scaled[:, 0]
    assert np.allclose(kappa_before, kappa_after)
    with pytest.raises(ValueError):
        ev._apply_plan_speed_scale(plan, 0.0)
    assert ev._plan_scale_factor(1.0, "expert", 4.0, 2.0) == pytest.approx(0.5)
    assert ev._plan_scale_factor(0.75, "expert", 0.5, 100.0) == pytest.approx(0.75 * 2.0)
    assert ev._plan_scale_factor(0.75, "expert", 4.0, None) == pytest.approx(0.75)
    assert ev._plan_scale_factor(0.75, "off", 4.0, 2.0) == pytest.approx(0.75)


def _synthetic_truth(tmp_path: Path, spec_id: int = 1, *, n_frames: int = 4) -> Path:
    """合成 truth（扩展 schema）：位姿线性推进 + 命令/实测动作一致 + RNG 恒值。"""
    poses = np.stack([np.arange(n_frames, dtype=np.float64), np.zeros(n_frames), np.zeros(n_frames)], axis=-1)
    plan_exec = np.tile(np.asarray([1.0, 0.0]), (n_frames, 1, 1))
    path = ev._truth_path(tmp_path, spec_id)
    np.savez_compressed(
        str(path),
        frame_env_step=np.arange(n_frames, dtype=np.float64) * 5,
        ego_pose_world=poses,
        ego_speed=np.full(n_frames, 5.0),
        od=np.zeros((n_frames, 16, 9), dtype=np.float32),
        od_mask=np.zeros((n_frames, 16), dtype=np.float32),
        od_presence=np.zeros((n_frames, 16), dtype=np.float32),
        ld=np.zeros((n_frames, 16, 7), dtype=np.float32),
        ld_mask=np.zeros((n_frames, 16), dtype=np.float32),
        kappa_expert=np.full(n_frames, 0.05), ds_expert=np.full(n_frames, 5.0),
        expert_valid=np.ones(n_frames),
        action_exec=plan_exec[:, 0, :], plan_exec=plan_exec,
        rng_digest=np.full(n_frames, 999.0),
        step_index=np.arange(n_frames * 2, dtype=np.float64),
        step_pose=np.repeat(poses, 2, axis=0),
        step_speed=np.full(n_frames * 2, 5.0),
        step_rng_digest=np.full(n_frames * 2, 999.0),
        termination_code=np.asarray([ev.TERMINATION_CODES["max_step"]]),
        n_steps=np.asarray([n_frames * 2]),
        meta_json=np.asarray(json.dumps({"protocol_hash": "h"})),
    )
    return path


def _bare_controller(tmp_path: Path, spec_id: int = 1) -> Any:
    """object.__new__ 构造的 controller（仅注入 finish/episode 聚合所需字段；不建 env）。"""
    controller = object.__new__(ev._CkptController)
    controller.spec = SimpleNamespace(id=spec_id, seed=1)
    controller.cf_active = True
    controller.teacher_force_mode = "all"
    controller.record_truth_dir = None
    controller.inject_truth_dir = str(tmp_path)
    controller.speed_scale = None
    controller.speed_oracle = "off"
    controller.cf_thresholds = dict(ev.DEFAULT_DRIFT_THRESHOLDS)
    controller.cf_frame_gate = dict(ev.DEFAULT_FRAME_GATE)
    controller.cf_out_dir = str(tmp_path / "out")
    controller.cf_protocol_hash = "h"
    controller.device = "cpu"
    controller.recorder = None
    controller.model = SimpleNamespace(set_teacher_force_hooks=lambda hooks: None)
    controller._cf_reset_episode()
    controller.record = ev._TruthRecord(_synthetic_truth(tmp_path, spec_id))
    controller.hook = ev._OracleInjectionHook(spec_id=spec_id, truth_summary=controller.record.summary())
    controller._oracle_route = {"route": "two_pass_replay", "supported": False}
    return controller


def _emit_event(hook: Any, *, frame: int, step: int, surface: str = "ego") -> None:
    hook.on_injection({
        "module": f"net.model._rollout.analytic_update.{surface}",
        "surface": surface, "step": step, "frame": frame, "status": "injected",
        "meta": {"truth_file": "synthetic.npz", "recorded_frame": frame + step},
        "before_state": {"ego_speed": torch.tensor([5.0])},
        "after_state": {"ego_speed": torch.tensor([5.1])},
        "before_policy": {"action": torch.tensor([[1.0, 0.0]])},
        "after_policy": {"action": torch.tensor([[1.1, 0.0]])},
    })


def test_oracle_hook_classifies_and_writes_detail(tmp_path: Path) -> None:
    hook = ev._OracleInjectionHook(
        spec_id=3, truth_summary={"path": "x.npz", "n_frames": 4}, detail_cap=2
    )
    _emit_event(hook, frame=0, step=1)
    _emit_event(hook, frame=0, step=2)
    _emit_event(hook, frame=1, step=1)
    hook.note_action_condition(0, {"steps": {
        1: {"legal": True, "unproven": False, "criteria": {}},
        2: {"legal": False, "unproven": False, "criteria": {}},
    }})
    hook.note_action_condition(1, {"steps": {1: {"legal": False, "unproven": True, "criteria": {}}}})
    summary = hook.finalize()
    assert summary["counts"]["n_events"] == 3
    assert summary["counts"]["n_legal"] == 1
    assert summary["counts"]["n_illegal"] == 1
    assert summary["counts"]["n_unproven"] == 1
    assert summary["n_detail_recorded"] == 2 and summary["n_detail_dropped"] == 1
    statuses = {record["status"] for record in hook.events}
    assert statuses == {"legal", "auto_rejected"}
    for record in hook.events:
        assert record["state_hash_before"] != record["state_hash_after"]
        assert record["policy_hash_before"] != record["policy_hash_after"]
    detail = hook.write_detail(tmp_path / "hook" / "spec_0003.jsonl")
    lines = [json.loads(line) for line in detail.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    # 收尾批量裁定：attribute 按 (frame, step) 分派；缺键 ⇒ unproven
    hook2 = ev._OracleInjectionHook(spec_id=4, truth_summary={})
    _emit_event(hook2, frame=0, step=1)
    _emit_event(hook2, frame=0, step=2)
    _emit_event(hook2, frame=0, step=3)
    summary2 = hook2.attribute({(0, 1): {"legal": True, "unproven": False, "criteria": {}}})
    assert summary2["counts"]["n_legal"] == 1 and summary2["counts"]["n_unproven"] == 2
    # 未裁定事件兜底 ⇒ unproven（不默认合法）
    hook3 = ev._OracleInjectionHook(spec_id=5, truth_summary={})
    _emit_event(hook3, frame=9, step=1)
    summary3 = hook3.finalize()
    assert summary3["counts"]["n_unproven"] == 1


def test_cf_verdict_semantics_one_step_and_chain(tmp_path: Path) -> None:
    """动作条件 = 执行命令链：k=1 单帧、k≥2 全链；缺帧/legacy ⇒ unproven（不默认合法）。"""
    controller = _bare_controller(tmp_path)
    # truth（合成）：帧 0..3 的 plan_exec[g][0] = (1.0, 0.0)
    controller._executed_first = {0: np.asarray([1.0, 0.0]), 1: np.asarray([1.0, 0.0])}
    assert controller._cf_verdict(0, 1)["legal"] is True
    assert controller._cf_verdict(0, 2)["legal"] is True
    # 重放首动作偏离（> 容差 5e-3）⇒ 自动拒绝
    controller._executed_first[0] = np.asarray([1.5, 0.0])
    verdict = controller._cf_verdict(0, 1)
    assert verdict["legal"] is False and verdict["unproven"] is False
    assert verdict["criteria"]["executed_chain"]["max_abs_diff_ds_m"] == pytest.approx(0.5)
    # 链步任一帧缺失 ⇒ unproven
    controller._executed_first = {0: np.asarray([1.0, 0.0])}
    assert controller._cf_verdict(0, 2)["unproven"] is True
    # legacy（无 plan_exec ⇒ 命令链不可判）⇒ unproven
    controller._executed_first = {9: np.asarray([1.0, 0.0])}
    assert controller._cf_verdict(9, 1)["unproven"] is True


def test_finish_episode_prunes_by_drift_and_illegal_fraction(tmp_path: Path) -> None:
    controller = _bare_controller(tmp_path)
    controller._drift_frames = [
        {"frame": f, "available": True, "pose_pos_m": 0.01 * f, "heading_rad": 0.001 * f,
         "speed_abs": 0.0, "od_mean_abs": 0.0, "od_max_abs": 0.0, "rng_match": True}
        for f in range(4)
    ]
    controller._replay_step_pose = [(float(i) // 2.0, 0.0, 0.0) for i in range(8)]  # 与记录逐步轨迹一致
    controller._replay_step_speed = [5.0] * 8
    controller._replay_step_index = list(range(8))
    controller._replay_step_rng = [999] * 8
    # 动作条件证据：执行命令链逐帧一致（合成 truth plan_exec[g][0] = (1.0, 0.0)）
    controller._executed_first = {f: np.asarray([1.0, 0.0]) for f in range(4)}
    controller._realized_intervals = {
        f: {"available": True, "consistent": True, "abs_diff_ds_m": 0.0,
            "abs_diff_dtheta_rad": 0.0, "interval": [f, f + 1]}
        for f in range(3)
    }
    for f in range(4):
        _emit_event(controller.hook, frame=f, step=1)
    result = controller.finish_episode(
        _stub_env(), steps=8, termination="max_step", success=False, collision=False, off_road=False,
    )
    oracle = result["oracle"]
    assert oracle["mode"] == "inject"
    assert oracle["eligible"] is True, oracle["excluded_reasons"]
    assert oracle["action_condition"]["n_legal_steps"] == 4
    assert oracle["drift"]["rng_match_frac"] == pytest.approx(1.0)
    assert oracle["termination"]["consistent"] is True
    assert (tmp_path / "out" / "oracle_hook" / "spec_0001.jsonl").is_file()

    # 超标：位置漂移 p95 > 阈 ⇒ 剔除
    controller2 = _bare_controller(tmp_path, spec_id=2)
    controller2._drift_frames = [
        {"frame": f, "available": True, "pose_pos_m": 3.0, "heading_rad": 0.0,
         "speed_abs": 0.0, "od_mean_abs": 0.0, "od_max_abs": 0.0, "rng_match": True}
        for f in range(2)
    ]
    controller2._replay_step_pose = [(0.0, 0.0, 0.0)]
    controller2._replay_step_speed = [5.0]
    controller2._executed_first = {}
    controller2._realized_intervals = {}
    result2 = controller2.finish_episode(
        _stub_env(), steps=100, termination="collision", success=False, collision=True, off_road=False,
    )
    oracle2 = result2["oracle"]
    assert oracle2["eligible"] is False
    assert any(reason.startswith("pose_p95") for reason in oracle2["excluded_reasons"])
    assert "termination_mismatch" in oracle2["excluded_reasons"]

    # 非法注入占比超阈 ⇒ 剔除（hook 计数口径）
    controller3 = _bare_controller(tmp_path, spec_id=3)
    controller3._drift_frames = []
    controller3._replay_step_pose = []
    controller3._executed_first = {}
    controller3._realized_intervals = {}
    for index in range(4):
        controller3.hook.counts["n_events"] += 1
        controller3.hook.counts["n_illegal"] += 1
    result3 = controller3.finish_episode(
        _stub_env(), steps=0, termination="max_step", success=False, collision=False, off_road=False,
    )
    assert result3["oracle"]["eligible"] is False
    assert any(reason.startswith("illegal_frac") for reason in result3["oracle"]["excluded_reasons"])


# =========================================================================== #
# 6. 五要件之 metrics/manifest + 裁决视图
# =========================================================================== #
def test_oracle_audit_report_excludes_ineligible() -> None:
    eligible_ep = {
        "id": 1, "primary": "straight", "difficulty": "easy", "success": True,
        "collision": False, "off_road": False, "route_completion": 1.0, "steps": 10,
        "termination": "arrival", "compound": False, "geometry": ["straight"],
        "oracle": {
            "mode": "inject", "eligible": True, "excluded_reasons": [],
            "route": {"route": "two_pass_replay"},
            "thresholds": {"pose_p95_m": 0.5},
            "action_condition": {"n_injected_steps": 2, "n_legal_steps": 2,
                                 "n_illegal_steps": 0, "n_unproven_steps": 0},
        },
    }
    excluded_ep = {
        "id": 2, "primary": "curve", "difficulty": "easy", "success": False,
        "collision": True, "off_road": False, "route_completion": 0.2, "steps": 30,
        "termination": "collision", "compound": False, "geometry": ["curve"],
        "oracle": {
            "mode": "inject", "eligible": False, "excluded_reasons": ["pose_p95>0.5"],
            "route": {"route": "two_pass_replay"},
            "thresholds": {"pose_p95_m": 0.5},
            "action_condition": {"n_injected_steps": 3, "n_legal_steps": 0,
                                 "n_illegal_steps": 3, "n_unproven_steps": 0},
        },
    }
    report = ev.build_report([eligible_ep, excluded_ep], meta={})
    audit = report["oracle_audit"]
    assert audit["n_inject_episodes"] == 2
    assert audit["n_eligible"] == 1 and audit["n_excluded"] == 1
    assert audit["excluded"][0]["id"] == 2
    assert audit["action_condition"]["n_injected_steps"] == 5
    assert audit["action_condition"]["n_illegal_steps"] == 3
    assert audit["action_condition"]["illegal_frac"] == pytest.approx(0.6)
    assert audit["eligible_view"]["n"] == 1, "裁决视图必须只含 eligible episode"
    # 默认关（无 oracle 键）⇒ 不追加 oracle_audit（metrics 形状不变）
    assert "oracle_audit" not in ev.build_report([dict(eligible_ep, oracle=None)], meta={}) \
        or ev._oracle_audit_report([dict(eligible_ep, oracle=None)]) is None
    assert ev._oracle_audit_report([{"id": 1}]) is None


def test_manifest_and_run_meta_counterfactual_only_when_active(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "manifest.txt").write_text("stage: eval\n", encoding="utf-8")
    hashes = {"evaluation_protocol_hash": "h", "model_tree_hash": "m", "schema_hash": "s",
              "checkpoint_hash": "c"}
    ev._record_run_meta(out_dir, {
        "hashes": hashes,
        "counterfactual": {"active": False, "teacher_force": "off"},
    })
    manifest = (out_dir / "manifest.txt").read_text(encoding="utf-8")
    assert "r41a1_counterfactual:" not in manifest
    ev._record_run_meta(out_dir, {
        "hashes": hashes,
        "counterfactual": {
            "active": True, "teacher_force": "all", "record_truth": None,
            "inject_truth": "/tmp/truth", "inject_truth_sha16": "d" * 16,
            "speed_scale": 0.5, "speed_oracle": "off",
        },
    })
    manifest = (out_dir / "manifest.txt").read_text(encoding="utf-8")
    assert manifest.count("r41a1_counterfactual:") == 1
    line = [row for row in manifest.splitlines() if row.startswith("r41a1_counterfactual:")][0]
    payload = json.loads(line.split(":", 1)[1])
    assert payload["teacher_force"] == "all" and payload["inject_truth_sha16"] == "d" * 16
    assert payload["speed_scale"] == pytest.approx(0.5)
    # 幂等：二次写不重复
    ev._record_run_meta(out_dir, {
        "hashes": hashes,
        "counterfactual": {"active": True, "teacher_force": "all", "inject_truth": "/tmp/truth",
                           "inject_truth_sha16": "d" * 16, "speed_scale": 0.5, "speed_oracle": "off"},
    })
    assert (out_dir / "manifest.txt").read_text(encoding="utf-8").count("r41a1_counterfactual:") == 1


def test_config_declares_counterfactual_defaults() -> None:
    import yaml

    config = yaml.safe_load((_ROOT / "config" / "eval.yaml").read_text(encoding="utf-8"))
    section = config["counterfactual"]
    assert section["teacher_force"] == "off" and section["speed_oracle"] == "off"
    assert section["record_truth"] is None and section["inject_truth"] is None
    assert set(ev.DEFAULT_DRIFT_THRESHOLDS) <= set(section["drift_thresholds"])
    assert set(ev.DEFAULT_FRAME_GATE) <= set(section["frame_gate"])
    for key, value in ev.DEFAULT_DRIFT_THRESHOLDS.items():
        assert float(section["drift_thresholds"][key]) == pytest.approx(float(value)), key
    for key, value in ev.DEFAULT_FRAME_GATE.items():
        assert float(section["frame_gate"][key]) == pytest.approx(float(value)), key


def test_teacher_force_modes_match_net_and_surfaces() -> None:
    assert TEACHER_FORCE_SURFACES["all"] == ("ego", "od", "ld", "lateral")
    assert TEACHER_FORCE_SURFACES["off"] == ()
    assert set(TEACHER_FORCE_SURFACES) == set(TEACHER_FORCE_MODES)
    # snapshot 助手语义（detach + 独立副本）
    x = torch.ones(2, requires_grad=True)
    snap = teacher_force_snapshot({"x": x * 2})
    assert snap["x"].grad_fn is None and not snap["x"].requires_grad
    assert torch.equal(snap["x"], torch.full((2,), 2.0))
