"""DAgger-lite（``tools/dagger_collect.py``）回归：**策略驱动 roll-in + 专家空问标签**。

覆盖（lane D2 + lane U5 契约）：
1. ``[steer, throttle] → (ds, dθ)`` 换算链与 ``pipeline.trainer.expand_policy_action`` 互逆（round-trip）；
2. 驱动 = 学生策略（``controller.action`` 每 env step；``env.step`` 收到其动作）；专家只在**策略步**
   被空问（``act()`` 次数 = 帧数），且**从未注册进 engine**（fake env 无 engine → 注册即崩）；
3. 标签重标注：``action[0]`` = 专家首步、``action[1:]`` 常量重复、``traj6/traj30`` 由运动学插值、
   ``roundtrip_*`` = NaN；
4. 病态过滤（行级 stuck / yaw_outlier，只改 ``train_weight`` 不删行）；
5. schema 往返：行 → ``save_dataset`` → ``BCDataset.load`` + ``dagger`` meta 块（``roll_in: true``）；
6. **lane U5 场景池**：``--specs`` 必填、``--from-eval`` 已删除、隔离守卫（池与 eval500/val 交集 → SystemExit）；
7. **lane U5 窗口模式**：失败 episode 尾部窗口行（不足全取）/成功与 max_step 零行/过滤松弛语义；
8. **lane U5 吞吐剖分**：``timing`` 键齐备且不改 episode 结果；report 聚合段正确；
9. 任务分块在 specs < chunk_size 时也按 worker 数铺开。
"""

from __future__ import annotations

import json
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tools import dagger_collect as dg
from tests.v2_synthetic import write_v2_dataset  # noqa: F401 - 保持与仓库合成数据同源导入习惯


# --------------------------------------------------------------------------- 1. 换算链
@pytest.mark.parametrize(
    ("action", "speed"),
    [([0.2, 0.5], 6.0), ([-0.35, -0.8], 8.0), ([0.0, 0.0], 3.0), ([0.9, 1.0], 12.0), ([-0.6, 0.4], 0.3)],
)
def test_expert_action_to_ds_dtheta_roundtrip_with_expand_policy_action(action, speed) -> None:
    """同一参数下：``expand_policy_action(label)`` 的首个子步必须还原原动作（运动学执行模型）。"""
    from pipeline.trainer import expand_policy_action

    label = dg.expert_action_to_ds_dtheta(action, speed=speed, accel_scale=2.0, brake_scale=2.0)
    expanded = expand_policy_action(label, speed=speed)  # dt=0.5 / hz=10 / accel_scale=2.0（默认）
    assert expanded.shape == (5, 2)
    np.testing.assert_allclose(expanded[0], action, atol=1e-6)
    assert label[0] >= 0.0


def test_expert_action_to_ds_dtheta_rejects_bad_action() -> None:
    with pytest.raises(ValueError):
        dg.expert_action_to_ds_dtheta([0.1], speed=5.0)


# --------------------------------------------------------------------------- 2. 策略驱动 + 空问
class _FakeLane:
    width = 3.5

    def local_coordinates(self, position):  # noqa: ANN001
        return 0.0, 0.0


class _FakeAgent:
    def __init__(self) -> None:
        self.position = np.zeros(3, dtype=np.float64)
        self.heading_theta = 0.0
        self.speed = 5.0
        self.lane = _FakeLane()


class _FakeEnv:
    def __init__(self, steps: int) -> None:
        self.agent = _FakeAgent()
        self.episode_step = 0
        self._steps = int(steps)
        self.step_calls: list = []
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1
        self.episode_step = 0
        return None

    def step(self, action):  # noqa: ANN001
        self.step_calls.append(list(action))
        self.episode_step += 1
        terminated = self.episode_step >= self._steps
        return None, 0.0, terminated, False, {"max_step": False}


class _FakeController:
    """最小驱动接口（`_CkptController` 的替身）：每 env step 决策并返回占位动作。"""

    def __init__(self) -> None:
        self.bind_calls = self.action_calls = self.post_calls = 0

    def bind(self, env) -> None:  # noqa: ANN001
        self.bind_calls += 1

    def action(self, env):  # noqa: ANN001
        self.action_calls += 1
        env.agent.position = env.agent.position + np.array([0.5, 0.0, 0.0])
        return [0.0, 0.0]

    def post_step(self, env) -> None:  # noqa: ANN001
        self.post_calls += 1


class _FakeBuilder:
    def build(self, env, spec):  # noqa: ANN001
        return {
            "ego": np.zeros((1, 8), dtype=np.float32),
            "hist_valid": np.zeros(6, dtype=np.float32),
            "od_id_hist": np.full((6, 16), -1, dtype=np.int64),
            "od_presence_hist": np.zeros((6, 16), dtype=np.float32),
        }


class _FakeLabeler:
    kind = "fake"

    def __init__(self) -> None:
        self.calls: list = []

    def label(self, ego):  # noqa: ANN001
        self.calls.append(round(float(ego.position[0]), 3))
        return {"action": np.array([2.0, 0.1]), "raw_action": np.array([0.1, 0.5])}

    def params(self):
        return {"kind": "fake"}


def test_dagger_episode_is_policy_driven_and_expert_is_empty_queried(monkeypatch) -> None:
    monkeypatch.setattr(dg, "compute_step_labels", lambda env, spec: {})
    monkeypatch.setattr(dg, "event_state", lambda env: {})
    env, controller, labeler = _FakeEnv(steps=12), _FakeController(), _FakeLabeler()
    episode = dg._dagger_episode(
        env, None, controller=controller, labeler_factory=lambda: labeler,
        builder=_FakeBuilder(), max_steps=12,
    )
    # 驱动 = 学生策略：每 env step 一次 controller.action，env.step 收到它的动作
    assert controller.bind_calls == 1 and controller.action_calls == 12 and controller.post_calls == 12
    assert env.step_calls == [[0.0, 0.0]] * 12
    # 观测/空问只在策略步（每 5 个 env step）：12 步 → 帧 0/5/10
    assert sorted(episode["frames"]) == [0, 5, 10]
    assert sorted(episode["expert_labels"]) == [0, 5, 10]
    assert labeler.calls == [0.5, 3.0, 5.5]  # 空问发生在学生访问状态上（位置随策略步推进）
    assert np.allclose(episode["expert_labels"][0]["action"], [2.0, 0.1])
    assert episode["labeler_params"] == {"kind": "fake"}
    assert len(episode["poses"]) == 13  # 每 env step 记录位姿
    assert episode["termination"] == "terminal"  # fake env 终止（非 max_step）


# --------------------------------------------------------------------------- 3. 标签重标注
def _obs_frame(step: float) -> dict:
    return {
        "ego": np.full((1, 8), step, dtype=np.float32),
        "od": np.zeros((16, 9), dtype=np.float32),
        "od_mask": np.zeros(16, dtype=np.float32),
        "od_presence": np.zeros(16, dtype=np.float32),
        "od_id": np.full(16, -1, dtype=np.int64),
        "ld": np.zeros((16, 7), dtype=np.float32),
        "ld_mask": np.zeros(16, dtype=np.float32),
        "nav": np.zeros((1, 11), dtype=np.float32),
        "nav_mask": np.zeros(1, dtype=np.float32),
        "signal": np.zeros((1, 4), dtype=np.float32),
        "signal_mask": np.zeros(1, dtype=np.float32),
        "others": np.zeros((1, 28), dtype=np.float32),
        "others_mask": np.zeros(1, dtype=np.float32),
    }


def _synthetic_episode(frames_steps=(0, 5, 10, 15), total_steps: int = 40, label=(2.5, 0.05)) -> dict:
    poses = [np.array([0.5 * i, 0.0, 0.0], dtype=np.float64) for i in range(total_steps + 1)]
    frames = {
        step: {
            "obs": _obs_frame(step),
            "hist_valid": np.ones(6, dtype=np.float32),
            "od_id_hist": np.full((6, 16), -1, dtype=np.int64),
            "od_presence_hist": np.zeros((6, 16), dtype=np.float32),
            "labels_raw": np.zeros(9, dtype=np.float32),
            "events": [],
            "pose": poses[step],
            "on_lane": True,
            "lane_lat": 0.1,
            "lane_width": 3.5,
        }
        for step in frames_steps
    }
    return {
        "frames": frames,
        "poses": poses,
        "flags": [None] + [{"bad": False, "max_step": False} for _ in range(total_steps)],
        "termination": "max_step",
        "steps": total_steps,
        "ended_by_env": False,
        "expert_labels": {step: {"action": np.array(label), "raw_action": np.array([0.1, 0.5])} for step in frames_steps},
        "labeler_params": {"kind": "test"},
    }


def test_relabel_rows_uses_expert_first_step_and_constant_window() -> None:
    from tools.collect_expert import extract_samples, resolve_interpolate

    interpolate_fn, _ = resolve_interpolate()
    label_order = ("cutin_active", "cutout_active", "crowded", "car_following",
                   "on_curve", "merging", "roundabout_near", "near_intersection")
    rows = extract_samples(
        _synthetic_episode(), None, episode_id=0, label_order=label_order, interpolate_fn=interpolate_fn,
        on_lane_frac=0.5, on_lane_margin=0.3, roundtrip_key_mean=float("inf"), roundtrip_key_max=float("inf"),
        filter_counter=Counter(),
    )
    assert len(rows) == 4
    relabeled = dg._relabel_rows_with_expert(rows, _synthetic_episode(), interpolate_fn=interpolate_fn)
    assert relabeled == 4
    for row in rows:
        np.testing.assert_allclose(row["action"][0], [2.5, 0.05], atol=1e-6)
        np.testing.assert_allclose(row["action"], np.repeat(np.asarray(row["action"][0])[None, :], 6, axis=0))
        assert row["action"].shape == (6, 2) and row["traj6"].shape == (6, 2) and row["traj30"].shape == (30, 2)
        assert np.isfinite(row["traj6"]).all()
        assert np.isnan(row["roundtrip_key_err"]) and np.isnan(row["roundtrip_dense_err"])
    # 前三帧窗口完整（t+30 ≤ limit=40），最后一帧命中 terminal_window
    assert [row["train_weight"] for row in rows] == [1.0, 1.0, 1.0, 0.0]
    assert rows[3]["filter_reason"] == "terminal_window"


# --------------------------------------------------------------------------- 4. 病态过滤
def _moving_measured(distance_m: float = 7.5) -> np.ndarray:
    return np.stack([np.linspace(0.0, distance_m, 30), np.zeros(30)], axis=1).astype(np.float32)


def test_pathological_filters_zero_weight_without_dropping_rows() -> None:
    rows = [
        {"train_weight": 1.0, "action": np.array([[2.5, 0.0]]), "traj30_measured": _moving_measured()},
        {"train_weight": 1.0, "action": np.array([[2.5, 0.0]]), "traj30_measured": np.zeros((30, 2), dtype=np.float32)},
        {"train_weight": 1.0, "action": np.array([[2.5, 0.9]]), "traj30_measured": _moving_measured(10.0)},
        {"train_weight": 0.0, "action": np.array([[2.5, 0.9]]), "traj30_measured": np.zeros((30, 2), dtype=np.float32)},
    ]
    counter: Counter = Counter()
    stats = dg._apply_pathological_filters(rows, filter_counter=counter)
    assert stats["stuck"] == 1 and stats["yaw_outlier"] == 1
    assert stats["rows"] == 4 and stats["trainable_after"] == 1
    assert [row["train_weight"] for row in rows] == [1.0, 0.0, 0.0, 0.0]  # 不删行
    assert rows[1]["filter_reason"] == "stuck" and rows[2]["filter_reason"] == "yaw_outlier"
    assert dict(counter) == {"stuck": 1, "yaw_outlier": 1}
    assert stats["thresholds"] == {"stuck_speed_mps": 0.5, "yaw_outlier_rad": 0.35}


# --------------------------------------------------------------------------- 5. schema 往返
def test_rows_save_and_load_with_dagger_meta(tmp_path: Path) -> None:
    from env.obs.builder import ObservationBuilder
    from pipeline.trainer import BCDataset
    from tools.collect_expert import extract_samples, resolve_interpolate, save_dataset

    interpolate_fn, kinematics = resolve_interpolate()
    label_order = ("cutin_active", "cutout_active", "crowded", "car_following",
                   "on_curve", "merging", "roundabout_near", "near_intersection")
    episode = _synthetic_episode()
    rows = extract_samples(
        episode, None, episode_id=0, label_order=label_order, interpolate_fn=interpolate_fn,
        on_lane_frac=0.5, on_lane_margin=0.3, roundtrip_key_mean=float("inf"), roundtrip_key_max=float("inf"),
        filter_counter=Counter(),
    )
    dg._relabel_rows_with_expert(rows, episode, interpolate_fn=interpolate_fn)
    out = tmp_path / "BTC_test_dagger"
    paths = save_dataset(
        out, rows, label_order=label_order, builder=ObservationBuilder({}),
        config={"max_steps": 40}, report={"kinematics_source": kinematics},
        sample_weight=np.ones(len(rows), dtype=np.float32),
        balance_group=np.asarray(["easy/curve"] * len(rows), dtype="U64"),
    )
    dagger = {
        "protocol": "dagger-lite: student(policy) roll-in + expert act() empty-query labels",
        "roll_in": True,
        "driver_ckpt": {"path": "x.pt", "sha256": "deadbeef"},
        "labeler": "pure_pursuit",
        "label_scope": "first_step_only",
        "window_fill": "constant_repeat(first_step)",
        "labeler_params": {"kind": "fake"},
    }
    dg._augment_meta(Path(paths["meta"]), dagger)
    meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
    assert meta["created_by"] == "tools/dagger_collect.py"
    assert meta["dagger"]["roll_in"] is True and meta["dagger"]["label_scope"] == "first_step_only"
    assert meta["label_names"] == list(label_order)
    dataset = BCDataset.load(str(out))
    assert dataset.count == len(rows)
    assert list(dataset.arrays["action"].shape) == [len(rows), 6, 2]
    np.testing.assert_allclose(
        dataset.arrays["action"][:, 0, :], np.tile(np.asarray([2.5, 0.05], dtype=np.float32), (len(rows), 1)), atol=1e-6
    )
    assert dataset.arrays["roundtrip_key_err"].shape == (len(rows),)
    assert np.isnan(dataset.arrays["roundtrip_key_err"]).all()


# --------------------------------------------------- 6. 场景池（lane U5：--specs 必填 + 隔离守卫）
def test_parser_requires_specs_and_has_no_from_eval() -> None:
    """``--specs`` 必填；``--from-eval`` 已删除（argparse 不再认识该参数）。"""
    parser = dg._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ckpt", "x.pt", "--out", "/tmp/x"])  # 缺 --specs → 报错
    with pytest.raises(SystemExit):
        parser.parse_args(["--ckpt", "x.pt", "--out", "/tmp/x", "--specs", "s.json", "--from-eval", "e.csv"])
    assert not any(action.dest == "from_eval" for action in parser._actions)


def test_select_specs_requires_pool_file(tmp_path: Path) -> None:
    args = Namespace(specs="", limit=0)
    with pytest.raises(SystemExit, match="--specs"):
        dg._select_specs(args)
    args2 = Namespace(specs=str(tmp_path / "missing.json"), limit=0)
    with pytest.raises(SystemExit, match="不存在"):
        dg._select_specs(args2)


def test_isolation_guard_blocks_eval_and_val_pools(tmp_path: Path) -> None:
    """池 (id,seed) 与 eval500/val 有交集 → SystemExit（两文件情形都覆盖）；干净 train 池 → 记录 provenance。"""
    from env.scenario.spec import load_specs, save_specs

    eval_spec = list(load_specs(str(dg.EVAL_SPEC_SOURCES[0])))[0]
    val_spec = list(load_specs(str(dg.EVAL_SPEC_SOURCES[1])))[0]
    train_specs = list(load_specs("env/specs/scenarios_smoke16.json"))[:2]

    eval_pool = tmp_path / "pool_eval.json"
    save_specs([eval_spec, *train_specs], eval_pool)
    with pytest.raises(SystemExit, match="隔离守卫"):
        dg._select_specs(Namespace(specs=str(eval_pool), limit=0))

    val_pool = tmp_path / "pool_val.json"
    save_specs([val_spec], val_pool)
    with pytest.raises(SystemExit, match="隔离守卫"):
        dg._select_specs(Namespace(specs=str(val_pool), limit=0))

    # val 文件情形：自定义 sources（隔离守卫函数级覆盖）
    with pytest.raises(SystemExit, match="隔离守卫"):
        dg.assert_no_eval_val_overlap([val_spec], context="unit", sources=(str(val_pool),))

    clean_pool = tmp_path / "pool_train.json"
    save_specs(train_specs, clean_pool)
    specs, provenance = dg._select_specs(Namespace(specs=str(clean_pool), limit=0))
    assert len(specs) == len(train_specs)
    assert provenance["specs_sha256"] and provenance["pool_specs"] == len(train_specs)
    assert provenance["isolation_guard"]["overlap"] == 0
    assert provenance["isolation_guard"]["checked"][dg.EVAL_SPEC_SOURCES[0]]["specs"] > 0


# --------------------------------------------------- 6b. 窗口模式（lane U5）
def _window_rows(count: int, *, start_step: int = 0, reason: str = "") -> list:
    return [
        {
            "step": int(start_step + index * 5),
            "train_weight": 1.0,
            "filter_reason": reason,
            "action": np.asarray([[2.5, 0.0]], dtype=np.float32),
        }
        for index in range(count)
    ]


def _window_episode(rows: list, termination: str, *, off_lane_steps=()) -> dict:
    return {
        "termination": termination,
        "frames": {
            int(row["step"]): {
                "on_lane": int(row["step"]) not in set(off_lane_steps),
                "lane_lat": 5.0 if int(row["step"]) in set(off_lane_steps) else 0.1,
                "lane_width": 3.5,
            }
            for row in rows
        },
    }


def test_window_select_rows_keeps_tail_and_full_when_short() -> None:
    rows = _window_rows(25)
    episode = _window_episode(rows, "collision")
    kept, stats = dg._window_select_rows(
        rows, episode, window_s=10.0, fail_terminations=dg.DEFAULT_FAIL_TERMINATIONS, strict_filters=False
    )
    assert stats["frames_per_window"] == 20 and stats["is_failure"] is True
    assert [row["step"] for row in kept] == [row["step"] for row in rows[-20:]], "窗口 = 尾部 ceil(10/0.5)=20 帧"
    assert stats["kept_rows"] == 20 and stats["dropped_rows"] == 5

    short = _window_rows(8)
    kept_short, stats_short = dg._window_select_rows(
        short, _window_episode(short, "out_of_road"), window_s=10.0,
        fail_terminations=dg.DEFAULT_FAIL_TERMINATIONS, strict_filters=False,
    )
    assert len(kept_short) == 8 and stats_short["kept_rows"] == 8, "不足窗口 → 全取"


def test_window_select_rows_skips_success_and_max_step() -> None:
    for termination in ("arrive_dest", "max_step"):
        rows = _window_rows(30)
        kept, stats = dg._window_select_rows(
            rows, _window_episode(rows, termination), window_s=10.0,
            fail_terminations=dg.DEFAULT_FAIL_TERMINATIONS, strict_filters=False,
        )
        assert kept == [], f"{termination} 必须零行"
        assert stats["kept_rows"] == 0 and stats["is_failure"] is False and stats["reason"] == "not_failure"
        assert stats["dropped_rows"] == 30  # 只记统计


def test_window_filter_relaxation_default_and_strict() -> None:
    """窗口模式默认松弛 on_lane/yaw；``--window-strict-filters`` 恢复 v1；非窗口恒 v1。"""
    window_cfg = {"window_fail_before": 10.0, "on_lane_frac": 0.5, "yaw_outlier": 0.35}
    strict_cfg = dict(window_cfg, window_strict_filters=True)
    plain_cfg = {"window_fail_before": 0.0, "on_lane_frac": 0.5, "yaw_outlier": 0.35}
    assert dg._effective_on_lane_frac(window_cfg) == float("inf")
    assert dg._effective_yaw_threshold(window_cfg) == float("inf")
    assert dg._effective_on_lane_frac(strict_cfg) == 0.5 and dg._effective_yaw_threshold(strict_cfg) == 0.35
    assert dg._effective_on_lane_frac(plain_cfg) == 0.5 and dg._effective_yaw_threshold(plain_cfg) == 0.35
    # terminal_window 松弛 + "本会被拦掉"计数
    rows = _window_rows(6, reason="terminal_window")
    rows[5]["action"] = np.asarray([[2.5, 0.9]], dtype=np.float32)  # 大转向（would_filter: yaw）
    counter: Counter = Counter({"terminal_window": 6})
    kept, stats = dg._window_select_rows(
        rows, _window_episode(rows, "collision"), window_s=3.0,
        fail_terminations=dg.DEFAULT_FAIL_TERMINATIONS, strict_filters=False, filter_counter=counter,
    )
    assert [row["train_weight"] for row in kept] == [1.0] * 6 and all(not row["filter_reason"] for row in kept)
    assert "terminal_window" not in counter, "松弛后计数应扣回"
    assert stats["would_filter"]["terminal_window"] == 6 and stats["would_filter"]["yaw_outlier"] == 1
    # on_lane 复算（v1 阈值）：
    off = _window_rows(4)
    off_episode = _window_episode(off, "collision", off_lane_steps={off[0]["step"], off[1]["step"]})
    _, stats_off = dg._window_select_rows(
        off, off_episode, window_s=10.0, fail_terminations=dg.DEFAULT_FAIL_TERMINATIONS, strict_filters=False,
    )
    assert stats_off["would_filter"]["on_lane"] == 2


def test_on_lane_relaxation_mechanism_in_window_mode() -> None:
    """松弛机制：``_relax_episode_on_lane`` 后 extract_samples 不再产生 not_on_lane（原值留 ``_on_lane_raw``）。"""
    from tools.collect_expert import extract_samples, resolve_interpolate

    interpolate_fn, _ = resolve_interpolate()
    label_order = ("cutin_active", "cutout_active", "crowded", "car_following",
                   "on_curve", "merging", "roundabout_near", "near_intersection")

    def _run(relax: bool):
        episode = _synthetic_episode()
        for frame in episode["frames"].values():
            frame["on_lane"] = False
            frame["lane_lat"] = 5.0
        if relax:
            relaxed = dg._relax_episode_on_lane(episode)
            assert relaxed == len(episode["frames"])
        rows = extract_samples(
            episode, None, episode_id=0, label_order=label_order, interpolate_fn=interpolate_fn,
            on_lane_frac=float("inf") if relax else 0.5, on_lane_margin=0.3,
            roundtrip_key_mean=float("inf"), roundtrip_key_max=float("inf"), filter_counter=Counter(),
        )
        return episode, rows

    _, strict_rows = _run(relax=False)
    relaxed_episode, relaxed_rows = _run(relax=True)
    assert any(row["filter_reason"] == "not_on_lane" for row in strict_rows)
    assert all(row["filter_reason"] != "not_on_lane" for row in relaxed_rows), "松弛后 on_lane 不再拦行"
    assert all(frame["_on_lane_raw"] is False for frame in relaxed_episode["frames"].values())


# --------------------------------------------------- 6c. 吞吐剖分（lane U5）
def test_dagger_episode_timing_does_not_change_result(monkeypatch) -> None:
    monkeypatch.setattr(dg, "compute_step_labels", lambda env, spec: {})
    monkeypatch.setattr(dg, "event_state", lambda env: {})
    plain = dg._dagger_episode(
        _FakeEnv(steps=12), None, controller=_FakeController(), labeler_factory=_FakeLabeler,
        builder=_FakeBuilder(), max_steps=12,
    )
    timing: dict = {}
    timed = dg._dagger_episode(
        _FakeEnv(steps=12), None, controller=_FakeController(), labeler_factory=_FakeLabeler,
        builder=_FakeBuilder(), max_steps=12, timing=timing,
    )
    assert sorted(timed["frames"]) == sorted(plain["frames"])
    assert sorted(timed["expert_labels"]) == sorted(plain["expert_labels"])
    assert timed["termination"] == plain["termination"] and timed["steps"] == plain["steps"]
    for key in ("student_step_s", "expert_query_s", "other_s"):
        assert timing.get(key, 0.0) > 0.0, f"缺少 timing 键 {key}"


def test_aggregate_timing_and_window_sections() -> None:
    entries = [
        {"report": {"timing": {"env_build_s": 1.0, "student_step_s": 2.0, "expert_query_s": 0.5,
                               "extract_s": 0.5, "other_s": 0.0, "total_s": 4.0},
                    "window": {"termination": "collision", "is_failure": True, "window_frames": 20,
                               "kept_rows": 20, "dropped_rows": 5,
                               "would_filter": {"terminal_window": 20, "on_lane": 1, "yaw_outlier": 0}}}},
        {"report": {"timing": {"env_build_s": 1.0, "student_step_s": 1.0, "expert_query_s": 0.5,
                               "extract_s": 0.5, "other_s": 0.0, "total_s": 3.0},
                    "window": {"termination": "max_step", "is_failure": False, "window_frames": 0,
                               "kept_rows": 0, "dropped_rows": 30, "would_filter": {}}}},
    ]
    timing = dg._aggregate_timing(entries, shard_write_s=0.25)
    assert timing["specs"] == 2 and timing["shard_write_s"] == 0.25
    assert timing["total_s"]["student_step_s"] == 3.0 and timing["mean_per_spec_s"]["student_step_s"] == 1.5
    window = dg._aggregate_window(entries, window_s=10.0)
    assert window["failures"] == 1 and window["window_frames"] == {"min": 20, "p50": 20, "max": 20}
    assert window["terminations"]["max_step"] == 1 and window["would_filter"]["terminal_window"] == 20
    assert dg._aggregate_window(entries, window_s=0.0) is None, "非窗口模式 → 无 window 段"


# --------------------------------------------------------------------------- 7. 分块铺满 worker
def test_chunk_tasks_spreads_small_pools_across_workers() -> None:
    specs = list(range(5))
    chunks = dg._chunk_tasks(specs, 10, workers=2)
    assert [len(chunk) for chunk in chunks] == [3, 2]
    assert sorted(index for chunk in chunks for index, _ in chunk) == [0, 1, 2, 3, 4]
    assert len(dg._chunk_tasks(specs, 10, workers=5)) == 5
    assert len(dg._chunk_tasks(range(25), 10, workers=2)) == 3  # 大批量仍按 chunk_size=10
