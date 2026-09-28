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
9. **lane U6 窗口 × stuck 交互**：末帧附近 ``future_truncated`` 行跳过 ``stuck``（无观测不判决）、
   完整未来的真 stuck 行仍被清零、计数分开（``stuck`` / ``stuck_skipped_truncated``）；
10. **lane U7 窗口模式报告口径**：yield 分母 = 窗口行数（gate 不再误报）、``filter_counts``
   只统计最终被置零的行（松弛/丢弃行不计）、整段模式口径逐位不变；
11. 任务分块在 specs < chunk_size 时也按 worker 数铺开。
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


# ------------------------------------------- 9. lane U6：窗口 × stuck 交互
def test_future_truncation_mask_boundary_and_alignment() -> None:
    """掩码 = ``last_step - step < span``（末帧附近为 True）；无 poses 回退 episode["steps"]。"""
    rows = [{"step": step} for step in (0, 5, 70, 75, 80, 100)]
    episode = {"poses": [None] * 106, "steps": 105}  # last_state = 105
    mask = dg.future_truncation_mask(rows, episode, span=30)
    assert mask.tolist() == [False, False, False, False, True, True], "step > last-30 → truncated"
    assert dg.future_truncation_mask([{"step": 80}], {"steps": 105}, span=30).tolist() == [True]
    assert dg.future_truncation_mask([{"step": 80}], {"steps": 105}, span=25).tolist() == [False]
    assert dg.TRAJ30_SPAN == 30, "保守跨度 = traj30 的 env-step 跨度（30）"


def test_truncated_rows_skip_stuck_but_yaw_still_applies() -> None:
    """① 截断行（measured 补齐=0）不被 stuck 清零；② 完整未来的真 stuck 行仍清零；
    ③ yaw 与未来轨迹无关（截断行仍生效）；④ 计数分开。"""
    truncated_measured = np.zeros((30, 2), dtype=np.float32)      # 补齐值：首末距 0.0
    real_stuck = _moving_measured(0.5)                            # 0.5 m / 2.5 s = 0.2 m/s < 0.5 → 真 stuck
    normal = _moving_measured(7.5)                                # 7.5 m / 2.5 s = 3 m/s → 正常
    rows = [
        {"step": 100, "train_weight": 1.0, "action": np.array([[2.5, 0.0]]), "traj30_measured": truncated_measured},
        {"step": 50, "train_weight": 1.0, "action": np.array([[2.5, 0.0]]), "traj30_measured": real_stuck},
        {"step": 30, "train_weight": 1.0, "action": np.array([[2.5, 0.0]]), "traj30_measured": normal},
        {"step": 100, "train_weight": 1.0, "action": np.array([[2.5, 0.9]]), "traj30_measured": truncated_measured},
    ]
    episode = {"poses": [None] * 126}  # last_state = 125 → truncated ⟺ step > 95
    mask = dg.future_truncation_mask(rows, episode, span=30)
    assert mask.tolist() == [True, False, False, True]
    counter: Counter = Counter()
    stats = dg._apply_pathological_filters(rows, filter_counter=counter, future_truncated=mask)
    assert rows[0]["train_weight"] == 1.0, "截断行不得因 stuck 被清零（无观测不判决）"
    assert rows[1]["train_weight"] == 0.0 and rows[1]["filter_reason"] == "stuck", "完整未来的真 stuck 仍清零"
    assert rows[2]["train_weight"] == 1.0
    assert rows[3]["train_weight"] == 0.0 and rows[3]["filter_reason"] == "yaw_outlier", "yaw 规则不受截断影响"
    # 截断行中"本会命中 stuck"的共 2 行：第 0 行被救回（权重保持 1.0），第 3 行另被 yaw 命中
    assert stats["stuck"] == 1 and stats["stuck_skipped_truncated"] == 2, "计数必须分开"
    assert stats["yaw_outlier"] == 1 and stats["future_truncated_rows"] == 2
    assert stats["trainable_after"] == 2, "可训练 = 截断低速行 + 正常行"
    assert dict(counter) == {"stuck": 1, "yaw_outlier": 1}, "计数器不含被跳过的 stuck"
    assert stats["traj30_span_steps"] == 30


def test_stuck_without_mask_keeps_full_episode_behavior() -> None:
    """整段模式（不传掩码）行为不变：同样的补齐行仍按 stuck 清零；新计数为 0。"""
    rows = [
        {"step": 100, "train_weight": 1.0, "action": np.array([[2.5, 0.0]]),
         "traj30_measured": np.zeros((30, 2), dtype=np.float32)},
    ]
    stats = dg._apply_pathological_filters(rows)
    assert rows[0]["train_weight"] == 0.0 and rows[0]["filter_reason"] == "stuck"
    assert stats["stuck"] == 1 and stats["stuck_skipped_truncated"] == 0
    assert stats["future_truncated_rows"] == 0
    with pytest.raises(ValueError, match="future_truncated 长度"):
        dg._apply_pathological_filters(rows, future_truncated=np.asarray([True, False]))


def test_window_tail_rows_flagged_truncated_from_real_episode() -> None:
    """integration：真实 episode 的窗口尾行被标 future_truncated（r1 bug 的直接回归）。"""
    from tools.collect_expert import extract_samples, resolve_interpolate

    interpolate_fn, _ = resolve_interpolate()
    label_order = ("cutin_active", "cutout_active", "crowded", "car_following",
                   "on_curve", "merging", "roundabout_near", "near_intersection")
    episode = _synthetic_episode(total_steps=40)  # poses 41 → last_state 40
    rows = extract_samples(
        episode, None, episode_id=0, label_order=label_order, interpolate_fn=interpolate_fn,
        on_lane_frac=0.5, on_lane_margin=0.3, roundtrip_key_mean=float("inf"),
        roundtrip_key_max=float("inf"), filter_counter=Counter(),
    )
    assert [row["step"] for row in rows] == [0, 5, 10, 15]
    mask = dg.future_truncation_mask(rows, episode, span=30)
    # distance = 40 - step → 40/35/30/25；truncated ⟺ distance < 30 → 仅 step 15（distance=30 仍完整）
    assert mask.tolist() == [False, False, False, True]
    for row, truncated in zip(rows, mask.tolist()):
        row["future_truncated"] = bool(truncated)
    stats = dg._apply_pathological_filters(rows, future_truncated=mask)
    assert stats["future_truncated_rows"] == 1
    assert stats["stuck"] == 0, "尾行 traj30_measured 为补齐值 → 不得被判 stuck"
    assert stats["stuck_skipped_truncated"] >= 0
    assert rows[3]["filter_reason"] == "terminal_window", "截断行的既有原因不得被 stuck 覆盖"
    assert rows[3]["train_weight"] == 0.0, "该行仍由 terminal_window 过滤（非 stuck）"


# ------------------------------------------- 10. lane U7：窗口模式报告口径
_LABELS8 = ("cutin_active", "cutout_active", "crowded", "car_following",
            "on_curve", "merging", "roundabout_near", "near_intersection")


def _min_balance() -> dict:
    return {"stats": {"mode": "weights", "ratio": 3.0, "group_counts_before": {},
                      "group_counts_after": {}, "group_trainable_before": {}, "counts": {}}}


def _report_arrays(n_trainable: int, n_zeroed: int, labels8: bool = True):
    n = n_trainable + n_zeroed
    train_weight = np.ones(n, dtype=np.float32)
    train_weight[n_trainable:] = 0.0
    return {
        "train_weight": train_weight,
        "balance_weight": np.ones(n, dtype=np.float32),
        "labels": np.zeros((n, 8), dtype=np.float32),
    }


def _build_report_for(arrays, entries, *, window_fail_before: float, filter_counter: Counter):
    return dg._build_report(
        entries=entries, arrays=arrays, balance=_min_balance(), filter_counter=filter_counter,
        config={"window_fail_before": window_fail_before, "specs_path": "x.json"},
        provenance={}, label_order=_LABELS8, kinematics_source="k", elapsed_s=0.1,
    )


def test_window_mode_yield_denominator_is_window_rows_and_gate_passes() -> None:
    """① 窗口模式：分母 = 窗口行数（20 行，1 行真 stuck 置零）→ yield 0.95、gate PASS。"""
    entries = [{
        "candidates": 100,  # 整段 episode 候选帧（仅参考）
        "report": {"timing": {}, "window": {
            "termination": "collision", "is_failure": True, "window_frames": 20,
            "kept_rows": 20, "dropped_rows": 80,
            "would_filter": {"terminal_window": 20, "on_lane": 0, "yaw_outlier": 0},
        }},
    }]
    report = _build_report_for(
        _report_arrays(n_trainable=19, n_zeroed=1), entries,
        window_fail_before=10.0, filter_counter=Counter({"stuck": 1}),
    )
    assert report["stored_rows"] == 20 and report["retained_steps"] == 19
    assert report["yield_denominator"] == "window_rows" and report["yield_denominator_value"] == 20
    assert report["bc_retained_step_yield"] == pytest.approx(19 / 20)
    assert report["dataset_gate"]["bc_retained_step_yield"]["pass"] is True, "gate 不应再误报 FAIL"
    assert report["dataset_gate"]["bc_retained_step_yield"]["denominator"] == "window_rows"
    assert report["total_candidate_steps"] == 100, "整段候选帧仍保留为参考字段（不进分母）"
    assert report["counts"]["yield_denominator"] == "window_rows"


def test_window_mode_filter_counts_exclude_relaxed_and_dropped_rows() -> None:
    """② 松弛保留的 terminal_window 行与窗口丢弃行都不进 filter_counts；would_filter 计数不变。"""
    normal = _moving_measured(7.5)
    real_stuck = _moving_measured(0.5)
    rows = [
        {"step": 0, "train_weight": 1.0, "filter_reason": "terminal_window",
         "action": np.array([[2.5, 0.0]]), "traj30_measured": normal},
        {"step": 5, "train_weight": 1.0, "filter_reason": "terminal_window",
         "action": np.array([[2.5, 0.0]]), "traj30_measured": normal},
        {"step": 10, "train_weight": 1.0, "filter_reason": "",
         "action": np.array([[2.5, 0.0]]), "traj30_measured": real_stuck},
        {"step": 15, "train_weight": 1.0, "filter_reason": "",
         "action": np.array([[2.5, 0.0]]), "traj30_measured": normal},
    ]
    episode = {"termination": "collision", "poses": [None] * 30,
               "frames": {int(row["step"]): {"on_lane": True, "lane_lat": 0.1, "lane_width": 3.5}
                          for row in rows}}
    counter: Counter = Counter({"terminal_window": 6})  # extract_samples 计了 6（含被窗口丢弃的 2 行）
    kept, window_stats = dg._window_select_rows(
        rows, episode, window_s=10.0, fail_terminations=dg.DEFAULT_FAIL_TERMINATIONS,
        strict_filters=False, filter_counter=counter,
    )
    dg._apply_pathological_filters(kept, filter_counter=counter)
    final = dg._final_filter_counts(kept)
    assert [row["train_weight"] for row in kept] == [1.0, 1.0, 0.0, 1.0], "仅真 stuck 行被置零"
    assert dict(final) == {"stuck": 1}, "filter_counts 只统计最终被置零的行"
    assert "terminal_window" not in dict(final), "松弛/丢弃行不得进最终计数"
    assert dict(counter).get("terminal_window") == 4, "原始计数器含被窗口丢弃行的陈旧计数（修复前误报来源）"
    assert window_stats["would_filter"]["terminal_window"] == 2, "would_filter 语义不变（本会被过滤）"


def test_full_mode_yield_and_filter_counts_bitwise_unchanged() -> None:
    """③ 整段模式（无窗口参数）：分母 = 整段候选帧、filter_counts 原样透传（修复前口径）。"""
    entries = [{"candidates": 100, "report": {"timing": {}, "window": None}}]
    counter: Counter = Counter({"terminal_window": 4, "stuck": 2})
    report = _build_report_for(
        _report_arrays(n_trainable=14, n_zeroed=6), entries,
        window_fail_before=0.0, filter_counter=counter,
    )
    assert report["yield_denominator"] == "candidates" and report["yield_denominator_value"] == 100
    assert report["bc_retained_step_yield"] == pytest.approx(14 / 100)
    assert report["dataset_gate"]["bc_retained_step_yield"]["value"] == pytest.approx(14 / 100)
    assert report["dataset_gate"]["bc_retained_step_yield"]["pass"] is False, "整段模式 gate 口径不变"
    assert report["filter_counts"] == {"terminal_window": 4, "stuck": 2}, "整段模式计数原样透传"
    assert report["window"] is None and report["total_candidate_steps"] == 100
