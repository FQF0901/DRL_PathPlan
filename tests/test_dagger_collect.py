"""DAgger-lite（``tools/dagger_collect.py``）回归：**策略驱动 roll-in + 专家空问标签**。

覆盖（lane D2 契约）：
1. ``[steer, throttle] → (ds, dθ)`` 换算链与 ``pipeline.trainer.expand_policy_action`` 互逆（round-trip）；
2. 驱动 = 学生策略（``controller.action`` 每 env step；``env.step`` 收到其动作）；专家只在**策略步**
   被空问（``act()`` 次数 = 帧数），且**从未注册进 engine**（fake env 无 engine → 注册即崩）；
3. 标签重标注：``action[0]`` = 专家首步、``action[1:]`` 常量重复、``traj6/traj30`` 由运动学插值、
   ``roundtrip_*`` = NaN；
4. 病态过滤（行级 stuck / yaw_outlier，只改 ``train_weight`` 不删行）；
5. schema 往返：行 → ``save_dataset`` → ``BCDataset.load`` + ``dagger`` meta 块（``roll_in: true``）；
6. 场景池选择：``--from-eval`` 的 ``episodes.csv``（``id``/``spec_id``、``success`` 各种写法）+ spec 源解析；
7. 任务分块在 specs < chunk_size 时也按 worker 数铺开。
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


# --------------------------------------------------------------------------- 6. 场景池选择
def test_select_specs_from_eval_csv_supports_id_and_spec_id(tmp_path: Path) -> None:
    from env.scenario.spec import load_specs, save_specs

    specs = list(load_specs("env/specs/scenarios_smoke16.json"))[:3]
    spec_source = tmp_path / "specs.json"
    save_specs(specs, spec_source)
    csv_path = tmp_path / "episodes.csv"
    csv_path.write_text(
        "id,success,termination\n"
        f"{specs[0].id},False,collision\n"
        f"{specs[1].id},true,arrive_dest\n"
        f"{specs[2].id},0,off_road\n",
        encoding="utf-8",
    )
    args = Namespace(specs="", from_eval=str(csv_path), spec_source=str(spec_source), limit=0)
    selected, provenance = dg._select_specs(args)
    assert [spec.id for spec in selected] == [specs[0].id, specs[2].id]
    assert provenance["failed_ids"] == 2 and provenance["matched_specs"] == 2
    # 兼容 spec_id 列名
    csv_spec_id = tmp_path / "episodes_spec_id.csv"
    csv_spec_id.write_text(f"spec_id,success\n{specs[1].id},False\n", encoding="utf-8")
    args2 = Namespace(specs="", from_eval=str(csv_spec_id), spec_source=str(spec_source), limit=1)
    selected2, _ = dg._select_specs(args2)
    assert [spec.id for spec in selected2] == [specs[1].id]


def test_select_specs_requires_pool(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dg, "_latest_eval_csv", lambda: None)
    args = Namespace(specs="", from_eval="", spec_source="", limit=0)
    with pytest.raises(SystemExit):
        dg._select_specs(args)


# --------------------------------------------------------------------------- 7. 分块铺满 worker
def test_chunk_tasks_spreads_small_pools_across_workers() -> None:
    specs = list(range(5))
    chunks = dg._chunk_tasks(specs, 10, workers=2)
    assert [len(chunk) for chunk in chunks] == [3, 2]
    assert sorted(index for chunk in chunks for index, _ in chunk) == [0, 1, 2, 3, 4]
    assert len(dg._chunk_tasks(specs, 10, workers=5)) == 5
    assert len(dg._chunk_tasks(range(25), 10, workers=2)) == 3  # 大批量仍按 chunk_size=10
