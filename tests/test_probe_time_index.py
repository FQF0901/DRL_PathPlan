"""P1-0C 时间索引探针单测（含恒速合成 episode 的链积分锚点）。

恒速 episode：ds=5m、dθ=0 ×6 步链经 ``env.tracking.interpolate`` 积分 →
政策步端点 (5..30, 0)（容差 1e-9），且与合成 episode 相邻行位移一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from env.tracking import interpolate
from tools.controls import probe_time_index as probe


def _make_npz(path: Path, episodes_steps: list[list[int]]) -> Path:
    episode_id: list[int] = []
    step: list[int] = []
    for index, steps in enumerate(episodes_steps):
        episode_id += [index] * len(steps)
        step += steps
    episode_id_arr = np.array(episode_id, dtype=np.int64)
    step_arr = np.array(step, dtype=np.int64)
    n = len(step_arr)
    pose = np.stack(
        [step_arr.astype(np.float64), np.zeros(n), np.zeros(n)], axis=1
    ).astype(np.float32)
    action = np.tile(np.array([5.0, 0.0], dtype=np.float32), (n, 6, 1))
    traj6 = np.zeros((n, 6, 2), dtype=np.float32)
    for k in range(1, 7):
        traj6[:, k - 1, 0] = 5.0 * k
    np.savez(
        path,
        episode_id=episode_id_arr,
        step=step_arr,
        pose=pose,
        ego_world=pose[:, None, :].copy(),
        traj6=traj6,
        action=action,
        train_weight=np.ones(n, dtype=np.float32),
    )
    return path


def test_constant_speed_episode_pass(tmp_path: Path) -> None:
    # 链积分锚点：ds=5m、dθ=0 ×6 → 政策步端点 (5..30, 0)
    chain = np.array([[5.0, 0.0]] * 6, dtype=np.float64)
    dense = interpolate(chain, dt=0.5, hz=10)
    key_points = dense[[4, 9, 14, 19, 24, 29], :2]
    expected = np.stack([np.arange(1, 7) * 5.0, np.zeros(6)], axis=1)
    assert np.allclose(key_points, expected, atol=1e-9)

    steps = list(range(0, 31, 5))
    dataset = _make_npz(tmp_path / "const.npz", [steps, steps])
    payload = probe.run_probe(dataset, tmp_path / "nested" / "out.json", quiet=True)

    assert payload["status"] == "PASS"
    assert payload["stride"] == 5
    assert payload["n_violations"] == 0
    assert payload["n_episodes"] == 2
    assert payload["checks"]["monotonic_increasing"] is True
    assert payload["checks"]["all_diffs_equal_mode"] is True
    # 合成 episode 相邻行位移 == 链积分端点
    rel = np.array(steps[1:], dtype=np.float64) - steps[0]
    assert np.allclose(rel, key_points[:, 0], atol=1e-9)
    assert (tmp_path / "nested" / "out.json").is_file()

    assert probe.main(["--dataset", str(dataset), "--out", str(tmp_path / "cli.json")]) == 0


def test_stride_gap_fail(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "gap.npz", [[0, 5, 10, 17, 20, 25, 30]])
    payload = probe.run_probe(dataset, tmp_path / "gap.json", quiet=True)

    assert payload["status"] == "FAIL"
    assert payload["n_violations"] == 2  # 17→20 与 20→25 两侧各一次（注入 17 后差 7/3）
    assert payload["checks"]["monotonic_increasing"] is True
    assert probe.main(["--dataset", str(dataset), "--out", str(tmp_path / "gap_cli.json")]) == 1


def test_non_monotonic_fail(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "back.npz", [[0, 5, 10, 5, 10, 15, 20]])
    payload = probe.run_probe(dataset, tmp_path / "back.json", quiet=True)

    assert payload["status"] == "FAIL"
    assert payload["checks"]["monotonic_increasing"] is False
    assert payload["n_violations"] >= 1


def test_missing_dataset_not_available(tmp_path: Path) -> None:
    payload = probe.run_probe(tmp_path / "nope.npz", tmp_path / "na.json", quiet=True)
    assert payload["status"] == "NOT_AVAILABLE"
    assert "reason" in payload
    assert probe.main(["--dataset", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "na2.json")]) == 2
