"""P1-0C 跨行 action 链 ↔ ego_world 对齐探针 离线单测（合成小 npz）。

覆盖：直行链 PASS；篡改一条链 FAIL；零链但 train_weight>0 FAIL；stride 非 5 FAIL；
数据集缺失 → NOT_AVAILABLE。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.controls import probe_action_ego_alignment as probe


def _make_npz(
    path: Path,
    *,
    stride: int = 5,
    n_eps: int = 2,
    n_valid: int = 7,
    n_tail: int = 6,
    tamper: tuple[int, int] | None = None,
    zero_chain_valid: bool = False,
) -> Path:
    """直行链合成数据：前 n_valid 行为有效行（weight=1），其后 n_tail 行为占位行（weight=0）。

    所有有效行的 k=1..6 未来行都存在 → coverage=100%；traj6 精确等于未来行相对位姿。
    """
    n_per_ep = n_valid + n_tail
    episode_id = np.repeat(np.arange(n_eps), n_per_ep)
    step = np.tile(np.arange(n_per_ep) * stride, n_eps).astype(np.int64)
    n = len(episode_id)
    x = 0.1 * step.astype(np.float64)
    pose = np.stack([x, np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    ego_world = pose[:, None, :].copy()

    index = np.arange(n)
    in_valid = (index % n_per_ep) < n_valid
    weight = np.where(in_valid, 1.0, 0.0).astype(np.float32)
    action = np.zeros((n, 6, 2), dtype=np.float32)
    action[in_valid] = np.array([0.1, 0.0], dtype=np.float32)

    traj6 = np.zeros((n, 6, 2), dtype=np.float32)
    for e in range(n_eps):
        base = e * n_per_ep
        for i in range(n_valid):
            for k in range(1, 7):
                traj6[base + i, k - 1] = (x[base + i + k] - x[base + i], 0.0)

    if tamper is not None:
        row, k = tamper
        traj6[row, k - 1] += np.array([10.0, 0.0], dtype=np.float32)
    if zero_chain_valid:
        action[np.flatnonzero(in_valid)[0]] = 0.0

    np.savez(
        path,
        episode_id=episode_id,
        step=step,
        pose=pose,
        ego_world=ego_world,
        traj6=traj6,
        action=action,
        train_weight=weight,
    )
    return path


def test_straight_chain_pass(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "straight.npz")
    out = tmp_path / "nested" / "alignment.json"
    payload = probe.run_probe(dataset, out, quiet=True)

    assert payload["status"] == "PASS"
    assert payload["stride"] == 5
    assert payload["n_total"] == 26
    assert payload["n_weight_pos"] == 14
    assert payload["zero_chain"] == {"count": 12, "with_weight_pos": 0}
    assert payload["sample"] == 14
    for k in ("1", "2", "3", "4", "5", "6"):
        stats = payload["per_k"][k]
        assert stats["coverage"] == 1.0
        assert stats["median"] == 0.0
        assert stats["p99"] == 0.0
    assert set(
        (
            "status",
            "dataset",
            "n_total",
            "n_weight_pos",
            "stride",
            "zero_chain",
            "per_k",
            "thresholds",
            "sample",
            "seed",
            "generated_at",
        )
    ) <= set(payload)
    assert out.is_file()  # 目标目录自动创建

    assert probe.main(["--dataset", str(dataset), "--out", str(tmp_path / "cli.json")]) == 0


def test_tampered_chain_fail(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "tamper.npz", tamper=(0, 3))
    payload = probe.run_probe(dataset, tmp_path / "tamper.json", quiet=True)

    assert payload["status"] == "FAIL"
    assert payload["per_k"]["3"]["p99"] > 1.0
    assert payload["per_k"]["3"]["max"] >= 10.0
    assert any("p99" in reason for reason in payload["fail_reasons"])
    assert probe.main(["--dataset", str(dataset), "--out", str(tmp_path / "tamper_cli.json")]) == 1


def test_zero_chain_with_weight_fail(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "zero.npz", zero_chain_valid=True)
    payload = probe.run_probe(dataset, tmp_path / "zero.json", quiet=True)

    assert payload["status"] == "FAIL"
    assert payload["zero_chain"]["count"] == 13  # 12 占位行 + 1 条被置零的有效行
    assert payload["zero_chain"]["with_weight_pos"] == 1
    assert any("全零" in reason for reason in payload["fail_reasons"])


def test_stride_not_5_fail(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "stride.npz", stride=6)
    payload = probe.run_probe(dataset, tmp_path / "stride.json", quiet=True)

    assert payload["stride"] == 6
    assert payload["status"] == "FAIL"
    assert any("stride" in reason for reason in payload["fail_reasons"])


def test_missing_dataset_not_available(tmp_path: Path) -> None:
    missing = tmp_path / "nope.npz"
    payload = probe.run_probe(missing, tmp_path / "na.json", quiet=True)

    assert payload["status"] == "NOT_AVAILABLE"
    assert "reason" in payload
    assert probe.main(["--dataset", str(missing), "--out", str(tmp_path / "na_cli.json")]) == 2
