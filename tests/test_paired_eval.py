"""v7-P0 配对评测工具（``tools/paired_eval.py``）单测：配对统计 / CI / 分层 / 多 run 汇总。

口径对齐：``net = fixed − broken``、``z = |net|/√(fixed+broken)``（v6 §7.3）；McNemar 精确
检验与 scipy 交叉核对（可用时）；bootstrap 固定 seed 可复现。

运行：``tools/venv-python -m pytest tests/test_paired_eval.py -q``。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from pipeline import eval_runner as er
from tools import paired_eval as pe


# --------------------------------------------------------------------------- #
# 合成数据工具
# --------------------------------------------------------------------------- #
def _episode(
    spec_id: int,
    success: bool,
    *,
    seed: int = 1000,
    collision: bool = False,
    off_road: bool = False,
    termination: str | None = None,
    primary: str = "straight",
    difficulty: str = "easy",
    split: str = "val",
    policy: str = "x",
    error: str = "",
) -> dict:
    return {
        "id": spec_id,
        "seed": seed,
        "split": split,
        "difficulty": difficulty,
        "primary": primary,
        "geometry": [primary],
        "extra_geometry": [],
        "compound": False,
        "success": success,
        "collision": collision,
        "off_road": off_road,
        "termination": termination or ("arrive_dest" if success else "out_of_road"),
        "steps": 10,
        "route_completion": 0.9 if success else 0.4,
        "policy": policy,
        "error": error,
    }


def _write_csv(path: Path, episodes: list[dict]) -> Path:
    er.write_episodes_csv(episodes, path)
    return path


def _record(
    spec_id: int,
    success: bool,
    *,
    collision: bool = False,
    off_road: bool = False,
    termination: str | None = None,
    primary: str = "straight",
    difficulty: str = "easy",
) -> pe.Episode:
    return pe.Episode(
        spec_id=spec_id,
        seed=1000,
        split="val",
        difficulty=difficulty,
        primary=primary,
        geometry=(primary,),
        success=success,
        collision=collision,
        off_road=off_road,
        termination=termination or ("arrive_dest" if success else "out_of_road"),
        route_completion=0.9 if success else 0.4,
        steps=10,
        policy="x",
        error="",
    )


# --------------------------------------------------------------------------- #
# McNemar / bootstrap / IQM
# --------------------------------------------------------------------------- #
def test_mcnemar_exact_reference_values() -> None:
    """手算参照：p = min(1, 2·Σ_{i≤min(b,c)} C(n,i)/2^n)。"""
    assert pe.mcnemar_exact_p(0, 0) == 1.0
    assert pe.mcnemar_exact_p(3, 0) == pytest.approx(0.25)
    assert pe.mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert pe.mcnemar_exact_p(10, 1) == pytest.approx(12 / 1024)
    assert pe.mcnemar_exact_p(2, 2) == 1.0  # 2·(1+4+6)/16 = 1.375 → 截断 1


def test_mcnemar_exact_matches_scipy() -> None:
    """与 scipy binomtest（two-sided, p=0.5）交叉核对。"""
    scipy_stats = pytest.importorskip("scipy.stats")
    for fixed, broken in ((1, 0), (5, 2), (10, 3), (30, 12), (50, 50), (120, 90)):
        expected = scipy_stats.binomtest(min(fixed, broken), fixed + broken, 0.5).pvalue
        assert pe.mcnemar_exact_p(fixed, broken) == pytest.approx(expected, abs=1e-12)


def test_bootstrap_ci_deterministic_and_bounds() -> None:
    values = [1.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    first = pe.bootstrap_mean_ci(values, n_boot=500, seed=7)
    second = pe.bootstrap_mean_ci(values, n_boot=500, seed=7)
    assert first == second  # 固定 seed 可复现
    lo, hi = first
    assert lo is not None and hi is not None
    assert lo <= sum(values) / len(values) <= hi
    assert pe.bootstrap_mean_ci([], n_boot=10, seed=1) == (None, None)
    # 不同 seed → 重采样序列不同（同数据同 B 的百分位 CI 通常仍接近，只断言不报错且有序）
    lo2, hi2 = pe.bootstrap_mean_ci(values, n_boot=500, seed=8)
    assert lo2 <= hi2


def test_iqm_reference_values() -> None:
    """IQM：排序后两侧各裁 int(n*0.25)（rliable 口径）。"""
    assert pe.iqm([]) is None
    assert pe.iqm([0.01, 0.02, 0.03]) == pytest.approx(0.02)  # trim=0 → 全均值
    assert pe.iqm([1, 2, 3, 4, 5, 6, 7, 8]) == pytest.approx(4.5)  # trim=2 → (3+4+5+6)/4


# --------------------------------------------------------------------------- #
# 配对统计：2×2 / net / z / CI
# --------------------------------------------------------------------------- #
def test_paired_stats_2x2_and_alignment() -> None:
    """已知翻牌：fixed/broken/both/net/z 与既有口径逐位一致。"""
    base_flags = [True, True, True, False, False, False]
    agent_flags = [True, False, True, False, True, False]
    rows = [(_record(i, b), _record(i, a)) for i, (b, a) in enumerate(zip(base_flags, agent_flags))]
    stats = pe.paired_stats(rows, "success", n_boot=200, seed=1)
    assert (stats["fixed"], stats["broken"], stats["both_pass"], stats["both_fail"]) == (1, 1, 2, 2)
    assert stats["net"] == 0
    assert stats["z"] == 0.0
    assert stats["delta"] == 0.0
    assert stats["delta_pp"] == 0.0
    assert stats["mcnemar_exact_p"] == 1.0
    assert stats["rate_base"] == pytest.approx(0.5)
    assert stats["rate_agent"] == pytest.approx(0.5)
    assert stats["ci95"][0] <= 0.0 <= stats["ci95"][1]
    assert stats["direction"] == "higher_is_better"

    # 全翻转：fixed=10、broken=0；z = |net|/√(fixed+broken)（无连续性校正）
    rows = [(_record(i, False), _record(i, True)) for i in range(10)]
    stats = pe.paired_stats(rows, "success", n_boot=200, seed=1)
    assert stats["fixed"] == 10 and stats["broken"] == 0 and stats["net"] == 10
    assert stats["z"] == pytest.approx(10 / math.sqrt(10))
    assert stats["delta_pp"] == pytest.approx(100.0)
    assert stats["mcnemar_exact_p"] == pytest.approx(2 / 2**10)
    assert stats["ci95"][0] > 0.0


def test_paired_stats_z_formula_and_zero_denominator() -> None:
    """z = |net|/√(fixed+broken)；分母为 0（无翻牌）→ z=None（不伪造数值）。"""
    rows = [(_record(i, True), _record(i, False)) for i in range(4)]
    rows += [(_record(100 + i, True), _record(100 + i, True)) for i in range(3)]
    stats = pe.paired_stats(rows, "success", n_boot=100, seed=2)
    assert stats["fixed"] == 0 and stats["broken"] == 4 and stats["net"] == -4
    assert stats["z"] == pytest.approx(4 / 2.0)
    same = [(_record(i, True), _record(i, True)) for i in range(3)]
    no_flip = pe.paired_stats(same, "success", n_boot=100, seed=2)
    assert no_flip["z"] is None
    assert no_flip["delta_pp"] == 0.0


# --------------------------------------------------------------------------- #
# 读取 / 配对（fail-closed）
# --------------------------------------------------------------------------- #
def test_load_episodes_csv_roundtrip(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "episodes.csv",
        [
            _episode(4, True, primary="roundabout", difficulty="hard"),
            _episode(5, False, collision=True, termination="collision"),
        ],
    )
    loaded = pe.load_episodes_csv(path)
    assert set(loaded) == {(4, 1000), (5, 1000)}
    first = loaded[(4, 1000)]
    assert first.success is True and first.primary == "roundabout" and first.difficulty == "hard"
    assert first.geometry == ("roundabout",)
    assert loaded[(5, 1000)].collision is True and loaded[(5, 1000)].termination == "collision"


def test_load_episodes_csv_duplicate_key_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "seed", "success", "collision", "off_road", "termination"]
        )
        writer.writeheader()
        for _ in range(2):
            writer.writerow(
                {
                    "id": 1,
                    "seed": 2,
                    "success": "True",
                    "collision": "False",
                    "off_road": "False",
                    "termination": "arrive_dest",
                }
            )
    with pytest.raises(SystemExit):
        pe.load_episodes_csv(path)


def test_pair_episodes_fail_closed_and_allow_mismatch() -> None:
    baseline = {(i, 1000): _record(i, True) for i in (1, 2)}
    agent = {(1, 1000): _record(1, False), (3, 1000): _record(3, True)}
    with pytest.raises(SystemExit):
        pe.pair_episodes(baseline, agent)
    rows, info = pe.pair_episodes(baseline, agent, allow_mismatch=True)
    assert len(rows) == 1
    assert info["n_common"] == 1 and info["n_baseline_only"] == 1 and info["n_agent_only"] == 1


def test_pair_episodes_label_mismatch_fails_closed() -> None:
    baseline = {(1, 1000): _record(1, True, primary="curve", difficulty="easy")}
    agent = {(1, 1000): _record(1, True, primary="merge", difficulty="easy")}
    with pytest.raises(SystemExit):
        pe.pair_episodes(baseline, agent)


# --------------------------------------------------------------------------- #
# 单对分析：分项 + 分层
# --------------------------------------------------------------------------- #
def test_analyze_pair_items_and_strata() -> None:
    baseline = {
        (i, 1000): _record(
            i,
            success=(i < 2),
            collision=(i == 3),
            off_road=(i == 4),
            termination="max_step" if i == 5 else None,
            primary="curve" if i < 3 else "merge",
            difficulty="easy" if i % 2 == 0 else "hard",
        )
        for i in range(6)
    }
    agent = {
        (i, 1000): _record(
            i,
            success=True,
            collision=(i == 4),
            off_road=(i == 5),
            termination="max_step" if i == 3 else None,
            primary="curve" if i < 3 else "merge",
            difficulty="easy" if i % 2 == 0 else "hard",
        )
        for i in range(6)
    }
    result = pe.analyze_pair(baseline, agent, n_boot=200, seed=3, min_stratum=5)
    assert result["n_pairs"] == 6
    assert result["items"]["success"]["fixed"] == 4 and result["items"]["success"]["broken"] == 0
    assert result["items"]["collision"]["net"] == 0  # 3→4 与 4→3 各一
    assert result["items"]["off_road"]["net"] == 0
    assert result["items"]["max_step"]["fixed"] == 1 and result["items"]["max_step"]["broken"] == 1
    assert result["items"]["collision"]["direction"] == "lower_is_better"
    assert set(result["strata"]["primary_x_difficulty"]) == {"curve×easy", "curve×hard", "merge×easy", "merge×hard"}
    assert result["strata"]["primary_x_difficulty"]["curve×easy"]["n"] == 2
    assert result["strata"]["primary_x_difficulty"]["curve×easy"]["judged"] is False  # n < 5
    assert result["verdict"]["positive_3pt_z196"] is True
    assert result["verdict"]["ci_lower_gt_zero"] is True


# --------------------------------------------------------------------------- #
# 多 run 汇总
# --------------------------------------------------------------------------- #
def _fake_run_pair(
    delta_pp: float,
    *,
    collision_delta_pp: float = 0.0,
    off_road_delta_pp: float = 0.0,
    collision_rate: float = 0.05,
    off_road_rate: float = 0.05,
) -> dict:
    return {
        "items": {
            "success": {
                "delta_pp": delta_pp,
                "rate_agent": 0.8,
                "rate_base": 0.75,
                "net": int(delta_pp),
                "z": 2.0,
                "mcnemar_exact_p": 0.01,
            },
            "collision": {"rate_agent": collision_rate, "delta_pp": collision_delta_pp},
            "off_road": {"rate_agent": off_road_rate, "delta_pp": off_road_delta_pp},
            "max_step": {"delta_pp": 0.0},
        },
        "verdict": {"positive_3pt_z196": delta_pp >= 3.0},
    }


def test_aggregate_runs_mean_median_iqm_ci() -> None:
    runs = [_fake_run_pair(delta) for delta in (1.0, 2.0, 3.0, 4.0, 5.0)]
    aggregate = pe.aggregate_runs(runs, n_boot=500, seed=5)
    delta = aggregate["metrics"]["delta_pp"]
    assert delta["n"] == 5
    assert delta["mean"] == pytest.approx(3.0)
    assert delta["median"] == pytest.approx(3.0)
    assert delta["iqm"] == pytest.approx(3.0)  # trim=1 → (2+3+4)/3
    assert delta["min"] == 1.0 and delta["max"] == 5.0
    assert delta["ci95"][0] > 0.0  # 全正 → 下界 > 0
    verdict = aggregate["verdict"]
    assert verdict["n_runs"] == 5
    assert verdict["primary_ci_lower_gt_zero"] is True
    assert verdict["primary_iqm_positive"] is True
    assert verdict["positive_runs"] == 3  # Δ ≥ 3 的 run
    assert verdict["collision_gate"]["passed"] is True
    assert verdict["off_road_gate"]["passed"] is True


def test_aggregate_runs_negative_and_gate_failure() -> None:
    runs = [
        _fake_run_pair(
            delta,
            collision_delta_pp=5.0,
            off_road_delta_pp=4.0,
            collision_rate=0.15,
            off_road_rate=0.15,
        )
        for delta in (-6.0, -4.0, -5.0)
    ]
    aggregate = pe.aggregate_runs(runs, n_boot=500, seed=5)
    verdict = aggregate["verdict"]
    assert verdict["primary_ci_lower_gt_zero"] is False
    assert verdict["primary_iqm_positive"] is False
    assert verdict["collision_gate"]["passed"] is False  # 绝对 15% > 10% 且 Δ=+5pt > 0
    assert verdict["off_road_gate"]["passed"] is False  # 绝对 15% > 10% 且 Δ=+4pt > +2pt


def test_aggregate_runs_deterministic() -> None:
    runs = [_fake_run_pair(delta) for delta in (1.0, 2.0, 3.0)]
    first = pe.aggregate_runs(runs, n_boot=300, seed=11)
    second = pe.aggregate_runs(runs, n_boot=300, seed=11)
    assert first == second


# --------------------------------------------------------------------------- #
# CLI 端到端
# --------------------------------------------------------------------------- #
def _scenario_episodes(offset: int, success_ids: set[int], total: int = 20) -> list[dict]:
    return [
        _episode(
            offset + i,
            success=i in success_ids,
            primary="curve" if i % 2 == 0 else "merge",
            difficulty="easy" if i % 3 else "hard",
        )
        for i in range(total)
    ]


def test_cli_end_to_end_multi_run(tmp_path: Path) -> None:
    baseline = _write_csv(tmp_path / "base" / "episodes.csv", _scenario_episodes(0, set(range(0, 12))))
    agents = []
    for index, wins in enumerate((13, 15, 17)):
        path = tmp_path / f"agent{index}" / "episodes.csv"
        agents.append(_write_csv(path, _scenario_episodes(0, set(range(0, wins)))))
    out_dir = tmp_path / "report"
    code = pe.main(
        [
            "--baseline",
            str(baseline),
            "--agent",
            *[str(path) for path in agents],
            "--bootstrap",
            "300",
            "--seed",
            "9",
            "--min-stratum",
            "5",
            "--out-dir",
            str(out_dir),
            "--quiet",
        ]
    )
    assert code == 0
    markdown = (out_dir / "paired_eval.md").read_text(encoding="utf-8")
    assert "McNemar" in markdown and "多 run 汇总" in markdown
    report = json.loads((out_dir / "paired_eval.json").read_text(encoding="utf-8"))
    assert report["aggregate"]["verdict"]["n_runs"] == 3
    assert report["meta"]["n_boot"] == 300 and report["meta"]["seed"] == 9
    # 每个 run 的 success 配对键一致（同 spec 同 (id, seed)）
    assert report["run_pairs"][0]["pair"]["n_common"] == 20
    assert report["run_pairs"][0]["items"]["success"]["fixed"] > 0


def test_cli_strict_mismatch_fails_closed(tmp_path: Path) -> None:
    baseline = _write_csv(tmp_path / "base" / "episodes.csv", _scenario_episodes(0, {0, 1}, total=4))
    agent = _write_csv(tmp_path / "agent" / "episodes.csv", _scenario_episodes(2, {2, 3}, total=4))
    with pytest.raises(SystemExit):
        pe.main(["--baseline", str(baseline), "--agent", str(agent), "--quiet"])
