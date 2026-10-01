"""P2 奖励审计工具（``tools/reward_audit.py``）口径单测：dry-run 端到端 + 分区互斥 + 重放语义。

- dry-run：合成样本每类 3 条（≥2），全档反解→重放应回到目标剖面（|Δ| ≤ 0.5）；
- 审计/反解样本同类内交替分配 → 互斥且确定性；
- 重放：CaRL 乘子只清零正向稠密、终局值只在终局帧、max_step 注入、low_speed 前车拆分。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import reward_audit
from tools.reward_audit import (
    CLASS_ORDER,
    TARGETS,
    _partition,
    _replay_episode,
    _synthetic_episode,
)


def test_dry_run_roundtrip_targets(tmp_path: Path) -> None:
    """dry-run 端到端：每类 3 条合成样本，全部 rc 档的审计剖面回到目标（|Δ| ≤ 0.5）。"""
    out = tmp_path / "dry"
    result = reward_audit.dry_run(argparse.Namespace(out=str(out)))
    assert result["meta"]["per_class_pass"] is False  # 合成样本 n=3 < 10（仅自检用）
    for tier in ("1.0", "3.0", "10.0", "30.0"):
        profile = result["tiers"][tier]["audit_profile"]
        for cls in CLASS_ORDER:
            item = profile[cls]
            assert item["n"] >= 2
            # 合成样本无噪声：反解→重放的往返差应回到整数舍入容差（≤ 0.5）
            assert abs(item["delta"]) <= reward_audit.TOLERANCE, (tier, cls, item["delta"])
            assert abs(item["delta"]) <= reward_audit.PROFILE_TOLERANCE
    assert (out / "reward_audit.md").is_file()
    assert (out / "reward_audit.json").is_file()
    assert (out / "config_draft_rc3.yaml").is_file()


def test_partition_is_mutually_exclusive_and_deterministic() -> None:
    episodes = []
    for cls_index, cls in enumerate(CLASS_ORDER):
        for repeat in range(5):
            episodes.append(_synthetic_episode(cls, spec_id=cls_index * 100 + repeat))
    audit, solve = _partition(episodes)
    keys_audit = {(e["meta"]["spec_id"], e["meta"]["spec_seed"]) for e in audit}
    keys_solve = {(e["meta"]["spec_id"], e["meta"]["spec_seed"]) for e in solve}
    assert not (keys_audit & keys_solve)
    assert len(audit) == 12 and len(solve) == 8  # 每类 5 条 → 交替 3/2
    audit2, solve2 = _partition(list(reversed(episodes)))
    assert [(e["meta"]["spec_id"]) for e in audit2] == [e["meta"]["spec_id"] for e in audit]
    assert [(e["meta"]["spec_id"]) for e in solve2] == [e["meta"]["spec_id"] for e in solve]


def test_replay_collision_carl_and_terminal_semantics() -> None:
    """碰撞 episode：正向稠密被乘子清零、同帧罚分保留、终局值只加一次。"""
    episode = _synthetic_episode("collision", spec_id=7, steps=4)
    replay = _replay_episode(
        episode,
        rc_weight=1.0,
        terminal_values={"arrive_dest": 31.0, "collision": -17.0, "out_of_road": -11.0,
                         "max_step": -19.0, "error": -5.0},
    )
    # 4 步：前 3 步 speed_ratio 0.4×0.625=0.25/步 + 塑形；末步正向清零
    assert replay.carl_hits == 1
    assert replay.terminal_hits == 1
    assert replay.terminal_value == pytest.approx(-17.0)
    assert replay.terminating_sum == pytest.approx(-10.0)  # crash 项
    assert replay.carl_zeroed_dense > 0.0
    # total = pos×mult+neg + terminating + terminal（carl_penalty=0）
    assert replay.total == pytest.approx(
        replay.dense_effective + replay.terminating_sum + replay.terminal_value
    )
    # 正稠密口径（忽略乘子）严格大于 effective（被清零）
    assert replay.dense_positive_only > replay.dense_effective


def test_replay_max_step_injection_and_low_speed_split() -> None:
    episode = _synthetic_episode("max_step", spec_id=9, steps=3)
    replay = _replay_episode(
        episode, rc_weight=1.0, terminal_values={"max_step": -19.0}
    )
    assert replay.terminal_hits == 1
    assert replay.terminal_value == pytest.approx(-19.0)
    # 合成 ctx 全帧有前车（lead_gap_m=20）且 v=5 → low_speed=0，且计入"有前车"桶
    assert replay.low_speed_lead_steps == 3
    assert replay.low_speed_nolead_steps == 0
    assert replay.low_speed_lead_sum == pytest.approx(0.0)

    # 无前车 + 低速：low_speed 计入无前车桶且为负
    episode_ls = _synthetic_episode("arrive_dest", spec_id=10, steps=3)
    for record in episode_ls["steps"]:
        record["ctx"]["speed"] = 0.0
        record["ctx"]["speed_ratio"] = 0.0
        record["ctx"]["lead_gap_m"] = -1.0
    replay_ls = _replay_episode(episode_ls, rc_weight=1.0, terminal_values={"arrive_dest": 0.0})
    assert replay_ls.low_speed_nolead_steps == 3
    assert replay_ls.low_speed_nolead_sum == pytest.approx(-0.6)  # 3 × (−0.2)


def test_solve_uses_effective_dense_semantics() -> None:
    """反解式：终局值 = 目标 − 稠密贡献(effective) − 终止项 − carl_penalty。"""
    episode = _synthetic_episode("collision", spec_id=11, steps=4)
    solve_set = [episode, _synthetic_episode("collision", spec_id=12, steps=4)]
    replays = [
        _replay_episode(
            item, rc_weight=1.0, terminal_values={"arrive_dest": 31.0, "collision": -17.0,
                                                   "out_of_road": -11.0, "max_step": -19.0,
                                                   "error": -5.0}
        )
        for item in solve_set
    ]
    solved = reward_audit._solve_terminal_values(replays, TARGETS)
    item = solved["collision"]
    assert item["n"] == 2
    raw = TARGETS["collision"] - item["dense_effective_mean"] - item["terminating_mean"] - item["carl_penalty_mean"]
    assert item["resolved_raw"] == pytest.approx(raw)
    assert abs(item["recompute_delta"]) <= reward_audit.TOLERANCE


def test_analyze_manifest_hashes(tmp_path: Path) -> None:
    """analyze 的样本清单含 id + ctx_sha256 + 文件 sha256（可追溯）。"""
    out = tmp_path / "dry2"
    result = reward_audit.dry_run(argparse.Namespace(out=str(out)))
    for label in ("audit", "solve"):
        for cls in CLASS_ORDER:
            for row in result[label]["manifest"][cls]:
                assert row["ctx_sha256"].startswith("sha256:")
                assert row["file_sha256"].startswith("sha256:")
                assert isinstance(row["spec_id"], int)
