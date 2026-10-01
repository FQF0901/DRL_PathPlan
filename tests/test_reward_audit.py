"""P2 奖励审计工具（``tools/reward_audit.py``）口径单测：dry-run 端到端 + 分区互斥 + 重放语义。

- dry-run：合成样本每类 3 条（≥2），全档反解→重放应回到目标剖面（|Δ| ≤ 0.5）；
- 审计/反解样本按终局类**分层随机**划分 → 互斥、seed 可复现、与输入顺序无关；
- **exclude 默认生效**（排除 eval500）与 ``--no-exclude``；analyze 兜底过滤；
- 报告 meta 记录 HEAD commit sha + split seed；
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


def _keys(rows) -> set:
    return {(e["meta"]["spec_id"], e["meta"]["spec_seed"]) for e in rows}


def test_partition_is_stratified_random_and_deterministic() -> None:
    """分层随机划分：同类内 seed 随机、同 seed 可复现、与输入顺序无关、两半互斥且覆盖全集。"""
    episodes = []
    for cls_index, cls in enumerate(CLASS_ORDER):
        for repeat in range(8):
            episodes.append(_synthetic_episode(cls, spec_id=cls_index * 100 + repeat))
    audit, solve = _partition(episodes, split_seed=1234)
    keys_audit, keys_solve = _keys(audit), _keys(solve)
    assert not (keys_audit & keys_solve)
    assert len(audit) == 16 and len(solve) == 16  # 每类 8 条 → 4/4（分层平衡）
    assert keys_audit | keys_solve == _keys(episodes)
    for cls in CLASS_ORDER:
        assert sum(1 for e in audit if e["meta"]["termination"] == cls) == 4
    # 同 seed 确定性 + 输入顺序无关
    audit2, solve2 = _partition(list(reversed(episodes)), split_seed=1234)
    assert _keys(audit2) == keys_audit and _keys(solve2) == keys_solve
    # 不同 seed 划分不同（8/类，同划分概率 ~ (1/C(8,4))^4 ≈ 4e-8）
    audit3, _ = _partition(episodes, split_seed=4321)
    assert _keys(audit3) != keys_audit
    # A/B 互换：两半对调
    audit4, solve4 = _partition(episodes, split_seed=1234, swap_ab=True)
    assert _keys(audit4) == keys_solve and _keys(solve4) == keys_audit


def test_exclude_default_removes_eval500_from_val_pool() -> None:
    """默认排除 eval500（Gate2 P4 MUST ①）：val 池候选 500 条且与 eval500 零交集。"""
    pool, exclude = reward_audit.DEFAULT_POOL, reward_audit.DEFAULT_EXCLUDE
    if not pool.is_file() or not exclude.is_file():
        pytest.skip("env/specs 未生成（gitignored 产物）")
    pairs = reward_audit._candidate_pairs(pool, exclude)
    assert len(pairs) == 500
    assert not (set(pairs) & set(reward_audit._load_spec_pairs(exclude)))
    assert reward_audit._candidate_pairs(pool, None) == sorted(reward_audit._load_spec_pairs(pool))


def test_collect_cli_exclude_default_and_no_exclude() -> None:
    """CLI：collect/analyze 默认排除 eval500；--no-exclude 显式关闭；划分参数有默认。"""
    parser = reward_audit._build_parser()
    ns = parser.parse_args(["collect"])
    assert ns.exclude == str(reward_audit.DEFAULT_EXCLUDE)
    assert ns.no_exclude is False
    ns = parser.parse_args(["collect", "--no-exclude"])
    assert ns.no_exclude is True
    ns = parser.parse_args(["analyze"])
    assert ns.exclude == str(reward_audit.DEFAULT_EXCLUDE)
    assert ns.no_exclude is False
    assert ns.split_seed == reward_audit.DEFAULT_SPLIT_SEED
    assert ns.swap_ab is False
    assert ns.min_per_class == reward_audit.MIN_PER_CLASS


def test_resolve_exclude_empty_is_error_and_no_exclude_wins() -> None:
    """空值 --exclude（历史污染根因）报错；--no-exclude 优先于 --exclude。"""
    with pytest.raises(SystemExit):
        reward_audit._resolve_exclude(argparse.Namespace(exclude="", no_exclude=False))
    assert (
        reward_audit._resolve_exclude(
            argparse.Namespace(exclude=str(reward_audit.DEFAULT_EXCLUDE), no_exclude=True)
        )
        is None
    )
    assert reward_audit._resolve_exclude(argparse.Namespace(exclude=None, no_exclude=False)) is None
    with pytest.raises(SystemExit):  # fail-closed：排除文件缺失报错
        reward_audit._load_excluded(Path("/nonexistent/exclude.json"))


def test_analyze_filter_excluded_episodes() -> None:
    """analyze 兜底过滤：采集未排除 eval500 时，复算仍剔除其 (id, seed)。"""
    episodes = [_synthetic_episode(cls, spec_id=i) for i, cls in enumerate(CLASS_ORDER)]
    kept = reward_audit._filter_excluded(episodes, {(0, 5000000)})
    assert len(kept) == len(episodes) - 1
    assert all((e["meta"]["spec_id"], e["meta"]["spec_seed"]) != (0, 5000000) for e in kept)
    assert reward_audit._filter_excluded(episodes, set()) == episodes


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


def test_report_records_head_commit_and_split_seed(tmp_path: Path) -> None:
    """报告 meta/markdown 记录 HEAD commit sha + split seed（Gate2 P4 MUST ④）。"""
    out = tmp_path / "dry_head"
    result = reward_audit.dry_run(argparse.Namespace(out=str(out)))
    meta = result["meta"]
    assert meta["split_seed"] == reward_audit.DEFAULT_SPLIT_SEED
    assert meta["swap_ab"] is False
    assert meta["min_per_class"] == reward_audit.MIN_PER_CLASS
    assert isinstance(meta["head_commit"], str) and len(meta["head_commit"]) >= 7
    markdown = (out / "reward_audit.md").read_text(encoding="utf-8")
    assert meta["head_commit"] in markdown
    assert f"split seed `{reward_audit.DEFAULT_SPLIT_SEED}`" in markdown


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


# --------------------------------------------------------------------------- #
# P4 前置-C：E-β″ 复算（current-code 模式 + seeded 切片 + 交集记录 + 审计/反解各自判定）
# --------------------------------------------------------------------------- #
def _write_specs(path: Path, pairs) -> Path:
    doc = {
        "schema_version": 1,
        "count": len(pairs),
        "specs": [{"id": int(i), "seed": int(s)} for i, s in pairs],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_pool_exclude_overlap_and_id_drop(tmp_path: Path) -> None:
    """池 ∩ 排除集计数（pair 级 + 数值 id 级）与可选 id 级剔除。"""
    pool = _write_specs(tmp_path / "pool.json", [(0, 10), (1, 11), (2, 12), (3, 13)])
    exclude = _write_specs(tmp_path / "exclude.json", [(2, 12), (3, 999)])
    assert reward_audit._pool_exclude_overlap(pool, exclude) == {
        "pair_overlap": 1,
        "id_overlap": 2,
    }
    pairs = reward_audit._candidate_pairs(pool, exclude)
    assert pairs == [(0, 10), (1, 11), (3, 13)]  # pair 级排除
    kept, dropped = reward_audit._drop_exclude_ids(pairs, exclude)
    assert dropped == 1 and kept == [(0, 10), (1, 11)]  # 数值 id 3 被剔除
    assert reward_audit._pool_exclude_overlap(pool, None) == {"pair_overlap": 0, "id_overlap": 0}
    assert reward_audit._drop_exclude_ids(pairs, None) == (pairs, 0)


def test_order_pairs_seeded_deterministic() -> None:
    """seeded 切片：同 seed 可复现、是置换、不同 seed 不同、None 保持原序。"""
    pairs = [(i, 1000 + i) for i in range(200)]
    a = reward_audit._order_pairs(pairs, 20261001)
    b = reward_audit._order_pairs(pairs, 20261001)
    c = reward_audit._order_pairs(pairs, 7)
    assert a == b and sorted(a) == sorted(pairs) and a != pairs and c != a
    assert reward_audit._order_pairs(pairs, None) == pairs


def test_collect_cli_current_code_and_slice_flags() -> None:
    """collect 支持 --code-mode current / --pool-shuffle-seed / --drop-exclude-ids；analyze --label。"""
    parser = reward_audit._build_parser()
    ns = parser.parse_args(
        [
            "collect",
            "--code-mode",
            "current",
            "--pool-shuffle-seed",
            "20261001",
            "--drop-exclude-ids",
            "--pool-extra",
            "env/specs/scenarios_train_5k.json",
        ]
    )
    assert ns.code_mode == "current"
    assert ns.pool_shuffle_seed == 20261001
    assert ns.drop_exclude_ids is True
    assert ns.pool_extra == ["env/specs/scenarios_train_5k.json"]
    ns2 = parser.parse_args(["collect"])
    assert ns2.code_mode == "pre"
    assert ns2.pool_shuffle_seed is None and ns2.drop_exclude_ids is False
    assert ns2.pool_extra is None
    ns3 = parser.parse_args(["analyze", "--label", "E-β″ 奖励审计报告（v6 基座）"])
    assert ns3.label.startswith("E-β")


def test_collector_cli_code_mode_requires_pre_root() -> None:
    """采集器：current 模式无需 --pre-root；pre 模式缺 --pre-root 报错（fail-closed）。"""
    from tools import reward_audit_collect

    assert reward_audit_collect._repo_root() == reward_audit.ROOT
    with pytest.raises(SystemExit):
        reward_audit_collect.main(
            ["--code-mode", "pre", "--specs", "x", "--spec-id", "1", "--spec-seed", "2",
             "--out", "/tmp/opencode/should_not_exist.json"]
        )


def test_collect_routes_each_pair_to_its_source_pool(tmp_path: Path, monkeypatch) -> None:
    """回归：补充池候选必须以自己的 --specs 传给采集器（曾全用主池 → 补充池全部失败）。"""
    pool_main = _write_specs(tmp_path / "val.json", [(0, 10), (1, 11)])
    pool_extra = _write_specs(tmp_path / "5k.json", [(100, 110), (101, 111)])
    exclude = _write_specs(tmp_path / "exclude.json", [(9, 99)])
    out_dir = tmp_path / "eps"
    calls: list = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        if "reward_audit_collect.py" not in cmd[1]:
            return _Result()
        calls.append(list(cmd))
        spec_id = int(cmd[cmd.index("--spec-id") + 1])
        seed = int(cmd[cmd.index("--spec-seed") + 1])
        specs = cmd[cmd.index("--specs") + 1]
        episode = _synthetic_episode("arrive_dest", spec_id=spec_id)
        episode["meta"]["spec_seed"] = seed
        episode["meta"]["spec_file"] = specs
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(episode), encoding="utf-8")
        return _Result()

    monkeypatch.setattr(reward_audit.subprocess, "run", fake_run)
    ns = argparse.Namespace(
        code_mode="current",
        pre_root=None,
        pre_commit="x",
        ckpt=str(tmp_path / "ckpt.pt"),
        pool=str(pool_main),
        pool_extra=[str(pool_extra)],
        exclude=str(exclude),
        no_exclude=False,
        drop_exclude_ids=False,
        pool_shuffle_seed=None,
        out_dir=str(out_dir),
        workers=1,
        batch=10,
        max_steps=1000,
        per_class_target=1,
        max_episodes=0,
        python="python3",
        config="config/default.yaml",
        tracker="lqr",
        eval_reference="plan",
        device="cpu",
    )
    summary = reward_audit.collect(ns)
    routed = {}
    for cmd in calls:
        routed[(int(cmd[cmd.index("--spec-id") + 1]), int(cmd[cmd.index("--spec-seed") + 1]))] = cmd[
            cmd.index("--specs") + 1
        ]
    assert routed[(0, 10)] == str(pool_main)
    assert routed[(1, 11)] == str(pool_main)
    assert routed[(100, 110)] == str(pool_extra)
    assert routed[(101, 111)] == str(pool_extra)
    assert summary["pool_candidates"] == 4
    assert summary["effective_pair_overlap"] == 0


def test_analyze_per_class_checks_audit_and_solve(tmp_path: Path) -> None:
    """§7.1 ④：每类 n ≥ 阈值对审计与反解**各自**判定（合成 6/类 → 3/3）。"""
    out = tmp_path / "out"
    episodes_dir = tmp_path / "eps"
    episodes_dir.mkdir(parents=True)
    for cls_index, cls in enumerate(CLASS_ORDER):
        for repeat in range(6):
            episode = _synthetic_episode(cls, spec_id=cls_index * 100 + repeat)
            path = episodes_dir / f"id{episode['meta']['spec_id']}_seed{episode['meta']['spec_seed']}.json"
            path.write_text(json.dumps(episode), encoding="utf-8")
            episode["path"] = str(path)
    ns = argparse.Namespace(
        episodes_dir=str(episodes_dir),
        out=str(out),
        tiers="1,3,10,30",
        mirror_md=None,
        exclude=None,
        no_exclude=True,
        split_seed=123,
        swap_ab=False,
        min_per_class=3,
        label="E-β″ 奖励审计报告（v6 基座）",
    )
    result = reward_audit.analyze(ns)
    assert result["meta"]["audit_per_class_pass"] is True
    assert result["meta"]["solve_per_class_pass"] is True
    assert result["meta"]["per_class_pass"] is True
    assert result["meta"]["solve_class_counts"] == {cls: 3 for cls in CLASS_ORDER}
    markdown = (out / "reward_audit.md").read_text(encoding="utf-8")
    assert markdown.startswith("# E-β″ 奖励审计报告（v6 基座）")
    assert "审计/反解各自" in markdown

    ns.min_per_class = 4  # 审计 3 / 反解 3 均不足
    result2 = reward_audit.analyze(ns)
    assert result2["meta"]["audit_per_class_pass"] is False
    assert result2["meta"]["solve_per_class_pass"] is False
    assert result2["meta"]["per_class_pass"] is False
