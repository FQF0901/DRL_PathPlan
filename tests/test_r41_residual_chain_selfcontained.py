"""R41 第一批-B：残差训练链**入库自包含守卫**（fixture 尺度；CI 可跑，不因研究工作区/大资源缺失 skip）。

背景
----
R41 复审（``docs/deepseekv4p1_argue.md`` §2.3 / §三 第一批②）：``tools/train_residual.py``
原硬依赖研究工作区 harness（a1_matrix/j1_train）。本测试锁定入库后的拆分与口径：

1. **自包含**：``tools/train_residual.py`` + ``pipeline/residual_dataset.py`` +
   ``pipeline/residual_features.py`` 零研究工作区引用（源码扫描）；
2. **数据/配对/split**：``tests/fixtures/residual_mini``（path_exec_v1 前 8 个 episode 子集，
   ~465 行）上跑 ``load_rows_filtered``/``build_step_map``/``build_pairs``：t→t+5 配对、
   episode 级 split 不相交、同种子可复现；
3. **obs**：``build_obs`` 合成分支键面/宽度/速度写入契约（无 base 数据集依赖）；
4. **特征/目标/ΔA_θ**：小模型（随机初始化，CPU）跑 ``transition_features`` →
   5 键/26 列/4 目标列契约、``tensor_to_features`` 往返逐位、``residual_targets`` 公式、
   ``head_delta`` 与 5 键调用逐位一致；
5. **fixture 资产自洽**：manifest shards sha256 == shard 文件 sha256。

大资源版同源守卫（train↔eval 想象链逐位）在 ``tests/test_r40_train_eval_feature_parity.py``
（资源缺失时 skip；本文件是其无资源核心子集替代）。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/residual_mini"
CHAIN_FILES = (
    ROOT / "tools/train_residual.py",
    ROOT / "pipeline/residual_dataset.py",
    ROOT / "pipeline/residual_features.py",
)

import pipeline.residual_dataset as rd  # noqa: E402
import pipeline.residual_features as rf  # noqa: E402
import tools.train_residual as runner  # noqa: E402
from net.world_residual import (  # noqa: E402
    READOUT_RESIDUAL_FIELDS,
    READOUT_RESIDUAL_INDICES,
    ReadoutResidualHead,
)


def _fixture_args(**overrides) -> Namespace:
    base = dict(data=str(FIXTURE), shards=None, max_pairs=None, seed=0, val_frac=0.25, chunk=64)
    base.update(overrides)
    return Namespace(**base)


def _tiny_model():
    """确定性小模型（随机初始化；CPU；结构化头开；MoE 关）。"""
    from net.model import DrivingModel

    torch.manual_seed(7)
    model = DrivingModel(
        hidden=32, num_experts=1, expert_hidden=16, router_hidden=16,
        primary_hidden=32, trunk_hidden=32, net_hidden=32, spatial_layers=1,
    )
    model.set_moe(enabled=False)
    model.set_path_head(True)
    return model.eval()


# --------------------------------------------------------------------------- #
# 1) 自包含（源码级守卫；CI 无研究工作区也可跑）
# --------------------------------------------------------------------------- #
def test_chain_sources_are_selfcontained() -> None:
    """链内三个文件不得含研究工作区引用/动态 harness 装载。"""
    for path in CHAIN_FILES:
        text = path.read_text(encoding="utf-8")
        assert ".slim" not in text, f"{path.name}: 残留研究工作区引用"
        assert "spec_from_file_location" not in text, f"{path.name}: 残留动态 harness 装载"
        assert "importlib.util" not in text, f"{path.name}: 残留 importlib.util"


def test_harness_info_reports_in_repo_modules() -> None:
    info = runner.harness_info()
    assert sorted(info) == ["residual_dataset", "residual_features", "runner"]
    for entry in info.values():
        path = Path(entry["path"]).resolve()
        assert path.is_file() and ROOT in path.parents
        assert ".slim" not in str(path)
        assert entry["sha256"] == runner._sha256(path)


def test_feature_constants_single_source_net_world_residual() -> None:
    """列/键契约 = net.world_residual 单一出处（train runner 与特征模块不许各写一份）。"""
    assert rf.TARGET_COLUMNS == tuple(READOUT_RESIDUAL_INDICES) == (0, 3, 4, 5)
    assert rf.COLUMN_NAMES == tuple(READOUT_RESIDUAL_FIELDS)
    assert runner.TARGET_COLUMNS == rf.TARGET_COLUMNS
    assert runner.COLUMN_NAMES == rf.COLUMN_NAMES
    assert runner.FEATURE_KEYS == rf.FEATURE_KEYS
    assert runner.FEATURE_WIDTHS == rf.FEATURE_WIDTHS == (8, 8, 2, 4, 4)
    assert sum(rf.FEATURE_WIDTHS) == 26


# --------------------------------------------------------------------------- #
# 2) fixture 资产 + 数据加载/配对/split
# --------------------------------------------------------------------------- #
def test_fixture_assets_self_consistent() -> None:
    import hashlib
    import json

    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shards"], "fixture manifest 缺 shards"
    for shard in manifest["shards"]:
        path = FIXTURE / str(shard["name"])
        assert path.is_file()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha == shard["sha256"], f"{path.name}: sha256 不符（fixture 损坏？）"
        assert int(shard["rows"]) == int(manifest["rows"])
        assert int(shard["episodes"]) == int(manifest["episodes"])


def test_load_rows_filtered_and_step_map_on_fixture() -> None:
    rows, train_weight, horizon_ok = rd.load_rows_filtered(_fixture_args())
    assert rows["source"] == "path_exec_v1" and rows["n"] > 0
    assert train_weight.shape == (rows["n"],) and horizon_ok.shape == (rows["n"],)
    spec = np.asarray(rows["spec_id"], dtype=np.int64)
    episode = np.asarray(rows["episode_id"], dtype=np.int64)
    step = np.asarray(rows["step"], dtype=np.int64)
    step_map = rd.build_step_map(rows)
    assert len(step_map) == rows["n"]
    for i in (0, rows["n"] - 1):
        assert step_map[(int(spec[i]), int(episode[i]), int(step[i]))] == i
    assert rows["long_exog"] is not None and rows["long_present"] is not None
    anchor_lens = np.asarray([len(np.asarray(a)) for a in rows["anchor"]], dtype=np.int64)
    assert anchor_lens.max() >= 3  # 空锚行为合法（ok 规则要求主行 ≥3）


def test_build_pairs_transition_and_split_contract() -> None:
    data = rd.build_pairs(_fixture_args())
    main_ids, succ_ids = data["main_ids"], data["succ_ids"]
    assert main_ids.size > 0 and succ_ids.shape == main_ids.shape
    # t → t+5（0.5 s），同 (spec, episode)
    assert np.array_equal(data["step"][succ_ids], data["step"][main_ids] + 5)
    assert np.array_equal(data["spec"][succ_ids], data["spec"][main_ids])
    assert np.array_equal(data["episode"][succ_ids], data["episode"][main_ids])
    # ok 规则：train_weight ∧ horizon_ok ∧ 速度 present（主/次行）+ 锚长 ≥3
    assert (data["present"][main_ids, 0] > 0.5).all()
    assert (data["present"][succ_ids, 0] > 0.5).all()
    anchor_lens = np.asarray([len(np.asarray(a)) for a in data["rows"]["anchor"]], dtype=np.int64)
    assert (anchor_lens[main_ids] >= 3).all()
    # episode 级 split：两集不相交 ∧ 均非空
    train_keys = set(zip(data["spec"][main_ids][data["train_mask"]].tolist(),
                         data["episode"][main_ids][data["train_mask"]].tolist()))
    val_keys = set(zip(data["spec"][main_ids][data["val_mask"]].tolist(),
                       data["episode"][main_ids][data["val_mask"]].tolist()))
    assert train_keys and val_keys and not (train_keys & val_keys)
    meta = data["meta"]
    assert meta["n_pairs"] == int(main_ids.size)
    assert meta["n_pairs_train"] + meta["n_pairs_val"] == meta["n_pairs"]
    assert meta["n_episodes"] == len(train_keys | val_keys)
    assert meta["n_episodes_val"] == len(val_keys)


def test_build_pairs_seed_deterministic_max_pairs_cap() -> None:
    a = rd.build_pairs(_fixture_args(seed=0, max_pairs=64))
    b = rd.build_pairs(_fixture_args(seed=0, max_pairs=64))
    assert np.array_equal(a["main_ids"], b["main_ids"])
    assert np.array_equal(a["train_mask"], b["train_mask"])
    assert np.array_equal(a["val_mask"], b["val_mask"])
    assert a["main_ids"].size == 64
    assert a["meta"]["max_pairs"] == 64


# --------------------------------------------------------------------------- #
# 3) obs（合成分支；无 base 数据集依赖）
# --------------------------------------------------------------------------- #
def test_build_obs_synthetic_branch_contract_on_fixture() -> None:
    rows, _tw, _hz = rd.load_rows_filtered(_fixture_args())
    data = rd.build_pairs(_fixture_args())
    ids = np.asarray(data["main_ids"][:4], dtype=np.int64)
    device = torch.device("cpu")
    obs = rd.build_obs(rows, ids, device=device, base_dataset=None)

    for key, suffix in rd.OBS_TEMPLATE.items():
        assert key in obs, f"obs 缺模板键 {key}"
        assert tuple(obs[key].shape) == (len(ids), *suffix), f"{key} 形状不符"
        assert obs[key].dtype == torch.float32
    assert tuple(obs["lane_anchor_xy"].shape[:2]) == (len(ids), obs["lane_anchor_xy"].shape[1])
    assert obs["lane_anchor_mask"].shape == obs["lane_anchor_xy"].shape[:2]
    # 锚 mask = 行内 1、批内 padding 0（全 1 不成立：行锚长不一）
    anchor_lens = [len(np.asarray(rows["anchor"][int(i)])) for i in ids]
    assert float(obs["lane_anchor_mask"].sum()) == float(sum(anchor_lens))
    assert set(torch.unique(obs["lane_anchor_mask"]).tolist()) <= {0.0, 1.0}
    assert torch.equal(obs["target_lane_valid"], torch.ones(len(ids)))
    # 合成分支速度写入 = action 首段 ds / DT（口径 = build_obs 迁移版）
    expected = torch.as_tensor(
        [float(np.asarray(rows["action"][int(i)])[0][0]) / rd.DT for i in ids], dtype=torch.float32
    )
    assert torch.equal(obs["ego"][:, 0], expected)
    # 外源列 present 键在（long_exog/long_present 存在 ⇒ 逐列写入）
    from net.path_head import LONG_EXOG_OBS_KEYS

    for key in LONG_EXOG_OBS_KEYS:
        if key is None:
            continue
        assert key in obs and f"{key}__present" in obs
        assert torch.equal(
            obs[f"{key}__present"],
            torch.as_tensor(np.asarray(rows["long_present"], dtype=np.float32)[ids, list(LONG_EXOG_OBS_KEYS).index(key)], device=device),
        )


# --------------------------------------------------------------------------- #
# 4) 特征/目标/ΔA_θ（小模型；CPU）
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def feature_setup():
    rows, _tw, _hz = rd.load_rows_filtered(_fixture_args())
    data = rd.build_pairs(_fixture_args())
    ids = np.asarray(data["main_ids"][:8], dtype=np.int64)
    model = _tiny_model()
    obs = rd.build_obs(rows, ids, device=torch.device("cpu"), base_dataset=None)
    with torch.no_grad():
        tfeat = rf.transition_features(model, obs)
    return {"rows": rows, "data": data, "ids": ids, "tfeat": tfeat}


def test_transition_features_contract(feature_setup) -> None:
    tfeat = feature_setup["tfeat"]
    features = tfeat["features"]
    assert sorted(features) == sorted(rf.FEATURE_KEYS)
    widths = tuple(int(features[key].shape[-1]) for key in rf.FEATURE_KEYS)
    assert widths == rf.FEATURE_WIDTHS
    batch = feature_setup["ids"].size
    assert tuple(tfeat["feature_tensor"].shape) == (batch, 26)
    assert tfeat["base_cols"].shape == (batch, len(rf.TARGET_COLUMNS))
    assert tfeat["base_present"].shape == (batch, len(rf.TARGET_COLUMNS))
    assert tfeat["action"].shape == (batch, 2)
    assert torch.isfinite(tfeat["feature_tensor"]).all()
    # 26 列 ↔ 5 键往返逐位
    back = rf.tensor_to_features(tfeat["feature_tensor"])
    for key in rf.FEATURE_KEYS:
        assert torch.equal(back[key], features[key]), f"{key}: 26 列切分往返非逐位"


def test_residual_target_mask_formula(feature_setup) -> None:
    tfeat = feature_setup["tfeat"]
    base_cols, base_present = tfeat["base_cols"], tfeat["base_present"]
    real_cols = base_cols + 1.0
    # 全 present ⇒ mask=1、target = real − base = 1
    real_present = torch.ones_like(base_present)
    target, mask = rf.residual_targets(
        base_cols=base_cols, base_present=base_present, real_cols=real_cols, real_present=real_present
    )
    expected_mask = (base_present > 0.5).to(torch.float32)
    assert torch.equal(mask, expected_mask)
    assert torch.equal(target, torch.ones_like(target) * expected_mask)
    # real present=0 ⇒ mask=0、target=0（掩码外严格 0）
    real_present = torch.zeros_like(base_present)
    target0, mask0 = rf.residual_targets(
        base_cols=base_cols, base_present=base_present, real_cols=real_cols, real_present=real_present
    )
    assert torch.equal(mask0, torch.zeros_like(mask0))
    assert torch.equal(target0, torch.zeros_like(target0))


def test_head_delta_matches_five_key_call_and_zero_init(feature_setup) -> None:
    raw = feature_setup["tfeat"]["feature_tensor"]
    # 零初始化头 ⇒ ΔA_θ 严格 0（S-B2-0 的 CI 可跑版本）
    zeros = ReadoutResidualHead(hidden=16).eval()
    with torch.no_grad():
        zero_delta = rf.head_delta(zeros, raw)
    assert torch.equal(zero_delta, torch.zeros_like(zero_delta))
    # 非平凡头：head_delta ≡ 5 键显式调用（逐位）
    torch.manual_seed(1234)
    head = ReadoutResidualHead(hidden=16).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.normal_(0.0, 0.1)
        via_helper = rf.head_delta(head, raw)
        explicit = head(**rf.tensor_to_features(raw))
    assert torch.equal(via_helper, explicit)


def test_runner_dry_run_flags_parse_without_resources() -> None:
    flags = runner.parse_args(["--dry-run"])
    assert flags.dry_run is True and flags.check_features is False
    assert runner.parse_args(["--check-features"]).check_features is True
