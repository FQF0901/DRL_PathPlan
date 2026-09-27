"""聚类 lane（fix-4）契约测试：特征口径 / 加权 / 容量 / 软目标 / 版本 / Hungarian / 两段式。

覆盖 ``docs/design-v1.2.md`` §4 的冻结契约；全部纯 CPU / 无 metadrive 依赖。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.clusters import (
    FEATURE_CONTRACT_V1,
    FEATURE_CONTRACT_V2,
    ClusterSpec,
    activity_statistics,
    align_clusters,
    clear_cache,
    density_weights,
    encode_obs,
    od_top6_indices,
    enforce_size_bounds,
    fit_capacity_kmeans,
    fit_feature_pipeline,
    load,
    remap_assignment,
    soft_targets_from_obs,
    top2_soft_assignments,
)
from tools.fit_clusters import fit_cluster_spec_from_dataset, run_fit

OD_SLOTS, OD_DIM, LD_SLOTS, LD_DIM = 16, 9, 16, 7


# --------------------------------------------------------------------------- #
# 合成 obs（v1 + v2 键）
# --------------------------------------------------------------------------- #


def make_obs(
    n: int = 128,
    seed: int = 0,
    *,
    with_presence: bool = True,
    with_others: bool = False,
    others_dim: int = 20,
) -> dict:
    rng = np.random.default_rng(seed)
    ego = rng.normal(size=(n, 8)).astype(np.float32)
    od = rng.normal(size=(n, OD_SLOTS, OD_DIM)).astype(np.float32) * 0.1
    od[:, :2, 6] = 4.5  # L/W：真实槽位非零（mask 之外仍可辨识）
    od[:, :2, 7] = 1.8
    od_mask = np.zeros((n, OD_SLOTS), np.float32)
    od_mask[:, :2] = 1.0
    ld = rng.normal(size=(n, LD_SLOTS, LD_DIM)).astype(np.float32) * 0.1
    ld[:, :3, 4] = 13.9  # speed_limit
    ld_mask = np.zeros((n, LD_SLOTS), np.float32)
    ld_mask[:, :3] = 1.0
    nav = rng.normal(size=(n, 11)).astype(np.float32)
    signal = np.tile(np.array([[0, 0, 0, 1]], np.float32), (n, 1))
    obs = {"ego": ego, "od": od, "od_mask": od_mask, "ld": ld, "ld_mask": ld_mask, "nav": nav, "signal": signal}
    if with_presence:
        obs["od_presence"] = od_mask.copy()
    if with_others:
        obs["others"] = rng.normal(size=(n, 1, others_dim)).astype(np.float32)
    return obs


def make_blobs(n_per: int = 40, k: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, 10)) * 4.0
    return np.concatenate([centers[j] + rng.normal(size=(n_per, 10)) * 0.2 for j in range(k)]).astype(np.float32)


def make_fixture_arrays(n: int = 720, seed: int = 0, *, enrich: bool = True) -> tuple[dict, dict]:
    """v2 等价 fixture：6 个可分布簇 + od_presence/od_id/others + 规则标签（仅体检用）。"""
    rng = np.random.default_rng(seed)
    k = 6
    rows = np.repeat(np.arange(k), n // k)
    rng.shuffle(rows)
    n = rows.size
    base = rng.normal(size=(k, 60))[rows] * 2.0 + rng.normal(size=(n, 60)) * 0.4
    od = np.zeros((n, OD_SLOTS, OD_DIM), np.float32)
    od[:, :3, :6] = base[:, 8:26].reshape(n, 3, 6)
    od[:, :3, 6] = 4.5
    od_mask = np.zeros((n, OD_SLOTS), np.float32)
    od_mask[:, :3] = 1.0
    ld = np.zeros((n, LD_SLOTS, LD_DIM), np.float32)
    ld[:, :4, :4] = base[:, 26:42].reshape(n, 4, 4)
    ld[:, :4, 4] = 13.9
    ld_mask = np.zeros((n, LD_SLOTS), np.float32)
    ld_mask[:, :4] = 1.0
    labels = np.zeros((n, 8), np.float32)
    if enrich:
        sel = np.flatnonzero(rows == 3)
        labels[sel[: max(1, int(0.3 * sel.size))], 0] = 1.0
    arrays = {
        "ego": base[:, :8].astype(np.float32),
        "od": od,
        "od_mask": od_mask,
        "od_presence": od_mask.copy(),
        "od_id": np.arange(OD_SLOTS, dtype=np.int64)[None, :].repeat(n, 0),
        "ld": ld,
        "ld_mask": ld_mask,
        "nav": base[:, 42:53].astype(np.float32),
        "signal": np.tile(np.array([[0, 0, 0, 1]], np.float32), (n, 1)),
        "others": np.concatenate([base[:, 53:60], rng.normal(size=(n, 21))], axis=1)
        .astype(np.float32)[:, None, :],  # v2 规范 others：nav(11)+speed_limit(1)+signal(4)+road_class(12)=28
        "labels": labels,
        "difficulty": np.array(["easy", "medium", "hard"])[rows % 3],
        "geometry": np.array(["straight", "curve", "merge", "roundabout"])[rows % 4],
    }
    meta = {
        "label_names": [
            "cutin_active",
            "cutout_active",
            "crowded",
            "car_following",
            "on_curve",
            "merging",
            "roundabout_near",
            "near_intersection",
        ],
        "obs_fingerprint": "fixture",
        "schema_version": 2,
    }
    return arrays, meta


def fit_small_spec(arrays: dict, meta: dict, **kwargs) -> ClusterSpec:
    # 本文件的历史用例都基于 v1 特征契约（v2 契约由 test_feature_contract_v2_* 单独覆盖）
    params = dict(k=6, pca_dim=10, density_k=6, cap_ratio=0.3, seed=0, max_iters=12,
                  feature_contract=FEATURE_CONTRACT_V1, restarts=1)
    params.update(kwargs)
    return fit_cluster_spec_from_dataset(arrays, meta, **params)["spec"]


# --------------------------------------------------------------------------- #
# 1. 特征口径（当前帧 / mask / 固定槽位）
# --------------------------------------------------------------------------- #


def test_encode_layout_and_determinism() -> None:
    obs = make_obs(n=32, seed=1)
    raw = encode_obs(obs, contract=FEATURE_CONTRACT_V1)
    assert raw.shape == (32, 8 + 16 * 9 + 16 * 7 + 16)
    assert raw.dtype == np.float32
    assert np.allclose(raw[:, :8], obs["ego"])
    od = raw[:, 8 : 8 + 16 * 9].reshape(32, 16, 9)
    assert np.allclose(od[:2, 2:], 0.0)  # 无效槽位清零
    assert np.allclose(raw, encode_obs(obs, contract=FEATURE_CONTRACT_V1))  # 确定性
    assert not np.allclose(obs["od"][:, 2], 0.0)  # 输入未被就地清零


def test_encode_v2_od_presence_precedence() -> None:
    obs = make_obs(n=8, seed=2, with_presence=True)
    stale = 7.5
    obs["od"][:, 3, :] = stale  # 出盒槽位保留陈旧特征
    obs["od_mask"][:, 3] = 1.0
    obs["od_presence"][:, 3] = 0.0
    raw = encode_obs(obs, contract=FEATURE_CONTRACT_V1)
    assert np.allclose(raw[:, 8 : 8 + 16 * 9].reshape(8, 16, 9)[:, 3], 0.0)
    # presence=1 / mask=0 → 以 presence 为准（保留特征）
    obs["od_presence"][:, 3] = 1.0
    raw = encode_obs(obs, contract=FEATURE_CONTRACT_V1)
    assert np.allclose(raw[:, 8 : 8 + 16 * 9].reshape(8, 16, 9)[:, 3], stale)


def test_encode_others_source_raw_and_fallback_speed_limit() -> None:
    obs = make_obs(n=6, seed=3, with_others=True, others_dim=20)
    raw = encode_obs(obs, contract=FEATURE_CONTRACT_V1)
    assert raw.shape[1] == 8 + 16 * 9 + 16 * 7 + 20  # raw others（含 road_class）
    assert np.allclose(raw[:, -20:], obs["others"][:, 0])
    fallback = encode_obs(make_obs(n=6, seed=3, with_others=False), contract=FEATURE_CONTRACT_V1)
    assert fallback.shape[1] == 8 + 16 * 9 + 16 * 7 + 16
    # speed_limit 在 ld 槽0 第4维，槽0 无效时置 0
    obs = make_obs(n=2, seed=4)
    obs["ld"][1, 0, 4] = 3.3
    obs["ld_mask"][1, 0] = 0.0
    raw = encode_obs(obs, contract=FEATURE_CONTRACT_V1)
    others = raw[:, -16:]
    assert others[0, 11] == pytest.approx(13.9)
    assert others[1, 11] == pytest.approx(0.0)


def test_fit_feature_pipeline_drops_constant_dims() -> None:
    raw = make_blobs(n_per=30, k=3, seed=5)
    raw = np.concatenate([raw, np.full((raw.shape[0], 1), 2.5, np.float32)], axis=1)
    features = fit_feature_pipeline(raw, pca_dim=6)
    assert raw.shape[1] - 1 not in features["keep_dims"]
    assert features["z"].shape == (raw.shape[0], 6)
    spec = ClusterSpec(
        others_source="fallback",
        raw_dim=raw.shape[1],
        keep_dims=features["keep_dims"],
        feature_mean=features["feature_mean"],
        feature_std=features["feature_std"],
        components=features["components"],
        whiten_scale=features["whiten_scale"],
        centroids=np.zeros((2, 6)),
    )
    assert np.allclose(spec.transform(raw), features["z"], atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. 稀有度加权
# --------------------------------------------------------------------------- #


def test_density_weights_bounds_and_rarity_monotone() -> None:
    rng = np.random.default_rng(6)
    dense = rng.normal(size=(200, 4)) * 0.1
    sparse = np.array([[5.0, 5.0, 5.0, 5.0], [-5.0, -5.0, -5.0, -5.0]], dtype=np.float64)
    x = np.vstack([dense, sparse])
    w = density_weights(x, k=8, alpha=0.75)
    assert w.shape == (x.shape[0],)
    assert np.all(w >= 0.1 - 1e-9) and np.all(w <= 10.0 + 1e-9)
    assert w[-1] > w[0] and w[-2] > w[0]  # 稀疏/离群点权重更高
    flat = density_weights(np.zeros((50, 3)), k=4)
    assert np.allclose(flat, 1.0)


# --------------------------------------------------------------------------- #
# 3. 容量约束加权 k-means + 体检后处理
# --------------------------------------------------------------------------- #


def test_capacity_kmeans_respects_cap_and_is_deterministic() -> None:
    x = make_blobs(n_per=60, k=3, seed=7)
    w = np.ones(x.shape[0])
    first = fit_capacity_kmeans(x, w, k=4, cap_ratio=0.3, seed=0, max_iters=12)
    second = fit_capacity_kmeans(x, w, k=4, cap_ratio=0.3, seed=0, max_iters=12)
    assert np.array_equal(first.assign, second.assign)
    counts = np.bincount(first.assign, minlength=4)
    assert counts.sum() == x.shape[0]
    assert counts.max() <= int(np.ceil(0.3 * x.shape[0])) + 1  # 容量上限
    assert counts.min() > 0


def test_enforce_size_bounds_splits_big_and_merges_tiny() -> None:
    rng = np.random.default_rng(8)
    big = rng.normal(size=(900, 4))
    small = rng.normal(size=(10, 4)) + 20.0
    others = rng.normal(size=(90, 4)) + 10.0
    x = np.vstack([big, small, others])
    w = np.ones(x.shape[0])
    centers = np.array([big.mean(0), small.mean(0)], dtype=np.float64)
    assign = np.zeros(x.shape[0], dtype=np.int64)
    assign[900:] = 1
    centers, assign, log = enforce_size_bounds(
        x, w, centers, assign, target_k=4, cap_ratio=0.3, max_share=0.3, min_share=0.02
    )
    counts = np.bincount(assign, minlength=centers.shape[0])
    shares = counts / x.shape[0]
    assert centers.shape[0] == 4
    assert shares.max() <= 0.30 + 1e-9  # 硬容量 = min(cap_ratio, max_share)
    assert shares.min() >= 0.02 - 1e-9
    assert counts.sum() == x.shape[0]
    assert any(entry["op"].startswith("split") for entry in log)
    assert any(entry["op"].startswith("merge") for entry in log)


# --------------------------------------------------------------------------- #
# 4. 软目标（top-2 softmax + 边界平滑）
# --------------------------------------------------------------------------- #


def test_top2_soft_targets_row_sum_and_boundary_smoothing() -> None:
    dist = np.array([[0.1, 5.0, 6.0], [1.0, 1.02, 9.0], [2.0, 2.0, 8.0]])
    soft = top2_soft_assignments(dist, temperature=0.5, boundary_margin=0.05, smooth_strength=0.5)
    assert soft.shape == (3, 3)
    assert np.allclose(soft.sum(axis=1), 1.0)
    assert soft[0].max() > 0.99  # 远离边界：接近 one-hot
    assert soft[1].max() < soft[0].max()  # 边界样本被平滑
    assert soft[1, 2] > 0.0  # 第三簇获得混合质量
    hard = top2_soft_assignments(dist, boundary_margin=0.0)
    assert np.isclose(hard[2, 0], 0.5, atol=1e-6)  # 等距无平滑 → 对称


def test_spec_assign_soft_rows_and_shapes() -> None:
    arrays, meta = make_fixture_arrays(seed=9)
    spec = fit_small_spec(arrays, meta)
    raw = encode_obs(arrays, contract=FEATURE_CONTRACT_V1)
    soft = spec.assign_soft(raw)
    assert soft.shape == (raw.shape[0], 6)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(soft >= 0.0)
    assert np.allclose(spec.soft_targets_from_obs(arrays), soft, atol=1e-6)


# --------------------------------------------------------------------------- #
# 5. 冻结/版本加载 + Hungarian 对齐
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip_and_validation(tmp_path: Path) -> None:
    arrays, meta = make_fixture_arrays(seed=10)
    spec = fit_small_spec(arrays, meta)
    path = spec.save(tmp_path / "cluster_test.npz")
    loaded = ClusterSpec.load(path)
    raw = encode_obs(arrays, contract=FEATURE_CONTRACT_V1)
    assert np.allclose(loaded.assign_soft(raw), spec.assign_soft(raw), atol=1e-6)
    assert loaded.data_fingerprint.get("fitted_at")
    clear_cache()
    cached = load(path)
    assert cached.raw_dim == spec.raw_dim
    # 维数不匹配 → 明确报错而不是静默错位
    with pytest.raises(ValueError):
        loaded.transform(raw[:, :-1])
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.npz")
    # 配置版本不一致 → 报错
    config = tmp_path / "cfg.yaml"
    config.write_text(f"cluster_version: v99\nspec: {path}\n", encoding="utf-8")
    clear_cache()
    with pytest.raises(ValueError):
        load(config)


def test_align_clusters_hungarian_and_remap() -> None:
    rng = np.random.default_rng(11)
    source = rng.normal(size=(5, 6)) * 50.0  # 分离良好 → 匹配唯一
    planting = rng.permutation(5)
    target = source[planting] + rng.normal(size=(5, 6)) * 0.01
    # align_clusters 返回 old→new 映射：perm[i] = 与 source[i] 最匹配的 target 行号
    expected = np.argsort(planting)
    recovered = align_clusters(source, target)
    assert np.array_equal(recovered, expected)
    old_of_new = np.empty(5, dtype=np.int64)
    old_of_new[recovered] = np.arange(5)  # 新编号 i → 对应旧编号
    assert np.array_equal(remap_assignment(np.arange(5), recovered), old_of_new)
    assert np.array_equal(remap_assignment(np.array([0, 1, 2, 3, 4, 0]), recovered), old_of_new[[0, 1, 2, 3, 4, 0]])


# --------------------------------------------------------------------------- #
# 6. 两段式退路
# --------------------------------------------------------------------------- #


def test_two_stage_gate_and_soft_targets() -> None:
    rng = np.random.default_rng(12)
    n = 400
    calm, active = n // 2, n - n // 2
    ego = np.zeros((n, 8), np.float32)
    ego[calm:, 2] = 2.0  # a_lat
    ego[calm:, 3] = 0.5  # yaw_rate
    od = np.zeros((n, OD_SLOTS, OD_DIM), np.float32)
    od[calm:, :6, 0] = 5.0  # 交互对象
    od[calm:, :6, 2] = -3.0  # 接近速度
    od[calm:, :6, 6] = 4.5
    od_presence = np.zeros((n, OD_SLOTS), np.float32)
    od_presence[calm:, :6] = 1.0
    obs = {
        "ego": ego,
        "od": od,
        "od_presence": od_presence,
        "ld": rng.normal(size=(n, LD_SLOTS, LD_DIM)).astype(np.float32) * 0.1,
        "ld_mask": np.ones((n, LD_SLOTS), np.float32),
        "nav": rng.normal(size=(n, 11)).astype(np.float32),
        "signal": np.tile(np.array([[0, 0, 0, 1]], np.float32), (n, 1)),
    }
    labels = np.zeros((n, 8), np.float32)
    labels[:calm, 0] = 1.0  # 稀有不影响路由（只体检）
    arrays = dict(obs)
    arrays["labels"] = labels
    meta = {"label_names": [f"label_{i}" for i in range(8)], "obs_fingerprint": "fixture", "schema_version": 2}
    result = fit_cluster_spec_from_dataset(
        arrays, meta, k=8, pca_dim=8, density_k=6, cap_ratio=0.25, max_share=0.30, min_share=0.02,
        seed=0, max_iters=12, two_stage=True, feature_contract=FEATURE_CONTRACT_V1, restarts=1,
    )
    spec = result["spec"]
    assert spec.mode == "two_stage" and spec.centroids.shape[0] == 7
    soft = spec.assign_soft(result["arrays"]["raw"])
    assert soft.shape == (n, 8) and np.allclose(soft.sum(axis=1), 1.0)
    assert soft[:calm, 1:].sum(axis=1).mean() < soft[calm:, 1:].sum(axis=1).mean()
    assert activity_statistics(result["arrays"]["raw"], contract=FEATURE_CONTRACT_V1).shape == (n, 3)


# --------------------------------------------------------------------------- #
# 7. 体检报告 / FLAG / 端到端落盘
# --------------------------------------------------------------------------- #


def test_report_flag_and_enrichment() -> None:
    good = fit_cluster_spec_from_dataset(*make_fixture_arrays(seed=13, enrich=True), k=6, pca_dim=10,
                                         feature_contract=FEATURE_CONTRACT_V1, restarts=1,
                                         density_k=6, cap_ratio=0.3, max_share=0.30, min_share=0.02,
                                         seed=0, max_iters=12, enrichment_min=3.0)
    assert good["report"]["flag"] is False
    assert good["report"]["rare"]["max_enrichment"] >= 3.0
    bad = fit_cluster_spec_from_dataset(*make_fixture_arrays(seed=13, enrich=False), k=6, pca_dim=10,
                                        feature_contract=FEATURE_CONTRACT_V1, restarts=1,
                                        density_k=6, cap_ratio=0.3, max_share=0.30, min_share=0.02,
                                        seed=0, max_iters=12, enrichment_min=3.0)
    assert bad["report"]["flag"] is True
    assert "two-stage" in bad["report"]["fallback"]["two_stage_command"]
    for report in (good["report"], bad["report"]):
        assert report["size"]["capacity_ok"]
        assert len(report["clusters"]) == 6
        assert report["soft_target"]["examples"]


def test_run_fit_end_to_end(tmp_path: Path) -> None:
    arrays, meta = make_fixture_arrays(seed=14)
    bc_dir = tmp_path / "fixture_bc"
    bc_dir.mkdir()
    np.savez_compressed(bc_dir / "expert_bc.npz", **arrays)
    (bc_dir / "expert_bc.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    out = tmp_path / "cluster_test.npz"
    targets = tmp_path / "router_targets.npz"
    report_path = run_fit(
        bc_dir, out=out, dump_targets=targets,
        k=6, pca_dim=10, density_k=6, cap_ratio=0.3, max_share=0.30, min_share=0.02,
        seed=0, max_iters=12, feature_contract=FEATURE_CONTRACT_V1, restarts=1,
    )
    assert out.is_file() and report_path.is_file() and targets.is_file()
    dumped = np.load(targets)
    assert dumped["router_soft_targets"].shape == (arrays["ego"].shape[0], 6)
    assert np.allclose(dumped["router_soft_targets"].sum(axis=1), 1.0, atol=1e-6)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["data"]["rows"] == arrays["ego"].shape[0]
    spec = ClusterSpec.load(out)
    soft = spec.soft_targets_from_obs(arrays)
    assert soft.shape == (arrays["ego"].shape[0], 6)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-6)


def test_default_config_loads_frozen_artifact() -> None:
    artifact = Path("config/clusters/cluster_v2.npz")
    if not artifact.is_file():
        pytest.skip("cluster_v2.npz 尚未拟合（先运行 tools/fit_clusters.py）")
    clear_cache()
    spec = load()  # config/clusters/default.yaml → spec（lane D：v2 契约 47 维）
    arrays, _ = make_fixture_arrays(n=60, seed=15)
    batch = {key: value for key, value in arrays.items() if key not in ("labels", "difficulty", "geometry")}
    soft = soft_targets_from_obs(batch)
    assert soft.shape == (60, spec.k)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-5)
    assert spec.cluster_version == "v2"
    assert (spec.feature_contract or {}).get("name") == "clusters.feature.v2"
    assert int(spec.raw_dim) == 47


def test_torch_passthrough_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    arrays, meta = make_fixture_arrays(seed=16)
    spec = fit_small_spec(arrays, meta)
    raw = encode_obs(arrays, contract=FEATURE_CONTRACT_V1)
    numpy_soft = spec.assign_soft(raw)
    obs_keys = ("ego", "od", "od_mask", "od_presence", "od_id", "ld", "ld_mask", "nav", "signal", "others")
    torch_batch = {key: torch.as_tensor(arrays[key]) for key in obs_keys if key in arrays}
    torch_soft = soft_targets_from_obs(torch_batch, spec=spec)
    assert isinstance(torch_soft, torch.Tensor)
    assert torch_soft.shape == numpy_soft.shape
    assert np.allclose(torch_soft.detach().cpu().numpy(), numpy_soft, atol=1e-6)


# --------------------------------------------------------------------------- #
# 8. 特征契约 v2（lane D）：top6 选择 / 维度顺序 / TTC 同源
# --------------------------------------------------------------------------- #


def test_feature_contract_v2_dims_slices_and_determinism() -> None:
    obs = make_obs(n=16, seed=11, with_presence=True, with_others=True, others_dim=28)
    obs["nav"] = obs["nav"].reshape(16, 1, 11)  # 真实 obs 形状
    raw = encode_obs(obs, contract=FEATURE_CONTRACT_V2)
    assert raw.shape == (16, 47) and raw.dtype == np.float32
    assert FEATURE_CONTRACT_V2["dims_total"] == 47
    assert np.allclose(raw[:, :8], obs["ego"])                      # ego 8
    indices, ok = od_top6_indices(obs["od"], obs["od_presence"])
    top6 = raw[:, 8:32].reshape(16, 6, 4)
    for row in range(16):
        for j in range(6):
            if ok[row, j]:
                assert np.allclose(top6[row, j], obs["od"][row, indices[row, j], :4])
            else:
                assert np.allclose(top6[row, j], 0.0)               # 零填充
    assert np.allclose(raw[:, 32:35], obs["nav"][:, 0, 4:7])        # 命令 one-hot(3)
    assert np.allclose(raw[:, 35:47], obs["others"][:, 0, 16:28])   # road_class one-hot(12)
    assert np.allclose(raw, encode_obs(obs, contract=FEATURE_CONTRACT_V2))  # 确定性


def test_od_top6_ttc_distance_tiebreak_and_pad() -> None:
    # 紧迫度公式（min(TTC, 5s)，与 env/obs/od.py 槽位分配同公式、同参数）由下方选择顺序断言覆盖：
    # env/obs/od.py 保持只读（不改观测指纹），本模块内置同公式的向量化实现。

    od = np.zeros((3, OD_SLOTS, OD_DIM), np.float32)
    presence = np.zeros((3, OD_SLOTS), np.float32)
    # 行 0：slot2 最紧迫（TTC=2）→ 先；slot0/slot1 urgency 并列 5 → 槽位升序 → slot0 先
    #       距离-3 从剩余 {slot1, slot3, slot4} 取最近（slot3 近、slot1 中、slot4 远）
    od[0, 0, :4] = [100.0, 5.0, 0.0, 0.0]     # urgency=5, dist≈100
    od[0, 1, :4] = [50.0, 0.0, -10.0, 0.0]    # TTC=5 → urgency=5, dist=50
    od[0, 2, :4] = [20.0, 0.0, -10.0, 0.0]    # TTC=2 → 最紧迫, dist=20
    od[0, 3, :4] = [10.0, 0.0, 0.0, 0.0]      # urgency=5, dist=10（非候选的 TTC 阶段之外）
    od[0, 4, :4] = [80.0, 0.0, 0.0, 0.0]      # dist=80
    presence[0, [0, 1, 2, 3, 4]] = 1.0
    # 行 1：只有 1 个候选 → 其余零填充；行 2：无候选 → 全零
    od[1, 5, :4] = [30.0, 1.0, -3.0, 2.0]
    presence[1, 5] = 1.0

    indices, ok = od_top6_indices(od, presence)
    assert list(indices[0]) == [2, 0, 1, 3, 4, -1]
    assert list(ok[0]) == [True, True, True, True, True, False]
    assert list(indices[1][ok[1]]) == [5] and int(ok[1].sum()) == 1
    assert not ok[2].any()
    # 非候选（presence=0）不参与，即使几何更近/更紧迫
    od2 = od.copy()
    od2[0, 6, :4] = [1.0, 0.0, -50.0, 0.0]  # 极近且 TTC 小，但 presence=0
    indices2, ok2 = od_top6_indices(od2, presence)
    assert list(indices2[0]) == list(indices[0]) and not ok2[0, 5]


def test_feature_contract_v1_still_encodes_legacy_spec() -> None:
    obs = make_obs(n=4, seed=12)
    assert encode_obs(obs, contract=FEATURE_CONTRACT_V1).shape[1] == 8 + 16 * 9 + 16 * 7 + 16
    with pytest.raises(ValueError, match="others"):
        encode_obs(obs, contract=FEATURE_CONTRACT_V2)  # v2 需要 others/od_presence 全键
