"""R21-27：B 行标签消费守卫 —— **NPZ 入口硬拒 v1**（唯一有效来源 = round26/baseline_labels.npz）。

背景（R21-26A errata）：``datasets/semantic_v1_*/samples.npz`` 中 B 行的旧 ``expert_action``
（v1）在恢复窗口后查询，状态与 obs 错配 ⇒ **不得用于任何动作监督/探针/mirror 分母**。
本模块把"覆盖"变成"入口强制"：

- :func:`load_v2_label_map`：读 v2 标签文件（``label_state_version=same_state_v2``，否则 fail-fast）；
- :func:`load_samples_v2`：读 ``samples.npz`` 并**替换** B 行 ``expert_action`` 为 v2；
  v2 缺失（B 行未覆盖且不在 ``excluded_b_rows``）→ **fail-fast**；返回的 ``arrays`` 中不存在
  未覆盖的旧 v1 B 行值（排除行置 NaN 并由 ``keep_mask`` 标出）；
- :func:`v1_exposure`：暴露计数（必须 0）。

消费方（训练/验证/选择/probe/mirror/统计）**必须**经此入口取 expert 动作；直接读
``samples.npz["expert_action"]`` 视为违规（历史产物按 errata 标记）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

V2_VERSION = "same_state_v2"
KAPPA_VALID_DS_M = 0.5  # κ 有效阈值（ds ≥ 0.5 m/0.5s；与 R21-28 K1 监督掩码一致）
EXPERT_BACKEND = "metadrive_expert_pure_pursuit"
QUERY_TIMING = "same_state_immediate_post_teleport"


def load_v2_label_map(labels_path: str | Path) -> dict:
    """读 v2 标签 → ``{(episode_id, step): (base_action, perturbed_action)}``；版本不符 fail-fast。"""
    path = Path(labels_path)
    if not path.exists():
        raise SystemExit(f"[semantic_labels] fail-fast：缺 v2 标签文件 {path}")
    with np.load(path) as payload:
        version = str(payload["label_state_version"][0]) if "label_state_version" in payload.files else "<missing>"
        if version != V2_VERSION:
            raise SystemExit(f"[semantic_labels] fail-fast：label_state_version={version}（要求 {V2_VERSION}）")
        keys = list(zip(payload["episode_id"], payload["step"]))
        return {
            "version": version,
            "perturbed": {(int(e), int(s)): payload["perturbed_action"][i] for i, (e, s) in enumerate(keys)},
            "base": {(int(e), int(s)): payload["action"][i] for i, (e, s) in enumerate(keys)},
        }


def _excluded_keys(contract_path: str | Path | None) -> set:
    if contract_path is None:
        return set()
    path = Path(contract_path)
    if not path.exists():
        raise SystemExit(f"[semantic_labels] fail-fast：缺数据契约 {path}（无法判定排除行）")
    contract = json.loads(path.read_text(encoding="utf-8"))
    return {(int(item["episode_id"]), int(item["step"])) for item in contract.get("excluded_b_rows", [])}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def kinematics_metadata(arrays: dict, legacy_actions_path: str | Path | None = None) -> dict:
    """R21-28：expert/legacy 的 ds/dθ/κ/valid 元数据（供训练/审计直接消费，避免重复重算）。

    ``κ = dθ/max(ds, ε)``；``kappa_valid = ds ≥ 0.5``（expert 与 legacy 同时满足）。
    """
    expert = np.asarray(arrays["expert_action"], dtype=np.float64)
    expert_ds, expert_dth = expert[:, 0], expert[:, 1]
    expert_kappa = expert_dth / np.maximum(expert_ds, 1e-6)
    out = {
        "expert_ds": expert_ds.astype(np.float32),
        "expert_dtheta": expert_dth.astype(np.float32),
        "expert_kappa": expert_kappa.astype(np.float32),
    }
    valid = np.abs(expert_ds) >= KAPPA_VALID_DS_M
    if legacy_actions_path is not None:
        path = Path(legacy_actions_path)
        if not path.exists():
            raise SystemExit(f"[semantic_labels] fail-fast：缺 legacy 动作 {path}")
        legacy = np.asarray(np.load(path)["action"], dtype=np.float64)
        if legacy.shape[0] != expert.shape[0]:
            raise SystemExit(f"[semantic_labels] fail-fast：legacy 行数 {legacy.shape[0]} != {expert.shape[0]}")
        legacy_ds, legacy_dth = legacy[:, 0], legacy[:, 1]
        legacy_kappa = legacy_dth / np.maximum(legacy_ds, 1e-6)
        out.update({
            "legacy_ds": legacy_ds.astype(np.float32),
            "legacy_dtheta": legacy_dth.astype(np.float32),
            "legacy_kappa": legacy_kappa.astype(np.float32),
        })
        valid = valid & (np.abs(legacy_ds) >= KAPPA_VALID_DS_M)
    out["kappa_valid"] = valid
    return out


def build_manifest(labels_path: Path, arrays: dict, *, rows: int, covered_b_rows: int,
                   excluded_b_rows: int) -> dict:
    """R21-28：正式训练 manifest（label/expert hash + backend + query timing），防数据版本混淆。"""
    expert = np.asarray(arrays["expert_action"], dtype=np.float32)
    return {
        "label_source": str(labels_path),
        "label_hash": _sha256(labels_path),
        "expert_hash": hashlib.sha256(expert.tobytes()).hexdigest()[:16],
        "backend": EXPERT_BACKEND,
        "query_timing": QUERY_TIMING,
        "label_state_version": V2_VERSION,
        "rows": int(rows),
        "covered_b_rows": int(covered_b_rows),
        "excluded_b_rows": int(excluded_b_rows),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def load_samples_v2(data_dir: str | Path, labels_path: str | Path,
                    contract_path: str | Path | None = None,
                    legacy_actions_path: str | Path | None = None) -> dict:
    """加载数据集 + **强制 v2 覆盖**：返回 ``{"arrays", "keep_mask", "n", "v1_exposed", "label_meta"}``。

    - B 行：``expert_action`` ← v2 置位态标签；``excluded_b_rows`` → ``keep_mask=False``（值为 NaN）；
    - 非 B 行：``expert_action`` 原样（A/C 行标签无状态错配问题；C 行为 learner 态专家标签）；
    - v1 暴露（B 行无 v2 且未排除）→ fail-fast（不返回）。
    """
    data_path = Path(data_dir)
    samples = data_path / "samples.npz" if data_path.is_dir() else data_path
    if not samples.exists():
        raise SystemExit(f"[semantic_labels] fail-fast：缺 {samples}")
    labels = load_v2_label_map(labels_path)
    excluded = _excluded_keys(contract_path)
    with np.load(samples) as payload:
        arrays = {key: payload[key] for key in payload.files}
    row_type = np.asarray(arrays["row_type"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    step = np.asarray(arrays["step"], dtype=np.int64)
    n = int(row_type.shape[0])
    expert = np.array(arrays["expert_action"], dtype=np.float32, copy=True)
    keep = np.ones(n, dtype=bool)
    exposed = 0
    covered = 0
    for row in range(n):
        if row_type[row] != 1:
            continue
        key = (int(episode[row]), int(step[row]))
        if key in excluded:
            keep[row] = False
            expert[row] = np.nan
            continue
        if key not in labels["perturbed"]:
            exposed += 1
            continue
        expert[row] = labels["perturbed"][key]
        covered += 1
    if exposed:
        raise SystemExit(f"[semantic_labels] fail-fast：B 行 v1 暴露={exposed}（要求 0；v2 唯一来源）")
    arrays["expert_action"] = expert
    meta = {
        "version": labels["version"], "covered_b_rows": int(covered),
        "excluded_b_rows": int((~keep & (row_type == 1)).sum()),
        "source": str(labels_path),
    }
    kinematics = kinematics_metadata(arrays, legacy_actions_path)
    meta["kappa_valid_rows"] = int(kinematics["kappa_valid"].sum())
    manifest = build_manifest(Path(labels_path), arrays, rows=n, covered_b_rows=int(covered),
                              excluded_b_rows=int((~keep & (row_type == 1)).sum()))
    return {
        "arrays": arrays, "keep_mask": keep, "n": n, "v1_exposed": int(exposed),
        "label_meta": meta, "kinematics": kinematics, "manifest": manifest,
    }


def v1_exposure(arrays: dict, label_map: dict, excluded: set | None = None) -> int:
    """B 行中无 v2 标签且未排除的数量（应恒 0）。"""
    excluded = excluded or set()
    row_type = np.asarray(arrays["row_type"], dtype=np.int64)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    step = np.asarray(arrays["step"], dtype=np.int64)
    exposed = 0
    for row in range(row_type.shape[0]):
        if row_type[row] != 1:
            continue
        key = (int(episode[row]), int(step[row]))
        if key not in label_map["perturbed"] and key not in excluded:
            exposed += 1
    return exposed
