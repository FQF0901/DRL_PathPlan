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

import json
from pathlib import Path

import numpy as np

V2_VERSION = "same_state_v2"


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


def load_samples_v2(data_dir: str | Path, labels_path: str | Path,
                    contract_path: str | Path | None = None) -> dict:
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
    return {
        "arrays": arrays, "keep_mask": keep, "n": n, "v1_exposed": int(exposed),
        "label_meta": {
            "version": labels["version"], "covered_b_rows": int(covered),
            "excluded_b_rows": int((~keep & (row_type == 1)).sum()),
            "source": str(labels_path),
        },
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
