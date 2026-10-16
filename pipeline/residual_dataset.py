#!/usr/bin/env python3
"""残差训练链（R41 第一批-B 入库）：**数据加载 / 连续 transition 配对 / episode 级 split / obs 组装**。

规格出处
--------
- R41 复审（``docs/deepseekv4p1_argue.md`` §2.3 / §三 第一批②）：``tools/train_residual.py``
  的残差训练链**完全入库**，移除对研究 harness 的硬依赖；
- 本模块 = 研究 harness 的 **逐位迁移版**（语义不变；对拍证据 = ``round41/b_bitwise.json``，
  max_abs_diff=0）：
  - 数据层（``load_rows_filtered``/``build_step_map``）迁移自 R38-A1 lane 的数据入口；
  - path_exec_v1 分片解码（``_concat_shards``/``_rows_from_path_exec``）与 obs 组装
    （``build_obs``）迁移自 R36-J1 lane；
  - 连续 transition 配对 + episode 级 split（``build_pairs``）迁移自 R40-V5 入库 runner
    （``tools/train_residual.py``，R40 §六.10）。

数据契约（path_exec_v1；分片 manifest）
--------------------------------------
``--data <dir>/manifest.json``：``shards[]``（每片含 ``anchor_xy (sumP,2)`` + ``anchor_off (N+1,)``
扁平锚 + 行数组）；整数码字段（``rel_code``/``geometry_code``/``target_src_code``）按 manifest
的名称表解码；``valid30`` → ``valid``、``long_exog_present`` → ``long_present``、
``anchor_shift_m`` → ``shift_m``。

配对/切分口径（与 ``tools/train_residual.py`` 同源）
----------------------------------------------------
- 样本 = 同 episode **连续行**（row t → row t+5 = 0.5 s）；``ok`` = ``train_weight>0 ∧
  horizon_ok ∧ 锚长≥3 ∧ 速度列 present=1``（主行与下一行）；
- train/val 按 **(spec_id, episode_id) 整段切**（episode 级；禁行级泄漏），固定 ``--seed``
  采样；落样本后断言两集不相交。

本模块**零研究工作区依赖**：只 import numpy/torch（惰性）与仓库内 ``pipeline``/``net``/``tools``。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DT = 0.5
EXPERT_STEPS = 30
#: v8 obs schema 模板（键 → 除 batch 维外形状；合成最小 obs 用；与 BCDataset.build_obs_batch 同键集）
OBS_TEMPLATE: dict[str, tuple[int, ...]] = {
    "ego": (8,), "ego_mask": (1,), "ego_hist": (6, 1, 8), "ego_hist_mask": (6, 1),
    "ego_world": (3,), "ego_world_mask": (1,), "ego_world_hist": (6, 1, 3), "ego_world_hist_mask": (6, 1),
    "hist_valid": (6,),
    "ld": (16, 7), "ld_mask": (16,), "ld_hist": (6, 16, 7), "ld_hist_mask": (6, 16),
    "od": (16, 9), "od_mask": (16,), "od_hist": (6, 16, 9), "od_hist_mask": (6, 16), "od_presence": (16,),
    "nav": (11,), "nav_mask": (1,), "nav_hist": (6, 1, 11), "nav_hist_mask": (6, 1),
    "signal": (4,), "signal_mask": (1,), "signal_hist": (6, 1, 4), "signal_hist_mask": (6, 1),
    "others": (33,), "others_mask": (1,), "others_hist": (6, 1, 33), "others_hist_mask": (6, 1),
    "route_world": (64, 2), "route_world_mask": (64,),
    "route_world_hist": (6, 64, 2), "route_world_hist_mask": (6, 64),
}


# --------------------------------------------------------------------------- #
# 0) 基础工具（迁移自 R36-J1；语义不变）
# --------------------------------------------------------------------------- #
def resolve_path(data_path: Path, base: Path = ROOT) -> Path:
    """相对路径 → 仓库根（``base``）下解析；绝对路径透传。"""
    return data_path if data_path.is_absolute() else (base / data_path)


def wrap_pi(value):
    """角度 wrap 到 ``(-π, π]``（标量/ndarray 通用）。"""
    arr = np.asarray(value, dtype=np.float64)
    out = (arr + np.pi) % (2.0 * np.pi) - np.pi
    return float(out) if np.isscalar(value) else out


def _json_load(value):
    """``str/bytes`` → JSON（容错 np.str_/np.bytes_）。"""
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="replace")
    return json.loads(str(value))


def _decode(value):
    """npz 值 → 常规 ndarray（``<U`` → object str；0 维 → 1 维）。"""
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.reshape(1)
    arr = np.asarray(value)
    if arr.dtype.kind in ("U", "S") and arr.ndim == 1:
        return np.array([str(item) for item in arr.ravel()], dtype=object)
    return np.asarray(arr)


def _decode_codes(codes, names) -> np.ndarray:
    """整数码 → 名称（越界给 ``?<code>``）。"""
    arr = np.asarray(codes).astype(np.int64).ravel()
    out = np.empty(arr.shape[0], dtype=object)
    for i, code in enumerate(arr):
        out[i] = names[int(code)] if 0 <= int(code) < len(names) else f"?{int(code)}"
    return out


def anchor_heading(anchor_xy: np.ndarray, anchor_theta: np.ndarray | None) -> np.ndarray:
    """锚折线航向：显式给（``(P,3)`` 第 3 列）则用之，否则中心差分切向（端点单侧）。"""
    xy = np.asarray(anchor_xy, dtype=np.float64)
    if anchor_theta is not None:
        return wrap_pi(np.asarray(anchor_theta, dtype=np.float64))
    if len(xy) < 2:
        return np.zeros(len(xy), dtype=np.float64)
    if len(xy) == 2:
        return np.full(2, math.atan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0]))
    nxt = np.vstack([xy[1:], xy[-1] + (xy[-1] - xy[-2])])
    prev = np.vstack([xy[0] - (xy[1] - xy[0]), xy[:-1]])
    return np.arctan2(nxt[:, 1] - prev[:, 1], nxt[:, 0] - prev[:, 0])


def world_to_ego_xy(xy_world: np.ndarray, ego_world) -> np.ndarray:
    """世界系 ``(P,2)`` → 自车 t0 系。"""
    xy = np.asarray(xy_world, dtype=np.float64).reshape(-1, 2)
    x0, y0, th = float(ego_world[0]), float(ego_world[1]), float(ego_world[2])
    cos_t, sin_t = math.cos(th), math.sin(th)
    dx, dy = xy[:, 0] - x0, xy[:, 1] - y0
    return np.column_stack([dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t])


def ego_to_world_xy(xy_ego: np.ndarray, ego_world) -> np.ndarray:
    """自车 t0 系 ``(P,2)`` → 世界系（``world_to_ego_xy`` 的逆）。"""
    xy = np.asarray(xy_ego, dtype=np.float64).reshape(-1, 2)
    x0, y0, th = float(ego_world[0]), float(ego_world[1]), float(ego_world[2])
    cos_t, sin_t = math.cos(th), math.sin(th)
    return np.column_stack(
        [
            x0 + xy[:, 0] * cos_t - xy[:, 1] * sin_t,
            y0 + xy[:, 0] * sin_t + xy[:, 1] * cos_t,
        ]
    )


# --------------------------------------------------------------------------- #
# 1) 数据层：path_exec_v1 分片 → rows
# --------------------------------------------------------------------------- #
def _concat_shards(data_path: Path, manifest: dict) -> dict:
    """I1 分片 manifest（``shards[]`` + 扁平 ``anchor_xy/anchor_off`` + 整数码字段）
    → 规范字段表（名称字段 + ``anchor_ego_list``）。"""
    shards = list(manifest.get("shards") or [])
    if not shards:
        raise SystemExit("[residual] fail-fast：manifest.shards 为空")
    blocks: dict[str, list] = {}
    xy_parts: list = []
    off_parts: list = []
    n_prev = 0
    const_keys = ("lane_table", "schema_json")
    for sh in shards:
        shard_path = data_path / str(sh["name"])
        if not shard_path.is_file():
            raise SystemExit(f"[residual] fail-fast：分片缺失：{shard_path}")
        with np.load(str(shard_path), allow_pickle=True) as data:
            block = {key: _decode(data[key]) for key in data.files}
        for key, value in block.items():
            if key in ("anchor_xy", "anchor_off"):
                continue
            if key in const_keys:
                blocks.setdefault(key, [value])
                continue
            blocks.setdefault(key, []).append(value)
        off = np.asarray(block["anchor_off"], dtype=np.int64)
        xy = np.asarray(block["anchor_xy"], dtype=np.float64)
        off_body = off[:-1]  # 每片 off 含末端哨兵（rows+1）；拼接时去哨兵，末端另补总长
        off_parts.append(off_body + n_prev)
        xy_parts.append(xy)
        n_prev += int(xy.shape[0])
    merged: dict = {}
    for key, values in blocks.items():
        merged[key] = values[0] if len(values) == 1 else np.concatenate([np.asarray(v) for v in values], axis=0)
    off = np.concatenate([*off_parts, np.array([n_prev], dtype=np.int64)], axis=0)
    xy = np.concatenate(xy_parts, axis=0)
    n_rows = int(off.shape[0]) - 1
    if int(off[-1]) != int(xy.shape[0]) or int(off[0]) != 0:
        raise SystemExit(
            f"[residual] fail-fast：分片锚偏移不一致（off[0]={int(off[0])}, off[-1]={int(off[-1])}, xy={int(xy.shape[0])}）"
        )
    merged["anchor_ego_list"] = [xy[int(off[i]) : int(off[i + 1])] for i in range(n_rows)]
    # 世界帧锚列表（评测 G1–G4 的 poses 为世界帧，须同帧比较）
    if "ego_world" in merged:
        _ego_world_rows = np.asarray(merged["ego_world"], dtype=np.float64).reshape(n_rows, -1)[:, :3]
        merged["anchor_world_list"] = [
            ego_to_world_xy(merged["anchor_ego_list"][i], _ego_world_rows[i]) for i in range(n_rows)
        ]
    if "anchor_shift_m" in merged:
        merged["shift_m"] = merged.pop("anchor_shift_m")
    if "valid30" in merged:
        merged["valid"] = merged.pop("valid30")
    if "rel_code" in merged:
        merged["rel"] = _decode_codes(merged.pop("rel_code"), list(manifest.get("rel_names") or []))
    if "geometry_code" in merged:
        merged["geometry"] = _decode_codes(merged.pop("geometry_code"), list(manifest.get("geometry_names") or []))
    if "target_src_code" in merged:
        merged["target_src"] = _decode_codes(merged.pop("target_src_code"), list(manifest.get("target_src_names") or []))
    if "long_exog_present" in merged:
        merged["long_present"] = merged.pop("long_exog_present")
    return merged


def _infer_n(arrays: dict) -> int:
    for key in ("lat_err", "head_err", "s_anchor", "valid", "spec_id", "step"):
        if key in arrays:
            return int(np.asarray(arrays[key]).shape[0])
    raise SystemExit(f"[residual] fail-fast：无法从字段推断行数（现有：{sorted(arrays)}）")


def _pick_anchor(arrays: dict, n: int):
    """锚折线：``anchor_ego``（(N,P,2|3)）或 ``anchor_world``(+``ego_world``)。"""
    if "anchor_ego_list" in arrays:
        anchors = list(arrays["anchor_ego_list"])
        if len(anchors) != n:
            raise SystemExit(f"[residual] fail-fast：anchor_ego_list 行数 {len(anchors)} != {n}")
        return anchors, [None] * n
    if "anchor_ego" in arrays:
        arr = np.asarray(arrays["anchor_ego"], dtype=np.float64)
        if arr.ndim != 3:
            raise SystemExit(f"[residual] fail-fast：anchor_ego 形状应为 (N,P,2|3)，收到 {arr.shape}")
        theta = [arr[i, :, 2] if arr.shape[2] > 2 else None for i in range(n)]
        return [arr[i, :, :2] for i in range(n)], theta
    if "anchor_world" in arrays and "ego_world" in arrays:
        world = np.asarray(arrays["anchor_world"], dtype=np.float64)
        ego = np.asarray(arrays["ego_world"], dtype=np.float64).reshape(n, -1)[:, :3]
        xy = [world_to_ego_xy(world[i, :, :2], ego[i]) for i in range(n)]
        theta = [world[i, :, 2] - ego[i, 2] for i in range(n)] if world.shape[2] > 2 else [None] * n
        return xy, theta
    raise SystemExit(
        "[residual] fail-fast：缺锚折线（需 anchor_ego (N,P,2|3)，或 anchor_world (N,P,3)+ego_world (N,3)）"
        f"；现有字段：{sorted(arrays)}"
    )


def _pick_str(arrays: dict, names, n: int, default=""):
    for name in names:
        if name in arrays:
            return np.array([str(item) for item in np.asarray(arrays[name]).ravel()], dtype=object)
    return np.array([default] * n, dtype=object)


def _pick_int(arrays: dict, names, n: int, default=0):
    for name in names:
        if name in arrays:
            return np.asarray(arrays[name]).astype(np.int64).reshape(-1)
    return None if default is None else np.full(n, int(default), dtype=np.int64)


def _pick_optional(arrays: dict, names, shape, allow_object=False):
    for name in names:
        if name in arrays:
            arr = np.asarray(arrays[name])
            if shape is not None and tuple(arr.shape) != tuple(shape):
                raise SystemExit(f"[residual] fail-fast：字段 {name} 形状应为 {shape}，收到 {tuple(arr.shape)}")
            return arr
    return None


def _rows_from_path_exec(arrays: dict, meta: dict) -> dict:
    """path_exec_v1 行数组 → rows（必需字段缺失即 fail-fast 并列出）。"""
    n = _infer_n(arrays)
    missing = [key for key in ("lat_err", "head_err", "s_anchor", "valid") if key not in arrays]
    if missing:
        raise SystemExit(
            f"[residual] fail-fast：path_exec_v1 缺必需字段 {missing}（H2 §1.1：专家 30 点在目标锚系的"
            f" lat_err/head_err/s_anchor/valid）；现有字段：{sorted(arrays)}"
        )
    anchor, anchor_theta = _pick_anchor(arrays, n)
    rows = {
        "source": "path_exec_v1", "n": n,
        "lat_err": np.asarray(arrays["lat_err"], dtype=np.float64),
        "head_err": np.asarray(arrays["head_err"], dtype=np.float64),
        "s_anchor": np.asarray(arrays["s_anchor"], dtype=np.float64),
        "valid": np.asarray(arrays["valid"]).astype(bool),
        "anchor": anchor, "anchor_theta": anchor_theta,
        "anchor_world": arrays.get("anchor_world_list"),
        "rel": _pick_str(arrays, ("rel", "relation"), n),
        "geometry": _pick_str(arrays, ("geometry",), n, default=""),
        "target_src": _pick_str(arrays, ("target_src", "target_source"), n, default=""),
        "task": _pick_str(arrays, ("task", "task_set"), n, default=""),
        "spec_id": _pick_int(arrays, ("spec_id",), n, default=-1),
        "step": _pick_int(arrays, ("step",), n, default=-1),
        "episode_id": _pick_int(arrays, ("episode_id",), n, default=None),
        "expert_ego": _pick_optional(arrays, ("expert_ego", "traj30"), (n, EXPERT_STEPS, 2)),
        "progress_expert": _pick_optional(arrays, ("progress_expert",), (n, EXPERT_STEPS)),
        "action": _pick_optional(arrays, ("action",), (n, 6, 2)),
        "long_exog": _pick_optional(arrays, ("long_exog",), (n, 8)),
        "long_present": _pick_optional(arrays, ("long_present",), (n, 8)),
        "lane_width": _pick_optional(arrays, ("lane_width",), (n,)),
        "shift_m": _pick_optional(arrays, ("shift_m",), (n,)),
        "dist_to_gate": _pick_optional(arrays, ("dist_to_gate_m",), (n,)),
        "ego_world": _pick_optional(arrays, ("ego_world",), (n, 3)),
        "toll_json": _pick_optional(arrays, ("toll_json",), (n,), allow_object=True),
        "row_index": _pick_int(arrays, ("row_index", "base_row"), n, default=None),
    }
    meta["fields"] = sorted(arrays)
    meta["missing_optional"] = sorted(
        key for key in ("episode_id", "expert_ego", "progress_expert", "long_exog") if rows.get(key) is None
    )
    return rows


def load_rows_filtered(args) -> tuple[dict, np.ndarray, np.ndarray]:
    """path_exec_v1（可分片子集）+ train_weight/horizon_ok；返回 ``(rows, tw, hz)``。"""
    data_dir = resolve_path(Path(args.data))
    manifest = _json_load((data_dir / "manifest.json").read_text(encoding="utf-8"))
    shards = list(manifest.get("shards") or [])
    if args.shards:
        wanted = set(args.shards)
        shards = [sh for sh in shards if str(sh["name"]) in wanted]
        missing = wanted - {str(sh["name"]) for sh in shards}
        if missing:
            raise SystemExit(f"[residual] fail-fast：分片不存在：{sorted(missing)}")
    arrays = _concat_shards(data_dir, {**manifest, "shards": shards})
    train_weight = np.asarray(arrays.pop("train_weight"), dtype=np.float64)
    horizon_ok = np.asarray(arrays.pop("horizon_ok"))
    rows = _rows_from_path_exec(arrays, {})
    return rows, train_weight, horizon_ok


def build_step_map(rows) -> dict:
    """``(spec_id, episode_id, step)`` → 行号。"""
    spec = np.asarray(rows["spec_id"], dtype=np.int64)
    episode = np.asarray(rows["episode_id"], dtype=np.int64)
    step = np.asarray(rows["step"], dtype=np.int64)
    return {(int(spec[i]), int(episode[i]), int(step[i])): i for i in range(rows["n"])}


# --------------------------------------------------------------------------- #
# 2) 连续 transition 配对 + episode 级 split（无泄漏）
# --------------------------------------------------------------------------- #
def build_pairs(args) -> dict:
    """row t → row t+5 配对 + episode 级 train/val 切分（口径 = R40-V5 入库 runner）。"""
    rows, train_weight, horizon_ok = load_rows_filtered(args)
    spec = np.asarray(rows["spec_id"], dtype=np.int64)
    episode = np.asarray(rows["episode_id"], dtype=np.int64)
    step = np.asarray(rows["step"], dtype=np.int64)
    exog = np.asarray(rows["long_exog"], dtype=np.float64)
    present = np.asarray(rows["long_present"], dtype=np.float64)
    anchor_len = np.asarray([len(np.asarray(a)) for a in rows["anchor"]], dtype=np.int64)
    step_map = build_step_map(rows)

    succ = np.full(rows["n"], -1, dtype=np.int64)
    for i in range(rows["n"]):
        found = step_map.get((int(spec[i]), int(episode[i]), int(step[i]) + 5))
        succ[i] = -1 if found is None else int(found)
    speed_ok = (present[:, 0] > 0.5) & np.isfinite(exog[:, 0]) & (exog[:, 0] > 0.0)
    next_speed_ok = np.zeros(rows["n"], dtype=bool)
    has = succ >= 0
    next_speed_ok[has] = (present[succ[has], 0] > 0.5) & np.isfinite(exog[succ[has], 0])
    ok = (
        (train_weight > 0.5)
        & (horizon_ok > 0)
        & (anchor_len >= 3)
        & speed_ok
        & has
        & next_speed_ok
    )
    main_ids = np.flatnonzero(ok).astype(np.int64)
    if main_ids.size == 0:
        raise SystemExit("[residual] fail-fast：无可用配对样本")
    rng = np.random.default_rng(int(args.seed))
    if args.max_pairs is not None and main_ids.size > int(args.max_pairs):
        pick = np.sort(rng.choice(main_ids.size, size=int(args.max_pairs), replace=False))
        main_ids = main_ids[pick]
    succ_ids = succ[main_ids]

    keys = np.asarray([f"{int(a)}|{int(b)}" for a, b in zip(spec[main_ids], episode[main_ids])])
    unique_keys = np.unique(keys)
    order = rng.permutation(unique_keys.size)
    n_val = max(1, int(round(float(args.val_frac) * unique_keys.size)))
    val_keys = set(unique_keys[order[:n_val]].tolist())
    is_val = np.asarray([key in val_keys for key in keys.tolist()])
    train_mask, val_mask = ~is_val, is_val
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise SystemExit("[residual] fail-fast：train/val 切分为空（调 --val-frac）")
    overlap = set(keys[train_mask].tolist()) & set(keys[val_mask].tolist())
    if overlap:  # pragma: no cover - 结构性排除
        raise SystemExit(f"[residual] fail-fast：episode 泄漏 {sorted(overlap)[:3]}")

    return {
        "rows": rows, "spec": spec, "episode": episode, "step": step,
        "exog": exog, "present": present,
        "main_ids": main_ids, "succ_ids": succ_ids,
        "train_mask": train_mask, "val_mask": val_mask,
        "n_episodes": int(unique_keys.size), "n_val_episodes": int(n_val),
        "meta": {
            "n_rows_total": int(rows["n"]),
            "n_pairs": int(main_ids.size),
            "n_pairs_train": int(train_mask.sum()),
            "n_pairs_val": int(val_mask.sum()),
            "n_episodes": int(unique_keys.size),
            "n_episodes_val": int(n_val),
            "seed": int(args.seed),
            "val_frac": float(args.val_frac),
            "max_pairs": None if args.max_pairs is None else int(args.max_pairs),
            "pair_rule": "row t → step t+5（0.5 s）；ok = train_weight ∧ horizon_ok ∧ 锚长≥3 ∧ 速度 present（主/次行）",
            "split_rule": "按 (spec_id, episode_id) 整段切（episode 级，禁行级泄漏；断言两集不相交）",
        },
    }


# --------------------------------------------------------------------------- #
# 3) base 数据集（obs 联合装载）
# --------------------------------------------------------------------------- #
def load_base_dataset(path: Path):
    """base BC 数据集 + ``(spec_id, episode_id, step)`` 行查找表。"""
    from pipeline.trainer import BCDataset

    dataset = BCDataset.load(str(path))
    arrays = dataset.arrays
    if not all(key in arrays for key in ("spec_id", "episode_id", "step")):
        raise SystemExit(f"[residual] fail-fast：base 数据集缺 spec_id/episode_id/step 列：{path}")
    lookup = {
        (int(spec), int(episode), int(step)): index
        for index, (spec, episode, step) in enumerate(zip(
            np.asarray(arrays["spec_id"]).tolist(),
            np.asarray(arrays["episode_id"]).tolist(),
            np.asarray(arrays["step"]).tolist(),
        ))
    }
    return {"dataset": dataset, "lookup": lookup}


# --------------------------------------------------------------------------- #
# 4) obs 组装
# --------------------------------------------------------------------------- #
def anchor_polyline_tensors(rows: dict, idx: np.ndarray, device):
    """锚折线（批内 padding）→ ``(anchor_xy (B,P,2), anchor_theta (B,P), anchor_mask (B,P))``。"""
    import torch

    anchors = [np.asarray(rows["anchor"][int(i)], dtype=np.float64) for i in idx]
    p_max = max(len(a) for a in anchors)
    xy = np.zeros((len(idx), p_max, 2), dtype=np.float32)
    theta = np.zeros((len(idx), p_max), dtype=np.float32)
    mask = np.zeros((len(idx), p_max), dtype=np.float32)
    theta_list = rows.get("anchor_theta")
    for row, a in enumerate(anchors):
        xy[row, : len(a)] = a
        mask[row, : len(a)] = 1.0
        given = theta_list[int(idx[row])] if theta_list else None
        theta[row, : len(a)] = (
            np.asarray(given, dtype=np.float64)[: len(a)] if given is not None else anchor_heading(a, None)
        )
    return (
        torch.as_tensor(xy, device=device),
        torch.as_tensor(theta, device=device),
        torch.as_tensor(mask, device=device),
    )


def build_obs(rows: dict, idx: np.ndarray, *, device, base_dataset, lane_width_default: float = 3.5):
    """行索引 → 模型 obs（base 联合装载优先；缺则合成最小 obs）。"""
    import torch

    from net.path_head import LONG_EXOG_OBS_KEYS

    batch = len(idx)
    obs: dict = {}
    use_base = base_dataset is not None and (
        rows.get("episode_id") is not None or rows.get("row_index") is not None
    )
    if use_base:
        base = base_dataset["dataset"]
        if rows.get("row_index") is not None:
            row_ids = np.asarray([rows["row_index"][int(i)] for i in idx], dtype=np.int64)
        else:
            lookup = base_dataset["lookup"]
            keys = [
                (int(rows["spec_id"][i]), int(rows["episode_id"][i]), int(rows["step"][i])) for i in idx
            ]
            missing = [key for key in keys if key not in lookup]
            if missing:
                raise SystemExit(
                    f"[residual] fail-fast：{len(missing)} 行无法按 (spec_id,episode_id,step) 对齐 base 数据集"
                    f"（前 3：{missing[:3]}）"
                )
            row_ids = np.asarray([lookup[key] for key in keys], dtype=np.int64)
        obs = {
            key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
            for key, value in base.build_obs_batch(row_ids).items()
        }
    else:
        obs = {
            key: torch.zeros((batch, *suffix), dtype=torch.float32, device=device)
            for key, suffix in OBS_TEMPLATE.items()
        }

    anchor_xy, _, anchor_mask = anchor_polyline_tensors(rows, idx, device)
    obs["lane_anchor_xy"] = anchor_xy
    obs["lane_anchor_mask"] = anchor_mask
    obs["target_lane_valid"] = torch.ones(batch, dtype=torch.float32, device=device)

    speed = np.zeros(batch, dtype=np.float32)
    for row, i in enumerate(idx):
        i = int(i)
        if rows.get("action") is not None:
            speed[row] = float(rows["action"][i][0][0]) / DT
        elif rows.get("expert_ego") is not None:
            speed[row] = float(np.linalg.norm(rows["expert_ego"][i][0])) / 0.1
    if not use_base and "ego" in obs:
        obs["ego"][:, 0] = torch.as_tensor(speed, device=device)
    if not use_base and rows.get("ego_world") is not None:
        obs["ego_world"] = torch.as_tensor(
            np.asarray(rows["ego_world"], dtype=np.float32)[idx], device=device
        )
        obs["ego_world_mask"] = torch.ones((batch, 1), dtype=torch.float32, device=device)

    exog, present, widths = rows.get("long_exog"), rows.get("long_present"), rows.get("lane_width")
    for column, key in enumerate(LONG_EXOG_OBS_KEYS):
        if key is None:
            continue
        if exog is not None:
            obs[key] = torch.as_tensor(np.asarray(exog, dtype=np.float32)[idx, column], device=device)
            if present is not None:
                obs[f"{key}__present"] = torch.as_tensor(
                    np.asarray(present, dtype=np.float32)[idx, column], device=device
                )
        elif key in ("path_left_margin", "path_right_margin"):
            half = 0.5 * np.asarray(
                widths if widths is not None else np.full(rows["n"], lane_width_default), dtype=np.float32
            )[idx]
            obs[key] = torch.as_tensor(half, device=device)
    return obs


__all__ = [
    "DT",
    "EXPERT_STEPS",
    "OBS_TEMPLATE",
    "ROOT",
    "anchor_heading",
    "anchor_polyline_tensors",
    "build_obs",
    "build_pairs",
    "build_step_map",
    "ego_to_world_xy",
    "load_base_dataset",
    "load_rows_filtered",
    "resolve_path",
    "world_to_ego_xy",
    "wrap_pi",
]
