#!/usr/bin/env python3
"""R41 第二批执行-H：**v7 同源 obs 重放存储**（训练装配 = 评测装配的 obs 侧单一出处）。

背景（台账 §2b / `g_report.md` §5-④）
-------------------------------------
G 批定位：训练装配（M0 非 semantic + v6 期 base obs〔加载器自带"须重新采集"警告〕+ path_exec
anchor）≠ 评测装配（v7 env obs + `--ld-mode semantic` K1e2 + `--lane-anchor inject`）⇒ 训练变锐
后 50/50 出界。本模块把 obs 侧对齐落地为：

1. **重放**（:func:`replay_episode`）：以 ``tools/collect_expert.py::collect_episode`` 的**同一驱动序**
   （reset → ``engine.add_policy(IDMPolicy, seed)`` → 每 5 env step 构建 obs；``env.prev_policy_action``
   = 实测窗口动作）重放专家 episode；obs 由 **``env/obs/builder.py`` 当前 v7 生产者**构建；
2. **锚注入**（与评测 ``--lane-anchor inject`` 同点同口径）：决策步在 ``builder.build`` 之后调用
   ``.slim/deepwork/p1_v2/round36/c_lane_anchor.py::LaneAnchorInjector.apply``（``select_target_lane``
   + E3 车道链采样）——注入器由调用方传入（模块本身**不 import .slim**，保持核心 CI 自包含）；
3. **存储**（:func:`write_shard` / :func:`merge_shards` / :class:`V7ObsStore`）：逐键 ``.npy``
   （mmap 随机访问）+ 索引数组 + ``meta.json``；锚为 ragged（flat xy + off）；
4. **对拍**（:func:`verify_store_vs_base`）：重放 vs ``BTC...expert5k_v8`` 记录的 v6 子集键
   **逐位**（max_abs_diff=0）证据（环境重放确定性门）。

纪律
----
- 默认关：仅显式调用触达 env；导入期零 env/零 IO；
- **禁静默缺省**：``obs_batch`` 对缺行/缺键/无锚 fail-fast；
- 数值保真：浮点键 float32、整型伴随 int64，逐帧原样落盘（不裁剪、不缩放）。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

#: 决策步 stride（env step；0.5 s @10 Hz，= collect_expert / eval 同口径）。
STEPS_PER_POLICY = 5
#: 存储 schema 版本（键集/语义变更时升版）。
STORE_SCHEMA = "r41h_v7obs_v1"
#: 逐帧整型键（其余键 float32；与 env 生产者 dtype 一致）。
INT64_KEYS = (
    "od_id",
    "od_id_hist",
    "ld_lane_role",
    "ld_lane_role_hist",
    "ld_group",
    "ld_group_hist",
)
#: 单槽通道（加载时去槽位维，= ``pipeline.trainer.SINGLE_SLOT_CHANNELS``）。
SINGLE_SLOT_CHANNELS = ("ego", "nav", "signal", "others", "ego_world")
#: v6 期 base 数据集实际落盘的 obs 键（逐位对拍门用）。
V6_COMPARE_KEYS = (
    "ego", "ego_mask", "od", "od_mask", "ld", "ld_mask", "nav", "nav_mask",
    "signal", "signal_mask", "others", "others_mask", "ego_world", "ego_world_mask",
    "route_world", "route_world_mask", "hist_valid", "od_id", "od_presence",
    "od_id_hist", "od_presence_hist",
)


def _wrap_pi(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _ego_pose(ego) -> np.ndarray:
    return np.array(
        [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)],
        dtype=np.float64,
    )


def window_actions(poses: Sequence[np.ndarray], t: int, n: int = 1) -> np.ndarray:
    """实测位姿窗口 → ``n`` 个 ``(ds, dθ)``（逐字 = ``tools/collect_expert.py::_window_actions``）。"""
    actions = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        a, b = t + i * STEPS_PER_POLICY, t + (i + 1) * STEPS_PER_POLICY
        length = 0.0
        for j in range(a, b):
            length += float(np.linalg.norm(poses[j + 1][:2] - poses[j][:2]))
        actions[i] = (length, _wrap_pi(float(poses[b][2]) - float(poses[a][2])))
    return actions


# --------------------------------------------------------------------------- #
# 1) 重放（驱动序 = collect_expert.collect_episode；obs 生产 = env v7 builder）
# --------------------------------------------------------------------------- #
def replay_episode(
    env,
    spec,
    *,
    builder,
    injector=None,
    expert_kind: str = "idm",
    max_steps: int = 600,
) -> dict:
    """重放单个专家 episode → 逐决策步 v7 obs 行（+ 注入锚 + 诊断）。

    ``env`` 必须为 ``env.metadrive_env.build_env(spec, traffic_density=None, use_render=False)``
    的产物（调用方构造与关闭）；本函数内 ``env.reset()`` + ``engine.add_policy``（与
    ``collect_expert`` 相同顺序；``policy.reset()`` 亦同——**缺一帧即可能轨迹漂移，实测**）。

    Returns:
        dict：``steps (n,)``、``obs {key: (n, *shape)}``（builder 键 + ``lane_anchor_*`` 已由
        注入器写入 + ``lane_anchor_valid`` 逐帧）、``anchor_meta``（n 项 list/dict）、
        ``termination``、``env_steps``。
    """
    env.reset()
    ego = env.agent
    if str(expert_kind).lower() == "idm":
        from metadrive.policy.idm_policy import IDMPolicy as _Expert
    else:
        from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy as _Expert
    policy = env.engine.add_policy(ego.id, _Expert, ego, int(getattr(spec, "seed", 0)))
    if hasattr(policy, "reset"):
        policy.reset()

    poses: list[np.ndarray] = [_ego_pose(ego)]
    rows: dict[int, dict] = {}
    anchor_metas: dict[int, dict] = {}
    termination = "max_step"
    step = 0
    env.prev_policy_action = np.zeros(2, dtype=np.float64)
    while step < int(max_steps):
        if step % STEPS_PER_POLICY == 0:
            if step >= STEPS_PER_POLICY:
                env.prev_policy_action = window_actions(poses, step - STEPS_PER_POLICY, 1)[0]
            else:
                env.prev_policy_action = np.zeros(2, dtype=np.float64)
            obs = builder.build(env, spec)
            valid = np.zeros(1, dtype=np.float32)
            if injector is not None:
                try:
                    anchor_metas[step] = dict(injector.apply(env, obs))  # 写 obs['lane_anchor_xy'/'_mask']
                    valid[0] = 1.0
                except RuntimeError as exc:  # 逐帧记账（禁静默）；训练消费前显式断言
                    anchor_metas[step] = {"error": repr(exc)[:300]}
            else:
                anchor_metas[step] = {"injector": None}
            row = {key: np.array(obs[key], copy=True) for key in obs}
            row["lane_anchor_valid"] = valid
            rows[step] = row
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        step += 1
        poses.append(_ego_pose(ego))
        if terminated or truncated:
            flag = info or {}
            if bool(flag.get("crash")):
                termination = "collision"
            elif bool(flag.get("out_of_road")):
                termination = "out_of_road"
            elif bool(flag.get("arrive_dest")):
                termination = "arrive_dest"
            else:
                termination = "max_step" if bool(flag.get("max_step")) else "terminal"
            break

    steps = np.asarray(sorted(rows), dtype=np.int64)
    keys = sorted({key for row in rows.values() for key in row})
    obs_arrays: dict[str, np.ndarray] = {}
    skip = {"lane_anchor_xy", "lane_anchor_mask"}
    for key in keys:
        if key in skip:
            continue
        values = [rows[int(s)][key] for s in steps]
        dtype = np.int64 if key in INT64_KEYS else np.float32
        obs_arrays[key] = np.asarray(values, dtype=dtype)
    # 锚 ragged：flat xy + 逐帧 off（无有效点 = 空段）
    anchor_off = np.zeros(steps.size + 1, dtype=np.int64)
    anchor_parts: list[np.ndarray] = []
    cursor = 0
    for i, step_key in enumerate(steps):
        anchor_off[i] = cursor
        xy = rows[int(step_key)].get("lane_anchor_xy")
        valid = float(np.asarray(rows[int(step_key)]["lane_anchor_valid"]).reshape(-1)[0])
        if xy is not None and valid > 0.5:
            xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2)
            anchor_parts.append(xy)
            cursor += int(xy.shape[0])
    anchor_off[steps.size] = cursor
    anchor_flat = (
        np.concatenate(anchor_parts, axis=0) if anchor_parts else np.zeros((0, 2), dtype=np.float32)
    )
    return {
        "steps": steps,
        "obs": obs_arrays,
        "anchor_flat": anchor_flat.astype(np.float32, copy=False),
        "anchor_off": anchor_off,
        "anchor_meta": [anchor_metas.get(int(s), {}) for s in steps],
        "termination": termination,
        "env_steps": int(step),
    }


# --------------------------------------------------------------------------- #
# 2) 分片写盘 / 合并（store = 逐键 .npy + 索引 + meta.json）
# --------------------------------------------------------------------------- #
def write_shard(path: str | Path, episodes: Sequence[Mapping[str, Any]]) -> dict:
    """若干 episode 重放结果 → 单分片 npz（帧级拼接 + ``__`` 前缀元数据）。

    ``episodes`` 每项 = :func:`replay_episode` 返回值 + ``spec_id / episode_id / seed /
    geometry_code``（驱动脚本注入）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for record in episodes for key in record["obs"]})
    payload: dict[str, np.ndarray] = {}
    for key in keys:
        parts: list[np.ndarray] = []
        for record in episodes:
            values = record["obs"].get(key)
            if values is None:
                raise SystemExit(f"[v7obs] fail-fast：episode 间键集不一致（缺 {key}）")
            parts.append(np.asarray(values))
        payload[key] = np.concatenate(parts, axis=0)
    payload["__step"] = np.concatenate([np.asarray(r["steps"], dtype=np.int64) for r in episodes])
    payload["__spec_id"] = np.repeat(
        np.asarray([int(r["spec_id"]) for r in episodes], dtype=np.int64),
        [int(np.asarray(r["steps"]).size) for r in episodes],
    )
    payload["__episode_id"] = np.repeat(
        np.asarray([int(r["episode_id"]) for r in episodes], dtype=np.int64),
        [int(np.asarray(r["steps"]).size) for r in episodes],
    )
    payload["__seed"] = np.repeat(
        np.asarray([int(r.get("seed", -1)) for r in episodes], dtype=np.int64),
        [int(np.asarray(r["steps"]).size) for r in episodes],
    )
    payload["__episode_frames"] = np.asarray(
        [int(np.asarray(r["steps"]).size) for r in episodes], dtype=np.int64
    )
    payload["__episode_ids"] = np.asarray([int(r["episode_id"]) for r in episodes], dtype=np.int64)
    payload["__termination"] = np.asarray([str(r.get("termination", "")) for r in episodes], dtype=object)
    payload["__geometry_name"] = np.repeat(
        np.asarray([str(r.get("geometry_name", "unknown")) for r in episodes], dtype=object),
        [int(np.asarray(r["steps"]).size) for r in episodes],
    )
    payload["__anchor_meta"] = np.asarray(
        [json.dumps(r.get("anchor_meta", []), ensure_ascii=False) for r in episodes], dtype=object
    )
    # 锚 ragged：分片内 flat + 逐帧 off（跨 episode 平移）
    flat_parts: list[np.ndarray] = []
    off_parts: list[np.ndarray] = []
    cursor = 0
    for record in episodes:
        off = np.asarray(record["anchor_off"], dtype=np.int64)
        off_parts.append(off[:-1] + cursor)
        flat_parts.append(np.asarray(record["anchor_flat"], dtype=np.float32))
        cursor += int(off[-1])
    off_parts.append(np.asarray([cursor], dtype=np.int64))
    payload["lane_anchor_xy_flat"] = (
        np.concatenate(flat_parts, axis=0) if flat_parts else np.zeros((0, 2), dtype=np.float32)
    )
    payload["lane_anchor_off"] = np.concatenate(off_parts, axis=0)
    np.savez(str(path), **payload)
    frames = int(payload["__step"].size)
    return {"shard": str(path), "episodes": len(episodes), "frames": frames}


def merge_shards(
    shards: Sequence[str | Path],
    out_dir: str | Path,
    *,
    meta: Mapping[str, Any] | None = None,
    replace_episodes: Sequence[int] | None = None,
    patch_shards: Sequence[str | Path] | None = None,
) -> dict:
    """分片 → store 目录：逐帧按 ``(episode_id, step)`` 稳定排序，逐键 `.npy` + 索引 + meta。

    ``replace_episodes`` + ``patch_shards``（可选）：把 ``replace_episodes`` 的行从**非 patch**
    分片中丢弃，patch 分片的行全部保留（per-episode hash-seed 补丁；合并后重复行仍 fail-fast）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(Path(p) for p in shards)
    if not shard_paths:
        raise SystemExit("[v7obs] fail-fast：无分片可合并")
    replace = {int(v) for v in (replace_episodes or [])}
    patch_set = {Path(p).resolve() for p in (patch_shards or [])}
    handles = [np.load(str(path), allow_pickle=True) for path in shard_paths]
    try:
        keys = sorted(key for key in handles[0].files if not key.startswith("__"))
        for handle in handles:
            missing = [key for key in keys if key not in handle.files]
            if missing:
                raise SystemExit(f"[v7obs] fail-fast：分片键集不一致（缺 {missing[:5]}）")

        # 行级元数据 + 锚 ragged（跨分片 cursor 平移）
        episode_parts, step_parts, spec_parts, seed_parts = [], [], [], []
        lens_parts, flat_parts, off_parts = [], [], []
        from_patch_parts = []
        cursor = 0
        for path, handle in zip(shard_paths, handles):
            n_i = int(np.asarray(handle["__step"]).size)
            episode_parts.append(np.asarray(handle["__episode_id"], dtype=np.int64))
            step_parts.append(np.asarray(handle["__step"], dtype=np.int64))
            spec_parts.append(np.asarray(handle["__spec_id"], dtype=np.int64))
            seed_parts.append(np.asarray(handle["__seed"], dtype=np.int64))
            from_patch_parts.append(np.full(n_i, path.resolve() in patch_set, dtype=bool))
            if "lane_anchor_off" in handle.files:
                off_i = np.asarray(handle["lane_anchor_off"], dtype=np.int64)
                lens_parts.append(off_i[1:] - off_i[:-1])
                flat_parts.append(np.asarray(handle["lane_anchor_xy_flat"], dtype=np.float32))
                off_parts.append(off_i[:-1] + cursor)
                cursor += int(flat_parts[-1].shape[0])
        off_all = (
            np.concatenate([*off_parts, np.asarray([cursor], dtype=np.int64)], axis=0)
            if off_parts else None
        )
        flat_all = np.concatenate(flat_parts, axis=0) if flat_parts else np.zeros((0, 2), dtype=np.float32)
        lens_all = np.concatenate(lens_parts, axis=0) if lens_parts else np.zeros(0, dtype=np.int64)

        episode = np.concatenate(episode_parts, axis=0)
        step = np.concatenate(step_parts, axis=0)
        spec_id = np.concatenate(spec_parts, axis=0)
        seed = np.concatenate(seed_parts, axis=0)
        from_patch = np.concatenate(from_patch_parts, axis=0)
        keep = np.ones(episode.size, dtype=bool)
        if replace:
            keep = (~np.isin(episode, np.asarray(sorted(replace), dtype=np.int64))) | from_patch

        def _cat(key: str) -> np.ndarray:
            return np.concatenate([np.asarray(h[key]) for h in handles], axis=0)

        order = np.lexsort((step[keep], episode[keep]))
        episode = episode[keep]
        step = step[keep]
        spec_id = spec_id[keep]
        seed = seed[keep]
        n = int(order.size)
        if np.unique(np.stack([episode, step], axis=1), axis=0).shape[0] != n:
            raise SystemExit("[v7obs] fail-fast：合并后出现重复 (episode_id, step) 行")

        per_key: dict[str, dict] = {}
        for key in keys:
            if key in ("lane_anchor_xy", "lane_anchor_mask", "lane_anchor_xy_flat", "lane_anchor_off",
                       "__geometry_name"):
                continue  # ragged/object：单独处理
            array = np.asarray(_cat(key))[keep][order]
            dtype = np.int64 if key in INT64_KEYS else np.float32
            array = array.astype(dtype, copy=False)
            np.save(str(out_dir / f"{key}.npy"), array)
            per_key[key] = {"shape": list(array.shape), "dtype": str(array.dtype)}
            del array

        # 锚 ragged：flat xy + off（N+1）；无锚帧 = 空段（长度 0）
        if off_all is not None:
            lens_kept = lens_all[keep]
            off_kept = off_all[:-1][keep]
            parts = []
            offs = np.zeros(n + 1, dtype=np.int64)
            pos = 0
            for i, row in enumerate(order):
                offs[i] = pos
                length = int(lens_kept[row])
                if length:
                    start = int(off_kept[row])
                    parts.append(flat_all[start: start + length])
                    pos += length
            offs[n] = pos
            flat_ordered = np.concatenate(parts, axis=0) if parts else np.zeros((0, 2), dtype=np.float32)
            np.save(str(out_dir / "lane_anchor_xy_flat.npy"), np.asarray(flat_ordered, dtype=np.float32))
            np.save(str(out_dir / "lane_anchor_off.npy"), offs)
            per_key["lane_anchor_xy"] = {
                "shape": ["ragged", 2], "dtype": "float32", "n_points_total": int(flat_ordered.shape[0])
            }
            per_key.pop("lane_anchor_mask", None)

        np.save(str(out_dir / "spec_id.npy"), spec_id[order])
        np.save(str(out_dir / "episode_id.npy"), episode[order])
        np.save(str(out_dir / "step.npy"), step[order])
        np.save(str(out_dir / "seed.npy"), seed[order])
        geometry_names: list[str] = []
        if "__geometry_name" in handles[0].files:
            raw_names = np.asarray(_cat("__geometry_name"), dtype=object)[keep][order]
            geometry_names = sorted({str(v) for v in raw_names.tolist()})
            geometry_map = {name: i for i, name in enumerate(geometry_names)}
            geometry = np.asarray([geometry_map[str(v)] for v in raw_names.tolist()], dtype=np.int8)
            np.save(str(out_dir / "geometry_code.npy"), geometry)
        payload = {
            "schema": STORE_SCHEMA,
            "n_rows": n,
            "n_episodes": int(np.unique(episode).size),
            "geometry_names": geometry_names,
            "keys": per_key,
            "source": dict(meta or {}),
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps({k: payload["keys"][k] for k in sorted(payload["keys"])}, sort_keys=True).encode("utf-8")
        ).hexdigest()
        (out_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return payload
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 3) 存储读取（mmap；训练/对拍共用）
# --------------------------------------------------------------------------- #
class V7ObsStore:
    """只读 store 视图：``.npy`` mmap + ``(episode_id, step)`` 行查找 + 批组装。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        meta_path = self.path / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(f"[v7obs] fail-fast：缺 store meta：{meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(self.meta.get("schema")) != STORE_SCHEMA:
            raise SystemExit(
                f"[v7obs] fail-fast：store schema={self.meta.get('schema')!r} != {STORE_SCHEMA!r}"
            )
        self.spec_id = np.load(self.path / "spec_id.npy", mmap_mode="r")
        self.episode_id = np.load(self.path / "episode_id.npy", mmap_mode="r")
        self.step = np.load(self.path / "step.npy", mmap_mode="r")
        self.n = int(self.spec_id.shape[0])
        self._arrays: dict[str, np.ndarray] = {}
        for key in self.meta["keys"]:
            if key == "lane_anchor_xy":
                continue
            self._arrays[key] = np.load(self.path / f"{key}.npy", mmap_mode="r")
        self.lane_anchor_xy_flat = np.load(self.path / "lane_anchor_xy_flat.npy", mmap_mode="r")
        self.lane_anchor_off = np.load(self.path / "lane_anchor_off.npy", mmap_mode="r")
        base_consistent_path = self.path / "base_consistent.npy"
        self.base_consistent = (
            np.load(base_consistent_path, mmap_mode="r") if base_consistent_path.is_file() else None
        )
        self.lookup: dict[tuple[int, int], int] = {
            (int(e), int(s)): i for i, (e, s) in enumerate(zip(self.episode_id, self.step))
        }

    def keys(self) -> list[str]:
        return sorted(self._arrays)

    def row_of(self, episode_id: int, step: int) -> int | None:
        return self.lookup.get((int(episode_id), int(step)))

    def rows_of(self, keys: Sequence[tuple[int, int]]) -> np.ndarray:
        """``(episode_id, step)`` 序列 → 行号；缺行 **fail-fast**（禁静默）。"""
        rows, present = self.map_rows(keys)
        if not bool(present.all()):
            missing = [key for key, ok in zip(keys, present.tolist()) if not ok]
            raise SystemExit(
                f"[v7obs] fail-fast：{len(missing)} 帧不在 store 中（前 3：{missing[:3]}）——"
                "训练装配禁静默跳过（先重放补全或显式裁剪 pair 集）"
            )
        return rows

    def map_rows(self, keys: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """``(episode_id, step)`` 序列 → ``(rows, present)``（缺行 ``rows=-1``，不抛）。"""
        out = np.full(len(keys), -1, dtype=np.int64)
        present = np.zeros(len(keys), dtype=bool)
        for i, (ep, st) in enumerate(keys):
            row = self.lookup.get((int(ep), int(st)))
            if row is not None:
                out[i] = int(row)
                present[i] = True
        return out, present

    def obs_batch(self, rows: np.ndarray, *, require_anchor: bool = True) -> dict[str, np.ndarray]:
        """行号批 → obs 字典（单槽通道去槽位维；锚按批内最大点数 pad + mask）。

        锚：``lane_anchor_xy (B,Pmax,2)`` / ``lane_anchor_mask (B,Pmax)``；``require_anchor=True``
        时任一帧 ``lane_anchor_valid=0`` ⇒ fail-fast（不静默给空锚）。
        """
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.n):
            raise SystemExit(f"[v7obs] fail-fast：行号越界（{rows.min()}..{rows.max()} / n={self.n}）")
        if require_anchor:
            valid = np.asarray(self._arrays["lane_anchor_valid"][rows], dtype=np.float32).reshape(-1)
            if not bool((valid > 0.5).all()):
                bad = rows[valid <= 0.5][:5]
                raise SystemExit(
                    f"[v7obs] fail-fast：{int((valid <= 0.5).sum())} 帧无有效注入锚（行 {bad.tolist()}）——"
                    "评测装配下这些帧不可训练（禁静默空锚）"
                )
        batch: dict[str, np.ndarray] = {}
        for key, array in self._arrays.items():
            if key == "lane_anchor_valid":
                continue
            values = np.asarray(array[rows])
            if key in SINGLE_SLOT_CHANNELS and values.ndim == 3 and values.shape[1] == 1:
                values = values[:, 0]
            batch[key] = np.array(values, copy=True)
        # ragged 锚 → pad
        offs = np.asarray(self.lane_anchor_off[rows])
        lengths = np.asarray(self.lane_anchor_off[rows + 1]) - offs
        p_max = int(lengths.max()) if lengths.size else 0
        xy = np.zeros((rows.size, p_max, 2), dtype=np.float32)
        mask = np.zeros((rows.size, p_max), dtype=np.float32)
        flat = self.lane_anchor_xy_flat
        for i, (start, length) in enumerate(zip(offs, lengths)):
            length = int(length)
            if length:
                xy[i, :length] = np.asarray(flat[int(start): int(start) + length])
                mask[i, :length] = 1.0
        batch["lane_anchor_xy"] = xy
        batch["lane_anchor_mask"] = mask
        return batch

    def digest(self) -> str:
        """store 指纹（meta 的 manifest_sha256；用于账本可比性）。"""
        return str(self.meta.get("manifest_sha256", ""))


def load_store(path: str | Path) -> V7ObsStore:
    """打开 store 目录（:class:`V7ObsStore`；缺 meta/schema 不符 fail-fast）。"""
    return V7ObsStore(path)


def compute_base_consistent(
    store: V7ObsStore,
    base_dataset_dir: str | Path,
    *,
    keys: Sequence[str] = V6_COMPARE_KEYS,
    chunk: int = 20000,
) -> np.ndarray:
    """逐行一致性（uint8）：该行在全部 ``keys`` 上与 base 记录帧**逐位相同** ∧ 存在 base 行。

    用于对齐训练排除门（重放不可复现行，禁静默；``h_patch_merge.py`` 落
    ``base_consistent.npy``）。
    """
    base_dir = Path(base_dataset_dir)
    with np.load(str(base_dir / "expert_bc.npz")) as data:
        episode = np.asarray(data["episode_id"], dtype=np.int64)
        step = np.asarray(data["step"], dtype=np.int64)
        base_lookup = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(episode, step))}
        base_arrays = {key: data[key] for key in keys if key in data.files}
    consistent = np.ones(store.n, dtype=np.uint8)
    base_rows = np.asarray(
        [
            base_lookup.get((int(store.episode_id[i]), int(store.step[i])), -1)
            for i in range(store.n)
        ],
        dtype=np.int64,
    )
    consistent[base_rows < 0] = 0
    valid = np.flatnonzero(base_rows >= 0)
    for key, ref in base_arrays.items():
        for start in range(0, valid.size, int(chunk)):
            sel = valid[start: start + int(chunk)]
            got = np.asarray(store._arrays[key][sel], dtype=np.float64)
            exp = np.asarray(ref[base_rows[sel]], dtype=np.float64)
            diff = np.abs(got - exp).reshape(sel.size, -1).max(axis=1)
            consistent[sel[diff > 0]] = 0
    return consistent


# --------------------------------------------------------------------------- #
# 4) 对拍（重放 vs base 记录帧）
# --------------------------------------------------------------------------- #
def verify_store_vs_base(
    store: V7ObsStore,
    base_dataset_dir: str | Path,
    *,
    keys: Sequence[str] = V6_COMPARE_KEYS,
    max_rows: int | None = None,
    chunk: int = 20000,
) -> dict:
    """v6 子集键逐位对拍（store 行 ↔ base 行，按 ``(episode_id, step)`` 对齐；分段向量化）。

    Returns:
        dict：``per_key {key: {max_abs_diff, n_compared, n_mismatch}}`` + ``n_missing`` + ``pass``
        （= 全部键 max_abs_diff == 0 且无缺行）。
    """
    base_dir = Path(base_dataset_dir)
    base_npz = base_dir / "expert_bc.npz" if base_dir.is_dir() else base_dir
    with np.load(str(base_npz)) as data:
        episode = np.asarray(data["episode_id"], dtype=np.int64)
        step = np.asarray(data["step"], dtype=np.int64)
        base_lookup = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(episode, step))}
        base_arrays = {key: data[key] for key in keys if key in data.files}
    rows_pick = np.arange(store.n, dtype=np.int64)
    if max_rows is not None and rows_pick.size > int(max_rows):
        rng = np.random.default_rng(0)
        rows_pick = np.sort(rng.choice(rows_pick, size=int(max_rows), replace=False))
    base_rows = np.asarray(
        [
            base_lookup.get((int(store.episode_id[row]), int(store.step[row])), -1)
            for row in rows_pick
        ],
        dtype=np.int64,
    )
    n_missing = int((base_rows < 0).sum())
    valid_rows = rows_pick[base_rows >= 0]
    valid_base = base_rows[base_rows >= 0]

    report: dict[str, Any] = {
        "n_rows_checked": int(rows_pick.size),
        "n_missing_rows": n_missing,
        "per_key": {},
    }
    # base 覆盖率（base 行 → store 是否齐全；训练 main_ids 的行映射依赖此门）
    store_keys = set(
        (int(e), int(s)) for e, s in zip(np.asarray(store.episode_id), np.asarray(store.step))
    )
    base_keys = list(zip(episode.tolist(), step.tolist()))
    missing_in_store = sum(1 for key in base_keys if key not in store_keys)
    report["base_coverage"] = {
        "n_base_rows": int(len(base_keys)),
        "n_missing_in_store": int(missing_in_store),
    }
    for key in keys:
        ref = base_arrays.get(key)
        if ref is None:
            report["per_key"][key] = {"status": "absent_in_base"}
            continue
        worst = 0.0
        mismatches = 0
        compared = 0
        for start in range(0, valid_rows.size, int(chunk)):
            sel = valid_rows[start: start + int(chunk)]
            base_sel = valid_base[start: start + int(chunk)]
            got = np.asarray(store._arrays[key][sel], dtype=np.float64)
            expected = np.asarray(ref[base_sel], dtype=np.float64)
            diff = np.abs(got - expected).reshape(sel.size, -1)
            per_row = diff.max(axis=1) if diff.size else np.zeros(sel.size)
            worst = max(worst, float(per_row.max()) if per_row.size else 0.0)
            mismatches += int((per_row > 0).sum())
            compared += int(sel.size)
        report["per_key"][key] = {
            "max_abs_diff": worst,
            "n_compared": compared,
            "n_mismatch": mismatches,
        }
    report["pass"] = bool(
        n_missing == 0
        and missing_in_store == 0
        and all(
            entry.get("max_abs_diff", 0.0) == 0.0
            for entry in report["per_key"].values()
            if isinstance(entry, dict) and "max_abs_diff" in entry
        )
    )
    return report


__all__ = [
    "INT64_KEYS",
    "SINGLE_SLOT_CHANNELS",
    "STEPS_PER_POLICY",
    "STORE_SCHEMA",
    "V6_COMPARE_KEYS",
    "V7ObsStore",
    "compute_base_consistent",
    "load_store",
    "merge_shards",
    "replay_episode",
    "verify_store_vs_base",
    "window_actions",
    "write_shard",
]
