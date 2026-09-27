"""难例挖掘（lane T）：冻结 primary 的逐行 IL 误差 → top-50% 难例 sidecar。

口径（用户定稿，不得改动）
--------------------------
- **参照模型**：primary 段训练结束、**已冻结**的 primary 权重（``--ckpt``）；
- **IL 误差主口径 = 动作加权误差**：逐行 ``err_l1 = mean|action_mu − 专家首步动作|``（L1），
  加权 ``err_weighted = w · err_l1``（``w = row_action_weights`` = ``train_weight×配平``）；
- **难例 = top-50%**：按 ``(-err_weighted, -err_l1, 行号升序)`` 确定性排序取前
  ``ceil(hard_frac · N)``（并列按行号升序；N=1 时取 1 行）；
- sidecar 记录**参照 primary ckpt 标识**（path/sha256/bytes）、数据集指纹、cluster spec
  标识（可选）、seed、阈值与规则 → 可追溯（训练侧只读，缺失/不符 fail-fast）。

产出（npz，供 ``tools/fit_clusters.py --rows-from`` 与 ``pipeline.stages`` 使用）：
``hard``(N,)uint8、``err_l1``(N,)f32、``err_weighted``(N,)f32、``weight``(N,)f32、
``meta``（0-d JSON：版本/规则/行数/阈值/ckpt/数据集/spec/seed/时间戳）。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

__all__ = [
    "HARD_SIDECAR_VERSION",
    "DEFAULT_HARD_FRAC",
    "ckpt_identity",
    "mine_hard_rows",
    "row_il_errors",
    "write_hard_sidecar",
    "load_hard_sidecar",
]

HARD_SIDECAR_VERSION = 1
DEFAULT_HARD_FRAC = 0.5


def ckpt_identity(path: Any) -> Dict[str, Any]:
    """ckpt 标识（可追溯）：``path`` + sha256 + 字节数 + mtime（文件缺失 → sha256 空）。"""
    target = Path(str(path))
    identity: Dict[str, Any] = {"path": str(target)}
    if target.is_file():
        payload = target.read_bytes()
        identity["sha256"] = hashlib.sha256(payload).hexdigest()
        identity["bytes"] = int(len(payload))
        identity["mtime"] = datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        )
    else:
        identity["sha256"] = ""
    return identity


def row_il_errors(
    model: Any,
    dataset: Any,
    *,
    device: str = "auto",
    batch_size: int = 256,
    logger: Any = print,
    obs_source: Any = None,
) -> Dict[str, np.ndarray]:
    """冻结模型逐行 IL 误差（``err_l1`` / ``weight`` / ``err_weighted``，行空间 = 数据集全量）。

    ``model`` 只做 ``rollout=False, world_model=False`` 的 cheap path 前向（``no_grad``）；
    ``weight = row_action_weights(dataset, indices)``（train_weight×配平，与训练主口径同源）。
    ``obs_source``（可选）= :class:`pipeline.trainer.MaterializedBCDataset` 物化快路径
    （只切片，避免逐行重拼历史）。
    """
    import torch

    from pipeline.trainer import resolve_device, row_action_weights, to_device_tensors

    resolved = resolve_device(device)
    torch_device = torch.device(resolved)
    model.to(torch_device).eval()
    count = int(dataset.count)
    err_l1 = np.empty(count, dtype=np.float32)
    weights = np.empty(count, dtype=np.float32)
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, count, step):
            stop = min(start + step, count)
            indices = np.arange(start, stop, dtype=np.int64)
            obs_np = (
                obs_source.obs_batch(indices)
                if obs_source is not None
                else dataset.build_obs_batch(indices)
            )
            obs = to_device_tensors(obs_np, torch_device, dtype=torch.float32)
            out = model(obs, rollout=False, world_model=False)
            mu = out["action_mu"]
            expert = torch.as_tensor(
                dataset.arrays["action"][indices, 0], dtype=torch.float32, device=torch_device
            )
            err_l1[start:stop] = (mu - expert).abs().mean(dim=-1).detach().cpu().numpy()
            weights[start:stop] = row_action_weights(dataset, indices).astype(np.float32)
    logger(
        f"[hard] 逐行 IL 误差完成：rows={count} · 加权误差均值="
        f"{float(np.average(err_l1, weights=np.maximum(weights, 0.0))):.4f}"
    )
    return {"err_l1": err_l1, "weight": weights, "err_weighted": err_l1 * weights}


def mine_hard_rows(
    err_l1: np.ndarray,
    err_weighted: np.ndarray,
    *,
    hard_frac: float = DEFAULT_HARD_FRAC,
) -> Dict[str, Any]:
    """确定性 top-``hard_frac`` 难例选择（主键 = 加权误差，并列按未加权误差→行号升序）。

    返回 ``{"hard": uint8 (N,), "hard_rows": int, "total_rows": int, "threshold": float,
    "order_rule": str}``；``hard_frac`` 必须 ∈ (0, 1]；N≥1 时至少取 1 行。
    """
    frac = float(hard_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"hard_frac 必须在 (0,1]，收到 {hard_frac!r}")
    primary = np.asarray(err_weighted, dtype=np.float64).reshape(-1)
    secondary = np.asarray(err_l1, dtype=np.float64).reshape(-1)
    if primary.size != secondary.size:
        raise ValueError(f"err_weighted/err_l1 行数不一致：{primary.size} vs {secondary.size}")
    total = int(primary.size)
    if total == 0:
        return {
            "hard": np.zeros(0, dtype=np.uint8),
            "hard_rows": 0,
            "total_rows": 0,
            "threshold": float("nan"),
            "order_rule": "empty",
        }
    n_hard = int(np.ceil(frac * total))
    n_hard = min(max(n_hard, 1), total)
    row_index = np.arange(total, dtype=np.int64)
    order = np.lexsort((row_index, -secondary, -primary))  # 末键为主键：先加权误差，再未加权，再行号
    selected = np.sort(order[:n_hard])
    hard = np.zeros(total, dtype=np.uint8)
    hard[selected] = 1
    threshold = float(primary[order[n_hard - 1]]) if n_hard > 0 else float("nan")
    return {
        "hard": hard,
        "hard_rows": int(n_hard),
        "total_rows": total,
        "threshold": threshold,
        "order_rule": "(-err_weighted, -err_l1, row_index asc)",
    }


def write_hard_sidecar(
    path: Any,
    *,
    hard: np.ndarray,
    err_l1: np.ndarray,
    err_weighted: np.ndarray,
    weight: np.ndarray,
    dataset_dir: Any = "",
    dataset_meta: Optional[Mapping[str, Any]] = None,
    ckpt: Any = "",
    cluster_spec: Any = "",
    seed: int = 0,
    hard_frac: float = DEFAULT_HARD_FRAC,
    threshold: float = float("nan"),
    order_rule: str = "",
    tool: str = "tools/mine_hard.py",
) -> Path:
    """写难例 sidecar（npz + meta JSON 字符串）；返回路径。"""
    target = Path(str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "version": HARD_SIDECAR_VERSION,
        "tool": str(tool),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(np.asarray(hard).reshape(-1).size),
        "hard_rows": int(np.count_nonzero(np.asarray(hard) > 0.5)),
        "hard_frac": float(hard_frac),
        "threshold": float(threshold),
        "order_rule": str(order_rule),
        "dataset": str(dataset_dir),
        "obs_fingerprint": str((dataset_meta or {}).get("obs_fingerprint") or ""),
        "schema_version": (dataset_meta or {}).get("schema_version"),
        "ckpt": ckpt_identity(ckpt) if ckpt else {},
        "cluster_spec": (
            dict(ckpt_identity(cluster_spec), version=str(cluster_spec)) if cluster_spec else {}
        ),
        "seed": int(seed),
    }
    np.savez_compressed(
        target,
        hard=np.asarray(hard, dtype=np.uint8).reshape(-1),
        err_l1=np.asarray(err_l1, dtype=np.float32).reshape(-1),
        err_weighted=np.asarray(err_weighted, dtype=np.float32).reshape(-1),
        weight=np.asarray(weight, dtype=np.float32).reshape(-1),
        meta=json.dumps(meta, ensure_ascii=False, sort_keys=True),
    )
    return target


def load_hard_sidecar(
    path: Any,
    *,
    rows: Optional[int] = None,
    obs_fingerprint: Optional[str] = None,
    ckpt_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """读难例 sidecar + **严格校验**（行数 / obs 指纹 / 可选 ckpt sha256）→ dict。

    校验不过 → ``ValueError``（含生成命令），绝不静默降级。
    """
    target = Path(str(path))
    if not target.is_file():
        raise ValueError(
            f"缺少难例 sidecar：{target}\n生成：tools/venv-python tools/mine_hard.py "
            f"--ckpt <primary.pt> --bc-dir <BC_DIR> --out {target}"
        )
    with np.load(target) as payload:
        files = set(payload.files)
        if "hard" not in files:
            raise ValueError(f"难例 sidecar 缺少 'hard' 键：{target}")
        hard = np.asarray(payload["hard"]).reshape(-1).astype(np.uint8)
        meta_raw = payload["meta"] if "meta" in files else np.asarray(["{}"])
        try:
            meta = json.loads(str(np.asarray(meta_raw).reshape(-1)[0]))
        except Exception:  # noqa: BLE001
            meta = {}
        result = {
            "hard": hard,
            "err_l1": np.asarray(payload["err_l1"], dtype=np.float32).reshape(-1) if "err_l1" in files else None,
            "err_weighted": (
                np.asarray(payload["err_weighted"], dtype=np.float32).reshape(-1)
                if "err_weighted" in files
                else None
            ),
            "weight": np.asarray(payload["weight"], dtype=np.float32).reshape(-1) if "weight" in files else None,
            "meta": meta,
            "path": str(target),
        }
    problems = []
    if rows is not None and int(rows) != int(hard.size):
        problems.append(f"rows: sidecar={int(hard.size)} != dataset={int(rows)}")
    expected_fp = str(obs_fingerprint or "")
    if expected_fp and str(meta.get("obs_fingerprint") or "") != expected_fp:
        problems.append(
            f"obs_fingerprint: sidecar={meta.get('obs_fingerprint')!r} != dataset={expected_fp!r}"
        )
    expected_ckpt = str(ckpt_sha256 or "")
    if expected_ckpt and str((meta.get("ckpt") or {}).get("sha256") or "") != expected_ckpt:
        problems.append(
            f"ckpt.sha256: sidecar={(meta.get('ckpt') or {}).get('sha256')!r} != expected={expected_ckpt!r}"
        )
    if problems:
        raise ValueError(
            "难例 sidecar 校验不过：" + "；".join(problems) + f"（{target}）\n"
            "重新挖掘：tools/venv-python tools/mine_hard.py --ckpt <primary.pt> "
            f"--bc-dir <BC_DIR> --out {target}"
        )
    return result
