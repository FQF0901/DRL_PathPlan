#!/usr/bin/env python3
"""正式训练 runner：**readout 列残差头 ΔA_θ 的 one-step 真值监督训练**（R40 §六.10 正式入库）。

规格出处
--------
- GPT ``docs/chatgpt5p6sol_review.md`` §六.10：训练 runner、配置、数据 split 与 train/eval
  feature parity **正式入库**（不能只留在 ``.slim/`` 或临时脚本）；
- GPT R39 ```647f909``` §二：``A_{k+1} = AnalyticUpdate(A_k, a_k) + ΔA_θ(S_k, a_k)``；
  B2 = analytic + learned residual；
- ``.slim/deepwork/p1_v2/round39/b1_report.md`` §5（B2 训练口径）与
  ``round39/b2_report.md``（lane 结论）；本文件 = ``round39/b2_train.py`` 的**逐位迁移版**
  （同参下 features/targets/masks/loss 与 ``.slim`` 版逐位一致，证据见
  ``.slim/deepwork/p1_v2/round40/v5_bitwise.json``）。
- 参数门：总量 ≤ :data:`net.world_residual.TOTAL_PARAM_CAP` = 2,000,000
  （= ``net/param_probe.TOTAL_CAP``，GPT R40 §五.5 绝对上限；R39「基线 ×1.02」乘性规则废止）。
  本 runner 在 ``net/world_residual.check_param_budget`` 之外再断言
  ``structured 基线 1,510,570 + head ≤ 2,000,000``。

数据/监督口径（与 ``b1_matrix.py`` 同列序/单位；单一口径）
----------------------------------------------------------
- 样本 = ``datasets/path_exec_v1`` 同 episode **连续行**（row t → row t+5 = 0.5 s）；
  ``ok`` 集合 = ``train_weight>0 ∧ horizon_ok ∧ 锚长≥3 ∧ 速度列 present=1``（主行与下一行）；
- ``prev`` = 行 t 真实 encode 的 ``WorldState.t0``（``world_state_from_encode``，**单一出处**）
  ⇒ ``readout`` 恒等位姿七量；``a_t`` = 生产结构化头在行 t 的动作 ``(ds,dθ)``（encode 内）；
- ``base`` = ``readout(analytic_update(t0, ds=a_t.ds, dθ=a_t.dθ))``（kin_b + 阻尼 α=1.0）；
- **真实下一状态** = 行 t+5 的真实列 ``long_exog/long_present`` 的四个目标列
  （``speed/static_distance/dynamic_distance/dynamic_ttc``）；
- 目标 ``Δ* = real_next − base``；loss = ``net.world_residual.column_residual_loss``
  （掩码 = ``base_present ∧ real_present``）；``--loss aligned`` 走 B2b 的
  ``column_residual_loss_aligned``（逐列权重 + TTC 事件式；B2b 的 2-step 预热 / ds-dt
  speed 目标研究线仍留在 ``.slim/.../round39/b2b_train.py``，未入库）；
- **禁自产状态回灌**：特征/目标只来自真实行 encode 与数据集真实列；不闭环、不迭代自产链。

切分定义（split；无泄漏）
-------------------------
- train/val 按 **(spec_id, episode_id) 整段切**（episode 级；禁行级泄漏）：固定种子
  ``--seed`` ⇒ ``rng = np.random.default_rng(seed)``，对 episode 键去重后
  ``rng.permutation`` 取前 ``round(--val-frac × n_episodes)`` 为 val；
- 落样本后**断言** train ∩ val = ∅（见 :func:`build_pairs` 的 fail-fast）；
- split 口径（seed/val_frac）记录在 ``config/residual_train.yaml`` 与产物 ``train_log.json``。

配置与可复现命令
----------------
配置：``config/residual_train.yaml``（seed/split/损失权重/batching/路径；CLI 同名键优先）。

::

    # 0) 装配验证（不训练、不写 runs/）
    tools/venv-python tools/train_residual.py --dry-run --max-pairs 1024
    tools/venv-python tools/train_residual.py --check-features --max-pairs 1024

    # 1) 冒烟训练（小样本 + 少 epoch；产物 runs/residual_train/smoke/）
    tools/venv-python tools/train_residual.py --smoke

    # 2) 正式训练（全量 pair 集；产物 <out>/residual_head.pt + train_log.json）
    tools/venv-python tools/train_residual.py --out runs/residual_train

    # 3) train/eval 特征同源守卫（pytest；小样本逐位）
    tools/venv-python -m pytest tests/test_r40_train_eval_feature_parity.py -q

研究 harness（只读装载，不复制实现）
------------------------------------
数据/obs 装配沿用 A1/J1 harness（**机制单一出处**）：
``.slim/deepwork/p1_v2/round38/a1_matrix.py``（``load_rows_filtered``/``build_step_map``）
与 ``.slim/deepwork/p1_v2/round36/j1_train.py``（``build_obs``/``build_model_and_load``）。
本 runner 记录两文件的 sha256（`harness` 字段）；缺失即 fail-fast 并给出路径。

产物（``--out``；默认 ``runs/residual_train``）
---------------------------------------------
``residual_head.pt``（state_dict + meta）/ ``train_log.json``（loss 曲线 + 分列残差 MAE +
训练账本 + split/特征指纹）。``--log-mirror`` 可选镜像摘要。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config/residual_train.yaml"
#: 研究 harness（只读装载；J1 由 A1 内部再装载）
HARNESS_A1 = ROOT / ".slim/deepwork/p1_v2/round38/a1_matrix.py"
HARNESS_J1 = ROOT / ".slim/deepwork/p1_v2/round36/j1_train.py"
#: 目标列（readout 列序中的列号）与列名（顺序敏感：贯穿特征/损失/账本）
TARGET_COLUMNS = (0, 3, 4, 5)
COLUMN_NAMES = ("speed", "static_distance", "dynamic_distance", "dynamic_ttc")
#: 特征键序/宽度（= ``net.world_residual.readout_residual_features`` 返回键；合计 26）
FEATURE_KEYS = ("prev_exog", "prev_present", "action", "base_exog", "base_present")
FEATURE_WIDTHS = (8, 8, 2, 4, 4)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"[residual] fail-fast：无法装载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def a1_module():
    """A1 harness（数据/obs/装配机制单一出处；importlib 只读装载）。"""
    global _A1
    if _A1 is None:
        if not HARNESS_A1.is_file():
            raise SystemExit(
                f"[residual] fail-fast：研究 harness 缺失 {HARNESS_A1}"
                "（数据/obs 机制单一出处；如迁移工作区请同步 round38/a1_matrix.py）"
            )
        _A1 = _load_module("r40_v5_a1_matrix", HARNESS_A1)
    return _A1


_A1 = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_info() -> dict[str, Any]:
    """harness 指纹（复现性证据：本 runner 依赖的只读研究模块）。"""
    return {
        "a1": {"path": str(HARNESS_A1), "sha256": _sha256(HARNESS_A1) if HARNESS_A1.is_file() else None},
        "j1": {"path": str(HARNESS_J1), "sha256": _sha256(HARNESS_J1) if HARNESS_J1.is_file() else None},
    }


# --------------------------------------------------------------------------- #
# 配置（config/residual_train.yaml；CLI 同名键优先）
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(f"[residual] fail-fast：config 不存在：{config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"[residual] fail-fast：config 顶层应为映射：{config_path}")
    return {str(key): value for key, value in payload.items()}


def _parse_floats(value: Any, *, expected: int | None = None) -> tuple[float, ...] | None:
    """``[..]``/``"a,b"`` → ``tuple[float, ...]``（``None`` 透传）。"""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part for part in value.replace(";", ",").split(",") if part.strip()]
        out = tuple(float(part) for part in parts)
    else:
        out = tuple(float(item) for item in value)
    if expected is not None and len(out) != expected:
        raise SystemExit(f"[residual] fail-fast：期望 {expected} 个数值，收到 {out!r}")
    return out


def build_parser(config: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    """CLI（default 取自 config；同名键 CLI 优先）。"""
    parser = argparse.ArgumentParser(
        prog="train_residual.py",
        description="readout 列残差头 one-step 训练（R40 §六.10 正式 runner）",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 配置（键 = 本 parser dest）")
    parser.add_argument("--data", default="datasets/path_exec_v1")
    parser.add_argument("--base-dataset", default="datasets/BTC20261007-2202_expert5k_v8")
    parser.add_argument("--init", default="runs/p1_v2/round22/M0/ckpt_epoch002.pt")
    parser.add_argument("--model-config", default="runs/p1_v2/round22/M0/model.snapshot.yaml")
    parser.add_argument("--ckpt", default="runs/round36/j2/t1_keep_res/ckpt_epoch001.pt")
    parser.add_argument("--out", default=str(ROOT / "runs/residual_train"))
    parser.add_argument("--log-mirror", default=None, help="摘要镜像路径（默认不写）")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=256, help="特征 pass 的 encode 批（split 后同 batching 口径）")
    parser.add_argument("--cache", default="", help="特征/目标缓存（存在 ∧ n ∧ 指纹相符 ⇒ 复用；空串禁用）")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--loss", default="smooth_l1", choices=("smooth_l1", "mse", "l1", "aligned"))
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--column-weights", default=None,
                        help="--loss aligned 逐列权重（如 2.0,0.5,1.0,1.0；默认 = net.world_residual.COLUMN_LOSS_WEIGHTS）")
    parser.add_argument("--ttc-event-weight", type=float, default=0.5, help="--loss aligned TTC 事件项权重")
    parser.add_argument("--ttc-regression-weight", type=float, default=0.5, help="--loss aligned TTC 回归项权重")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--shards", nargs="*", default=None, help="path_exec_v1 分片子集（默认全量）")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="冒烟：小样本 + 少 epoch（产物 <out>/smoke）")
    parser.add_argument("--dry-run", action="store_true", help="装配验证：pairs/split/特征 pass/参数门；不训练、不写 runs/")
    parser.add_argument("--check-features", action="store_true",
                        help="--dry-run + 特征契约逐位复算（键面/宽度/掩码/目标公式）")
    if config:
        # 配置键 = parser dest（CLI 显式给出时覆盖；未知键在 parse_args 里 fail-fast）
        parser.set_defaults(**dict(config))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """两段解析：先 ``--config`` → 配置作 parser 默认值 → CLI 覆盖。"""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre.parse_known_args(argv)
    config = load_config(known.config)
    parser = build_parser(config)
    args = parser.parse_args(argv)
    declared = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - declared)
    if unknown:
        raise SystemExit(f"[residual] fail-fast：{known.config} 含未知键 {unknown}（键名 = CLI dest）")
    return args


# --------------------------------------------------------------------------- #
# 数据：连续行配对 + episode 切分（无泄漏）
# --------------------------------------------------------------------------- #
def build_pairs(args) -> dict:
    a1 = a1_module()
    rows, train_weight, horizon_ok = a1.load_rows_filtered(args)
    spec = np.asarray(rows["spec_id"], dtype=np.int64)
    episode = np.asarray(rows["episode_id"], dtype=np.int64)
    step = np.asarray(rows["step"], dtype=np.int64)
    exog = np.asarray(rows["long_exog"], dtype=np.float64)
    present = np.asarray(rows["long_present"], dtype=np.float64)
    anchor_len = np.asarray([len(np.asarray(a)) for a in rows["anchor"]], dtype=np.int64)
    step_map = a1.build_step_map(rows)

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


def build_model(args, device):
    """结构化模型装载（口径 = A1 harness：正式 loader + fail-fast）。"""
    from tools.legacy.checkpoint_migration import load_checkpoint_state

    import torch

    from net.model import DrivingModel
    from net.path_head import PATH_HEAD_DEFAULT_MODE

    a1 = a1_module()
    jt = a1.jt_module()
    model, _ = jt.build_model_and_load(
        a1._Args(args.init, args.model_config), path_head=True
    )
    if not isinstance(model, DrivingModel):  # pragma: no cover - 装配契约
        raise SystemExit("[residual] fail-fast：build_model_and_load 未返回 DrivingModel")
    ckpt_path = jt._resolve(Path(args.ckpt))
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    report = load_checkpoint_state(model, state, on_partial="fail")
    model.to(device).eval()
    model.set_target_lane_dropout(0.0)
    if model.path_head_mode() != PATH_HEAD_DEFAULT_MODE:
        raise SystemExit(f"[residual] fail-fast：path_head_mode={model.path_head_mode()!r} != structured")
    if model._is_semantic():
        raise SystemExit("[residual] fail-fast：本 runner 口径 = 非 semantic（M0 装配）")
    if model.ego_warp_mode != "off":
        raise SystemExit(f"[residual] fail-fast：ego_warp_mode={model.ego_warp_mode!r} 非默认")

    return model, {
        "ckpt": str(ckpt_path),
        "ckpt_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "missing_keys": list(report.missing_keys),
        "unexpected_keys": list(report.reject_keys),
        "migration": report.to_dict(),
    }


# --------------------------------------------------------------------------- #
# 特征/目标 pass（真实行 encode → prev/base 列 → one-step 目标）
# --------------------------------------------------------------------------- #
def _cache_fingerprint(args, load_info: Mapping[str, Any]) -> str:
    """缓存指纹（data/子集/seed/chunk/base_dataset/ckpt；不符 ⇒ 重算，禁静默复用）。"""
    payload = {
        "data": str(args.data),
        "shards": sorted(str(shard) for shard in args.shards) if args.shards else None,
        "base_dataset": str(args.base_dataset),
        "max_pairs": None if args.max_pairs is None else int(args.max_pairs),
        "seed": int(args.seed),
        "chunk": int(args.chunk),
        "ckpt_sha256": load_info.get("ckpt_sha256"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_supervision(args, device, data: dict) -> dict:
    import torch

    from net.world_residual import READOUT_RESIDUAL_INDICES, readout_residual_features
    from net.world_state import analytic_update, readout, world_state_from_encode

    a1 = a1_module()
    jt = a1.jt_module()
    rows = data["rows"]
    main_ids, succ_ids = data["main_ids"], data["succ_ids"]
    exog, present = data["exog"], data["present"]
    n = int(main_ids.size)

    model, load_info = build_model(args, device)
    fingerprint = _cache_fingerprint(args, load_info)
    cache = Path(args.cache) if args.cache else None
    loaded = None
    if cache is not None and cache.exists():
        payload = np.load(cache)
        stored_fp = str(payload["fingerprint"]) if "fingerprint" in payload.files else ""
        if int(payload["n"]) == n and stored_fp == fingerprint:
            print(f"[residual] 复用特征缓存 {cache}（n={n}，指纹 {fingerprint}）", flush=True)
            loaded = {
                key: payload[key]
                for key in ("features", "targets", "masks", "base_abs", "base_cols", "real_cols")
            }
        else:
            print(
                f"[residual] 缓存失效（n={int(payload['n'])}/{n}，指纹 {stored_fp or '缺失'}/"
                f"{fingerprint}）⇒ 重算",
                flush=True,
            )
    if loaded is not None:
        return {
            "model": model, "load_info": load_info, "cache_fingerprint": fingerprint,
            **loaded, "feature_time_s": 0.0, "cache_hit": True,
        }
    base_dataset = jt.load_base_dataset(jt._resolve(Path(args.base_dataset)))

    chunk = int(args.chunk)
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    base_mae_parts: list[np.ndarray] = []
    base_col_parts: list[np.ndarray] = []
    real_col_parts: list[np.ndarray] = []
    started = time.time()
    with torch.no_grad():
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            obs = jt.build_obs(rows, main_ids[start:stop], device=device, base_dataset=base_dataset)
            encoded = model.encode(obs)
            state = world_state_from_encode(
                model.st_gnn, encoded, dt=model.dt, speed_alpha=1.0, speed_max=20.0
            )
            prev_view = readout(state)
            action = encoded["path"]["action"]
            ds, dtheta = action[:, 0], action[:, 1]
            analytic_update(state, ds=ds, dtheta=dtheta, pose=None,
                            z_od=encoded["z_od"], z_ld=encoded["z_ld"], step=1)
            base_view = readout(state)
            features = readout_residual_features(
                prev_view=prev_view, action=action, base_view=base_view
            )
            index = list(READOUT_RESIDUAL_INDICES)
            base_cols = base_view["long_exog"][:, index]
            base_present = base_view["long_present"][:, index]
            succ = succ_ids[start:stop]
            real_cols = torch.as_tensor(exog[succ][:, index], dtype=torch.float32, device=device)
            real_present = torch.as_tensor(
                present[succ][:, index], dtype=torch.float32, device=device
            )
            mask = ((base_present > 0.5) & (real_present > 0.5)).to(torch.float32)
            target = (real_cols - base_cols) * mask
            feature_parts.append(
                torch.cat(
                    [
                        features["prev_exog"], features["prev_present"], features["action"],
                        features["base_exog"], features["base_present"],
                    ],
                    dim=-1,
                ).detach().cpu().numpy().astype(np.float32)
            )
            target_parts.append(target.detach().cpu().numpy().astype(np.float32))
            mask_parts.append(mask.detach().cpu().numpy().astype(np.float32))
            base_mae_parts.append((target.abs()).detach().cpu().numpy().astype(np.float32))
            base_col_parts.append(base_cols.detach().cpu().numpy().astype(np.float32))
            real_col_parts.append(real_cols.detach().cpu().numpy().astype(np.float32))
            if (start // chunk) % 100 == 0:
                print(
                    f"[residual] features {stop}/{n}（{time.time() - started:.1f}s）", flush=True
                )
    features = np.concatenate(feature_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    masks = np.concatenate(mask_parts, axis=0)
    base_abs = np.concatenate(base_mae_parts, axis=0)
    base_cols_all = np.concatenate(base_col_parts, axis=0)
    real_cols_all = np.concatenate(real_col_parts, axis=0)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise SystemExit("[residual] fail-fast：特征/目标出现非有限值")
    print(f"[residual] supervision 完成：n={n}（{time.time() - started:.1f}s）", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, n=np.asarray(n), fingerprint=np.asarray(fingerprint),
            features=features, targets=targets, masks=masks,
            base_abs=base_abs, base_cols=base_cols_all, real_cols=real_cols_all,
        )
        print(f"[residual] 特征缓存 → {cache}", flush=True)
    return {
        "model": model, "load_info": load_info, "cache_fingerprint": fingerprint,
        "features": features,
        "targets": targets, "masks": masks, "base_abs": base_abs,
        "base_cols": base_cols_all, "real_cols": real_cols_all,
        "feature_time_s": round(time.time() - started, 1), "cache_hit": False,
    }


# --------------------------------------------------------------------------- #
# 训练（只训头；optimizer allowlist 断言）
# --------------------------------------------------------------------------- #
def train_head(args, device, data: dict, supervision: dict) -> dict:
    import torch

    from net.world_residual import (
        TARGET_LOSS_SCALE,
        ReadoutResidualHead,
        column_residual_loss,
        column_residual_loss_aligned,
    )

    torch.manual_seed(int(args.seed))
    train_mask, val_mask = data["train_mask"], data["val_mask"]
    features = torch.as_tensor(supervision["features"], dtype=torch.float32)
    targets = torch.as_tensor(supervision["targets"], dtype=torch.float32)
    masks = torch.as_tensor(supervision["masks"], dtype=torch.float32)
    base_abs = torch.as_tensor(supervision["base_abs"], dtype=torch.float32)
    base_cols = torch.as_tensor(supervision["base_cols"], dtype=torch.float32)
    real_cols = torch.as_tensor(supervision["real_cols"], dtype=torch.float32)

    aligned = str(args.loss) == "aligned"
    column_weights = _parse_floats(args.column_weights, expected=len(COLUMN_NAMES)) if aligned else None

    head = ReadoutResidualHead(hidden=int(args.hidden)).to(device)
    head.train()
    model = supervision["model"]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.set_world_residual_head(head)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    head_ids = {id(p) for p in head.parameters()}
    if opt_ids != head_ids:  # pragma: no cover - allowlist 断言
        raise SystemExit("[residual] fail-fast：optimizer 参数集 ≠ 残差头参数集（allowlist 违约）")
    foreign = [
        name for name, p in model.named_parameters() if p.requires_grad and not name.startswith("world_residual.")
    ]
    if foreign:  # pragma: no cover - 冻结断言
        raise SystemExit(f"[residual] fail-fast：存在未冻结参数 {foreign[:3]}")

    device_features = features.to(device)
    device_targets = targets.to(device)
    device_masks = masks.to(device)
    device_base = base_cols.to(device)
    device_real = real_cols.to(device)

    def batch_loss(delta, index):
        # 字面 one-step：L = loss(base + ΔA_θ, 实际下一状态)（掩码 = base_present ∧ real_present）
        if aligned:
            # B2b 对齐损失路径（逐列权重 + TTC 事件式；speed_target/2-step 预热留在 .slim 研究线）
            return column_residual_loss_aligned(
                delta,
                base_exog=device_base[index],
                base_present=device_masks[index],
                real_exog=device_real[index],
                real_present=device_masks[index],
                speed_target=None,
                mode="smooth_l1",
                beta=float(args.beta),
                scale=TARGET_LOSS_SCALE,
                column_weights=column_weights,
                ttc_event_weight=float(args.ttc_event_weight),
                ttc_regression_weight=float(args.ttc_regression_weight),
            )
        return column_residual_loss(
            delta,
            base_exog=device_base[index],
            base_present=device_masks[index],
            real_exog=device_real[index],
            real_present=device_masks[index],
            mode=args.loss, beta=float(args.beta), scale=TARGET_LOSS_SCALE,
        )

    train_index = np.flatnonzero(train_mask)
    val_index = np.flatnonzero(val_mask)
    target_stats = {}
    for index, name in enumerate(COLUMN_NAMES):
        msk = masks[:, index] > 0.5
        values = targets[msk, index]
        target_stats[name] = {
            "n": int(msk.sum()),
            "mean": float(values.mean()) if values.numel() else None,
            "std": float(values.std()) if values.numel() else None,
            "mae_abs_mean": float(values.abs().mean()) if values.numel() else None,
            "p95_abs": float(values.abs().quantile(0.95)) if values.numel() else None,
        }
    print(f"[residual] target stats {json.dumps(target_stats, ensure_ascii=False)}", flush=True)
    history: list[dict] = []
    best = {"epoch": -1, "val_loss": float("inf"), "state": None}
    patience = int(args.patience)
    stale = 0
    started = time.time()
    for epoch in range(1, int(args.epochs) + 1):
        head.train()
        perm = np.random.default_rng(int(args.seed) + epoch).permutation(train_index.size)
        order = train_index[perm]
        epoch_loss = 0.0
        batches = 0
        for start in range(0, order.size, int(args.batch_size)):
            index = torch.as_tensor(order[start : start + int(args.batch_size)], dtype=torch.long)
            base = device_features[index]
            prev_exog, prev_present = base[:, :8], base[:, 8:16]
            action, base_exog = base[:, 16:18], base[:, 18:22]
            base_present = base[:, 22:26]
            delta = head(
                prev_exog=prev_exog, prev_present=prev_present, action=action,
                base_exog=base_exog, base_present=base_present,
            )
            out = batch_loss(delta, index)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.grad_clip))
            optimizer.step()
            epoch_loss += float(out["loss"].item())
            batches += 1
        head.eval()
        with torch.no_grad():
            def eval_split(index: np.ndarray) -> dict:
                pred_parts, target_parts_ = [], []
                for start in range(0, index.size, 4096):
                    idx = torch.as_tensor(index[start : start + 4096], dtype=torch.long)
                    base = device_features[idx]
                    delta = head(
                        prev_exog=base[:, :8], prev_present=base[:, 8:16],
                        action=base[:, 16:18], base_exog=base[:, 18:22],
                        base_present=base[:, 22:26],
                    )
                    pred_parts.append(delta.cpu())
                    target_parts_.append(device_targets[idx].cpu())
                pred = torch.cat(pred_parts)
                tgt = torch.cat(target_parts_)
                msk = torch.as_tensor(masks[index], dtype=torch.float32)
                err = (pred - tgt).abs() * msk
                counts = msk.sum(dim=0).clamp_min(1.0)
                return {
                    "mae_per_column": [float(v) for v in (err.sum(0) / counts)],
                    "n_per_column": [int(v) for v in msk.sum(0)],
                }

            tr = eval_split(train_index)
            va = eval_split(val_index)
            va_err = (
                torch.as_tensor(va["mae_per_column"]) * torch.as_tensor(va["n_per_column"])
            ).sum() / max(1.0, float(sum(va["n_per_column"])))
            item = {
                "epoch": epoch,
                "train_loss": epoch_loss / max(1, batches),
                "train_mae_per_column": tr["mae_per_column"],
                "val_mae_per_column": va["mae_per_column"],
                "val_mae_weighted": float(va_err),
            }
        history.append(item)
        if item["val_mae_weighted"] < best["val_loss"] - 1e-6:
            best = {
                "epoch": epoch,
                "val_loss": item["val_mae_weighted"],
                "state": {k: v.detach().clone() for k, v in head.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % int(args.log_every) == 0:
            print(
                f"[residual] epoch {epoch}: train={item['train_loss']:.5f} "
                f"val_wMAE={item['val_mae_weighted']:.5f}",
                flush=True,
            )
        if stale >= patience:
            print(f"[residual] early stop @ epoch {epoch}（patience={patience}）", flush=True)
            break
    if best["state"] is None:  # pragma: no cover - 至少一个 epoch
        raise SystemExit("[residual] fail-fast：训练未记录任何 best 状态")
    head.load_state_dict(best["state"])
    head.eval()
    train_time_s = round(time.time() - started, 1)
    with torch.no_grad():
        base_mae = {}
        for index, name in enumerate(COLUMN_NAMES):
            msk = masks[:, index]
            base_mae[name] = float((base_abs[:, index] * msk).sum() / max(1.0, float(msk.sum())))
        val_base_abs = base_abs[val_index]
        val_masks = masks[val_index]
        val_pred_parts = []
        with torch.no_grad():
            for start in range(0, val_index.size, 4096):
                idx = torch.as_tensor(val_index[start : start + 4096], dtype=torch.long)
                base = device_features[idx]
                val_pred_parts.append(
                    head(
                        prev_exog=base[:, :8], prev_present=base[:, 8:16],
                        action=base[:, 16:18], base_exog=base[:, 18:22],
                        base_present=base[:, 22:26],
                    ).cpu()
                )
        val_pred = torch.cat(val_pred_parts)
        per_column = {}
        for index, name in enumerate(COLUMN_NAMES):
            msk = val_masks[:, index]
            count = max(1.0, float(msk.sum()))
            tgt = targets[val_index, index]
            mean = float((tgt * msk).sum() / count)
            per_column[name] = {
                "base_mae": float((val_base_abs[:, index] * msk).sum() / count),
                "corrected_mae": float(((val_pred[:, index] - tgt).abs() * msk).sum() / count),
                "constant_mae": float((((tgt - mean).abs()) * msk).sum() / count),
                "n": int(msk.sum()),
            }
    return {
        "head": head, "history": history, "best_epoch": best["epoch"],
        "best_val_mae_weighted": best["val_loss"], "train_time_s": train_time_s,
        "per_column_val": per_column,
        "base_mae_all_split": base_mae,
        "target_stats": target_stats,
    }


# --------------------------------------------------------------------------- #
# 装配验证（--dry-run / --check-features；不训练、不写 runs/）
# --------------------------------------------------------------------------- #
def self_proof(args, device, data, supervision, head) -> dict:
    """S-B2-0：零初始化头 ⇒ Δ=0（逐位）；S-B2-1：episode 切分不相交；S-B2-2：参数/冻结核对。"""
    import torch

    from net.world_residual import ReadoutResidualHead, residual_head_param_count

    zeros = ReadoutResidualHead(hidden=int(args.hidden)).to(device).eval()
    features = torch.as_tensor(supervision["features"][:64], dtype=torch.float32, device=device)
    with torch.no_grad():
        delta_zero = zeros(
            prev_exog=features[:, :8], prev_present=features[:, 8:16],
            action=features[:, 16:18], base_exog=features[:, 18:22],
            base_present=features[:, 22:26],
        )
    proof = {
        "zero_init_delta_is_zero": bool(torch.equal(delta_zero, torch.zeros_like(delta_zero))),
        "zero_init_max_abs": float(delta_zero.abs().max().item()),
        "episode_split_disjoint": not bool(
            (data["train_mask"] & data["val_mask"]).any()
        ),
        "note": "S-B2-0 = 零初始化头严格零残差；S-B2-1 = 划分子集互斥（episode 级）",
    }
    if head is not None:
        proof["head_params"] = residual_head_param_count(head)
    return proof


def feature_contract(args, device, data, supervision) -> dict:
    """特征契约逐位复算（键面/宽度/复算 features/掩码/目标公式；--check-features 用）。"""
    import torch

    from net.world_residual import (
        READOUT_RESIDUAL_FIELDS,
        READOUT_RESIDUAL_INDICES,
        readout_residual_features,
    )
    from net.world_state import analytic_update, readout, world_state_from_encode

    a1 = a1_module()
    jt = a1.jt_module()
    rows, main_ids = data["rows"], np.asarray(data["main_ids"], dtype=np.int64)
    chunk = max(1, min(int(args.chunk), int(main_ids.size)))
    ids = main_ids[:chunk]
    model = supervision["model"]
    base_dataset = jt.load_base_dataset(jt._resolve(Path(args.base_dataset)))
    with torch.no_grad():
        obs = jt.build_obs(rows, ids, device=device, base_dataset=base_dataset)
        encoded = model.encode(obs)
        state = world_state_from_encode(
            model.st_gnn, encoded, dt=model.dt, speed_alpha=1.0, speed_max=20.0
        )
        prev_view = readout(state)
        action = encoded["path"]["action"]
        analytic_update(state, ds=action[:, 0], dtheta=action[:, 1], pose=None,
                        z_od=encoded["z_od"], z_ld=encoded["z_ld"], step=1)
        base_view = readout(state)
        features = readout_residual_features(prev_view=prev_view, action=action, base_view=base_view)
        recon = (
            torch.cat([features[key] for key in FEATURE_KEYS], dim=-1)
            .detach().cpu().numpy().astype(np.float32)
        )
    stored = np.asarray(supervision["features"][:chunk], dtype=np.float32)
    index = list(READOUT_RESIDUAL_INDICES)
    present_next = np.asarray(data["present"], dtype=np.float32)[data["succ_ids"][:chunk]][:, index]
    mask_re = ((recon[:, 22:26] > 0.5) & (present_next > 0.5)).astype(np.float32)
    target_re = (
        np.asarray(supervision["real_cols"][:chunk], dtype=np.float32)
        - np.asarray(supervision["base_cols"][:chunk], dtype=np.float32)
    ) * mask_re
    report = {
        "chunk": int(chunk),
        "keys": sorted(features),
        "keys_match": tuple(sorted(features)) == tuple(sorted(FEATURE_KEYS)),
        "widths": [int(features[key].shape[-1]) for key in FEATURE_KEYS],
        "widths_match": [int(features[key].shape[-1]) for key in FEATURE_KEYS] == list(FEATURE_WIDTHS),
        "indices_match": tuple(READOUT_RESIDUAL_INDICES) == TARGET_COLUMNS
        and tuple(READOUT_RESIDUAL_FIELDS) == COLUMN_NAMES,
        "recompute_bitwise_vs_stored": bool(np.array_equal(recon, stored)),
        "recompute_max_abs_diff": float(
            np.abs(recon.astype(np.float64) - stored.astype(np.float64)).max()
        ),
        "mask_formula_bitwise": bool(
            np.array_equal(mask_re, np.asarray(supervision["masks"][:chunk], dtype=np.float32))
        ),
        "target_formula_bitwise": bool(
            np.array_equal(target_re, np.asarray(supervision["targets"][:chunk], dtype=np.float32))
        ),
        "finite": bool(np.isfinite(stored).all()),
    }
    report["pass"] = bool(
        report["keys_match"]
        and report["widths_match"]
        and report["indices_match"]
        and report["recompute_bitwise_vs_stored"]
        and report["mask_formula_bitwise"]
        and report["target_formula_bitwise"]
        and report["finite"]
    )
    return report


def dry_run_report(args, device, data, supervision, contract, started: float) -> dict:
    """--dry-run 汇总（装配验证；零写盘、零训练）。"""
    from net.world_residual import (
        TOTAL_PARAM_BASELINE,
        TOTAL_PARAM_CAP,
        ReadoutResidualHead,
        check_param_budget,
        residual_head_param_count,
    )

    head = ReadoutResidualHead(hidden=int(args.hidden)).to(device).eval()
    model_total = int(sum(p.numel() for p in supervision["model"].parameters()))
    head_params = residual_head_param_count(head)
    budget = check_param_budget(head, model_total=model_total)
    structured_with_head = TOTAL_PARAM_BASELINE + head_params
    proof = self_proof(args, device, data, supervision, head)
    report = {
        "mode": "check-features" if args.check_features else "dry-run",
        "trained": False,
        "wrote_runs": False,
        "device": str(device),
        "runtime_s": round(time.time() - started, 1),
        "harness": harness_info(),
        "runner": str(Path(__file__).resolve()),
        "config": str(DEFAULT_CONFIG),
        "data": data["meta"],
        "supervision": {
            "n": int(supervision["features"].shape[0]),
            "feature_width": int(supervision["features"].shape[1]),
            "cache_hit": bool(supervision.get("cache_hit")),
            "cache_fingerprint": supervision.get("cache_fingerprint"),
            "load_info_ckpt": supervision["load_info"]["ckpt"],
            "load_info_ckpt_sha256": supervision["load_info"]["ckpt_sha256"],
        },
        "param_gate": {
            **budget,
            "structured_total_baseline": TOTAL_PARAM_BASELINE,
            "structured_with_head_total": structured_with_head,
            "structured_cap": TOTAL_PARAM_CAP,
            "structured_cap_ok": structured_with_head <= TOTAL_PARAM_CAP,
        },
        "self_proof": proof,
        "feature_contract": contract,
    }
    report["pass"] = bool(
        report["supervision"]["feature_width"] == sum(FEATURE_WIDTHS)
        and report["param_gate"]["structured_cap_ok"]
        and proof["zero_init_delta_is_zero"]
        and proof["episode_split_disjoint"]
        and (contract is None or contract["pass"])
    )
    return report


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    if args.smoke and not (args.dry_run or args.check_features):
        args.max_pairs = 4000
        args.epochs = 6
        args.patience = 3
        args.log_every = 1
        args.chunk = 64
        args.batch_size = 512
        args.out = str(Path(args.out) / "smoke")
        args.log_mirror = None
        args.cache = ""

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data = build_pairs(args)
    print(f"[residual] pairs {json.dumps(data['meta'], ensure_ascii=False)}", flush=True)
    supervision = build_supervision(args, device, data)
    if args.dry_run or args.check_features:
        contract = feature_contract(args, device, data, supervision) if args.check_features else None
        report = dry_run_report(args, device, data, supervision, contract, started)
        print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
        if not report["pass"]:
            raise SystemExit("[residual] fail-fast：--dry-run/--check-features 装配验证未通过")
        print("[residual] 装配验证通过（不训练、不写 runs/）", flush=True)
        return 0

    result = train_head(args, device, data, supervision)
    head = result["head"]

    from net.world_residual import (
        TOTAL_PARAM_BASELINE,
        TOTAL_PARAM_CAP,
        check_param_budget,
        residual_head_param_count,
        save_residual_head,
    )
    from net.world_state import POLICY_STATE_FIELDS

    model_total = int(sum(p.numel() for p in supervision["model"].parameters()))
    head_params = residual_head_param_count(head)
    budget = check_param_budget(head, model_total=model_total)
    # 参数门（R40 §五.5 绝对上限）：structured 基线 1,510,570 + head ≤ 2,000,000
    structured_with_head = TOTAL_PARAM_BASELINE + head_params
    if structured_with_head > TOTAL_PARAM_CAP:  # pragma: no cover - 结构性排除
        raise SystemExit(
            f"[residual] fail-fast：structured 总量 {structured_with_head} > 上限 {TOTAL_PARAM_CAP}"
        )
    budget["structured_with_head_total"] = structured_with_head
    meta = {
        "hidden": int(args.hidden),
        "target_fields": list(COLUMN_NAMES),
        "target_indices": list(TARGET_COLUMNS),
        "policy_state_fields": list(POLICY_STATE_FIELDS),
        "base": "WorldState.t0(real row t) + analytic_update(kin_b + 阻尼 α=1.0)",
        "target": "row t+5 真实列 long_exog/long_present（speed/static/dyn/ttc）",
        "loss": args.loss,
        "data": data["meta"],
        "epochs_ran": len(result["history"]),
        "best_epoch": int(result["best_epoch"]),
        "train_time_s": result["train_time_s"],
        "script": str(Path(__file__).resolve()),
        "harness": harness_info(),
        "cache_fingerprint": supervision.get("cache_fingerprint"),
    }
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_residual_head(out_dir / "residual_head.pt", head, meta=meta)
    payload = {
        "meta": {
            "runner": "r40-v5 正式训练 runner（tools/train_residual.py；GPT §六.10 入库）",
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "harness": harness_info(),
            "device": str(device),
            "runtime_s": round(time.time() - started, 1),
            "ckpt_base": supervision["load_info"],
            "head_ckpt": str(head_path),
            "head_sha256": _sha256(head_path),
            "config": str(DEFAULT_CONFIG),
            "moe": "不启用", "ppo": "不启用", "bptt": "不启用（one-step）",
        },
        "param_ledger": {
            **budget,
            "structured_total_baseline": TOTAL_PARAM_BASELINE,
            "structured_total_cap": TOTAL_PARAM_CAP,
            "lane_model_total_note": "model_total = 本 runner 实际装配（非 semantic M0 + 头）；"
                                     "预算判定按 R40 §五.5 绝对上限 2,000,000（param_probe.TOTAL_CAP 同源）",
            "head_param_names_allowlist": sorted(name for name, _ in head.named_parameters()),
            "head_only_optimizer": True,
            "frozen": "其余全部（policy/encoder/Mem/st_gnn/旧头）",
        },
        "data": data["meta"],
        "train": {
            "history": result["history"],
            "best_epoch": int(result["best_epoch"]),
            "best_val_mae_weighted": result["best_val_mae_weighted"],
            "train_time_s": result["train_time_s"],
            "feature_pass_time_s": supervision["feature_time_s"],
            "per_column_val": result["per_column_val"],
            "base_mae_all_split": result["base_mae_all_split"],
            "target_stats": result["target_stats"],
            "cache_hit": bool(supervision.get("cache_hit")),
            "cache_fingerprint": supervision.get("cache_fingerprint"),
            "columns": list(COLUMN_NAMES),
            "note": "per_column_val.base_mae = 未修正（analytic base）分列 MAE；corrected_mae = ΔA_θ 修正后",
        },
        "self_proof": self_proof(args, device, data, supervision, head),
        "caveats": [
            "one-step teacher forcing：训练输入 = 真实行 encode 的 readout（禁自产状态回灌）；"
            "评测 = 想象链自回归 ⇒ 存在分布偏移（面级证据在 b2_matrix）。",
            "detach 边界（D-WS01）未改：残差头输出不穿越跨步链；训练/评测都是逐帧调用。",
            "真实下一状态列 = 行 t+5 的数据集列（env 生产者口径，与 b1_matrix v_real 同源；"
            "speed present 在长 horizon 处有衰减 ⇒ 掩码）。",
            "--loss aligned = B2b 的 column_residual_loss_aligned（逐列权重 + TTC 事件式）；"
            "B2b 的 2-step 预热 / ds-dt speed 目标研究线未入库（.slim/.../round39/b2b_train.py）。",
        ],
    }
    log_path = out_dir / "train_log.json"
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.log_mirror:
        mirror = Path(args.log_mirror)
        if not mirror.is_absolute():
            mirror = ROOT / mirror
        mirror.parent.mkdir(parents=True, exist_ok=True)
        mirror.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"[residual] → {head_path}（head params {head_params}；总量 {model_total} ≤ "
        f"{budget['model_total_cap']}；{payload['meta']['runtime_s']}s）",
        flush=True,
    )
    return 0


def _main(argv=None) -> int:  # pragma: no cover - 占位（保持 main 名字约定）
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
