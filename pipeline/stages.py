"""阶段 A/B/C 编排（v1.1：WM 教师强制 → planner BC → PPO RL）。

阶段语义（user v1.1，2026-09-25 修订）
------------------------------------
- **A = world model 训练（教师强制）**：ego 条件 = 专家 GT 动作序列 ``action (6,2)``
  （harvest 数据集），目标 = 由 ``(episode, step+k)`` 查表重建的未来 OD/LD 帧
  （对齐到 t0 帧、mask+valid，见 :class:`FrameWindows.build_future`），
  **直接多步损失**（Huber + 角度 ``1-cos``）+ **ego plan 噪声增强**。
  可训练：encoders/temporal/spatial/MoE + world model；policy/value 头不参与。
- **B = planner BC**：可训练 backbone+MoE+policy head，**primary→specific** 两段：
  先训练 shared+primary（experts 冻结），再冻结 primary+shared、训练 8 个 specific experts。
  损失 = **动作**（``action_mu`` vs 专家即时动作，主项）+ **小权重 rollout 轨迹辅助**
  （``traj_xy`` vs 专家 ``traj6``，**WM 冻结且 rollout 内输出 detach**，因果链有效）+ router BCE。
- **C = PPO RL**：KL 锚 = **阶段 B 策略快照**（``--ckpt``，冻结参考模型），系数**线性衰减**
  （默认 0.05 → 0）；primary lr ×0.1；world model 初始冻结、``--wm-freeze-updates`` 后解冻
  （PPO 损失本身不消费 WM 输出，解冻 = 参数重新进入优化器，供后续 WM 辅助损失使用）。

环境约束（§8.1）
----------------
MetaDrive 每进程只能有一个 engine，``LocalEnvPool`` 因此恒为 1 env；多 env 训练用
``--pool vector``（``pipeline.vector_env`` 子进程池）。任一时刻只跑一个 env-heavy 任务。

用法::

    tools/venv-python tools/train.py --stage A --bc-dir runs/bc_expert_full \\
        --wm-epochs 10 --out runs/train/stage_a
    tools/venv-python tools/train.py --stage B --ckpt runs/train/stage_a/final.pt \\
        --bc-dir runs/bc_expert_full --bc-epochs 10 --out runs/train/stage_b
    tools/venv-python tools/train.py --stage C --ckpt runs/train/stage_b/final.pt \\
        --spec env/specs/scenarios_train_slice200.json --envs 2 --updates 20 --out runs/train/stage_c
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 允许 `python pipeline/stages.py` 直接运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.trainer import (  # noqa: E402
    BCConfig,
    BCDataset,
    PPOConfig,
    PPOTrainer,
    apply_freeze_prefixes,
    apply_thread_limits,
    build_pool,
    build_reward_adapter,
    load_checkpoint,
    pretrain_bc,
    resolve_device,
    sanitize_masked_od,
    save_checkpoint,
    stack_history,
    squeeze_single_slot,
    trim_memory,
)

__all__ = ["main", "run_stage_a", "run_stage_b", "run_stage_c", "load_config", "build_model", "FrameWindows"]

_DEFAULT_MODEL_CFG = "config/model.yaml"
_DEFAULT_TRAIN_CFG = "config/train.yaml"


# --------------------------------------------------------------------------- #
# 配置 / 模型
# --------------------------------------------------------------------------- #

def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml  # type: ignore

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str = "config/default.yaml") -> Dict[str, Any]:
    """读主配置（``includes`` 列表逐个子配置合并；后者覆盖前者同名键）。"""
    payload = _load_yaml(path)
    merged: Dict[str, Any] = {}
    for include in payload.get("includes", []) or []:
        include_path = include if os.path.isabs(str(include)) else os.path.join(_PROJECT_ROOT, str(include))
        try:
            merged.update(_load_yaml(include_path))
        except FileNotFoundError:
            continue
    merged.update({key: value for key, value in payload.items() if key != "includes"})
    return merged


def build_model(config: Mapping[str, Any]) -> Any:
    """按 ``config/model.yaml`` 构造 ``net.model.DrivingModel``。

    ``load_config`` 的 includes 是**平铺合并**（model.yaml 的键在根层），因此这里也接受
    根层键（``hidden_dim``/``moe``/``world_model``）或显式的 ``config["model"]`` 子映射。
    """
    from net.model import DrivingModel

    section = config.get("model")
    section = dict(section) if isinstance(section, Mapping) else dict(config)
    moe = dict(section.get("moe", {}) or {})
    experts = dict(moe.get("experts", {}) or {})
    world_model = dict(section.get("world_model", {}) or {})
    kwargs = {
        "hidden": int(section.get("hidden_dim", 128)),
        "num_experts": int(experts.get("count", 8)),
        "expert_hidden": int(experts.get("hidden_dim", 256)),
        "wm_steps": int(world_model.get("rollout_steps", 6)),
    }
    return DrivingModel(**kwargs)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _monitor_enabled(args: argparse.Namespace, config: Mapping[str, Any]) -> bool:
    """monitoring 开关：CLI 显式优先，否则读 ``config/train.yaml::monitoring``。"""
    explicit = getattr(args, "monitor", None)
    if explicit is not None:
        return bool(explicit)
    cfg = dict(config.get("monitoring") or {})
    return any(bool(cfg.get(key)) for key in ("tensorboard", "csv", "scene_labels", "moe_routing"))


def _make_monitor(log_dir: Path, enabled: bool) -> Optional[Any]:
    if not enabled:
        return None
    try:
        from pipeline.monitoring import TrainingMonitor

        return TrainingMonitor(str(log_dir))
    except Exception as exc:  # noqa: BLE001
        print(f"[stages] monitoring 不可用（{type(exc).__name__}: {exc}）→ 跳过", flush=True)
        return None


def _resolve_specs(args: argparse.Namespace, config: Mapping[str, Any]) -> List[Any]:
    from env.scenario.spec import load_specs

    spec_path = args.spec or (config.get("data", {}) or {}).get("spec", "env/specs/scenarios_train.json")
    specs = load_specs(str(spec_path))
    if args.geometry:
        wanted = {item.strip() for item in str(args.geometry).split(",") if item.strip()}
        specs = [spec for spec in specs if str(getattr(spec, "labels", {}).get("geometry", "")) in wanted]
    if args.limit is not None:
        specs = specs[: max(1, int(args.limit))]
    return specs


def _stage_section(config: Mapping[str, Any], stage: str) -> Dict[str, Any]:
    section = (config.get("stages", {}) or {}).get(stage, {}) or {}
    return dict(section) if isinstance(section, Mapping) else {}


def _limit_dataset(dataset: BCDataset, limit: Optional[int]) -> BCDataset:
    """``--limit-dataset`` 取**前缀**（harvest 按 episode 顺序追加 → 前缀 = 完整 episode）。

    与旧的 ``linspace`` 抽样不同：阶段 A 的未来目标依赖 ``(episode, step+k)`` 查表，
    随机抽样会把同 episode 的相邻帧拆散 → 目标大面积无效。
    """
    if limit is None or dataset.count <= int(limit):
        return dataset
    keep = np.arange(int(limit), dtype=np.int64)
    return BCDataset({key: value[keep] for key, value in dataset.arrays.items()}, dataset.meta)


# --------------------------------------------------------------------------- #
# 按帧数据的窗口构建（阶段 A：BC 专家数据集的未来目标）
# --------------------------------------------------------------------------- #

_REPLAY_CHANNELS = ("ego", "od", "ld", "nav", "signal")
#: BC harvest 的 ``step`` 单位是 env step（0.1 s），帧只在策略步边界记录
#: （``collect_expert.STEPS_PER_POLICY = 5``）→ 未来第 k 个策略步 = ``step + 5k``。
_BC_STEP_STRIDE = 5
#: 默认 schema = PPO replay（``obs.od`` / ``mask.od``）；BC 数据集用 ``_bc_channel_keys()``
_REPLAY_CHANNEL_KEYS: Dict[str, Tuple[str, Optional[str]]] = {
    name: (f"obs.{name}", f"mask.{name}") for name in _REPLAY_CHANNELS
}


def _bc_channel_keys() -> Dict[str, Tuple[str, Optional[str]]]:
    """BC 数据集（``tools/collect_expert.py``）的通道键映射。"""
    return {name: (name, f"{name}_mask") for name in _REPLAY_CHANNELS}


class FrameWindows:
    """把"每帧当前通道 + 位姿"的数组建成模型输入窗口与未来目标。

    - 历史窗口：同 episode 内向前最多 5 帧，按 ``stack_history`` SE(2) 对齐到当前帧；
    - 未来目标：向后 1..K 帧（对齐到当前帧），缺失步 ``valid=0`` 并在 mask 中清零。

    键名可配置（``episode_key``/``step_key``/``keys``）：PPO replay 用默认
    （``episode``/``step_index``/``obs.*``）；BC 专家数据集用
    ``episode_id``/``step`` + :func:`_bc_channel_keys`。
    """

    def __init__(
        self,
        arrays: Mapping[str, np.ndarray],
        alignments: Optional[Dict[str, Any]] = None,
        *,
        episode_key: str = "episode",
        step_key: str = "step_index",
        keys: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
        step_stride: int = 1,
    ):
        from pipeline.trainer import _alignment_from_meta

        self.arrays = arrays
        self.alignments = alignments if alignments is not None else _alignment_from_meta({})
        self.keys = dict(keys) if keys is not None else dict(_REPLAY_CHANNEL_KEYS)
        self.step_stride = max(1, int(step_stride))
        self.episode = np.asarray(arrays[episode_key], dtype=np.int64)
        self.step_index = np.asarray(arrays[step_key], dtype=np.int64)
        self.pose = np.asarray(arrays["pose"], dtype=np.float32)
        self.count = int(len(self.episode))
        self._lookup = {
            (int(self.episode[i]), int(self.step_index[i])): i for i in range(self.count)
        }
        self._episode_starts: Dict[int, int] = {}
        for index in range(self.count):
            self._episode_starts.setdefault(int(self.episode[index]), index)

    # ---------------------------------------------------------------- 单帧
    def _frame_obs(self, index: int) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        for name, (feat_key, mask_key) in self.keys.items():
            value = self.arrays.get(feat_key)
            if value is None:
                continue
            obs[name] = np.asarray(value[index], dtype=np.float32)
            mask = self.arrays.get(mask_key) if mask_key else None
            if mask is not None:
                obs[f"{name}_mask"] = np.asarray(mask[index], dtype=np.float32)
        return obs

    def _entry(self, index: int) -> Dict[str, Any]:
        return {"obs": self._frame_obs(index), "pose": np.asarray(self.pose[index], dtype=np.float32)}

    # ---------------------------------------------------------------- 组装
    def build_obs(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """当前帧 + 历史窗口（与训练器 ``BCDataset.build_obs_batch`` 同口径）。"""
        batch: Dict[str, List[np.ndarray]] = {}
        hist_valid: List[np.ndarray] = []
        for row, index in enumerate(np.asarray(indices, dtype=np.int64)):
            episode = int(self.episode[index])
            start = self._episode_starts[episode]
            entries = [self._entry(j) for j in range(max(start, index - 5), index + 1)]
            valid = np.zeros(6, dtype=np.float32)
            valid[6 - len(entries) :] = 1.0
            if len(entries) < 6:
                entries = [entries[0]] * (6 - len(entries)) + entries
            patch = stack_history(entries, self.alignments, current_pose=self.pose[index], valid=valid)
            frame = self._frame_obs(index)
            for key, value in frame.items():
                batch.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
            for key, value in patch.items():
                batch.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
            hist_valid.append(valid)
        batch["hist_valid"] = [np.asarray(item) for item in hist_valid]
        return sanitize_masked_od(
            squeeze_single_slot({key: np.stack(values, axis=0).astype(np.float32) for key, values in batch.items()})
        )

    def build_future(self, indices: np.ndarray, future: int = 6) -> Dict[str, np.ndarray]:
        """未来 1..K 步目标（对齐到当前帧）+ 掩码 + 有效步（``(B,K,16,...)``）。

        通过 ``(episode, step + k·step_stride)`` 查表取**未来帧的原始通道**，因此 mask 是
        未来帧自身的槽位掩码；缺失步（episode 结束 / 帧被过滤）``valid=0`` 且 mask 清零。

        ``step_stride``：harvest 数据集只在策略步边界记录帧（``STEPS_PER_POLICY=5`` env steps
        = 0.5 s），因此 BC schema 用 stride=5；PPO replay（逐 env step 记录）用默认 1。
        """
        od_fut: List[np.ndarray] = []
        ld_fut: List[np.ndarray] = []
        od_masks: List[np.ndarray] = []
        ld_masks: List[np.ndarray] = []
        valid: List[np.ndarray] = []
        for index in np.asarray(indices, dtype=np.int64):
            episode = int(self.episode[index])
            step = int(self.step_index[index])
            entries: List[Dict[str, Any]] = []
            step_valid = np.zeros(future, dtype=np.float32)
            for k in range(1, future + 1):
                target = self._lookup.get((episode, step + k * self.step_stride))
                if target is None:
                    entries.append(entries[-1] if entries else self._entry(index))
                else:
                    entries.append(self._entry(target))
                    step_valid[k - 1] = 1.0
            patch = stack_history(entries, self.alignments, current_pose=self.pose[index], valid=None)
            od_fut.append(patch["od_hist"])
            ld_fut.append(patch["ld_hist"])
            od_masks.append(patch["od_hist_mask"] * step_valid[:, None])
            ld_masks.append(patch["ld_hist_mask"] * step_valid[:, None])
            valid.append(step_valid)
        return {
            "od_fut": np.stack(od_fut, axis=0).astype(np.float32),
            "ld_fut": np.stack(ld_fut, axis=0).astype(np.float32),
            "od_mask": np.stack(od_masks, axis=0).astype(np.float32),
            "ld_mask": np.stack(ld_masks, axis=0).astype(np.float32),
            "valid": np.stack(valid, axis=0).astype(np.float32),
        }


# --------------------------------------------------------------------------- #
# 阶段 A：world model（教师强制）
# --------------------------------------------------------------------------- #

def _ade_fde(pred_xy: "Any", target_xy: "Any", weight: "Any") -> Tuple["Any", "Any"]:
    """掩码加权 ADE/FDE（``(B,K,16,2)`` + ``(B,K,16)``）；无有效槽位时返回 ``nan``。"""
    import torch

    error = torch.linalg.norm(pred_xy - target_xy, dim=-1)  # (B,K,16)
    weight_sum = weight.sum()
    if float(weight_sum) <= 0.0:
        nan = torch.tensor(float("nan"))
        return nan, nan
    ade = (error * weight).sum() / weight_sum
    fde_weight = weight[:, -1, :]
    if float(fde_weight.sum()) <= 0.0:
        return ade, torch.tensor(float("nan"))
    fde = (error[:, -1, :] * fde_weight).sum() / fde_weight.sum()
    return ade, fde


def _episode_split(
    episode_ids: np.ndarray, val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 episode 切训练/留出（返回 ``(train_idx, val_idx, val_episodes)``，确定性）。"""
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(episodes))
    n_val = int(round(float(val_frac) * len(episodes)))
    n_val = min(max(n_val, 0), len(episodes) - 1) if len(episodes) > 1 else 0
    val_episodes = episodes[order[:n_val]] if n_val > 0 else np.zeros(0, dtype=np.int64)
    if val_episodes.size == 0:
        index = np.arange(episode_ids.shape[0], dtype=np.int64)
        return index, np.zeros(0, dtype=np.int64), val_episodes
    is_val = np.isin(episode_ids, val_episodes)
    return np.where(~is_val)[0], np.where(is_val)[0], val_episodes


def match_future_od_slots(
    future: Mapping[str, np.ndarray],
    current_od: np.ndarray,
    current_mask: np.ndarray,
    *,
    gate_m: float = 8.0,
    dt: float = 0.5,
) -> Dict[str, np.ndarray]:
    """把未来帧 OD 槽位按 **t0 帧最近邻** 匹配到当前槽位（identity association）。

    为什么需要：OD 通道按 ``min(TTC, cap) + 距离`` 排序 → 未来帧的槽位顺序会变
    （同一对象可能从 slot i 换到 slot j）。直接按 index 回归是 ill-posed：实测未匹配时
    匀速基线 ADE ≈ 9–14 m，其中大部分是槽位错配而非物理误差，世界模型的残差头
    也学不到"哪个槽位是哪个对象"。匹配后每个当前槽位的目标是"**该对象**未来在 t0 帧的
    状态"，物理可学（匹配键用 WM 自己的匀速先验，gate 过滤对象离场/新入场）。

    仅改 ``od_fut``/``od_mask``（槽位轴重排 + 有效门控）；``valid``（步存在性）与 LD 不变。
    """
    cur = np.asarray(current_od, dtype=np.float32)  # (B,16,9)
    mask = np.asarray(current_mask, dtype=np.float32)  # (B,16)
    od_fut = np.asarray(future["od_fut"], dtype=np.float32)  # (B,K,16,9)
    od_mask = np.asarray(future["od_mask"], dtype=np.float32)  # (B,K,16)
    batch, steps, slots, _ = od_fut.shape
    horizon = np.arange(1, steps + 1, dtype=np.float32).reshape(1, steps, 1, 1)
    prior = cur[:, None, :, :2] + horizon * float(dt) * cur[:, None, :, 2:4]  # (B,K,16,2)
    distance = np.linalg.norm(prior[:, :, :, None, :] - od_fut[:, :, None, :, :2], axis=-1)  # (B,K,S,S)
    nearest = distance.argmin(axis=-1)  # (B,K,S)
    min_distance = np.take_along_axis(distance, nearest[..., None], axis=-1)[..., 0]  # (B,K,S)
    batch_index = np.arange(batch)[:, None, None]
    step_index = np.arange(steps)[None, :, None]
    matched_feat = od_fut[batch_index, step_index, nearest]  # (B,K,S,9)
    matched_mask = od_mask[batch_index, step_index, nearest]  # (B,K,S)
    out = dict(future)
    out["od_fut"] = matched_feat.astype(np.float32)
    out["od_mask"] = (mask[:, None, :] * matched_mask * (min_distance <= float(gate_m))).astype(np.float32)
    return out


def run_stage_a(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """阶段 A：world model 教师强制训练（专家动作序列为 ego 条件，未来 OD/LD 为目标）。"""
    import torch

    from net.world_model import WorldModel, direct_multi_step_loss

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_thread_limits(workers=1, config=config)
    device = torch.device(resolve_device(args.device, config))
    model = build_model(_load_yaml(args.model_config))
    if args.ckpt and Path(args.ckpt).exists():
        meta = load_checkpoint(args.ckpt, model)
        print(f"[stageA] 载入 {args.ckpt}（missing={len(meta.get('missing_keys', []))}）", flush=True)

    dataset = _limit_dataset(BCDataset.load(args.bc_dir), args.limit_dataset)
    arrays = dataset.arrays
    if "step" not in arrays:
        raise SystemExit("[stageA] BC 数据集缺少 'step'（未来目标需要 (episode, step+k) 查表）")
    windows = FrameWindows(
        arrays,
        dataset.alignments,
        episode_key="episode_id",
        step_key="step",
        keys=_bc_channel_keys(),
        step_stride=_BC_STEP_STRIDE,
    )
    train_idx, val_idx, val_episodes = _episode_split(arrays["episode_id"], float(args.val_frac), int(args.seed))
    if train_idx.size < 2:
        raise SystemExit(f"[stageA] 训练帧不足（{train_idx.size}）")
    if val_idx.size == 0:
        print("[stageA] 警告：留出集为空（episode 数过少/val_frac=0）→ 用训练帧自评", flush=True)
        val_idx = train_idx

    # 可训练：encoders/temporal/spatial/MoE + world model；policy/value 头不参与（v1.1 契约）。
    trainable: List["torch.Tensor"] = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith(("policy.", "value."))
    ]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(trainable, lr=float(args.lr))
    model.to(device).train()

    batch_size = int(args.batch_size or 128)
    noise_std = torch.tensor(
        [float(args.plan_noise_ds), float(args.plan_noise_dtheta)], dtype=torch.float32, device=device
    )
    action_all = np.asarray(arrays["action"], dtype=np.float32)

    def _to_tensor(batch: Mapping[str, np.ndarray]) -> Dict[str, "torch.Tensor"]:
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in batch.items()}

    def _future_targets(batch_indices: np.ndarray, obs_np: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """未来目标；``--match-future-slots`` 时把 OD 槽位按 t0 帧最近邻匹配到当前槽位。"""
        future = windows.build_future(batch_indices)
        if bool(args.match_future_slots):
            future = match_future_od_slots(
                future, obs_np["od"], obs_np["od_mask"], gate_m=float(args.match_gate_m)
            )
        return future

    def _forward_loss(batch_indices: np.ndarray, *, noise: bool) -> "torch.Tensor":
        obs_np = dataset.build_obs_batch(batch_indices)
        obs = _to_tensor(obs_np)
        future = _future_targets(batch_indices, obs_np)
        encoded = model.encode(obs)
        frame = encoded["frame"]
        od_state = WorldModel.od_state_from_features(frame.od_feat, frame.od_mask)
        ld_state = WorldModel.ld_state_from_features(frame.ld_feat, frame.ld_mask)
        plan = torch.as_tensor(action_all[batch_indices], dtype=torch.float32, device=device)
        if noise and float(args.plan_noise_p) > 0.0:
            hit = (torch.rand_like(plan) < float(args.plan_noise_p)).to(plan.dtype)
            plan = plan + torch.randn_like(plan) * noise_std * hit
        od_pred, ld_pred, _ = model.world_model(encoded["latent"], plan, od_state, ld_state)
        od_target, ld_target = WorldModel.targets_from_features(
            torch.as_tensor(future["od_fut"], device=device),
            torch.as_tensor(future["ld_fut"], device=device),
        )
        return direct_multi_step_loss(
            od_pred,
            ld_pred,
            od_target,
            ld_target,
            torch.as_tensor(future["od_mask"], device=device),
            torch.as_tensor(future["ld_mask"], device=device),
            torch.as_tensor(future["valid"], device=device),
        )

    @torch.no_grad()
    def _evaluate(indices: np.ndarray, limit: int) -> Dict[str, float]:
        model.eval()
        eval_idx = np.asarray(indices, dtype=np.int64)[: max(1, int(limit))]
        loss = float(_forward_loss(eval_idx, noise=False))
        obs_np = dataset.build_obs_batch(eval_idx)
        obs = _to_tensor(obs_np)
        future = _future_targets(eval_idx, obs_np)
        encoded = model.encode(obs)
        frame = encoded["frame"]
        od_state = WorldModel.od_state_from_features(frame.od_feat, frame.od_mask)
        ld_state = WorldModel.ld_state_from_features(frame.ld_feat, frame.ld_mask)
        plan = torch.as_tensor(action_all[eval_idx], dtype=torch.float32, device=device)
        od_pred, ld_pred, _ = model.world_model(encoded["latent"], plan, od_state, ld_state)
        weight = torch.as_tensor(future["od_mask"], device=device)
        od_target = torch.as_tensor(future["od_fut"], device=device)[..., 0:2]
        model_ade, model_fde = _ade_fde(od_pred[..., 0:2], od_target, weight)
        # 匀速基线（t0 帧内：位置 + k·dt·相对速度；未学习时 WM 的先验与其同源）
        current = obs["od"]
        offset = current[..., 0:2].unsqueeze(1) + current[..., 2:4].unsqueeze(1) * 0.5 * torch.arange(
            1, 7, device=device
        ).reshape(1, 6, 1, 1)
        cv_ade, cv_fde = _ade_fde(offset, od_target, weight)
        model.train()
        return {
            "eval_loss": loss,
            "model_ade": float(model_ade),
            "model_fde": float(model_fde),
            "cv_ade": float(cv_ade),
            "cv_fde": float(cv_fde),
        }

    stage_cfg = _stage_section(config, "A")
    wm_cfg = dict(stage_cfg.get("world_model", {}) or {})
    epochs = int(args.wm_epochs or wm_cfg.get("epochs") or 10)
    metrics: Dict[str, Any] = {
        "stage": "A",
        "kind": "world_model_teacher_forcing",
        "bc_dir": str(args.bc_dir),
        "samples": int(dataset.count),
        "train_frames": int(train_idx.size),
        "val_frames": int(val_idx.size),
        "val_episodes": int(val_episodes.size),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": float(args.lr),
        "match_future_slots": bool(args.match_future_slots),
        "match_gate_m": float(args.match_gate_m),
        "plan_noise": {"p": float(args.plan_noise_p), "ds": float(args.plan_noise_ds), "dtheta": float(args.plan_noise_dtheta)},
    }
    monitor = _make_monitor(out_dir / "monitor", enabled=_monitor_enabled(args, config))
    rng = np.random.default_rng(int(args.seed))
    loss_curve: List[float] = []
    for epoch in range(epochs):
        order = rng.permutation(train_idx)
        totals, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            loss = _forward_loss(batch_indices, noise=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            totals += float(loss.detach())
            batches += 1
            if args.max_batches and batches >= int(args.max_batches):
                break
        train_loss = totals / max(1, batches)
        loss_curve.append(train_loss)
        eval_metrics = _evaluate(val_idx, int(args.eval_frames))
        metrics.update(
            {
                "last_epoch": epoch + 1,
                "wm_loss": train_loss,
                "loss_curve": list(loss_curve),
                "val_loss": eval_metrics["eval_loss"],
                "model_ade": eval_metrics["model_ade"],
                "model_fde": eval_metrics["model_fde"],
                "cv_ade": eval_metrics["cv_ade"],
                "cv_fde": eval_metrics["cv_fde"],
                "beats_cv_ade": bool(eval_metrics["model_ade"] < eval_metrics["cv_ade"]),
                "beats_cv_fde": bool(eval_metrics["model_fde"] < eval_metrics["cv_fde"]),
                "batches": batches,
            }
        )
        if monitor is not None:
            monitor.on_train_step(
                {
                    "wm_loss": train_loss,
                    "val_loss": eval_metrics["eval_loss"],
                    "model_ade": eval_metrics["model_ade"],
                    "model_fde": eval_metrics["model_fde"],
                    "cv_ade": eval_metrics["cv_ade"],
                    "cv_fde": eval_metrics["cv_fde"],
                },
                step=epoch + 1,
            )
            monitor.flush(step=epoch + 1)
        print(
            f"[stageA] epoch {epoch + 1}/{epochs} loss={train_loss:.4f} val={eval_metrics['eval_loss']:.4f} "
            f"ADE(WM/CV)={eval_metrics['model_ade']:.3f}/{eval_metrics['cv_ade']:.3f} "
            f"FDE={eval_metrics['model_fde']:.3f}/{eval_metrics['cv_fde']:.3f}",
            flush=True,
        )
    if monitor is not None:
        monitor.close()
    save_checkpoint(out_dir / "world_model.pt", model, meta=metrics)
    save_checkpoint(out_dir / "final.pt", model, meta={"stage": "A", "epochs": epochs})
    metrics["checkpoint"] = str(out_dir / "final.pt")
    _write_json(out_dir / "metrics.json", metrics)
    print(f"[stageA] DONE → {out_dir}", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# 阶段 B：planner BC（primary → specific）
# --------------------------------------------------------------------------- #

#: primary 段冻结：WM（阶段 A 产物）、value 头、8 个 specific experts
_PRIMARY_PHASE_FREEZE: Tuple[str, ...] = ("world_model.", "value.", "moe.experts.")
#: specific 段冻结：primary+shared backbone 冻结，只训练 8 个 experts（+router+policy head）
_SPECIFIC_PHASE_FREEZE: Tuple[str, ...] = (
    "world_model.",
    "value.",
    "moe.primary.",
    "encoders.",
    "temporal.",
    "spatial.",
    "latent_mlp.",
    "latent_norm.",
)


def _action_mu_stats(model: Any, dataset: BCDataset, device: Any, *, batch_size: int = 256) -> Dict[str, float]:
    """全数据集确定性前向（cheap path）：``action_mu`` 的 ds 均值 + 专家 ds 均值。"""
    import torch

    model.eval()
    total_ds, total_dtheta, count = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, dataset.count, max(1, int(batch_size))):
            indices = np.arange(start, min(start + max(1, int(batch_size)), dataset.count), dtype=np.int64)
            obs = {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in
                   dataset.build_obs_batch(indices).items()}
            out = model(obs, rollout=False, world_model=False)
            mu = out["action_mu"]
            total_ds += float(mu[:, 0].sum())
            total_dtheta += float(mu[:, 1].sum())
            count += int(mu.shape[0])
    expert = np.asarray(dataset.arrays["action"], dtype=np.float32)[:, 0, :]
    return {
        "action_mu_ds_mean": total_ds / max(count, 1),
        "action_mu_dtheta_mean": total_dtheta / max(count, 1),
        "expert_action_ds_mean": float(expert[:, 0].mean()),
        "expert_action_dtheta_mean": float(expert[:, 1].mean()),
    }


def run_stage_b(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """阶段 B：planner BC（动作主损失 + rollout 轨迹辅助（WM detach）+ router BCE）。"""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_thread_limits(workers=1, config=config)
    device = resolve_device(args.device, config)
    model = build_model(_load_yaml(args.model_config))
    ckpt = args.ckpt or "runs/train/stage_a/final.pt"
    if Path(ckpt).exists():
        meta = load_checkpoint(ckpt, model)
        print(f"[stageB] 载入阶段 A 产物 {ckpt}（missing={len(meta.get('missing_keys', []))}）", flush=True)
    else:
        print(f"[stageB] 警告：checkpoint {ckpt} 不存在 → 从随机初始化开始（world model 未训练）", flush=True)

    dataset = _limit_dataset(BCDataset.load(args.bc_dir), args.limit_dataset)
    stage_cfg = _stage_section(config, "B")
    bc_cfg = dict(stage_cfg.get("bc", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    epochs = int(args.bc_epochs or bc_cfg.get("epochs") or 10)
    split = float(args.bc_phase_split if args.bc_phase_split is not None else bc_cfg.get("primary_phase_split", 0.5))
    primary_epochs = max(1, int(round(epochs * split))) if epochs > 1 else epochs
    specific_epochs = max(0, epochs - primary_epochs)
    action_weight = float(args.action_weight if args.action_weight is not None else bc_cfg.get("action_weight", 1.0))
    traj_weight = float(args.traj_aux_weight if args.traj_aux_weight is not None else bc_cfg.get("traj_aux_weight", 0.1))
    loss_type = str(args.loss_type if args.loss_type is not None else bc_cfg.get("loss_type", "l2"))
    router_coef = float(args.router_coef if args.router_coef is not None else bc_cfg.get("router_coef", 0.1))
    batch_size = int(args.batch_size or (train_cfg.get("bc", {}) or {}).get("batch_size") or 256)

    # WM 冻结：阶段 A 已训练；rollout 内输出再 detach（wm_detach=True）→ 轨迹辅助损失不回传 WM。
    apply_freeze_prefixes(model, ("world_model.",))
    metrics: Dict[str, Any] = {
        "stage": "B",
        "kind": "planner_bc",
        "bc_dir": str(args.bc_dir),
        "samples": int(dataset.count),
        "device": device,
        "epochs": epochs,
        "primary_epochs": primary_epochs,
        "specific_epochs": specific_epochs,
        "batch_size": batch_size,
        "lr": float(args.lr),
        "action_weight": action_weight,
        "traj_aux_weight": traj_weight,
        "router_coef": router_coef,
        "wm_detach": True,
    }
    monitor = _make_monitor(out_dir / "monitor", enabled=_monitor_enabled(args, config))

    def _run_phase(phase: str, phase_epochs: int, freeze_prefixes: Sequence[str]) -> Dict[str, Any]:
        if phase_epochs <= 0:
            return {"skipped": True, "epochs": 0, "phase": phase}
        cfg = BCConfig(
            epochs=phase_epochs,
            batch_size=batch_size,
            lr=float(args.lr),
            loss_type=loss_type,
            traj_weight=traj_weight,
            action_weight=action_weight,
            router_coef=router_coef,
            seed=int(args.seed),
            device=device,
            max_batches=args.max_batches,
            wm_detach=True,
            freeze_prefixes=tuple(freeze_prefixes),
            phase=phase,
        )
        result = pretrain_bc(model, dataset, cfg, logger=print)
        if monitor is not None:
            monitor.on_train_step(
                {f"{phase}_{key}": value for key, value in result.items() if isinstance(value, (int, float))},
                step=phase_epochs,
            )
            monitor.flush(step=phase_epochs)
        return result

    metrics["primary"] = _run_phase("primary", primary_epochs, _PRIMARY_PHASE_FREEZE)
    metrics["specific"] = _run_phase("specific", specific_epochs, _SPECIFIC_PHASE_FREEZE)
    metrics.update(_action_mu_stats(model, dataset, device, batch_size=batch_size))
    if monitor is not None:
        monitor.close()
    save_checkpoint(out_dir / "bc.pt", model, meta=metrics)
    save_checkpoint(out_dir / "final.pt", model, meta={"stage": "B", "epochs": epochs})
    metrics["checkpoint"] = str(out_dir / "final.pt")
    _write_json(out_dir / "metrics.json", metrics)
    print(
        f"[stageB] DONE → {out_dir} action_mu_ds={metrics['action_mu_ds_mean']:.3f}m "
        f"（专家 {metrics['expert_action_ds_mean']:.3f}m）",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------------- #
# 阶段 C：PPO RL（Stage-B 快照 KL 锚 + 衰减）
# --------------------------------------------------------------------------- #

def run_stage_c(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """PPO：LqrTracker 闭环 + Stage-B 快照 KL 锚（衰减）+ primary lr ×0.1 + WM 冻结/解冻。"""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _resolve_specs(args, config)
    if not specs:
        raise SystemExit("[stages] 阶段 C 没有可用 spec")
    apply_thread_limits(workers=1, config=config)
    train_cfg = dict(config.get("train", {}) or {})
    device = resolve_device(args.device, config)
    model = build_model(_load_yaml(args.model_config))
    ckpt = args.ckpt or "runs/train/stage_b/final.pt"
    if Path(ckpt).exists():
        load_checkpoint(ckpt, model)
        print(f"[stageC] 载入阶段 B 策略快照 {ckpt}", flush=True)
    else:
        print(f"[stageC] 警告：checkpoint {ckpt} 不存在 → 从随机初始化开始", flush=True)
    stage_cfg = _stage_section(config, "C")
    primary_lr_scale = float(stage_cfg.get("primary_lr_scale", 0.1) or 0.1)
    router_coef = float(args.router_coef if args.router_coef is not None else stage_cfg.get("router_coef", 0.1))
    updates = int(args.updates)
    # KL 锚 = 阶段 B 快照（冻结参考模型）；系数线性衰减（默认 0.05 → 0）
    ref_model = copy.deepcopy(model).eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    kl_initial = float(args.kl_anchor_coef)
    kl_final = float(args.kl_anchor_final_coef)
    kl_decay = bool(args.kl_anchor_decay)
    # WM 初始冻结：--wm-freeze-updates（None = updates//4）；解冻 = 参数重新进入优化器
    wm_freeze_updates = args.wm_freeze_updates
    if wm_freeze_updates is None:
        wm_freeze_updates = max(1, updates // 4) if updates > 0 else 0
    wm_freeze_updates = int(wm_freeze_updates)
    if wm_freeze_updates > 0:
        apply_freeze_prefixes(model, ("world_model.",))

    bc_dataset = None
    if args.bc_anchor and Path(args.bc_dir).exists():
        bc_dataset = BCDataset.load(args.bc_dir)
        print(f"[stageC] BC 锚：{bc_dataset.count} 样本", flush=True)

    pool = build_pool(
        specs,
        kind=str(args.pool),
        num_envs=int(args.envs),
        tracker="lqr",
        traffic_density=args.traffic_density,
        mem_floor_mb=args.mem_floor_mb,
        logger=print,
    )
    metrics: Dict[str, Any] = {
        "stage": "C",
        "specs": len(specs),
        "primary_lr_scale": primary_lr_scale,
        "kl_anchor": {"initial": kl_initial, "final": kl_final, "decay": kl_decay, "source": str(ckpt)},
        "wm_freeze_updates": wm_freeze_updates,
        "device": device,
    }
    monitor = _make_monitor(out_dir / "monitor", enabled=_monitor_enabled(args, config))
    try:
        reward_adapter, reward_source = build_reward_adapter(logger=print)
        ppo_cfg = PPOConfig(
            lr=float(args.lr),
            epochs=int(args.ppo_epochs),
            minibatch_size=int(args.minibatch_size or (train_cfg.get("ppo", {}) or {}).get("minibatch_size") or 1024),
            kl_anchor_coef=kl_initial,
            router_coef=router_coef,
            bc_anchor_coef=float(args.bc_anchor_coef) if bc_dataset is not None else 0.0,
            primary_lr_scale=primary_lr_scale,
            seed=int(args.seed),
            device=device,
        )
        trainer = PPOTrainer(
            model,
            pool,
            ppo_cfg,
            reward_adapter=reward_adapter,
            ref_model=ref_model,
            bc_dataset=bc_dataset,
            logger=print,
        )
        initial = pool.reset()
        trainer.adopt_obs([record["obs"] for record in initial])
        history: List[Dict[str, Any]] = []
        kl_schedule: List[float] = []
        for update in range(updates):
            update_started = time.perf_counter()
            # WM 解冻：requires_grad 打开 + 参数加入优化器（保留已有 Adam 状态）
            if wm_freeze_updates > 0 and update == wm_freeze_updates:
                apply_freeze_prefixes(model, ())
                wm_params = [
                    parameter for name, parameter in model.named_parameters() if name.startswith("world_model.")
                ]
                existing = {
                    id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
                }
                fresh = [p for p in wm_params if id(p) not in existing and p.requires_grad]
                if fresh:
                    trainer.optimizer.add_param_group({"params": fresh, "lr": float(args.lr)})
                print(f"[stageC] update {update}: world model 解冻（{len(fresh)} 个参数进入优化器）", flush=True)
            progress = update / max(updates - 1, 1) if updates > 1 else 0.0
            trainer.config.kl_anchor_coef = kl_initial + (kl_final - kl_initial) * progress if kl_decay else kl_initial
            kl_schedule.append(float(trainer.config.kl_anchor_coef))
            trainer.collect_rollout(int(args.rollout_steps))
            update_metrics = trainer.update()
            update_metrics["kl_anchor_coef"] = float(trainer.config.kl_anchor_coef)
            trim_memory()
            history.append(update_metrics)
            if monitor is not None:
                monitor.on_train_step(update_metrics, step=update + 1)
                monitor.flush(step=update + 1)
            print(
                f"[stageC] update {update + 1}/{updates} loss={update_metrics['total_loss']:+.3f} "
                f"kl_anchor={update_metrics.get('kl_anchor', 0.0):.4f} coef={update_metrics['kl_anchor_coef']:.4f} "
                f"router={update_metrics.get('router_loss', 0.0):.4f} "
                f"steps/s={int(args.rollout_steps) * int(getattr(pool, 'num_envs', 1)) / max(time.perf_counter() - update_started, 1e-9):.1f}",
                flush=True,
            )
        metrics["ppo_updates"] = len(history)
        metrics["kl_anchor_coef_schedule"] = kl_schedule
        metrics["last"] = {key: history[-1].get(key) for key in ("total_loss", "approx_kl", "kl_anchor")}
        metrics["reward_source"] = reward_source
    finally:
        pool.close()
        if monitor is not None:
            monitor.close()
    save_checkpoint(out_dir / "final.pt", model, meta={"stage": "C"})
    metrics["checkpoint"] = str(out_dir / "final.pt")
    _write_json(out_dir / "metrics.json", metrics)
    print(f"[stageC] DONE → {out_dir}", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipeline.stages", description="阶段 A/B/C 编排（v1.1）")
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True,
                        help="A=WM 教师强制训练；B=planner BC（primary→specific）；C=PPO RL")
    parser.add_argument("--config", default="config/default.yaml", help="主配置（includes 合并）")
    parser.add_argument("--model-config", default=_DEFAULT_MODEL_CFG)
    parser.add_argument("--spec", default=None, help="阶段 C 训练 spec（默认取 config data.spec）")
    parser.add_argument("--geometry", default=None, help="阶段 C 按主几何标签过滤（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="阶段 C spec 数上限")
    parser.add_argument("--out", default="runs/train")
    parser.add_argument("--ckpt", default=None,
                        help="A=初始化权重；B=阶段 A 产物（默认 runs/train/stage_a/final.pt）；"
                             "C=阶段 B 快照（默认 runs/train/stage_b/final.pt）")
    parser.add_argument("--bc-dir", default="runs/bc_expert_full", help="阶段 A/B 的 BC 专家数据集目录")
    parser.add_argument("--limit-dataset", type=int, default=None, help="BC 数据集前缀抽样上限（冒烟用）")
    # ---- 阶段 A ----
    parser.add_argument("--wm-epochs", type=int, default=None, help="阶段 A world model 轮数（默认 config/stages.A.world_model.epochs=10）")
    parser.add_argument("--val-frac", type=float, default=0.15, help="阶段 A 按 episode 留出比例（默认 0.15）")
    parser.add_argument("--eval-frames", type=int, default=256, help="阶段 A 每轮评估帧数")
    parser.add_argument("--plan-noise", action=argparse.BooleanOptionalAction, default=True,
                        help="阶段 A ego plan 噪声增强（默认开）")
    parser.add_argument("--plan-noise-p", type=float, default=0.5, help="噪声命中概率（逐元素 Bernoulli）")
    parser.add_argument("--plan-noise-ds", type=float, default=0.3, help="ds 噪声标准差（m/0.5s）")
    parser.add_argument("--plan-noise-dtheta", type=float, default=0.05, help="dθ 噪声标准差（rad/0.5s）")
    parser.add_argument("--match-future-slots", action=argparse.BooleanOptionalAction, default=True,
                        help="阶段 A 未来 OD 槽位按 t0 帧最近邻匹配（默认开；关闭 = 原始 TTC 排序槽位）")
    parser.add_argument("--match-gate-m", type=float, default=8.0, help="槽位匹配距离门限（m）")
    # ---- 阶段 B ----
    parser.add_argument("--bc-epochs", type=int, default=None, help="阶段 B 总轮数（默认 config/stages.B.bc.epochs=10）")
    parser.add_argument("--bc-phase-split", type=float, default=None,
                        help="阶段 B primary 段占比（默认 config/stages.B.bc.primary_phase_split=0.5）")
    parser.add_argument("--action-weight", type=float, default=None, help="阶段 B 动作损失权重（默认 1.0，主项）")
    parser.add_argument("--traj-aux-weight", type=float, default=None, help="阶段 B rollout 轨迹辅助权重（默认 0.1）")
    parser.add_argument("--loss-type", choices=("l1", "l2"), default=None,
                        help="阶段 B 损失口径（默认取 config/stages.B.bc.loss_type=l2；l1 可减轻转向回归均值）")
    # ---- 阶段 C ----
    parser.add_argument("--pool", choices=("auto", "vector", "local"), default="local")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=None,
                        help="默认取 config/train.yaml train.ppo.minibatch_size（1024）")
    parser.add_argument("--kl-anchor-coef", type=float, default=0.05, help="阶段 C KL 锚初始系数（默认 0.05）")
    parser.add_argument("--kl-anchor-final-coef", type=float, default=0.0, help="阶段 C KL 锚末值（默认 0）")
    parser.add_argument("--kl-anchor-decay", action=argparse.BooleanOptionalAction, default=True,
                        help="阶段 C KL 锚系数线性衰减到末值（默认开）")
    parser.add_argument("--wm-freeze-updates", type=int, default=None,
                        help="阶段 C 前 N 个 update 冻结 WM（默认 updates//4；0=不冻结）")
    parser.add_argument("--bc-anchor", action="store_true", help="阶段 C 启用 BC 动作锚（默认关）")
    parser.add_argument("--bc-anchor-coef", type=float, default=0.1)
    # ---- 通用 ----
    parser.add_argument("--batch-size", type=int, default=None,
                        help="A 默认 128；B 默认取 config/train.yaml train.bc.batch_size（256）")
    parser.add_argument("--max-batches", type=int, default=None, help="A/B 每轮批数上限（冒烟用）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--router-coef", type=float, default=None, help="B/C router BCE 权重（默认 0.1）")
    parser.add_argument("--traffic-density", type=float, default=None)
    parser.add_argument("--mem-floor-mb", type=float, default=2000.0)
    parser.add_argument("--device", default=None,
                        help="默认取 config/train.yaml train.device（auto=cuda 可用则 cuda，否则 cpu）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=None,
                        help="使用 pipeline.monitoring（tensorboard/CSV）；默认读 config/train.yaml::monitoring")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    if args.stage == "A":
        run_stage_a(args, config)
    elif args.stage == "B":
        run_stage_b(args, config)
    else:
        run_stage_c(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
