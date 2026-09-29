"""阶段 A/B/C 编排（v1.2：WM 教师强制 → planner BC → PPO RL）。

阶段语义（2026-09-26 schema v2 修订）
------------------------------------
- **A = world model 训练（教师强制）**：ego 条件 = 专家 GT 动作序列 + **目标帧真实 ego**
  （``build_future.ego_fut``，退回解析运动学）；目标 = ``(episode_id, base+5k)`` 精确查表
  的未来 OD 帧（对齐到 t0、``od_id`` 身份匹配、``wm_valid`` 门控）。
  **直接多步 OD 损失**（``train_weight × wm_valid`` 显式加权）+ **未来 LD 直接多步损失**
  （``ld_fut`` 前 4 维 + ``ld_mask``/``wm_valid`` 掩码，与 OD 同构；权重
  ``stages.A.world_model.ld_coef`` **默认 0.0**（2026-09-30 拍板）——LD 损失仅监控、不监督 LD 头；
  可经配置/``--wm-ld-coef`` 开启）+ **plan head ``ego_next``
  监督**（方案①：每步挤入 GT 帧前用同一 mem 预测第 k 帧 ego 前 6 维，plan head/MoE 因此在
  A 阶段有梯度，design-v1.2 §2.3）+ **presence/entry BCE + AUC（id 轴，见
  :func:`presence_entry_targets`）**。
  监控（Tier-1，lane B）：``loss/wm|od|ld|ego_next|presence|entry``（训练目标分解，宏 batch 均值）+
  OD KPI ``val/od/ade_m|fde_m/h*``（含 ``cv_h*`` 匀速基线）+ ``val/od/presence_auc|entry_auc`` +
  ego KPI ``val/ego/action/err_weighted`` / ``val/ego/traj/mae_m/h*`` / ``val/ego/traj/fde_m``；
  val 口径 loss 曲线/有效样本计数等已移除（``docs/metrics.md``）。
  可训练：encoders/mem-encoder/plan head/MoE/ST-GNN；policy/value 头不参与。
- **B = planner BC**：可训练 backbone+MoE+policy head，**primary→specific** 两段。
  损失 = **首步动作**（``action_mu`` vs 专家即时动作，权重感知）+ **6 点 rollout 轨迹辅助**
  （``traj_xy`` vs 专家 ``traj6``，WM 冻结；新 net 的合成帧 detach 语义）+ **router 硬标签
  **无聚类监督**（lane U1：无簇标签/router CE；MoE 负载均衡 aux + worst/mild 权重化 specific）。
  监控（Tier-1，lane B）：``loss/planner/<phase>/{total,traj,action,router}``（留出
  ``val/loss/planner/...``）、``ego/action/err_weighted``、``ego/traj/mae_m|fde_m``（留出
  ``val/ego/...``）、``router/{ce,acc,acc_majority}``（留出 ``val/router/...``）；动作误差
  median/p95、全部 slice/label、软目标 KL/温度/专家混合权重等已移除（``docs/metrics.md``）。
- **C = PPO RL（实验口径）**：P0-1（整条 plan 执行 vs 首动作记账 / WM 解冻无信号）已按
  **A-hold** 修复（references = repeat(a_t)；WM 全期冻结），P0-2（router 标签错位）已对齐
  （标签步前取 + 终局 record/reset 分离）；W2 WM 自监督未接入；P2 性能优化进行中
  （V9：``plan_reference=repeat_action`` 时 collect 走 cheap path）。KL 锚 =
  阶段 B 快照（``--ckpt``）系数线性衰减；primary lr ×0.1；critic warmup。

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

（串行 A→B + 冻结评测协议见 ``tools/train.sh`` / ``tools/test.sh``。）
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 允许 `python pipeline/stages.py` 直接运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline.hard_mining import ckpt_identity, row_il_errors  # noqa: E402
from pipeline.trainer import (  # noqa: E402
    BCConfig,
    BCDataset,
    DEFAULT_PROBE_BATCH,
    DEFAULT_STAGE_BATCH_SIZE,
    MaterializedBCDataset,
    PHASE3_WM_LOSS_KEYS,
    Phase3Config,
    PPOConfig,
    PPOTrainer,
    STAGE_C_DESIGN_PREFIXES,
    STAGE_C_TRAINABLE_SCOPES,
    apply_freeze_prefixes,
    apply_thread_limits,
    apply_trainable_allowlist,
    build_pool,
    build_reward_adapter,
    capture_rng_state,
    config_snapshot_hash,
    dataset_weight_report,
    ego_kpi_arrays,
    load_checkpoint,
    load_optimizer_state,
    load_training_checkpoint,
    move_optimizer_state_to_device,
    presence_entry_loss,
    pretrain_bc,
    pretrain_bc_phase3,
    resolve_device,
    restore_rng_state,
    row_action_weights,
    sanitize_masked_od,
    save_checkpoint,
    stack_history,
    squeeze_single_slot,
    to_device_tensor,
    to_device_tensors,
    trainable_param_groups,
    trim_memory,
    weighted_ld_multi_step_loss,
    weighted_od_multi_step_loss,
)

__all__ = ["main", "run_stage_a", "run_stage_b", "run_stage_b_phase3", "run_stage_c", "load_config",
           "build_model", "FrameWindows", "validate_bc_dataset"]

_DEFAULT_MODEL_CFG = "config/model.yaml"


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


def _make_monitor(log_dir: Path, enabled: bool, *, legacy_tags: bool = False) -> Optional[Any]:
    """构造 :class:`pipeline.monitoring.TrainingMonitor`（``legacy_tags`` = 回退旧 tag 口径）。"""
    if not enabled:
        return None
    try:
        from pipeline.monitoring import TrainingMonitor

        return TrainingMonitor(str(log_dir), legacy_tags=legacy_tags)
    except Exception as exc:  # noqa: BLE001
        print(f"[stages] monitoring 不可用（{type(exc).__name__}: {exc}）→ 跳过", flush=True)
        return None


def _monitor_legacy_tags(args: argparse.Namespace) -> bool:
    """``--monitor-legacy-tags``（默认关）：监控回退到旧 tag/旧写入行为。"""
    return bool(getattr(args, "monitor_legacy_tags", False))


def _wm_trainable(model: Any) -> bool:
    """W1 fail-fast 判据：``st_gnn.*`` 是否存在可训练参数（阶段 C 期望恒 False）。"""
    return any(parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith("st_gnn."))


#: 监控瘦身清单（docs/metrics.md）：这些阶段结果键不再进 metrics.json（CSV/TB 侧由
#: ``pipeline.monitoring`` 的白名单过滤）。保留项示例：``bc_traj_mse*``（损失口径字段）、
#: ``bc_load_*``（MoE 负载元数据）。
_SLIMMED_RESULT_RES = (
    re.compile(r"^bc_action_err_slice_"),
    re.compile(r"^bc_action_err_label_"),
    re.compile(r"^bc_action_err_(median|p95)$"),
    re.compile(r"^bc_router_expert_(util|mix_util)_\d+$"),
)


def _slim_phase_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """递归剔除已移除的切片/分位数/利用率键（metrics.json 只保留损失/KPI 口径字段）。"""
    slim: Dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(key, str) and any(rx.match(key) for rx in _SLIMMED_RESULT_RES):
            continue
        if key in ("slices", "labels"):
            continue
        slim[key] = _slim_phase_result(value) if isinstance(value, Mapping) else value
    return slim


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


def _resolve_val_dir(args: argparse.Namespace, bc_dir: Any = None) -> Optional[Path]:
    """独立留出目录（lane T）：``--val-dir`` 显式优先；否则在 **train-dir 同级**探测
    ``*_expert500val``（同一数据集族，多个取 mtime 最新）；都没有 → ``None``（调用方告警并
    回退 legacy 比例切分）。

    只在 train-dir 的同级目录探测（**不扫仓库 datasets/**）：避免测试/临时数据集误用无关的
    真实 val 集，也避免跨数据集族串用。
    """
    explicit = getattr(args, "val_dir", None)
    if explicit:
        path = Path(str(explicit))
        if not path.exists():
            raise SystemExit(f"[val] --val-dir 不存在：{path}")
        return path
    if not bc_dir:
        return None
    root = Path(str(bc_dir)).resolve().parent
    candidates = [item for item in root.glob("*_expert500val") if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _merge_dagger_rows(
    dataset: Any,
    train_idx: np.ndarray,
    worst_flags: Optional[np.ndarray],
    dagger_dirs: Sequence[str],
    *,
    logger: Any = print,
) -> "tuple[Any, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]":
    """lane U1/U5：把**多份** DAgger 数据集行并入 phase 2 训练集（行权重 = 1.0，与 worst 行同权）。

    - 同 schema 契约：每份都必须覆盖主集的全部数组键（缺键 → ``SystemExit``）；
    - ``episode_id`` **累积偏移**（主集 max+1 起、逐份递增，互不冲突）；
    - 返回 ``(merged, merged_train_idx, merged_worst, traj_aux_valid, info)``：
      merged 行 = ``[主集全部行, dagger1 全部行, dagger2 全部行, …]``；dagger 行 ``worst=1``；
    - **traj-aux 掩码（lane U4）**：``traj_aux_valid`` 主集 1.0 + **所有 dagger 行 0.0**
      （dagger 的 ``traj6/traj30`` 是常量外推合成值，不得进 traj 损失）；
    - **防线（lane U5）**：每份 dagger 目录若能从 ``report.json`` / ``expert_bc.meta.json``
      读到 spec 来源，则校验其不含 eval/val（命中 → ``SystemExit``；字段缺失 → 警告日志）。
    """
    from env.scenario.spec import load_specs

    from pipeline.trainer import BCDataset
    from tools.dagger_collect import (
        _looks_like_eval_or_val,
        assert_no_eval_val_overlap,
        dagger_spec_sources,
    )

    dirs = [str(item) for item in (dagger_dirs or []) if str(item).strip()]
    if not dirs:
        raise SystemExit("[stageB] --dagger-dir 为空（应给至少一个 dagger 数据集目录）")
    dagger_datasets: list = []
    per_dir: list = []
    guard_records: list = []
    for dagger_dir in dirs:
        # 防线：spec 来源可读 → 过隔离守卫；不可读 → 警告（不阻塞，但留痕）
        sources = dagger_spec_sources(dagger_dir)
        if not sources:
            logger(
                f"[stageB] 警告：DAgger 目录 {dagger_dir} 读不到 spec 来源"
                "（report.json/expert_bc.meta.json 缺字段）→ 无法校验隔离，请自行确认为 train spec 数据"
            )
        for source in sources:
            path = Path(str(source))
            if path.is_file():
                guard_records.append(
                    assert_no_eval_val_overlap(
                        list(load_specs(str(path))), context=f"dagger:{dagger_dir}:{source}"
                    )
                )
            elif _looks_like_eval_or_val(source):
                raise SystemExit(
                    f"[stageB] DAgger 目录 {dagger_dir} 记录的 spec 来源疑似 eval/val：{source}"
                    "（文件缺失无法复核）→ 拒绝合并（训练数据只允许取自 train spec）"
                )
            else:
                logger(f"[stageB] 警告：DAgger 目录 {dagger_dir} 的 spec 来源不存在：{source}（跳过隔离校验）")
        dagger = BCDataset.load(dagger_dir)
        missing = [key for key in dataset.arrays if key not in dagger.arrays]
        if missing:
            raise SystemExit(
                f"[stageB] --dagger-dir {dagger_dir} 与主集 schema 不一致（缺键 {missing[:6]}）→ 拒绝合并"
                f"（DAgger 必须与 BC 同 schema；见 tools/dagger_collect.py）"
            )
        dagger_datasets.append(dagger)
        per_dir.append({"path": dagger_dir, "rows": int(dagger.count)})

    n_train = int(dataset.count)
    # episode_id 累积偏移（主集 max+1 起，逐份递增）
    offsets: list = []
    running = int(np.max(np.asarray(dataset.arrays["episode_id"]))) + 1
    for dagger in dagger_datasets:
        offsets.append(int(running))
        running += int(np.max(np.asarray(dagger.arrays["episode_id"]))) + 1

    merged_arrays: Dict[str, np.ndarray] = {}
    for key, value in dataset.arrays.items():
        left = np.asarray(value)
        parts = [left]
        for dagger, offset in zip(dagger_datasets, offsets):
            right = np.asarray(dagger.arrays[key])
            if key == "episode_id":
                right = right + int(offset)
            parts.append(right.astype(left.dtype))
        merged_arrays[key] = np.concatenate(parts, axis=0)
    merged = BCDataset(merged_arrays, dict(dataset.meta))

    idx_parts = [np.asarray(train_idx, dtype=np.int64)]
    worst_parts = [
        np.asarray(worst_flags, dtype=np.float32)
        if worst_flags is not None
        else np.full(n_train, -1.0, dtype=np.float32)
    ]
    valid_parts = [np.ones(n_train, dtype=np.float32)]
    row_base = n_train
    for index, (dagger, offset) in enumerate(zip(dagger_datasets, offsets)):
        rows = int(dagger.count)
        idx_parts.append(np.arange(row_base, row_base + rows, dtype=np.int64))
        worst_parts.append(np.ones(rows, dtype=np.float32))
        valid_parts.append(np.zeros(rows, dtype=np.float32))
        per_dir[index]["episode_id_offset"] = int(offset)
        per_dir[index]["row_base"] = int(row_base)
        row_base += rows
    merged_train_idx = np.concatenate(idx_parts)
    merged_worst = np.concatenate(worst_parts)
    traj_aux_valid = np.concatenate(valid_parts)
    total_dagger = int(sum(int(dagger.count) for dagger in dagger_datasets))
    info = {
        "dagger_dirs": per_dir,
        "dagger_rows": total_dagger,
        "train_rows": n_train,
        "episode_id_offset": int(offsets[0]),
        "episode_id_offsets": [int(item) for item in offsets],
        "dagger_weight": 1.0,
        "traj_aux_masked_rows": total_dagger,
        "isolation_guard": guard_records,
    }
    for item in per_dir:
        logger(
            f"[stageB] DAgger 行合并：{item['path']} → {item['rows']} 行"
            f"（episode_id 偏移 +{item['episode_id_offset']}，merged 行 [{item['row_base']},"
            f"{item['row_base'] + item['rows']})；权重 = 1.0）"
        )
    logger(
        f"[stageB] DAgger 合并汇总：主集 {n_train} + DAgger {total_dagger}（{len(dirs)} 份）= "
        f"{n_train + total_dagger} 行；traj-aux 掩码：DAgger {total_dagger} 行 = 0"
        f"；隔离守卫 {len(guard_records)} 项通过"
    )
    return merged, merged_train_idx, merged_worst, traj_aux_valid, info


def _stage_section(config: Mapping[str, Any], stage: str) -> Dict[str, Any]:
    section = (config.get("stages", {}) or {}).get(stage, {}) or {}
    return dict(section) if isinstance(section, Mapping) else {}


#: 冒烟（``--limit-dataset``）CPU batch/eval 上限（见 :func:`_apply_smoke_batch_caps`）。
_SMOKE_CPU_BATCH_CAP = 128
_SMOKE_CPU_MICRO_CAP = 8
#: 阶段 A 逐 epoch 评估帧上限（host no_grad 前向实测 ~2.4MB/帧 × 256 帧 ≈ 0.6GB）。
_SMOKE_CPU_EVAL_FRAMES = 64


def _smoke_cpu_guard(args: argparse.Namespace, device: Any) -> bool:
    """``--limit-dataset`` + CPU → 启用冒烟 host 内存保护（GPU/正常训练不启用）。"""
    return (
        getattr(args, "limit_dataset", None) is not None
        and str(getattr(device, "type", device)) == "cpu"
    )


def _apply_smoke_batch_caps(
    args: argparse.Namespace,
    device: Any,
    batch_size: int,
    micro_batch: Optional[int],
    count: int,
) -> Tuple[int, Optional[int]]:
    """``--limit-dataset`` 冒烟 + CPU 时的 host 内存保护（返回 ``(宏 batch, micro batch)``）。

    ``--limit-dataset`` 的前缀截断保证数据集/物化内存 ∝ N；但训练前向图仍按配置 batch
    展开：CPU 无显存上限，ST-GNN 6 步展开实测 macro 1024/micro 256 峰值 RSS ≈ 10.4 GB
    （A/B 均如此）→ 冒烟仍会挤爆机器、拖慢并行训练。故冒烟（limit 设置）且 ``device=cpu``
    时把 batch 收敛到 host 安全上限（macro ≤ 128 / micro ≤ 8，实测峰值 ≈ 1.2–1.6 GB）。

    GPU / 未设置 limit 时**原样返回**（正常训练语义不变；GPU 冒烟不受影响）。
    """
    if not _smoke_cpu_guard(args, device):
        return int(batch_size), micro_batch
    macro = max(1, min(int(batch_size), _SMOKE_CPU_BATCH_CAP, max(1, int(count))))
    micro = max(1, min(int(micro_batch) if micro_batch else macro, _SMOKE_CPU_MICRO_CAP, macro))
    return macro, micro


# --------------------------------------------------------------------------- #
# 周期检查点 / resume（2026-09-27：防"跑 19/20 epoch 被静默杀掉 → 全部白跑"）
# --------------------------------------------------------------------------- #

def _resolve_ckpt_every(args: argparse.Namespace, config: Mapping[str, Any]) -> int:
    """周期 ckpt 间隔（epoch）：CLI ``--ckpt-every`` 优先，否则 ``config train.ckpt_every``。

    默认 5（config/train.yaml）；0 = 关闭周期保存（阶段末 ``final.pt`` 仍写）。
    """
    value = getattr(args, "ckpt_every", None)
    if value is None:
        value = dict(config.get("train", {}) or {}).get("ckpt_every", 5)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        print(f"[stages] 无法解析 ckpt_every={value!r} → 回退 5", flush=True)
        return 5


def _resume_path(args: argparse.Namespace) -> Optional[str]:
    """``--resume <ckpt>``（空 = 不了 resume）。"""
    path = getattr(args, "resume", None)
    return str(path) if path else None


def _periodic_ckpt_path(out_dir: Path, global_epoch: int) -> Path:
    """周期 ckpt 命名（与 ``final.pt`` 同格式/payload）。"""
    return out_dir / f"ckpt_epoch{int(global_epoch):03d}.pt"


def _resume_epoch_of(resume_info: Mapping[str, Any], total_epochs: int) -> int:
    """resume ckpt 的已完成 epoch 数（0 基；``epoch`` 键缺失/None → 0，超出总轮数 → 截断）。"""
    epoch = resume_info.get("epoch") if resume_info else None
    return min(max(0, int(epoch or 0)), max(0, int(total_epochs)))


# --------------------------------------------------------------------------- #
# Stage A/B 启动数据集契约守卫（2026-09-26 事故防复发）
# --------------------------------------------------------------------------- #
#
# 事故：``STAGE=B BC_EPOCHS=20 STAGE_A_OUT=... bash tools/train.sh`` 漏设 ``BC_DIR``
# → 静默回退默认 ``runs/bc_expert_full``（schema v1，10,777 行，无 ``others``）→
# Stage B 照常训练，router 损失全程为 0，只打印一行
# ``[bc] router 软目标不可用（...）→ 本轮跳过 router 损失``。
# 防复发：Stage A/B 在训练循环开始前硬校验数据集契约（schema≥2 + v2 通道），
# 不满足即 ``raise SystemExit``；显式逃生门（``--allow-legacy-dataset`` /
# ``ALLOW_LEGACY_DATASET=1``）放行时打印醒目警告并在 metrics.json 标注
# ``legacy_dataset=true``。

#: v2 采集（``tools/collect_expert.py``）必有、v1 数据集（``runs/bc_expert_full``）完全没有的通道。
#: ``hist_valid`` 在 v1 也存在（无区分度）故不入列；``others`` 缺失正是 router 损失静默归零的根因。
_BC_V2_REQUIRED_ARRAYS: Tuple[str, ...] = ("others", "od_id", "od_presence")
#: 逃生门环境变量（与 CLI ``--allow-legacy-dataset`` 等价；放行时打印醒目警告）。
_LEGACY_DATASET_ENV = "ALLOW_LEGACY_DATASET"


def _allow_legacy_dataset(args: argparse.Namespace) -> bool:
    """逃生门：CLI ``--allow-legacy-dataset`` 或环境变量 ``ALLOW_LEGACY_DATASET=1``。"""
    if bool(getattr(args, "allow_legacy_dataset", False)):
        return True
    return str(os.environ.get(_LEGACY_DATASET_ENV, "")).strip().lower() in ("1", "true", "yes", "on")


def validate_bc_dataset(
    dataset: BCDataset, bc_dir: str, stage: str, *, allow_legacy: bool = False
) -> Dict[str, Any]:
    """Stage A/B 训练前校验 BC 数据集契约；返回需并入 metrics 的元数据。

    - 通过（``schema_version>=2`` 且含 :data:`_BC_V2_REQUIRED_ARRAYS`）→ 打印启动横幅
      ``[stage{X}] dataset=<dir> rows=<n> fingerprint=<fp> schema=v2``，返回
      ``{"legacy_dataset": False, "dataset_fingerprint": ...}``；
    - 不通过 → ``allow_legacy`` 时打印醒目警告并返回 ``legacy_dataset=True``；
      否则 ``raise SystemExit``（消息含当前 BC_DIR、实际 schema/缺通道/行数、
      正确示例命令）。
    """
    meta = dict(getattr(dataset, "meta", {}) or {})
    schema = int(getattr(dataset, "schema_version", 0) or 0)
    fingerprint = str(meta.get("obs_fingerprint") or "")
    rows = int(getattr(dataset, "count", 0))
    arrays = getattr(dataset, "arrays", {}) or {}
    missing = tuple(key for key in _BC_V2_REQUIRED_ARRAYS if key not in arrays)
    problems: List[str] = []
    if schema < 2:
        problems.append(f"schema_version={schema}（要求 ≥2）")
    if missing:
        problems.append("缺少 v2 通道：" + ", ".join(missing))
    tag = f"[stage{stage}]"
    if problems:
        detail = "；".join(problems)
        if allow_legacy:
            print(
                f"{tag} ⚠⚠ 警告：放行 legacy/不兼容 BC 数据集（{detail}）—— "
                f"dataset={bc_dir} rows={rows} fingerprint={fingerprint or '<缺失>'} schema=v{schema}；"
                "router/others 相关损失可能全程无效（结果作废风险），metrics.json 标注 legacy_dataset=true",
                flush=True,
            )
            return {"legacy_dataset": True, "dataset_fingerprint": fingerprint}
        raise SystemExit(
            f"{tag} BC 数据集契约校验失败：BC_DIR={bc_dir} 不满足 Stage {stage} 要求。\n"
            f"{tag} 实际内容：{detail}，rows={rows}，obs_fingerprint={fingerprint or '<缺失>'}\n"
            f"{tag} 继续训练会导致 router/others 相关损失静默为 0（2026-09-26 事故：漏设 BC_DIR 落到 v1 数据集）。\n"
            f"{tag} 正确示例：BC_DIR=runs/bc_expert_5k_v2 STAGE={stage} bash tools/train.sh\n"
            f"{tag} 确认要用 legacy 数据集（结果可能作废）：追加 --allow-legacy-dataset 或 ALLOW_LEGACY_DATASET=1"
        )
    print(
        f"{tag} dataset={bc_dir} rows={rows} fingerprint={fingerprint or '<none>'} schema=v{schema}",
        flush=True,
    )
    return {"legacy_dataset": False, "dataset_fingerprint": fingerprint}


# --------------------------------------------------------------------------- #
# 按帧数据的窗口构建（阶段 A/B：精确查表历史 + 未来目标）
# --------------------------------------------------------------------------- #
#
# **禁止按行位置取窗口**（旧实现）：等距假设在帧被过滤后失效（实测 16.8% 窗口时间
# 不均匀），且 episode 头部会伪造 ``hist_valid=1``。v2 起一律按
# ``(episode_id, step − 5j)`` / ``(episode_id, step + 5k)`` 精确查表 + per-slot valid。
#
# lane A 的 ``pipeline/frames.py::build_history/build_future`` 落地后优先使用；
# 未落地时用本文件的等价实现（语义一致，测试覆盖）。
# TODO(lane): 若 lane A 的函数签名/返回键不同，请同步 ``_lane_frames_api`` 适配器。

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


def _lane_frames_api() -> Optional[Any]:
    """lane A 的 ``pipeline/frames.py`` 适配器（不可用 → None，用本地等价实现）。

    lane A 的接口是**逐帧**的（``FrameLookup.build_history(episode_id, step)`` /
    ``build_future(...)`` + ``lookup_from_arrays``）；本文件把它批量化成
    ``build_history(arrays, indices)`` / ``build_future(arrays, indices)``。
    lane A 尚在迁移中（异常/缺键）时自动回退到本地精确查表实现并告警一次。
    """
    try:
        from pipeline import frames as lane  # type: ignore
    except Exception:  # noqa: BLE001 - lane 未落地
        return None
    if callable(getattr(lane, "lookup_from_arrays", None)) and hasattr(lane, "FrameLookup"):
        return lane
    return None


#: lane A FrameLookup 缓存（按 arrays 身份 + stride/episode/step 键；构造 O(N)，避免每 batch 重建）
_FRAMES_LOOKUP_CACHE: Dict[Any, Tuple[Any, Any]] = {}
_FRAMES_FALLBACK_WARNED: Dict[str, bool] = {}


def _warn_frames_fallback(where: str, exc: BaseException) -> None:
    if not _FRAMES_FALLBACK_WARNED.get(where):
        print(
            f"[stages] WARN：pipeline.frames.{where} 不可用（{type(exc).__name__}: {exc}）"
            "→ 使用本地精确查表等价实现（TODO(lane)：接口稳定后移除此回退）",
            flush=True,
        )
        _FRAMES_FALLBACK_WARNED[where] = True


def _frames_lookup(
    arrays: Mapping[str, np.ndarray],
    *,
    episode_key: str,
    step_key: str,
    stride: int,
) -> Optional[Any]:
    """构造/取缓存 lane A ``FrameLookup``；lane 不可用或构造失败 → None。"""
    lane = _lane_frames_api()
    if lane is None:
        return None
    key = (id(arrays), int(stride), episode_key, step_key)
    cached = _FRAMES_LOOKUP_CACHE.get(key)
    if cached is not None and cached[0] is arrays:
        return cached[1]
    usable_key = "frame_usable" if "frame_usable" in arrays else ("usable" if "usable" in arrays else None)
    try:
        lookup = lane.lookup_from_arrays(
            arrays,
            episode_key=episode_key,
            step_key=step_key,
            stride=max(1, int(stride)),
            usable_key=usable_key,
            check=True,
        )
    except Exception as exc:  # noqa: BLE001 - lane 迁移期（缺键/断言未定案）
        _warn_frames_fallback("lookup_from_arrays", exc)
        return None
    if len(_FRAMES_LOOKUP_CACHE) > 8:
        _FRAMES_LOOKUP_CACHE.clear()
    _FRAMES_LOOKUP_CACHE[key] = (arrays, lookup)  # 强引用 arrays，避免 id 复用
    return lookup


def _mem_history_available(arrays: Mapping[str, np.ndarray]) -> bool:
    """数据集是否内嵌 v2 mem 历史数组（od/ld + hist_valid 必需）。"""
    return all(key in arrays for key in ("od_hist", "od_hist_mask", "ld_hist", "ld_hist_mask", "hist_valid"))


def _entry_from_arrays(
    arrays: Mapping[str, np.ndarray], row: int, keys: Mapping[str, Tuple[str, Optional[str]]]
) -> Dict[str, Any]:
    """单行 → ``stack_history`` 需要的 entry（当前通道 + mask + pose）。"""
    obs: Dict[str, np.ndarray] = {}
    for name, (feat_key, mask_key) in keys.items():
        value = arrays.get(feat_key)
        if value is None:
            continue
        obs[name] = np.asarray(value[row], dtype=np.float32)
        if mask_key and mask_key in arrays:
            obs[f"{name}_mask"] = np.asarray(arrays[mask_key][row], dtype=np.float32)
    return {"obs": obs, "pose": np.asarray(arrays["pose"][row], dtype=np.float32)}


#: 历史输出里保留的键（其余 lane A 内部键丢弃，避免污染 net 输入）
_HISTORY_CHANNELS = tuple(dict.fromkeys(_REPLAY_CHANNELS + ("others",)))
#: 未来输出键（lane A 与本文件等价实现的统一契约）
_FUTURE_OUTPUT_KEYS = (
    "od_fut",
    "ld_fut",
    "od_mask",
    "od_mask_raw",
    "ld_mask",
    "valid",
    "wm_valid",
    "od_id_fut",
    "od_presence_fut",
    "od_id_t0",
    "od_presence_t0",
    "ego_fut",
)


def _batch_lane_history(
    lookup: Any, indices: np.ndarray, *, frames: int, stride: int, alignments: Optional[Mapping[str, Any]]
) -> Dict[str, np.ndarray]:
    """逐帧调用 lane A ``build_history`` 并批量化（旧→新，含 ``hist_valid``）。"""
    rows: List[Dict[str, np.ndarray]] = []
    for index in indices:
        episode = int(lookup.episode[index])
        step = int(lookup.step[index])
        kwargs: Dict[str, Any] = {}
        if alignments:
            kwargs["alignments"] = alignments
        if lookup.pose is not None:
            kwargs["target_pose"] = lookup.pose[index]
        rows.append(lookup.build_history(episode, step, stride=int(stride), k=int(frames), **kwargs))
    out: Dict[str, np.ndarray] = {}
    allowed = {f"{name}_hist" for name in _HISTORY_CHANNELS} | {
        f"{name}_hist_mask" for name in _HISTORY_CHANNELS
    } | {"hist_valid", "od_id_hist", "od_presence_hist"}
    for key in rows[0]:
        if key not in allowed:
            continue
        out[key] = np.stack([np.asarray(row[key], dtype=np.float32) for row in rows], axis=0)
    return out


def _batch_lane_future(
    lookup: Any,
    indices: np.ndarray,
    *,
    future: int,
    stride: int,
    wm_valid: Optional[np.ndarray],
    alignments: Optional[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """逐帧调用 lane A ``build_future`` 并批量化（+ 显式 ``wm_valid`` 覆盖 + ``ego_fut``）。"""
    rows: List[Dict[str, np.ndarray]] = []
    for index in indices:
        episode = int(lookup.episode[index])
        step = int(lookup.step[index])
        kwargs: Dict[str, Any] = {}
        if alignments:
            kwargs["alignments"] = alignments
        if lookup.pose is not None:
            kwargs["target_pose"] = lookup.pose[index]
        rows.append(lookup.build_future(episode, step, stride=int(stride), k=int(future), **kwargs))
    out: Dict[str, np.ndarray] = {}
    for key in _FUTURE_OUTPUT_KEYS:
        if key == "ego_fut":
            continue
        if key in rows[0]:
            out[key] = np.stack([np.asarray(row[key]) for row in rows], axis=0)
    # ego 未来帧（teacher forcing：目标帧真实 ego，reserved 维 = 该帧上一动作）
    ego = lookup.features.get("ego")
    if ego is not None:
        if ego.ndim == 3 and ego.shape[1] == 1:
            ego = ego[:, 0, :]  # 单槽通道 (N,1,8) → (N,8)
        ego_fut = np.zeros((indices.shape[0], int(future)) + tuple(ego.shape[1:]), dtype=ego.dtype)
        for row_index, index in enumerate(indices):
            episode = int(lookup.episode[index])
            step = int(lookup.step[index])
            base = (step // int(stride)) * int(stride)
            for k in range(1, int(future) + 1):
                target = lookup.index_of(episode, base + k * int(stride))
                if target is not None:
                    ego_fut[row_index, k - 1] = ego[target]
        out["ego_fut"] = ego_fut
    if wm_valid is not None:
        wm = np.asarray(wm_valid, dtype=np.float32).reshape(indices.shape[0], int(future))
        out["wm_valid"] = wm
        out["od_mask"] = out.get("od_mask", np.zeros_like(wm[:, :, None])) * wm[:, :, None]
        if "ld_mask" in out:
            out["ld_mask"] = out["ld_mask"] * wm[:, :, None]
        out["valid"] = out.get("valid", wm) * 1.0  # 步存在性不受可用性影响
    # 无身份伴随数组（旧 schema）时 lane A 的同 id 匹配会把 mask 全清零 → 回退原始 mask
    if not lookup.companions and "od_mask_raw" in out:
        out["od_mask"] = out["od_mask_raw"].copy()
        if "wm_valid" in out:
            out["od_mask"] = out["od_mask"] * out["wm_valid"][:, :, None]
    return out


def _local_build_history(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    episode_key: str,
    step_key: str,
    keys: Optional[Mapping[str, Tuple[str, Optional[str]]]],
    frames: int,
    stride: int,
    alignments: Optional[Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    """本地精确查表历史（lane A 不可用时的等价实现）。"""
    from pipeline.trainer import stack_history

    idx = np.asarray(indices, dtype=np.int64)
    if not alignments:
        from pipeline.trainer import _alignment_from_meta

        alignments = _alignment_from_meta({})
    if _mem_history_available(arrays):
        out: Dict[str, np.ndarray] = {}
        for name in ("ego", "others", "od", "ld"):
            feat_key, mask_key = f"{name}_hist", f"{name}_hist_mask"
            if feat_key in arrays and mask_key in arrays:
                out[feat_key] = np.asarray(arrays[feat_key], dtype=np.float32)[idx]
                out[mask_key] = np.asarray(arrays[mask_key], dtype=np.float32)[idx]
        out["hist_valid"] = np.asarray(arrays["hist_valid"], dtype=np.float32)[idx]
        return out
    channel_keys = dict(keys) if keys is not None else dict(_bc_channel_keys())
    episode = np.asarray(arrays[episode_key], dtype=np.int64)
    step = np.asarray(arrays[step_key], dtype=np.int64)
    lookup = {(int(episode[i]), int(step[i])): i for i in range(len(episode))}
    pose = np.asarray(arrays["pose"], dtype=np.float32)
    history: Dict[str, List[np.ndarray]] = {}
    valid_rows: List[np.ndarray] = []
    for index in idx:
        ep, st = int(episode[index]), int(step[index])
        rows = [lookup.get((ep, st - stride * j)) for j in range(frames - 1, -1, -1)]
        fallback = next((row for row in reversed(rows) if row is not None), int(index))
        entries = [
            _entry_from_arrays(arrays, fallback if row is None else row, channel_keys) for row in rows
        ]
        valid = np.asarray([1.0 if row is not None else 0.0 for row in rows], dtype=np.float32)
        patch = stack_history(entries, alignments, current_pose=pose[index], valid=valid)
        for key, value in patch.items():
            history.setdefault(key, []).append(value)
        valid_rows.append(valid)
    out = {key: np.stack(values, axis=0).astype(np.float32) for key, values in history.items()}
    out["hist_valid"] = np.stack(valid_rows, axis=0).astype(np.float32)
    return out


def _local_build_future(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    episode_key: str,
    step_key: str,
    future: int,
    stride: int,
    keys: Optional[Mapping[str, Tuple[str, Optional[str]]]],
    alignments: Optional[Mapping[str, Any]],
    wm_valid: Optional[np.ndarray],
) -> Dict[str, np.ndarray]:
    """本地精确查表未来目标（lane A 等价实现；含 id 匹配 / presence / ego_fut）。"""
    from pipeline.trainer import stack_history

    idx = np.asarray(indices, dtype=np.int64)
    if not alignments:
        from pipeline.trainer import _alignment_from_meta

        alignments = _alignment_from_meta({})
    channel_keys = dict(keys) if keys is not None else dict(_bc_channel_keys())
    episode = np.asarray(arrays[episode_key], dtype=np.int64)
    step = np.asarray(arrays[step_key], dtype=np.int64)
    lookup = {(int(episode[i]), int(step[i])): i for i in range(len(episode))}
    pose = np.asarray(arrays["pose"], dtype=np.float32)
    od_id = np.asarray(arrays["od_id"]) if "od_id" in arrays else None
    presence = np.asarray(arrays["od_presence"], dtype=np.float32) if "od_presence" in arrays else None
    ego = arrays.get("ego")
    if ego is not None:
        ego = np.asarray(ego)
        if ego.ndim == 3 and ego.shape[1] == 1:
            ego = ego[:, 0, :]
    slots = int(np.asarray(arrays["od"]).shape[1])
    od_fut: List[np.ndarray] = []
    ld_fut: List[np.ndarray] = []
    od_masks: List[np.ndarray] = []
    od_masks_raw: List[np.ndarray] = []
    ld_masks: List[np.ndarray] = []
    valid_rows: List[np.ndarray] = []
    wm_valid_rows: List[np.ndarray] = []
    od_presence_fut: List[np.ndarray] = []
    od_id_fut: List[np.ndarray] = []
    od_presence_t0: List[np.ndarray] = []
    od_id_t0: List[np.ndarray] = []
    ego_fut: List[np.ndarray] = []
    for row_index, index in enumerate(idx):
        ep, st = int(episode[index]), int(step[index])
        rows = [lookup.get((ep, st + stride * (k + 1))) for k in range(future)]
        step_valid = np.asarray([1.0 if row is not None else 0.0 for row in rows], dtype=np.float32)
        explicit_wm = (
            np.asarray(wm_valid, dtype=np.float32).reshape(idx.shape[0], future)[row_index]
            if wm_valid is not None
            else step_valid
        )
        fallback = next((row for row in rows if row is not None), int(index))
        entries = [
            _entry_from_arrays(arrays, fallback if row is None else row, channel_keys) for row in rows
        ]
        patch = stack_history(entries, alignments, current_pose=pose[index], valid=None)
        od_fut.append(patch["od_hist"])
        ld_fut.append(patch["ld_hist"])
        raw_mask = patch["od_hist_mask"]
        od_masks_raw.append(raw_mask * step_valid[:, None])
        ld_masks.append(patch["ld_hist_mask"] * explicit_wm[:, None])
        frame_presence = np.zeros((future, slots), dtype=np.float32)
        frame_ids = np.full((future, slots), -1, dtype=np.int64)
        for k, row in enumerate(rows):
            if row is None:
                continue
            frame_presence[k] = (
                presence[row] if presence is not None else np.asarray(arrays["od_mask"][row], dtype=np.float32)
            )
            if od_id is not None:
                frame_ids[k] = od_id[row]
        t0_presence = (
            presence[index] if presence is not None else np.asarray(arrays["od_mask"][index], dtype=np.float32)
        )
        t0_ids = od_id[index] if od_id is not None else np.full(slots, -1, dtype=np.int64)
        if od_id is not None:
            same_identity = (frame_ids == t0_ids[None, :]) & (t0_ids[None, :] >= 0)
            od_masks.append(explicit_wm[:, None] * frame_presence * same_identity.astype(np.float32))
        else:
            od_masks.append(explicit_wm[:, None] * raw_mask)
        valid_rows.append(step_valid)
        wm_valid_rows.append(explicit_wm)
        od_presence_fut.append(frame_presence)
        od_id_fut.append(frame_ids)
        od_presence_t0.append(np.asarray(t0_presence, dtype=np.float32))
        od_id_t0.append(np.asarray(t0_ids, dtype=np.int64))
        if ego is not None:
            frame_ego = np.zeros((future, ) + tuple(np.asarray(ego).shape[1:]), dtype=np.float32)
            for k, row in enumerate(rows):
                if row is not None:
                    frame_ego[k] = np.asarray(ego[row], dtype=np.float32)
            ego_fut.append(frame_ego)
    result: Dict[str, np.ndarray] = {
        "od_fut": np.stack(od_fut, axis=0).astype(np.float32),
        "ld_fut": np.stack(ld_fut, axis=0).astype(np.float32),
        "od_mask": np.stack(od_masks, axis=0).astype(np.float32),
        "od_mask_raw": np.stack(od_masks_raw, axis=0).astype(np.float32),
        "ld_mask": np.stack(ld_masks, axis=0).astype(np.float32),
        "valid": np.stack(valid_rows, axis=0).astype(np.float32),
        "wm_valid": np.stack(wm_valid_rows, axis=0).astype(np.float32),
        "od_presence_fut": np.stack(od_presence_fut, axis=0).astype(np.float32),
        "od_id_fut": np.stack(od_id_fut, axis=0),
        "od_presence_t0": np.stack(od_presence_t0, axis=0).astype(np.float32),
        "od_id_t0": np.stack(od_id_t0, axis=0),
    }
    if ego_fut:
        result["ego_fut"] = np.stack(ego_fut, axis=0).astype(np.float32)
    return result


def build_history(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    episode_key: str = "episode_id",
    step_key: str = "step",
    keys: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
    frames: int = 6,
    stride: int = _BC_STEP_STRIDE,
    alignments: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """6 帧历史窗口（旧→新）+ ``hist_valid``（精确查表；**禁止按行位置取窗口**）。

    优先级：lane A ``pipeline/frames``（逐帧查表，权威）→ 数据集内嵌 mem 历史
    （``od_hist/...`` + ``hist_valid``）→ 本地精确查表等价实现。
    """
    idx = np.asarray(indices, dtype=np.int64)
    lookup = _frames_lookup(arrays, episode_key=episode_key, step_key=step_key, stride=stride)
    if lookup is not None:
        try:
            return _batch_lane_history(
                lookup, idx, frames=int(frames), stride=int(stride), alignments=alignments
            )
        except Exception as exc:  # noqa: BLE001 - lane 迁移期
            _warn_frames_fallback("build_history", exc)
    return _local_build_history(
        arrays,
        idx,
        episode_key=episode_key,
        step_key=step_key,
        keys=keys,
        frames=int(frames),
        stride=int(stride),
        alignments=alignments,
    )


def build_future(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    episode_key: str = "episode_id",
    step_key: str = "step",
    future: int = 6,
    stride: int = _BC_STEP_STRIDE,
    keys: Optional[Mapping[str, Tuple[str, Optional[str]]]] = None,
    alignments: Optional[Dict[str, Any]] = None,
    wm_valid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """未来 1..K 步目标（t0 帧对齐）+ mask + per-step valid（精确查表）。

    - 目标帧 = ``(episode_id, base + stride·k)``；缺失/越界 → ``valid=0``、mask 清零；
    - ``wm_valid (B,K)``（v2）优先于 lane A 的 ``usable``（显式传入时覆盖）；
    - ``od_mask`` 含**身份匹配**（lane A 同 id；本地回退用 ``od_id`` 同 id）；
    - ``ego_fut (B,K,8)``：目标帧真实 ego（teacher forcing 用）；
    - ``od_presence_fut/od_id_fut/od_presence_t0/od_id_t0``：presence/entry BCE 目标来源。
    """
    idx = np.asarray(indices, dtype=np.int64)
    lookup = _frames_lookup(arrays, episode_key=episode_key, step_key=step_key, stride=stride)
    if lookup is not None:
        try:
            return _batch_lane_future(
                lookup, idx, future=int(future), stride=int(stride), wm_valid=wm_valid, alignments=alignments
            )
        except Exception as exc:  # noqa: BLE001 - lane 迁移期
            _warn_frames_fallback("build_future", exc)
    return _local_build_future(
        arrays,
        idx,
        episode_key=episode_key,
        step_key=step_key,
        future=int(future),
        stride=int(stride),
        keys=keys,
        alignments=alignments,
        wm_valid=wm_valid,
    )


class FrameWindows:
    """把"每帧当前通道 + 位姿"的数组建成模型输入窗口与未来目标。

    - 历史窗口：``build_history``（v2 mem 历史 → 精确查表，见函数 docstring）；
    - 未来目标：``build_future``（精确查表 + ``wm_valid`` 门控）。

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
        self.episode_key = episode_key
        self.step_key = step_key
        self.step_stride = max(1, int(step_stride))
        self.episode = np.asarray(arrays[episode_key], dtype=np.int64)
        self.step_index = np.asarray(arrays[step_key], dtype=np.int64)
        self.pose = np.asarray(arrays["pose"], dtype=np.float32)
        self.count = int(len(self.episode))

    # ---------------------------------------------------------------- 组装
    def build_obs(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """当前帧 + 历史窗口（v2 mem 历史优先 → 精确查表；B 维在前）。"""
        from pipeline.trainer import sanitize_masked_od, squeeze_single_slot

        idx = np.asarray(indices, dtype=np.int64)
        batch: Dict[str, List[np.ndarray]] = {}
        for index in idx:
            frame = _entry_from_arrays(self.arrays, int(index), self.keys)["obs"]
            for key, value in frame.items():
                batch.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
        patch = build_history(
            self.arrays,
            idx,
            episode_key=self.episode_key,
            step_key=self.step_key,
            keys=self.keys,
            stride=self.step_stride,
            alignments=self.alignments,
        )
        for key, value in patch.items():
            batch[key] = [np.asarray(item, dtype=np.float32) for item in value]
        return sanitize_masked_od(
            squeeze_single_slot({key: np.stack(values, axis=0).astype(np.float32) for key, values in batch.items()})
        )

    def build_future(self, indices: np.ndarray, future: int = 6, wm_valid: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """未来 1..K 步目标（精确查表 + ``wm_valid`` 门控 + 对齐到当前帧）。"""
        return build_future(
            self.arrays,
            np.asarray(indices, dtype=np.int64),
            episode_key=self.episode_key,
            step_key=self.step_key,
            future=future,
            stride=self.step_stride,
            keys=self.keys,
            alignments=self.alignments,
            wm_valid=wm_valid,
        )


# --------------------------------------------------------------------------- #
# 阶段 A/B 快路径：逐样本窗口/目标的一次性物化（训练循环只做切片 + H2D）
# --------------------------------------------------------------------------- #
#
# 实测（2026-09-26，v2 数据 / H=128）：逐样本 build_obs_batch + build_future 占单 batch
# 墙钟 >95%（batch=256 时 Stage A ~0.44 s vs GPU 前反向 5–15 ms）。物化后循环内只做
# numpy 切片 + pin/non-blocking H2D（`MaterializedBCDataset` 见 pipeline.trainer）。

#: Stage A 未来目标物化保留的键（Stage A 消费：OD/LD 多步 + presence/entry + ego_next；od_mask_raw 仅内部诊断）
_MATERIALIZED_FUTURE_KEYS: Tuple[str, ...] = (
    "od_fut",
    "ld_fut",
    "od_mask",
    "ld_mask",
    "wm_valid",
    "valid",
    "ego_fut",
    "od_presence_fut",
    "od_id_fut",
    "od_presence_t0",
    "od_id_t0",
)

#: stage B phase 3（lane P3-B）未来目标物化键：A 的键 + presence/entry 目标 + 多步动作链
#: （见 :func:`phase3_action_chain_targets`；LD 键已并入 A 的 :data:`_MATERIALIZED_FUTURE_KEYS`）。
_PHASE3_FUTURE_KEYS: Tuple[str, ...] = _MATERIALIZED_FUTURE_KEYS + (
    "presence_target",
    "entry_target",
    "action_chain",
    "action_chain_valid",
)


class _ArrayBatchSource:
    """连续数组上的只读 batch 视图（``arr[idx]`` 切片；训练循环唯一的数据操作）。"""

    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self.arrays: Dict[str, np.ndarray] = {str(key): np.asarray(value) for key, value in arrays.items()}
        self.nbytes = int(sum(value.nbytes for value in self.arrays.values()))

    def batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        return {key: value[idx] for key, value in self.arrays.items()}


def _materialize_future_targets(
    future_fn: Any,
    obs_source: MaterializedBCDataset,
    count: int,
    *,
    batch_size: int = 1024,
    logger: Any = print,
    keys: Optional[Sequence[str]] = None,
) -> _ArrayBatchSource:
    """阶段 A 未来目标一次性物化（分块调用 ``future_fn``，与逐 batch 完全同路径）。

    ``future_fn(indices, obs_np) -> dict`` = ``run_stage_a`` 内的 ``_future_targets``
    （含 ``wm_valid`` 门控 / 身份匹配 / v1 最近邻回退），因此物化结果与旧逐 batch 路径
    逐值一致（等价性测试见 ``tests/test_fast_data_path.py``）。

    ``keys``：物化的键集（缺省 = Stage A 的 :data:`_MATERIALIZED_FUTURE_KEYS`；phase 3 传
    :data:`_PHASE3_FUTURE_KEYS` 以带上 LD/presence/动作链目标）。
    """
    started = time.perf_counter()
    chunk = max(1, int(batch_size))
    key_set = tuple(keys) if keys is not None else _MATERIALIZED_FUTURE_KEYS
    arrays: Dict[str, np.ndarray] = {}
    for start in range(0, int(count), chunk):
        stop = min(start + chunk, int(count))
        indices = np.arange(start, stop, dtype=np.int64)
        future = future_fn(indices, obs_source.obs_batch(indices))
        if not arrays:
            arrays = {
                key: np.empty((int(count), ) + tuple(np.asarray(future[key]).shape[1:]), dtype=np.asarray(future[key]).dtype)
                for key in key_set
            }
        for key in key_set:
            arrays[key][start:stop] = future[key]
    source = _ArrayBatchSource(arrays)
    logger(
        f"[materialize] future targets {count} 行 → {source.nbytes / 1e6:.1f} MB "
        f"（{time.perf_counter() - started:.1f}s，chunk={chunk}）"
    )
    return source


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
    fde_weight = weight if weight.ndim == 2 else weight[:, -1, :]
    if float(fde_weight.sum()) <= 0.0:
        return ade, torch.tensor(float("nan"))
    fde_error = error if error.ndim == 2 else error[:, -1, :]
    fde = (fde_error * fde_weight).sum() / fde_weight.sum()
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
    current_od_id: Optional[np.ndarray] = None,
    current_presence: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """把未来帧 OD 槽位匹配到当前槽位（identity association）。

    两条路径：

    1. **id 精确匹配**（v2；``future["od_id_fut"]`` + ``current_od_id``）：同 id = 同对象，
       比匀速先验最近邻更可靠（变道/遮挡/排序切换时不串位）；``id<=0`` 或 presence=0
       视为空槽；无匹配 → mask=0（对象离场）。
    2. **t0 帧最近邻回退**（v1）：匹配键用 WM 自己的匀速先验，gate 过滤对象离场/新入场。

    为什么需要：OD 通道按 ``min(TTC, cap) + 距离`` 排序 → 未来帧的槽位顺序会变
    （同一对象可能从 slot i 换到 slot j）。直接按 index 回归是 ill-posed：实测未匹配时
    匀速基线 ADE ≈ 9–14 m，其中大部分是槽位错配而非物理误差。

    仅改 ``od_fut``/``od_mask``（槽位轴重排 + 有效门控）；``valid``（步存在性）与 LD 不变。
    """
    cur = np.asarray(current_od, dtype=np.float32)  # (B,16,9)
    mask = np.asarray(current_mask, dtype=np.float32)  # (B,16)
    od_fut = np.asarray(future["od_fut"], dtype=np.float32)  # (B,K,16,9)
    od_mask = np.asarray(future["od_mask"], dtype=np.float32)  # (B,K,16)
    if current_od_id is not None and "od_id_fut" in future:
        target_id = np.asarray(future["od_id_fut"])  # (B,K,S)
        cur_id = np.asarray(current_od_id)  # (B,S)
        presence = (
            np.asarray(current_presence, dtype=np.float32)
            if current_presence is not None
            else (mask > 0.5).astype(np.float32)
        )
        valid_current = (presence > 0.5) & (cur_id > 0)  # (B,S)
        matches = (target_id[:, :, None, :] == cur_id[:, None, :, None]) & (cur_id[:, None, :, None] > 0)
        any_match = matches.any(axis=-1)  # (B,K,S)
        nearest = matches.argmax(axis=-1)  # (B,K,S)
        batch_index = np.arange(od_fut.shape[0])[:, None, None]
        step_index = np.arange(od_fut.shape[1])[None, :, None]
        matched_feat = od_fut[batch_index, step_index, nearest]  # (B,K,S,9)
        matched_mask = od_mask[batch_index, step_index, nearest]  # (B,K,S)
        out = dict(future)
        out["od_fut"] = matched_feat.astype(np.float32)
        out["od_mask"] = (
            valid_current[:, None, :] * any_match * matched_mask
        ).astype(np.float32)
        return out
    batch, steps, _, _ = od_fut.shape
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


def presence_entry_targets(future: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """未来目标 → **id 轴** presence/entry 目标（``(B,K,S)`` float32，0/1）。

    v1.2 语义（与采集侧 id 轴一致；槽位轴版本已废弃）：

    - ``od_presence`` = 对象本帧在盒内被观测（0 = 出盒未释放/空槽）；``od_id`` = episode
      内稳定 track id（-1 = 空槽）。
    - **presence target[k, j]**：t0 槽位 j 的 track id（``od_id_t0``）**在 t0 被观测**且在
      目标帧 k 仍在盒内（同一 id 被观测到，``od_presence_fut > 0.5``，不限槽位）→ 1；
      t0 未观测（空槽/出盒未释放）或目标帧未观测 → 0。即"未来帧该 track id 是否仍在盒内"。
    - **entry target[k, j]**：目标帧 k 的槽位 j 上出现**新观测 id**——该 id 在 t0 未被观测
      （"t0 无、t+k 有"：既有全新 id 入场，也有出盒 id 的回入）→ 1；目标帧该 id 在 t0
      已被观测 → 0。即"新 id 是否出现"。（v2 固定槽位下 id 不迁移，presence/entry 互斥。）
    - 无 id 伴随数组（v1 数据 / 全 -1）时回退槽位轴：presence = t0 presence 且未来
      presence；entry = 未来 presence 且 t0 无 presence。**新数据一律走 id 轴。**

    ``train_weight × wm_valid`` 加权由调用方施加（本函数只产出 0/1 目标）。
    """
    presence_fut = np.asarray(future["od_presence_fut"], dtype=np.float32)
    presence_t0 = np.asarray(future.get("od_presence_t0", future["od_mask"]), dtype=np.float32)
    id_fut = np.asarray(
        future.get("od_id_fut", np.full_like(presence_fut, -1, dtype=np.int64)), dtype=np.int64
    )
    id_t0 = np.asarray(
        future.get(
            "od_id_t0",
            np.full((presence_t0.shape[-1],), -1, dtype=np.int64)
            if presence_t0.ndim == 1
            else np.full_like(presence_t0, -1, dtype=np.int64),
        ),
        dtype=np.int64,
    )
    if presence_fut.ndim != 3:
        raise ValueError(f"od_presence_fut 形状应为 (B,K,S)，收到 {presence_fut.shape}")
    batch = int(presence_fut.shape[0])
    if presence_t0.ndim == 1:
        presence_t0 = np.broadcast_to(presence_t0[None, :], (batch, presence_t0.shape[0]))
    if id_t0.ndim == 1:
        id_t0 = np.broadcast_to(id_t0[None, :], (batch, id_t0.shape[0]))
    if not (np.any(id_fut >= 0) and np.any(id_t0 >= 0)):
        # v1 回退：无 id 轴信息 → 槽位轴（旧语义，仅兼容旧数据）
        presence_target = (presence_t0 > 0.5)[:, None, :] * (presence_fut > 0.5)
        entry_target = (presence_fut > 0.5) & (presence_t0[:, None, :] <= 0.5)
        return presence_target.astype(np.float32), entry_target.astype(np.float32)
    observed_fut = (presence_fut > 0.5) & (id_fut > 0)  # (B,K,S)
    observed_t0 = (presence_t0 > 0.5) & (id_t0 > 0)  # (B,S)
    # t0 槽位 j 的 id 是否在目标帧 k 被观测（id 轴持续存在；不限槽位）
    same_id_fut = (
        (id_fut[:, :, None, :] == id_t0[:, None, :, None]) & observed_fut[:, :, None, :]
    ).any(axis=-1)  # (B,K,S)
    presence_target = (observed_t0[:, None, :] & same_id_fut).astype(np.float32)
    # 目标帧槽位 j 的 id 是否在 t0 已被观测（是 → 不是 entry）
    known_at_t0 = (
        (id_fut[:, :, :, None] == id_t0[:, None, None, :]) & observed_t0[:, None, None, :]
    ).any(axis=-1)  # (B,K,S)
    entry_target = (observed_fut & ~known_at_t0).astype(np.float32)
    return presence_target, entry_target


def phase3_action_chain_targets(
    arrays: Mapping[str, np.ndarray],
    indices: np.ndarray,
    *,
    future: int = 6,
    stride: int = _BC_STEP_STRIDE,
) -> Tuple[np.ndarray, np.ndarray]:
    """phase 3 多步动作链目标（lane P3-B）：``(action_chain (B,K,2), valid (B,K))``。

    - ``action_chain[:, k]`` = 同 episode ``(episode_id, step + stride·k)`` 帧的**专家首步
      动作标签**（``arrays["action"][row, 0, :]``；``k=0`` = 当前帧）——与 rollout
      ``plan[:, k]`` 时间对齐（``plan[:,0] = action_mu`` 已由首步动作损失监督，本项只吃 ``k≥1``）；
    - ``valid``：目标帧存在（精确查表）→ 1；窗口末端/缺帧 → 0（逐帧 mask）；
    - 动作 ``(ds, dθ)`` 是车体量，与 SE(2) 对齐无关（无需 t0 系重建）。

    未来 LD/OD 目标由 :func:`build_future`（t0 自车系对齐）提供，本函数只补动作链标签。
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    episode = np.asarray(arrays["episode_id"], dtype=np.int64)
    step = np.asarray(arrays["step"], dtype=np.int64)
    action = np.asarray(arrays["action"], dtype=np.float32)
    if action.ndim == 2:  # 旧 schema：单步动作 (N,2) → (N,1,2)
        action = action[:, None, :]
    lookup = {(int(episode[i]), int(step[i])): i for i in range(len(episode))}
    horizon = max(1, int(future))
    chain = np.zeros((idx.size, horizon, action.shape[-1]), dtype=np.float32)
    valid = np.zeros((idx.size, horizon), dtype=np.float32)
    for i, row in enumerate(idx):
        ep, st = int(episode[row]), int(step[row])
        chain[i, 0] = action[row, 0]
        valid[i, 0] = 1.0
        for k in range(1, horizon):
            target = lookup.get((ep, st + int(stride) * k))
            if target is not None:
                chain[i, k] = action[target, 0]
                valid[i, k] = 1.0
    return chain, valid


def _grad_norms(model: Any, prefixes: Sequence[str]) -> Dict[str, float]:
    """按参数名前缀汇总梯度 L2 范数（Stage A 可微性自检；无梯度 = 0）。

    用于验证 design-v1.2 §2.3：Stage A 里 plan head/MoE/ST-GNN/编码器必须有梯度，
    policy/value（冻结）必须为 0。嵌套前缀（``plan_head.`` 与 ``plan_head.moe.router.``）
    同时累计（单次遍历，无梯度参数跳过）。
    """
    out: Dict[str, float] = {str(prefix).rstrip("."): 0.0 for prefix in prefixes}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().norm())
        for prefix in prefixes:
            if name.startswith(prefix):
                out[str(prefix).rstrip(".")] += norm
    return out


def run_stage_a(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """阶段 A：world model 教师强制训练（专家动作序列为 ego 条件，未来 OD/LD 为目标）。"""
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_thread_limits(workers=1, config=config)
    device = torch.device(resolve_device(args.device, config))
    model = build_model(_load_yaml(args.model_config))
    resume_path = _resume_path(args)
    resume_info: Dict[str, Any] = {}
    if resume_path:
        if not Path(resume_path).exists():
            raise SystemExit(f"[stageA] --resume 文件不存在：{resume_path}")
        resume_info = load_training_checkpoint(resume_path, model)
        done = _resume_epoch_of(resume_info, 10**9)
        print(
            f"[stageA] resume {resume_path}：模型已载入（missing={len(resume_info['missing_keys'])}）"
            f" · 已完成 epoch={done} → 从 epoch {done + 1} 继续",
            flush=True,
        )
    elif args.ckpt and Path(args.ckpt).exists():
        meta = load_checkpoint(args.ckpt, model)
        print(f"[stageA] 载入 {args.ckpt}（missing={len(meta.get('missing_keys', []))}）", flush=True)

    # lane U1+：Stage A 关闭 MoE（experts+gate 只在 B-phase2 训练；A 输出严格 = primary）。
    # 2026-09-30：MoE-in-A 路线已由 W2/T2 闭环实验证伪（version_ledger），实验开关 STAGE_A_MOE 移除。
    model.set_moe(enabled=False)
    print("[stageA] MoE 已关闭（experts/router 不参与）", flush=True)

    dataset = BCDataset.load(args.bc_dir, limit=args.limit_dataset)
    dataset_contract = validate_bc_dataset(
        dataset, str(args.bc_dir), "A", allow_legacy=_allow_legacy_dataset(args)
    )
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
    # lane T：与 Stage B 同一 val 口径（train-dir 全部行 + val-dir 全部行；缺 val-dir → legacy 切分）
    val_dataset = None
    val_dir = _resolve_val_dir(args, args.bc_dir)
    if val_dir is not None:
        val_dataset = BCDataset.load(str(val_dir))
        validate_bc_dataset(val_dataset, str(val_dir), "A", allow_legacy=_allow_legacy_dataset(args))
        train_idx = np.arange(dataset.count, dtype=np.int64)
        val_idx = np.arange(val_dataset.count, dtype=np.int64)
        val_episodes = np.unique(val_dataset.arrays["episode_id"])
        val_source = f"dir:{val_dir}"
        print(
            f"[stageA] val-dir={val_dir}：train={train_idx.size} 行（train-dir 全部行）· "
            f"val={val_idx.size} 行（val-dir 全部行 · {val_episodes.size} episodes）",
            flush=True,
        )
    else:
        train_idx, val_idx, val_episodes = _episode_split(
            arrays["episode_id"], float(args.val_frac), int(args.seed)
        )
        val_source = "episode_split(legacy)"
        print(
            "[stageA] 警告：未找到独立 val-dir（--val-dir 或 train-dir 同级 *_expert500val）→ 回退 "
            "legacy 按 episode 比例切分",
            flush=True,
        )
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
    if resume_info and load_optimizer_state(optimizer, resume_info.get("optimizer")):
        print("[stageA] resume：优化器状态已恢复", flush=True)
    model.to(device).train()
    # 跨设备 resume：优化器在 CPU 上恢复状态、模型刚搬到 device → 状态张量必须跟随参数设备（幂等）
    _moved = move_optimizer_state_to_device(optimizer)
    if _moved:
        print(f"[stageA] resume：优化器状态已迁移到 {device}（{_moved} 个张量）", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_cfg = dict(config.get("train", {}) or {})
    batch_size = int(
        args.batch_size or (train_cfg.get("wm", {}) or {}).get("batch_size") or DEFAULT_STAGE_BATCH_SIZE
    )
    micro_cfg = args.micro_batch_size if args.micro_batch_size is not None else (train_cfg.get("wm", {}) or {}).get(
        "micro_batch_size"
    )
    micro_batch = batch_size if not micro_cfg else max(1, min(batch_size, int(micro_cfg)))
    batch_size, micro_batch = _apply_smoke_batch_caps(args, device, batch_size, micro_batch, dataset.count)
    use_accum = micro_batch < batch_size
    eval_frames = max(1, int(args.eval_frames))
    if _smoke_cpu_guard(args, device):
        eval_frames = min(eval_frames, _SMOKE_CPU_EVAL_FRAMES)
        print(
            f"[stageA] 冒烟内存保护（--limit-dataset + CPU）：上限 macro≤{_SMOKE_CPU_BATCH_CAP} · "
            f"micro≤{_SMOKE_CPU_MICRO_CAP} · eval_frames≤{_SMOKE_CPU_EVAL_FRAMES} → 实际 "
            f"macro={batch_size} micro={micro_batch} eval_frames={eval_frames}"
            "（CPU 前向图/评估无显存上限；避免冒烟挤爆 host / 拖慢并行训练）",
            flush=True,
        )
    use_pin = device.type == "cuda"
    use_materialized = bool(getattr(args, "materialize", False))

    def _tensor(value: Any, *, dtype: Optional["torch.dtype"] = None) -> "torch.Tensor":
        """numpy → device（物化路径走 pin_memory + non_blocking H2D）。"""
        return to_device_tensor(value, device, dtype=dtype, pin=use_pin)
    noise_std = torch.tensor(
        [float(args.plan_noise_ds), float(args.plan_noise_dtheta)], dtype=torch.float32, device=device
    )
    action_all = np.asarray(arrays["action"], dtype=np.float32)
    traj6_all = np.asarray(arrays["traj6"], dtype=np.float32)
    # 权重/目标可用性（v2；v1 退化到 sample_weight/查表存在性）
    train_weight_all = row_action_weights(dataset, np.arange(dataset.count, dtype=np.int64))
    wm_valid_all = np.asarray(arrays["wm_valid"], dtype=np.float32) if "wm_valid" in arrays else None
    presence_coef = float(
        args.wm_presence_coef
        if args.wm_presence_coef is not None
        else dict(_stage_section(config, "A").get("world_model", {}) or {}).get("presence_coef", 0.1)
    )
    entry_coef = float(
        args.wm_entry_coef
        if args.wm_entry_coef is not None
        else dict(_stage_section(config, "A").get("world_model", {}) or {}).get("entry_coef", 0.1)
    )
    ego_next_coef = float(
        args.wm_ego_next_coef
        if args.wm_ego_next_coef is not None
        else dict(_stage_section(config, "A").get("world_model", {}) or {}).get("ego_next_coef", 0.1)
    )
    # lane P3-F：未来 LD 监督恢复（LD 属 WM，应在 A 学会）——权重与 od 同量级（od 隐式 1.0）
    ld_coef = float(
        args.wm_ld_coef
        if args.wm_ld_coef is not None
        else dict(_stage_section(config, "A").get("world_model", {}) or {}).get("ld_coef", 0.0)
    )
    presence_state = {"available": 0.0, "warning": False}

    def _to_tensor(batch: Mapping[str, np.ndarray]) -> Dict[str, "torch.Tensor"]:
        return to_device_tensors(batch, device, dtype=torch.float32, pin=use_pin)

    def _future_targets(
        batch_indices: np.ndarray,
        obs_np: Mapping[str, np.ndarray],
        view: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """未来目标（精确查表 + ``wm_valid`` 门控 + 身份匹配）。

        键契约（lane A 与本地等价实现一致）：``od_fut/ld_fut/od_mask/ld_mask/valid/wm_valid/
        od_mask_raw/od_presence_fut/od_presence_t0/od_id_fut/od_id_t0/ego_fut``。
        仅当数据集没有 ``od_id`` 时回退 v1 的最近邻槽位匹配。
        """
        view = view or {}
        wm_valid_all_local = view.get("wm_valid_all", wm_valid_all)
        windows_local = view.get("windows", windows)
        wm_valid = wm_valid_all_local[batch_indices] if wm_valid_all_local is not None else None
        future = windows_local.build_future(batch_indices, wm_valid=wm_valid)
        has_identity = "od_id_fut" in future and bool(np.any(np.asarray(future["od_id_fut"]) >= 0))
        if not has_identity and bool(args.match_future_slots):
            future = match_future_od_slots(
                future,
                obs_np["od"],
                obs_np["od_mask"],
                gate_m=float(args.match_gate_m),
            )
        future.setdefault("wm_valid", future["valid"])
        future.setdefault("od_mask_raw", future["od_mask"])
        # 目标帧可用性 = 存在 ∧ usable（显式 wm_valid 与查表存在性取交，口径统一）
        future["wm_valid"] = np.asarray(future["wm_valid"], dtype=np.float32) * np.asarray(
            future["valid"], dtype=np.float32
        )
        return future

    # 快路径：obs + 未来目标一次性物化（训练循环只做切片 + H2D；见 pipeline.trainer）
    obs_source: Optional[MaterializedBCDataset] = None
    future_source: Optional[_ArrayBatchSource] = None
    if use_materialized:
        obs_source = MaterializedBCDataset(dataset, include_targets=False)
        future_source = _materialize_future_targets(
            _future_targets, obs_source, dataset.count, batch_size=batch_size
        )

    def _build_val_view() -> Optional[Dict[str, Any]]:
        """Stage A 的独立 val 视图（lane T）：val-dir 的全部行按同一窗口/权重口径评估。"""
        if val_dataset is None:
            return None
        val_arrays = val_dataset.arrays
        return {
            "dataset": val_dataset,
            "windows": FrameWindows(
                val_arrays,
                val_dataset.alignments,
                episode_key="episode_id",
                step_key="step",
                keys=_bc_channel_keys(),
                step_stride=_BC_STEP_STRIDE,
            ),
            "train_weight_all": row_action_weights(
                val_dataset, np.arange(val_dataset.count, dtype=np.int64)
            ),
            "wm_valid_all": (
                np.asarray(val_arrays["wm_valid"], dtype=np.float32) if "wm_valid" in val_arrays else None
            ),
            "action_all": np.asarray(val_arrays["action"], dtype=np.float32),
            "traj6_all": np.asarray(val_arrays["traj6"], dtype=np.float32),
            "obs_source": None,
            "future_source": None,
        }

    def _wm_predictions(
        obs: Mapping[str, "torch.Tensor"],
        future: Mapping[str, np.ndarray],
        plan: "torch.Tensor",
        *,
        noise: bool,
    ) -> Dict[str, "torch.Tensor"]:
        """WM 前向：mem + ST-GNN 教师强制 + **plan head ``ego_next`` 监督**（net v2 唯一路径）。

        教师强制（design-v1.2 §2.3）：

        - 每步把**目标帧真实 ego**（``ego_fut``）挤入 mem 副本（无 ``ego_fut`` 时用
          ``ego_next_features`` 解析构造），reserved 两维写入专家 GT 动作（可加噪声）；
        - 合成/教师帧一律 ``detach``（state 切），但 st_gnn 输出到 encoder 的梯度保留；
        - 每步调用 ``model.st_gnn(step_index=k)`` 得 t0 帧预测（直接多步，无递归误差累积）；
        - **plan head/MoE 梯度（方案①）**：每步挤入 GT 帧**之前**，用同一教师强制 mem 跑
          plan head 得 ``ego_next_pred``（预测第 k 帧 ego 前 6 维，返回键 ``ego_next_pred``）。
          若只喂 GT ego 而不取该预测，plan head/MoE 在 Stage A 无任何梯度（旧 bug）。
        """
        encoded = model.encode(obs)
        mem = encoded["mem"].clone()
        enc0 = encoded["encoded"]
        nav_token = encoded["nav_token"]
        signal_token = encoded["signal_token"]
        anchor_od = model.st_gnn.od_state_from_features(enc0.od_now, enc0.od_live)
        anchor_ld = model.st_gnn.ld_state_from_features(enc0.ld_now, enc0.ld_live)
        ego_fut = future.get("ego_fut")
        horizon = int(plan.shape[1])
        ego_now = obs["ego"]
        if ego_now.ndim == 3:
            ego_now = ego_now[:, 0]
        predictions: Dict[str, List["torch.Tensor"]] = {
            "od_pred": [], "ld_pred": [], "presence_pred": [], "entry_pred": [], "ego_next_pred": []
        }
        analytic_ego = None
        if ego_fut is None:
            try:
                from net.model import ego_next_features  # type: ignore

                analytic_ego = ego_next_features
            except Exception:  # noqa: BLE001
                analytic_ego = None
        enc_plan = enc0  # plan head 视角：mem 含 ≤ k-1 帧（第 k 帧挤入前）
        for k in range(1, horizon + 1):
            # (i) plan head 预测第 k 帧 ego 特征（唯一梯度来源，权重见 ego_next_coef）
            _, ego_next_pred, _ = model.plan_step(enc_plan, nav_token, signal_token)
            predictions["ego_next_pred"].append(ego_next_pred)
            # (ii) 教师强制：挤入目标帧真实 ego（detach），ST-GNN 单步推演
            raw_action = plan[:, k - 1]
            if noise and bool(args.plan_noise) and float(args.plan_noise_p) > 0.0:
                hit = (torch.rand_like(raw_action) < float(args.plan_noise_p)).to(raw_action.dtype)
                raw_action = raw_action + torch.randn_like(raw_action) * noise_std * hit
            if ego_fut is not None:
                ego_frame = torch.as_tensor(
                    np.asarray(ego_fut)[:, k - 1], dtype=torch.float32, device=device
                )
                if ego_frame.ndim == 3 and ego_frame.shape[1] == 1:
                    ego_frame = ego_frame[:, 0, :]
            elif analytic_ego is not None:
                ego_frame = analytic_ego(
                    ego_now, raw_action[:, 0], raw_action[:, 1], dt=float(model.dt), prev_speed=ego_now[:, 0]
                )
            else:  # 兜底：复制当前帧 + GT 动作（无 ego 历史时仍保留动作条件）
                ego_frame = torch.cat([ego_now[:, :6], raw_action], dim=-1)
            ego_frame = torch.cat([ego_frame[..., :6], raw_action], dim=-1)  # reserved 维 = 该步 GT 动作
            mem.shift_ego(ego_frame.detach())
            enc_k = model.mem_encoder.encode(model.encoders, mem)
            od_pred_k, ld_pred_k, presence_k, entry_k = model.st_gnn(
                ego_ctx=enc_k.ego_ctx,
                od_ctx=enc_k.od_ctx,
                ld_ctx=enc_k.ld_ctx,
                node_mask=enc_k.frame.node_mask,
                pose=enc_k.frame.pose,
                step_index=k,
                od_anchor=anchor_od,
                ld_anchor=anchor_ld,
            )
            predictions["od_pred"].append(od_pred_k)
            predictions["ld_pred"].append(ld_pred_k)
            predictions["presence_pred"].append(presence_k)
            predictions["entry_pred"].append(entry_k)
            enc_plan = enc_k  # 下一轮 plan head 看到 ≤ k 帧
        return {
            key: torch.stack(values, dim=1) for key, values in predictions.items()
        }

    def _term_denominators(
        future: Mapping[str, np.ndarray], frame_weight_np: np.ndarray
    ) -> Dict[str, float]:
        """宏/micro 口径的损失分母（纯数据计算，无需前向）。

        - ``od``：``Σ w·valid·od_mask``（= ``weighted_od_multi_step_loss`` 的分母）；
        - ``ld``：``Σ w·valid·ld_mask``（= ``weighted_ld_multi_step_loss`` 的分母；lane P3-F）；
        - ``step``：``Σ w·valid``（= ``ego_next`` 的 ``step_weight`` 分母）；
        - ``presence``：``step × 槽位数``（= presence/entry BCE 的分母口径）。
        """
        wm_valid = np.asarray(future["wm_valid"], dtype=np.float64)
        od_mask = np.asarray(future["od_mask"], dtype=np.float64)
        ld_mask = np.asarray(future["ld_mask"], dtype=np.float64)
        weight = np.asarray(frame_weight_np, dtype=np.float64).reshape(-1, 1)
        od_weight = od_mask * wm_valid[:, :, None] * weight[:, :, None]
        ld_weight = ld_mask * wm_valid[:, :, None] * weight[:, :, None]
        step_weight = weight * wm_valid
        slots = int(od_mask.shape[-1]) if od_mask.ndim >= 2 else 0
        return {
            "od": float(od_weight.sum()),
            "ld": float(ld_weight.sum()),
            "step": float(step_weight.sum()),
            "presence": float(step_weight.sum()) * float(slots),
        }

    def _scale_factors(
        micro: Mapping[str, float], macro: Mapping[str, float]
    ) -> Dict[str, float]:
        """micro → 宏口径的精确缩放因子（``Σ_m s_m·L_m = L_macro``；宏分母 0 → 该 term 恒 0）。"""
        return {
            key: (float(micro[key]) / float(macro[key]) if float(macro[key]) > 0.0 else 0.0)
            for key in macro
        }

    def _prepare_inputs(
        batch_indices: np.ndarray,
        timing: Optional[Dict[str, float]] = None,
        view: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[
        Dict[str, "torch.Tensor"], Dict[str, np.ndarray], np.ndarray, "torch.Tensor", "torch.Tensor"
    ]:
        """obs/未来目标 + H2D（物化后只剩切片 + pin 拷贝）。

        返回 ``(obs, future_np, frame_weight_np, frame_weight, plan)``；``scale`` 型调用方
        用同一宏 batch 的输入，微批时再切张量/数组（GPU 切片廉价）。
        """
        data_started = time.perf_counter()
        view = view or {}
        obs_source_local = view.get("obs_source", obs_source)
        future_source_local = view.get("future_source", future_source)
        dataset_local = view.get("dataset", dataset)
        weight_all_local = view.get("train_weight_all", train_weight_all)
        action_all_local = view.get("action_all", action_all)
        if obs_source_local is not None:
            obs_np = obs_source_local.obs_batch(batch_indices)
        else:
            obs_np = dataset_local.build_obs_batch(batch_indices)
        if future_source_local is not None:
            future = future_source_local.batch(batch_indices)
        else:
            future = _future_targets(batch_indices, obs_np, view=view)
        frame_weight_np = np.asarray(weight_all_local[batch_indices], dtype=np.float32)
        obs = _to_tensor(obs_np)
        frame_weight = _tensor(frame_weight_np, dtype=torch.float32)
        plan = _tensor(action_all_local[batch_indices], dtype=torch.float32)
        if timing is not None:
            timing["data"] = timing.get("data", 0.0) + time.perf_counter() - data_started
        return obs, future, frame_weight_np, frame_weight, plan

    def _forward_terms(
        obs: Mapping[str, "torch.Tensor"],
        future: Mapping[str, np.ndarray],
        frame_weight: "torch.Tensor",
        plan: "torch.Tensor",
        *,
        noise: bool,
        scales: Optional[Mapping[str, float]] = None,
        auc_pool: Optional[Dict[str, List[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """一次前向（输入已备好）：OD 直接多步损失 + plan head ``ego_next`` + presence/entry BCE。

        ``scales``（梯度累积）：按宏 batch 口径缩放各损失项（``od``/``step``/``presence``），
        使 ``Σ_m s_m·L_m = L_macro``；``None`` = 不分片（数值与旧版逐位一致）。
        ``auc_pool``：presence/entry 的逐 micro scores/labels 汇入池，AUC 由调用方在宏 batch
        上一次性计算（训练路径 AUC 仅诊断；评估路径不传 → 旧行为）。
        """
        predictions = _wm_predictions(obs, future, plan, noise=noise)
        od_pred = predictions["od_pred"]
        od_target = model.st_gnn.od_state_from_features(_tensor(future["od_fut"]))
        wm_valid = _tensor(future["wm_valid"])
        od_loss, od_per_horizon = weighted_od_multi_step_loss(
            od_pred,
            od_target,
            _tensor(future["od_mask"]),
            frame_weight=frame_weight,
            valid=wm_valid,
        )
        # lane P3-F：未来 LD 监督恢复（与 OD 同构：smooth_l1[0,1,3] + 1-cos[2]；
        # 目标 = ``ld_fut`` 前 4 维（LD 预测空间 [dx,dy,heading,curvature]），掩码/valid 同 OD）
        ld_loss, ld_per_horizon = weighted_ld_multi_step_loss(
            predictions["ld_pred"],
            _tensor(future["ld_fut"])[..., :4],
            _tensor(future["ld_mask"]),
            frame_weight=frame_weight,
            valid=wm_valid,
        )
        terms: Dict[str, Any] = {
            "od": od_loss,
            "ld": ld_loss,
            "per_horizon": od_per_horizon,
            "ld_per_horizon": ld_per_horizon,
            "od_pred": od_pred,
            "future": future,
            "frame_weight": frame_weight,
            "obs": obs,
        }
        # plan head ``ego_next`` 监督（方案①）：目标 = 目标帧真实 ego 前 6 维；
        # 无 ``ego_fut``（旧 schema）→ 跳过并在 metrics 记 ego_next_available=0。
        ego_next_pred = predictions.get("ego_next_pred")
        ego_fut = future.get("ego_fut")
        if ego_next_pred is not None and ego_fut is not None:
            ego_target = _tensor(
                np.asarray(ego_fut)[:, :, : int(ego_next_pred.shape[-1])], dtype=torch.float32
            )
            horizon = min(int(ego_target.shape[1]), int(ego_next_pred.shape[1]))
            error = torch.nn.functional.smooth_l1_loss(
                ego_next_pred[:, :horizon] - ego_target[:, :horizon],
                torch.zeros_like(ego_target[:, :horizon]),
                beta=1.0,
                reduction="none",
            ).mean(dim=-1)  # (B,K)
            step_weight = frame_weight.reshape(-1, 1) * wm_valid[:, :horizon]
            terms["ego_next"] = (error * step_weight).sum() / step_weight.sum().clamp(min=1e-8)
            terms["ego_next_per_horizon"] = [
                float((error[:, k] * step_weight[:, k]).sum() / step_weight[:, k].sum().clamp(min=1e-8))
                if float(step_weight[:, k].sum()) > 0.0
                else float("nan")
                for k in range(horizon)
            ]
            terms["ego_next_available"] = True
        else:
            terms["ego_next_available"] = False
        if presence_coef > 0.0 or entry_coef > 0.0:
            presence_pred = predictions.get("presence_pred")
            if presence_pred is None:
                presence_state["available"] = 0.0
                if not presence_state["warning"]:
                    print(
                        "[stageA] WARN：net 无 od_presence_pred/od_entry_pred（TODO(lane)）→ "
                        "presence/entry 损失本轮跳过（presence_available=0）",
                        flush=True,
                    )
                    presence_state["warning"] = True
            else:
                presence_state["available"] = 1.0
                entry_pred = predictions["entry_pred"]
                # id 轴目标（v1.2；见 presence_entry_targets）：未来帧同 id 是否在盒内 / 新 id 是否出现
                presence_target_np, entry_target_np = presence_entry_targets(future)
                presence_target = _tensor(presence_target_np, dtype=torch.float32)
                entry_target = _tensor(entry_target_np, dtype=torch.float32)
                if presence_pred.ndim == 2:  # 兜底：只给当前帧的旧形状
                    presence_target = presence_target[:, 0, :]
                    entry_target = entry_target[:, 0, :]
                step_weight = frame_weight.reshape(-1, 1) * wm_valid if presence_pred.ndim == 3 else frame_weight
                presence_terms = presence_entry_loss(
                    presence_pred,
                    entry_pred,
                    presence_target,
                    entry_target,
                    frame_weight=step_weight,
                    collect=auc_pool,
                )
                terms["presence"] = presence_terms["presence"]
                terms["entry"] = presence_terms["entry"]
                terms["presence_auc"] = presence_terms["presence_auc"]
                terms["entry_auc"] = presence_terms["entry_auc"]
                terms["presence_pos_rate"] = presence_terms["presence_pos_rate"]
                terms["entry_pos_rate"] = presence_terms["entry_pos_rate"]
        if scales:
            # 梯度累积：把 micro 损失缩放到宏 batch 口径（MicroNum/MacroDen = L_micro·(Den_micro/Den_macro)）
            terms["od"] = terms["od"] * float(scales["od"])
            terms["ld"] = terms["ld"] * float(scales["ld"])
            if "ego_next" in terms:
                terms["ego_next"] = terms["ego_next"] * float(scales["step"])
            if "presence" in terms:
                terms["presence"] = terms["presence"] * float(scales["presence"])
                terms["entry"] = terms["entry"] * float(scales["presence"])
                terms["presence_auc"] = float("nan")
                terms["entry_auc"] = float("nan")
        total = terms["od"] + ld_coef * terms["ld"]
        if "ego_next" in terms:
            total = total + ego_next_coef * terms["ego_next"]
        if "presence" in terms:
            total = total + presence_coef * terms["presence"] + entry_coef * terms["entry"]
        terms["total"] = total
        return terms

    @torch.no_grad()
    def _evaluate(
        indices: np.ndarray, limit: int, view: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        model.eval()
        view = view or {}
        traj6_all_local = view.get("traj6_all", traj6_all)
        action_all_local = view.get("action_all", action_all)
        eval_idx = np.asarray(indices, dtype=np.int64)[: max(1, int(limit))]
        obs_eval, future_eval, _, frame_weight_eval, plan_eval = _prepare_inputs(eval_idx, view=view)
        terms = _forward_terms(obs_eval, future_eval, frame_weight_eval, plan_eval, noise=False)
        od_pred = terms["od_pred"]
        future = terms["future"]
        frame_weight = terms["frame_weight"]
        obs = terms["obs"]
        weight = _tensor(future["od_mask"])
        wm_valid = _tensor(future["wm_valid"])
        od_target = _tensor(future["od_fut"])[..., 0:2]
        model_ade, model_fde = _ade_fde(od_pred[..., 0:2], od_target, weight)
        # 匀速基线（t0 帧内：位置 + k·dt·相对速度；未学习时 WM 的先验与其同源）
        current = obs["od"]
        horizon = int(od_pred.shape[1])
        offset = current[..., 0:2].unsqueeze(1) + current[..., 2:4].unsqueeze(1) * 0.5 * torch.arange(
            1, horizon + 1, device=device
        ).reshape(1, horizon, 1, 1)
        cv_ade, cv_fde = _ade_fde(offset, od_target, weight)
        # lane B B2：A/B 同定义的 ego KPI（val 子集；WM rollout 6 点 vs 专家 traj6 + 首步动作）
        ego = ego_kpi_arrays(
            model(obs, rollout=True, world_model=True, wm_detach=True),
            {
                "traj6": _tensor(traj6_all_local[eval_idx], dtype=torch.float32),
                "action": _tensor(action_all_local[eval_idx], dtype=torch.float32),
            },
        )
        w_eval_sum = frame_weight.sum().clamp(min=1e-8)
        ego_action_err_weighted = float((ego["action_err"] * frame_weight).sum() / w_eval_sum)
        ego_traj_fde_m = float((ego["traj_fde_end"] * frame_weight).sum() / w_eval_sum)
        ego_traj_mae_weighted = [
            float((ego["traj_mae_point"][:, k] * frame_weight).sum() / w_eval_sum)
            for k in range(int(ego["traj_mae_point"].shape[1]))
        ]
        per_horizon: Dict[str, Dict[str, float]] = {}
        ego_next_per_horizon = terms.get("ego_next_per_horizon")
        for k in range(horizon):
            w_k = weight[:, k, :]
            model_ade_k, model_fde_k = _ade_fde(od_pred[:, k, :, 0:2], od_target[:, k], w_k)
            cv_ade_k, cv_fde_k = _ade_fde(offset[:, k], od_target[:, k], w_k)
            w_frame = frame_weight * wm_valid[:, k]
            slot_count = float(w_k.sum())
            per_horizon[f"h{k + 1}"] = {
                "loss": float(terms["per_horizon"][k]["loss"].detach()) if slot_count > 0.0 else float("nan"),
                "ld_loss": (
                    float(terms["ld_per_horizon"][k]["loss"].detach())
                    if float(terms["ld_per_horizon"][k]["weight"]) > 0.0
                    else float("nan")
                ),
                "model_ade": float(model_ade_k),
                "model_fde": float(model_fde_k),
                "cv_ade": float(cv_ade_k),
                "cv_fde": float(cv_fde_k),
                # 物理计数（与分组窗口的 n_updates 区分）：有效帧数 / 权重和 / 有效槽位和
                "valid_samples": float(future["valid"][:, k].sum()),
                "valid_weight_sum": float(w_frame.sum()),
                "slot_count": slot_count,
                "traj_mae_m": ego_traj_mae_weighted[k],
                "ego_next_loss": (
                    float(ego_next_per_horizon[k]) if ego_next_per_horizon is not None and k < len(ego_next_per_horizon)
                    else float("nan")
                ),
            }
        model.train()
        return {
            "eval_loss": float(terms["total"]),
            "eval_od_loss": float(terms["od"]),
            "eval_ld_loss": float(terms["ld"]),
            "ego_next_loss": float(terms["ego_next"]) if "ego_next" in terms else float("nan"),
            "ego_next_available": bool(terms.get("ego_next_available", False)),
            "presence_loss": float(terms["presence"]) if "presence" in terms else float("nan"),
            "entry_loss": float(terms["entry"]) if "entry" in terms else float("nan"),
            "presence_auc": float(terms.get("presence_auc", float("nan"))),
            "entry_auc": float(terms.get("entry_auc", float("nan"))),
            "presence_pos_rate": float(terms.get("presence_pos_rate", float("nan"))),
            "entry_pos_rate": float(terms.get("entry_pos_rate", float("nan"))),
            "model_ade": float(model_ade),
            "model_fde": float(model_fde),
            "cv_ade": float(cv_ade),
            "cv_fde": float(cv_fde),
            "ego_action_err_weighted": ego_action_err_weighted,
            "ego_traj_fde_m": ego_traj_fde_m,
            "per_horizon": per_horizon,
        }

    stage_cfg = _stage_section(config, "A")
    wm_cfg = dict(stage_cfg.get("world_model", {}) or {})
    epochs = int(args.wm_epochs or wm_cfg.get("epochs") or 10)
    ckpt_every = _resolve_ckpt_every(args, config)
    config_hash = config_snapshot_hash(config)
    start_epoch = _resume_epoch_of(resume_info, epochs)
    if resume_info:
        if start_epoch >= epochs:
            print(f"[stageA] 警告：resume ckpt 已完成 {start_epoch} ≥ {epochs} epochs → 无剩余训练", flush=True)
        if resume_info.get("config_hash") and resume_info["config_hash"] != config_hash:
            print(
                f"[stageA] 警告：resume ckpt 配置哈希 {resume_info['config_hash']} != 当前 {config_hash}"
                "（配置快照已变，续跑结果可能不可比）",
                flush=True,
            )
    metrics: Dict[str, Any] = {
        "stage": "A",
        "kind": "world_model_teacher_forcing",
        "bc_dir": str(args.bc_dir),
        "dataset_schema": int(dataset.schema_version),
        "limit_dataset": (int(args.limit_dataset) if args.limit_dataset is not None else None),
        "samples": int(dataset.count),
        "train_frames": int(train_idx.size),
        "val_frames": int(val_idx.size),
        "val_episodes": int(val_episodes.size),
        "val_source": str(val_source),
        "val_dir": (str(val_dir) if val_dir is not None else ""),
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch,
        "eval_frames": int(eval_frames),
        "grad_accum": bool(use_accum),
        "lr": float(args.lr),
        "match_future_slots": bool(args.match_future_slots),
        "match_gate_m": float(args.match_gate_m),
        "plan_noise": {"p": float(args.plan_noise_p), "ds": float(args.plan_noise_ds), "dtheta": float(args.plan_noise_dtheta)},
        "presence_coef": presence_coef,
        "entry_coef": entry_coef,
        "ego_next_coef": ego_next_coef,
        "ld_coef": ld_coef,
        "presence_entry_axis": "id",
        # lane P3-F：未来 LD 监督恢复（LD 属 WM；目标 = ld_fut 前 4 维，与 od 同构直接多步）
        "ld_loss": "direct_multi_step",
        "materialize": bool(obs_source is not None),
        "fast_data": bool(obs_source is not None and future_source is not None),
        "ckpt_every": ckpt_every,
        "resume_epoch": start_epoch,
        "config_hash": config_hash,
    }
    if resume_info:
        metrics["resumed_from"] = resume_path
    metrics.update(dataset_contract)
    metrics.update(dataset_weight_report(dataset, prefix="dataset"))
    monitor = _make_monitor(
        out_dir / "monitor",
        enabled=_monitor_enabled(args, config),
        legacy_tags=_monitor_legacy_tags(args),
    )
    val_view = _build_val_view()
    rng = np.random.default_rng(int(args.seed))
    if start_epoch:
        if resume_info.get("rng_state"):
            restore_rng_state(resume_info["rng_state"], numpy_generator=rng)
            print("[stageA] resume：RNG 状态已恢复（数据顺序/plan 噪声续跑）", flush=True)
        else:
            for _ in range(start_epoch):
                rng.permutation(train_idx)
            print(f"[stageA] resume：无 RNG 状态 → seed 重放 {start_epoch} 次 permutation", flush=True)
    loss_curve: List[float] = []
    last_val: Dict[str, Any] = {}
    grad_probe_first: Dict[str, float] = {}
    grad_probe_last: Dict[str, float] = {}
    grad_prefixes = (
        "encoders.", "mem_encoder.", "plan_head.", "plan_head.moe.router.",
        "plan_head.moe.experts.", "st_gnn.", "policy.", "value.",
    )
    if use_accum:
        print(
            f"[stageA] 梯度累积：宏 batch={batch_size} · micro batch={micro_batch}"
            "（ST-GNN 6 步展开显存 ~32MB/样本；损失按宏 batch 口径精确缩放）",
            flush=True,
        )
    for epoch in range(start_epoch, epochs):
        order = rng.permutation(train_idx)
        totals: Dict[str, float] = {
            "total": 0.0, "od": 0.0, "ld": 0.0, "presence": 0.0, "entry": 0.0, "ego_next": 0.0
        }
        batches = 0
        data_seconds = forward_seconds = backward_seconds = 0.0
        for start in range(0, len(order), batch_size):
            batch_indices = order[start : start + batch_size]
            timing = {"data": 0.0}
            obs_macro, future_macro, weight_np_macro, weight_macro, plan_macro = _prepare_inputs(
                batch_indices, timing
            )
            if not use_accum:
                forward_started = time.perf_counter()
                terms = _forward_terms(obs_macro, future_macro, weight_macro, plan_macro, noise=True)
                forward_seconds += time.perf_counter() - forward_started
                backward_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                if not grad_probe_first:  # 首个 batch：plan head/MoE/编码器可微性自检
                    grad_probe_first = _grad_norms(model, grad_prefixes)
                grad_probe_last = _grad_norms(model, grad_prefixes)  # 训练结束时（router 需 experts 非零）
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                backward_seconds += time.perf_counter() - backward_started
                totals["total"] += float(terms["total"].detach())
                totals["od"] += float(terms["od"].detach())
                totals["ld"] += float(terms["ld"].detach())
                if "ego_next" in terms:
                    totals["ego_next"] += float(terms["ego_next"].detach())
                if "presence" in terms:
                    totals["presence"] += float(terms["presence"].detach())
                    totals["entry"] += float(terms["entry"].detach())
            else:
                # 梯度累积：micro 前向/反向 + 宏口径精确缩放（Σ_m s_m·L_m = L_macro），
                # optimizer 每宏 batch 一次；grad 探针/clip 与单 batch 路径同一位置。
                macro_denom = _term_denominators(future_macro, weight_np_macro)
                n_macro = int(batch_indices.shape[0])
                auc_pool: Dict[str, List[np.ndarray]] = {}
                optimizer.zero_grad(set_to_none=True)
                for m_start in range(0, n_macro, micro_batch):
                    m_lo, m_hi = m_start, min(m_start + micro_batch, n_macro)
                    obs_micro = {key: value[m_lo:m_hi] for key, value in obs_macro.items()}
                    plan_micro = plan_macro[m_lo:m_hi]
                    weight_micro = weight_macro[m_lo:m_hi]
                    future_micro = {key: value[m_lo:m_hi] for key, value in future_macro.items()}
                    scales = _scale_factors(
                        _term_denominators(future_micro, weight_np_macro[m_lo:m_hi]), macro_denom
                    )
                    forward_started = time.perf_counter()
                    terms_m = _forward_terms(
                        obs_micro,
                        future_micro,
                        weight_micro,
                        plan_micro,
                        noise=True,
                        scales=scales,
                        auc_pool=auc_pool,
                    )
                    forward_seconds += time.perf_counter() - forward_started
                    backward_started = time.perf_counter()
                    terms_m["total"].backward()
                    backward_seconds += time.perf_counter() - backward_started
                    totals["total"] += float(terms_m["total"].detach())
                    totals["od"] += float(terms_m["od"].detach())
                    totals["ld"] += float(terms_m["ld"].detach())
                    if "ego_next" in terms_m:
                        totals["ego_next"] += float(terms_m["ego_next"].detach())
                    if "presence" in terms_m:
                        totals["presence"] += float(terms_m["presence"].detach())
                        totals["entry"] += float(terms_m["entry"].detach())
                backward_started = time.perf_counter()
                if not grad_probe_first:
                    grad_probe_first = _grad_norms(model, grad_prefixes)
                grad_probe_last = _grad_norms(model, grad_prefixes)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                backward_seconds += time.perf_counter() - backward_started
            data_seconds += timing["data"]
            batches += 1
            if args.max_batches and batches >= int(args.max_batches):
                break
        train_loss = totals["total"] / max(1, batches)
        loss_curve.append(train_loss)
        eval_metrics = _evaluate(val_idx, eval_frames, view=val_view)
        last_val = eval_metrics
        counts = {
            "valid_samples": int(sum(item["valid_samples"] for item in eval_metrics["per_horizon"].values())),
            "valid_weight_sum": float(
                sum(item["valid_weight_sum"] for item in eval_metrics["per_horizon"].values())
            ),
        }
        metrics.update(
            {
                "last_epoch": epoch + 1,
                "wm_loss": train_loss,
                "wm_loss_od": totals["od"] / max(1, batches),
                "wm_loss_ld": totals["ld"] / max(1, batches),
                "wm_loss_ego_next": totals["ego_next"] / max(1, batches),
                "wm_loss_presence": totals["presence"] / max(1, batches),
                "wm_loss_entry": totals["entry"] / max(1, batches),
                "ego_next_available": float(eval_metrics["ego_next_available"]),
                "ego_next_loss": eval_metrics["ego_next_loss"],
                "grad_norms_first_batch": dict(grad_probe_first),
                "grad_norms_last_batch": dict(grad_probe_last),
                "loss_curve": list(loss_curve),
                "val_loss": eval_metrics["eval_loss"],
                "val_loss_od": eval_metrics["eval_od_loss"],
                "val_loss_ld": eval_metrics["eval_ld_loss"],
                "presence_available": float(presence_state["available"]),
                "presence_loss": eval_metrics["presence_loss"],
                "entry_loss": eval_metrics["entry_loss"],
                "presence_auc": eval_metrics["presence_auc"],
                "entry_auc": eval_metrics["entry_auc"],
                "presence_pos_rate": eval_metrics["presence_pos_rate"],
                "entry_pos_rate": eval_metrics["entry_pos_rate"],
                "per_horizon": eval_metrics["per_horizon"],
                "model_ade": eval_metrics["model_ade"],
                "model_fde": eval_metrics["model_fde"],
                "cv_ade": eval_metrics["cv_ade"],
                "cv_fde": eval_metrics["cv_fde"],
                "beats_cv_ade": bool(eval_metrics["model_ade"] < eval_metrics["cv_ade"]),
                "beats_cv_fde": bool(eval_metrics["model_fde"] < eval_metrics["cv_fde"]),
                "batches": batches,
                "val_valid_samples": counts["valid_samples"],
                "val_valid_weight_sum": counts["valid_weight_sum"],
                "epoch_data_seconds": data_seconds,
                "epoch_forward_seconds": forward_seconds,
                "epoch_backward_seconds": backward_seconds,
                "epoch_batches_per_sec": batches / max(data_seconds + forward_seconds + backward_seconds, 1e-9),
            }
        )
        if monitor is not None:
            train_payload: Dict[str, Any] = {
                "wm_loss": train_loss,
                "wm_loss_od": metrics["wm_loss_od"],
                "wm_loss_ld": metrics["wm_loss_ld"],
                "wm_loss_ego_next": metrics["wm_loss_ego_next"],
                "wm_loss_presence": metrics["wm_loss_presence"],
                "wm_loss_entry": metrics["wm_loss_entry"],
                "presence_auc": eval_metrics["presence_auc"],
                "entry_auc": eval_metrics["entry_auc"],
                "ego_action_err_weighted": eval_metrics["ego_action_err_weighted"],
                "ego_traj_fde_m": eval_metrics["ego_traj_fde_m"],
            }
            # OD KPI（val 子集，lane B B1）：逐 horizon ADE/FDE + 匀速基线（同族同图）
            horizon_payload: Dict[str, Dict[str, float]] = {}
            for k, item in enumerate(eval_metrics["per_horizon"].values()):
                group = {"ade": item["model_ade"], "fde": item["model_fde"]}
                if epoch == 0:
                    # val 集常量：只记首个 epoch（cv 基线不逐 epoch 重复）
                    group.update({"cv_ade": item["cv_ade"], "cv_fde": item["cv_fde"]})
                horizon_payload[f"h{k + 1}"] = group
            # ego 轨迹 KPI（val 子集，lane B B2）：逐 horizon MAE
            ego_horizon = {
                f"h{k + 1}": {"traj_mae_m": item["traj_mae_m"]}
                for k, item in enumerate(eval_metrics["per_horizon"].values())
            }
            monitor.on_train_step(train_payload, step=epoch + 1)
            monitor.on_grouped_step(horizon=horizon_payload, step=epoch + 1)
            monitor.on_val_grouped_step(horizon=ego_horizon, step=epoch + 1)
            monitor.flush(step=epoch + 1)
        print(
            f"[stageA] epoch {epoch + 1}/{epochs} loss={train_loss:.4f} val={eval_metrics['eval_loss']:.4f} "
            f"od={metrics['wm_loss_od']:.4f} ld={metrics['wm_loss_ld']:.4f} "
            f"ADE(WM/CV)={eval_metrics['model_ade']:.3f}/{eval_metrics['cv_ade']:.3f} "
            f"FDE={eval_metrics['model_fde']:.3f}/{eval_metrics['cv_fde']:.3f} "
            f"presence_avail={int(presence_state['available'])} "
            f"ego_next={int(eval_metrics['ego_next_available'])} "
            f"grad(plan_head)={grad_probe_last.get('plan_head', 0.0):.3f} "
            f"data={data_seconds:.2f}s fwd={forward_seconds:.2f}s bwd={backward_seconds:.2f}s "
            f"it/s={batches / max(data_seconds + forward_seconds + backward_seconds, 1e-9):.2f}",
            flush=True,
        )
        global_epoch = epoch + 1
        if ckpt_every > 0 and global_epoch % ckpt_every == 0:
            ckpt_path = _periodic_ckpt_path(out_dir, global_epoch)
            save_checkpoint(
                ckpt_path,
                model,
                meta={"stage": "A", "epoch": global_epoch, "epochs": epochs},
                optimizer=optimizer,
                epoch=global_epoch,
                val_metrics=last_val,
                rng_state=capture_rng_state(rng),
                config_hash=config_hash,
            )
            print(f"[stageA] ckpt → {ckpt_path}（epoch {global_epoch}/{epochs}）", flush=True)
    if monitor is not None:
        monitor.close()
    if device.type == "cuda":
        metrics["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(device) / 1e6)
    save_checkpoint(
        out_dir / "world_model.pt", model, meta=metrics, optimizer=optimizer, epoch=epochs,
        val_metrics=last_val, rng_state=capture_rng_state(rng), config_hash=config_hash,
    )
    save_checkpoint(
        out_dir / "final.pt", model, meta={"stage": "A", "epochs": epochs}, optimizer=optimizer,
        epoch=epochs, val_metrics=last_val, rng_state=capture_rng_state(rng), config_hash=config_hash,
    )
    metrics["checkpoint"] = str(out_dir / "final.pt")
    _write_json(out_dir / "metrics.json", metrics)
    print(f"[stageA] DONE → {out_dir}", flush=True)
    return metrics


# --------------------------------------------------------------------------- #
# 阶段 B：planner BC（primary → specific）
# --------------------------------------------------------------------------- #

#: Stage B val 标量里的阶段常量（val 集样本计数/权重和）：只在首个 epoch 记一次
#: （监控降噪；完整值仍在 metrics.json 的 phase ``val`` 快照里）
_VAL_CONSTANT_KEYS = frozenset(
    {
        "bc_action_err_count",
        "bc_action_err_weight",
    }
)

#: primary 段冻结：WM（ST-GNN）、value 头、8 个 specific experts + router（lane U1：MoE 关闭）
#: （可训练 = encoders/mem_encoder/plan head（primary + policy 路径）/policy）
#: primary 段冻结（lane U1）：MoE 关闭 → 专家与 router 都不参与（输出 = primary）
_PRIMARY_PHASE_FREEZE: Tuple[str, ...] = (
    "st_gnn.",
    "value.",
    "plan_head.moe.experts.",
    "plan_head.moe.router.",
)
#: specific 段冻结（lane U1 用户定稿）：**只训 experts + gate（router）+ residual_scale**；
#: 主干/primary 策略头/专家/policy 全冻（primary 输出保持 phase 1 结束时的口径）。
_SPECIFIC_PHASE_FREEZE: Tuple[str, ...] = (
    "st_gnn.",
    "value.",
    "encoders.",
    "mem_encoder.",
    "plan_head.fusion.",
    "plan_head.norm.",
    "plan_head.ego_next.",
    "plan_head.moe.primary.",
    "policy.",
)


def _action_mu_stats(
    model: Any,
    dataset: BCDataset,
    device: Any,
    *,
    batch_size: int = 256,
    obs_source: Optional[MaterializedBCDataset] = None,
) -> Dict[str, float]:
    """全数据集确定性前向（cheap path）：``action_mu`` 与专家动作的**计数/加权双口径**统计。

    - count 口径：``*_mean``（旧键，逐样本算术均值）；
    - weighted 口径：``*_weighted_mean``（``Σ w·x/Σ w``，w = ``train_weight×balance_weight``）；
    - ``*_weight``：权重和；``action_mu_ds_abs_err_weighted_mean``：|μ−专家| 加权误差。

    ``obs_source``（物化快路径）传入时只切片，否则回退逐样本 ``build_obs_batch``。
    """
    import torch

    from pipeline.trainer import row_action_weights, weighted_stats

    model.eval()
    total_ds, total_dtheta, count = 0.0, 0.0, 0
    weighted_ds, weighted_dtheta, weight_sum = 0.0, 0.0, 0.0
    errors: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, dataset.count, max(1, int(batch_size))):
            indices = np.arange(start, min(start + max(1, int(batch_size)), dataset.count), dtype=np.int64)
            obs_batch = (
                obs_source.obs_batch(indices) if obs_source is not None else dataset.build_obs_batch(indices)
            )
            obs = to_device_tensors(obs_batch, device, dtype=torch.float32, pin=obs_source is not None)
            out = model(obs, rollout=False, world_model=False)
            mu = out["action_mu"]
            expert = torch.as_tensor(
                dataset.arrays["action"][indices, 0], dtype=torch.float32, device=device
            )
            w = torch.as_tensor(row_action_weights(dataset, indices), dtype=torch.float32, device=device)
            total_ds += float(mu[:, 0].sum())
            total_dtheta += float(mu[:, 1].sum())
            weighted_ds += float((mu[:, 0] * w).sum())
            weighted_dtheta += float((mu[:, 1] * w).sum())
            weight_sum += float(w.sum())
            count += int(mu.shape[0])
            errors.append((mu - expert).abs().mean(dim=-1).detach().double().cpu().numpy())
            weights.append(w.detach().double().cpu().numpy())
    expert = np.asarray(dataset.arrays["action"], dtype=np.float64)[:, 0, :]
    expert_w = row_action_weights(dataset, np.arange(dataset.count, dtype=np.int64))
    error_stats = weighted_stats(np.concatenate(errors), np.concatenate(weights), prefix="")
    return {
        # count 口径（与旧日志可比）
        "action_mu_ds_mean": total_ds / max(count, 1),
        "action_mu_dtheta_mean": total_dtheta / max(count, 1),
        "expert_action_ds_mean": float(expert[:, 0].mean()),
        "expert_action_dtheta_mean": float(expert[:, 1].mean()),
        # weighted 口径（权重感知会计主口径）
        "action_mu_ds_weighted_mean": weighted_ds / max(weight_sum, 1e-8),
        "action_mu_dtheta_weighted_mean": weighted_dtheta / max(weight_sum, 1e-8),
        "expert_action_ds_weighted_mean": float(np.dot(expert[:, 0], expert_w) / max(expert_w.sum(), 1e-8)),
        "expert_action_dtheta_weighted_mean": float(np.dot(expert[:, 1], expert_w) / max(expert_w.sum(), 1e-8)),
        "action_mu_weight_sum": float(weight_sum),
        "action_mu_count": float(count),
        "action_mu_abs_err_weighted_mean": error_stats["weighted_mean"],
        "action_mu_abs_err_count": error_stats["count"],
    }


def _train_grouped_groups(
    metrics: Mapping[str, Any], label_names: Sequence[str]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """逐 epoch 训练标量 → ``(horizon, slices, labels)`` 三组 grouped 监控载荷。

    轨迹度量带单位（2026-09-26）：``traj_mse_m2``（加权 MSE，m²）/ ``traj_mae_m``（加权 MAE，m）/
    ``traj_err``（loss_type 口径 legacy）；无标签/切片样本时对应组缺失（监控显示为空）。
    """
    horizon: Dict[str, Dict[str, float]] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)) or value != value:
            continue
        if key.startswith("bc_traj_mse_h"):
            group, metric = key[len("bc_traj_mse_h"):], "traj_mse_m2"
        elif key.startswith("bc_traj_mae_h") and key.endswith("_m"):
            group, metric = key[len("bc_traj_mae_h"):-2], "traj_mae_m"
        elif key.startswith("bc_traj_err_h"):
            group, metric = key[len("bc_traj_err_h"):], "traj_err"
        else:
            continue
        if group.isdigit():
            horizon.setdefault(f"h{group}", {})[metric] = float(value)
    slices: Dict[str, Dict[str, float]] = {}
    for name in ("brake", "turn", "curve"):
        value = metrics.get(f"bc_action_err_slice_{name}_weighted_mean")
        if isinstance(value, (int, float)) and value == value:
            slices[name] = {"action_err": float(value)}
    labels: Dict[str, Dict[str, float]] = {}
    for label_name in label_names:
        value = metrics.get(f"bc_action_err_label_{label_name}")
        if isinstance(value, (int, float)) and value == value:
            labels[label_name] = {"action_err": float(value)}
    return horizon, slices, labels


def _val_grouped_groups(
    val_metrics: Mapping[str, Any]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """留出集指标（``evaluate_bc`` 的嵌套返回）→ ``(horizon, slices, labels)`` 三组载荷。"""
    horizon: Dict[str, Dict[str, float]] = {}
    for key, item in (val_metrics.get("per_horizon") or {}).items():
        if not isinstance(item, Mapping):
            continue
        group = str(key)
        name = group if group.startswith("h") else f"h{group}"
        horizon[name] = {
            "traj_mse_m2": item.get("traj_mse_m2"),
            "traj_mae_m": item.get("traj_mae_m"),
            "traj_err": item.get("traj_err"),
        }
    slices = {
        str(name): {"action_err": stats.get("weighted_mean")}
        for name, stats in (val_metrics.get("slices") or {}).items()
        if isinstance(stats, Mapping)
    }
    labels = {
        str(name): {"action_err": stats.get("weighted_mean")}
        for name, stats in (val_metrics.get("labels") or {}).items()
        if isinstance(stats, Mapping)
    }
    return horizon, slices, labels


def run_stage_b(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """阶段 B：planner BC（动作主损失 + rollout 轨迹辅助（WM detach）+ router 硬标签 CE）。

    留出集（2026-09-26）：与阶段 A 同 ``--val-frac``/``--seed`` 按 episode 切分（A/B 同一批
    留出 episode）；每 epoch 末在留出集上做无梯度确定性评估（:func:`pipeline.trainer.evaluate_bc`），
    epoch 行打印 ``val=``；monitor 命名：训练 ``train/<phase>_*``、留出 ``val/<phase>_*``，
    分组指标训练 ``horizon|slice|label/*``、留出 ``val/horizon|val/slice|val/label/*``。
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_thread_limits(workers=1, config=config)
    import torch

    device = resolve_device(args.device, config)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(_load_yaml(args.model_config))
    resume_path = _resume_path(args)
    resume_info: Dict[str, Any] = {}
    if resume_path:
        if not Path(resume_path).exists():
            raise SystemExit(f"[stageB] --resume 文件不存在：{resume_path}")
        resume_info = load_training_checkpoint(resume_path, model)
        print(
            f"[stageB] resume {resume_path}：模型已载入（missing={len(resume_info['missing_keys'])}）",
            flush=True,
        )
    else:
        ckpt = args.ckpt or "runs/train/stage_a/final.pt"
        if Path(ckpt).exists():
            meta = load_checkpoint(ckpt, model)
            print(f"[stageB] 载入阶段 A 产物 {ckpt}（missing={len(meta.get('missing_keys', []))}）", flush=True)
        else:
            print(f"[stageB] 警告：checkpoint {ckpt} 不存在 → 从随机初始化开始（world model 未训练）", flush=True)

    dataset = BCDataset.load(args.bc_dir, limit=args.limit_dataset)
    dataset_contract = validate_bc_dataset(
        dataset, str(args.bc_dir), "B", allow_legacy=_allow_legacy_dataset(args)
    )
    # lane T：训练侧 val 口径 = train-dir **全部行** + val-dir **全部行**（默认探测
    # ``datasets/*_expert500val``）；缺 val-dir → 告警回退 legacy「按 episode 比例切分」。
    val_dataset = None
    val_dir = _resolve_val_dir(args, args.bc_dir)
    if val_dir is not None:
        val_dataset = BCDataset.load(str(val_dir))
        validate_bc_dataset(val_dataset, str(val_dir), "B", allow_legacy=_allow_legacy_dataset(args))
        train_idx = np.arange(dataset.count, dtype=np.int64)
        val_idx = np.arange(val_dataset.count, dtype=np.int64)
        val_episodes = np.unique(val_dataset.arrays["episode_id"])
        val_source = f"dir:{val_dir}"
        print(
            f"[stageB] val-dir={val_dir}：train={train_idx.size} 行（train-dir 全部行）· "
            f"val={val_idx.size} 行（val-dir 全部行 · {val_episodes.size} episodes）",
            flush=True,
        )
    else:
        train_idx, val_idx, val_episodes = _episode_split(
            dataset.arrays["episode_id"], float(args.val_frac), int(args.seed)
        )
        val_source = "episode_split(legacy)"
        print(
            "[stageB] 警告：未找到独立 val-dir（--val-dir 或 datasets/*_expert500val）→ 回退 "
            "legacy 按 episode 比例切分（默认口径已改为 val-dir 全部行）",
            flush=True,
        )
    if val_idx.size == 0:
        print("[stageB] 警告：留出集为空（episode 数过少/val_frac=0）→ 用训练帧自评", flush=True)
        val_idx = train_idx
    if train_idx.size == 0:
        raise SystemExit("[stageB] 训练帧为空（留出比例过高）")
    stage_cfg = _stage_section(config, "B")
    bc_cfg = dict(stage_cfg.get("bc", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    epochs = int(args.bc_epochs or bc_cfg.get("epochs") or 10)
    split = float(args.bc_phase_split if args.bc_phase_split is not None else bc_cfg.get("primary_phase_split", 0.5))
    primary_epochs = max(1, int(round(epochs * split))) if epochs > 1 else epochs
    specific_epochs = max(0, epochs - primary_epochs)
    ckpt_every = _resolve_ckpt_every(args, config)
    config_hash = config_snapshot_hash(config)
    # resume：``epoch`` = 已完成的**全局** epoch 数（primary + specific 连续计数）
    resume_epoch = _resume_epoch_of(resume_info, epochs)
    if resume_info:
        raw_done = int(resume_info.get("epoch") or 0)
        suffix = "（超出总轮数 → 截断）" if raw_done > epochs else ""
        print(
            f"[stageB] resume：已完成全局 epoch={raw_done} → 从 epoch {resume_epoch + 1} 继续"
            f"（primary={primary_epochs} + specific={specific_epochs}）{suffix}",
            flush=True,
        )
        if resume_info.get("config_hash") and resume_info["config_hash"] != config_hash:
            print(
                f"[stageB] 警告：resume ckpt 配置哈希 {resume_info['config_hash']} != 当前 {config_hash}"
                "（配置快照已变，续跑结果可能不可比）",
                flush=True,
            )
    action_weight = float(args.action_weight if args.action_weight is not None else bc_cfg.get("action_weight", 1.0))
    traj_weight = float(args.traj_aux_weight if args.traj_aux_weight is not None else bc_cfg.get("traj_aux_weight", 0.1))
    loss_type = str(args.loss_type if args.loss_type is not None else bc_cfg.get("loss_type", "l2"))
    # lane U1：MoE 负载均衡 α + worst/mild 行权重（phase 2；CLI 优先，config 兜底）
    load_balance_coef = float(
        args.load_balance_coef
        if args.load_balance_coef is not None
        else bc_cfg.get("load_balance_coef", 0.01)
    )
    hard_weight = float(args.hard_weight if args.hard_weight is not None else bc_cfg.get("hard_weight", 1.0))
    mild_weight = float(args.mild_weight if args.mild_weight is not None else bc_cfg.get("mild_weight", 0.1))
    batch_size = int(
        args.batch_size or (train_cfg.get("bc", {}) or {}).get("batch_size") or DEFAULT_STAGE_BATCH_SIZE
    )
    micro_cfg = args.micro_batch_size if args.micro_batch_size is not None else (train_cfg.get("bc", {}) or {}).get(
        "micro_batch_size"
    )
    micro_batch_size = max(1, min(batch_size, int(micro_cfg))) if micro_cfg else None
    batch_size, micro_batch_size = _apply_smoke_batch_caps(
        args, device, batch_size, micro_batch_size, dataset.count
    )
    if _smoke_cpu_guard(args, device):
        print(
            f"[stageB] 冒烟内存保护（--limit-dataset + CPU）：上限 macro≤{_SMOKE_CPU_BATCH_CAP} · "
            f"micro≤{_SMOKE_CPU_MICRO_CAP} → 实际 macro={batch_size} micro={micro_batch_size}"
            "（CPU 前向图无显存上限；避免冒烟挤爆 host / 拖慢并行训练）",
            flush=True,
        )
    history_stride = int(args.history_stride if args.history_stride is not None else dataset.history_stride)
    use_materialized = bool(getattr(args, "materialize", False))

    # ---- lane U1：权重化 specific（去聚类）----
    # phase 1（primary）：MoE 关闭（专家/router 不参与，输出 = primary）→ 冻结 primary 策略头/专家；
    # phase 2（specific）：只训 experts+gate；worst-50% 行权重 1.0 / 其余 0.1（全量曝光）+ 负载均衡 aux。
    weight_sidecar_arg = str(getattr(args, "weight_sidecar", "") or "")
    mine_only = bool(getattr(args, "mine_only", False))
    worst_flags: Optional[np.ndarray] = None
    weight_meta: Dict[str, Any] = {}
    if weight_sidecar_arg:
        from pipeline.hard_mining import load_weight_sidecar

        side = load_weight_sidecar(
            weight_sidecar_arg,
            rows=int(dataset.count),
            obs_fingerprint=str(dataset.meta.get("obs_fingerprint") or ""),
        )
        worst_flags = (np.asarray(side["worst"]) > 0.5).astype(np.float32)
        weight_meta = dict(side.get("meta") or {})
        print(
            f"[stageB] 权重 sidecar={weight_sidecar_arg}：worst={int(worst_flags.sum())}/{worst_flags.size}"
            f"（ckpt={(weight_meta.get('ckpt') or {}).get('sha256', '')[:12]} · "
            f"hard_weight={hard_weight} · mild_weight={mild_weight}）",
            flush=True,
        )

    obs_source: Optional[MaterializedBCDataset] = None
    if use_materialized:
        obs_source = MaterializedBCDataset(dataset, include_targets=True)

    # WM 冻结：阶段 A 已训练；rollout 内合成帧再 detach（固定语义，wm_detach 为 no-op）→
    # 轨迹辅助损失不回传 ST-GNN。
    apply_freeze_prefixes(model, ("st_gnn.",))
    metrics: Dict[str, Any] = {
        "stage": "B",
        "kind": "planner_bc",
        "bc_dir": str(args.bc_dir),
        "dataset_schema": int(dataset.schema_version),
        "limit_dataset": (int(args.limit_dataset) if args.limit_dataset is not None else None),
        "samples": int(dataset.count),
        "device": device,
        "epochs": epochs,
        "primary_epochs": primary_epochs,
        "specific_epochs": specific_epochs,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "grad_accum": bool(micro_batch_size is not None and micro_batch_size < batch_size),
        "lr": float(args.lr),
        "action_weight": action_weight,
        "traj_aux_weight": traj_weight,
        # lane U1：去聚类（无 cluster/router 监督）；MoE 负载均衡 + 权重化 specific
        "moe_phase1_enabled": False,
        "load_balance_coef": float(load_balance_coef),
        "hard_weight": float(hard_weight),
        "mild_weight": float(mild_weight),
        "weight_sidecar": (weight_sidecar_arg or (str(out_dir / "weight_sidecar.npz") if not mine_only else "")),
        "worst_rows": (int(np.count_nonzero(worst_flags > 0.5)) if worst_flags is not None else None),
        "hard_frac": float(args.hard_frac),
        "dagger_dir": str(getattr(args, "dagger_dir", "") or ""),
        "wm_detach": True,
        "history_stride": history_stride,
        "materialize": bool(obs_source is not None),
        # lane U1：留出口径 = 独立 val-dir（全部行）或 legacy 比例切分
        "val_frac": float(args.val_frac),
        "val_source": str(val_source),
        "val_dir": (str(val_dir) if val_dir is not None else ""),
        "train_frames": int(train_idx.size),
        "val_frames": int(val_idx.size),
        "val_episodes": int(val_episodes.size),
        "ckpt_every": ckpt_every,
        "resume_epoch": resume_epoch,
        "config_hash": config_hash,
    }
    if resume_info:
        metrics["resumed_from"] = resume_path
    metrics.update(dataset_contract)
    metrics.update(dataset_weight_report(dataset, prefix="dataset"))
    monitor = _make_monitor(
        out_dir / "monitor",
        enabled=_monitor_enabled(args, config),
        legacy_tags=_monitor_legacy_tags(args),
    )
    if monitor is not None:
        monitor.on_train_step(
            {
                "load_balance_coef": float(load_balance_coef),
                "hard_weight": float(hard_weight),
                "mild_weight": float(mild_weight),
                "worst_flags_available": 1.0 if worst_flags is not None else 0.0,
            },
            step=0,
        )

    phase_state: Dict[str, Dict[str, Any]] = {}  # phase → 最近一次 epoch 回调的 optimizer/val/rng

    def _run_phase(
        phase: str,
        phase_epochs: int,
        freeze_prefixes: Sequence[str],
        phase_offset: int,
        *,
        start_epoch: int = 0,
        optimizer_state: Optional[Mapping[str, Any]] = None,
        rng_state: Optional[Mapping[str, Any]] = None,
        moe_enabled: bool = True,
        worst_flags: Optional[np.ndarray] = None,
        val_worst_flags: Optional[np.ndarray] = None,
        traj_aux_valid: Optional[np.ndarray] = None,
        val_dataset: Any = None,
        dataset_override: Any = None,
        train_idx_override: Optional[np.ndarray] = None,
        batch_source_override: Any = None,
    ) -> Dict[str, Any]:
        """跑一个 BC 相位：逐 epoch 落盘（step = ``phase_offset + epoch``，全局单调 1..2N）。

        - 训练侧：``train/<phase>_*`` 标量 + ``horizon|slice|label/*``（逐 epoch 增量）；
        - 留出侧：``val/<phase>_*`` 标量 + ``val/horizon|val/slice|val/label/*``；
        - 喂入 tag 均为旧命名，由 ``pipeline.monitoring`` 按 Tier-1 清单重命名/过滤
          （``stageB/<phase>/loss_terms`` / ``ego/*`` / ``router/*``；见 docs/metrics.md）；
          ``--monitor-legacy-tags`` 时全量落盘；
        - 阶段末只补记逐 epoch 未覆盖的标量（MoE 负载汇总/元数据），避免同 (step, tag) 重复；
        - resume：``start_epoch`` = 本相位已完成的本地 epoch 数（0 基）→ ``pretrain_bc`` 从该处续跑；
          ``optimizer_state``/``rng_state`` 只在 ckpt 与当前相位同源时传入（跨相位 → 全新优化器，
          与原跑法一致：optimizer/rng 本就按相位重建）。
        """
        if phase_epochs <= 0 or int(start_epoch) >= int(phase_epochs):
            return {"skipped": True, "epochs": 0, "phase": phase, "start_epoch": int(start_epoch)}
        epoch_keys: set = set()
        # lane U1：specific 段可换数据集/索引/物化源（--dagger-dir 行合并；None = 主集）
        phase_dataset = dataset if dataset_override is None else dataset_override
        phase_train_idx = train_idx if train_idx_override is None else train_idx_override
        phase_batch_source = obs_source if batch_source_override is None else batch_source_override

        def _on_checkpoint(
            epoch_index: int,
            ckpt_model: Any,
            ckpt_optimizer: Any,
            ckpt_val_metrics: Mapping[str, Any],
            ckpt_rng: Any,
        ) -> None:
            """周期 ckpt（全局 epoch 计数；与 final.pt 同格式/payload）。"""
            # 供阶段末 bc.pt/final.pt 复用（优化器/RNG/val 与训练循环同源）
            phase_state[phase] = {
                "optimizer": ckpt_optimizer,
                "val_metrics": dict(ckpt_val_metrics or {}),
                "rng": ckpt_rng,
            }
            global_epoch = phase_offset + epoch_index + 1
            if ckpt_every <= 0 or global_epoch % ckpt_every != 0:
                return
            ckpt_path = _periodic_ckpt_path(out_dir, global_epoch)
            save_checkpoint(
                ckpt_path,
                ckpt_model,
                meta={"stage": "B", "phase": phase, "epoch": global_epoch, "epochs": epochs},
                optimizer=ckpt_optimizer,
                epoch=global_epoch,
                val_metrics=ckpt_val_metrics,
                rng_state=capture_rng_state(ckpt_rng),
                config_hash=config_hash,
            )
            print(
                f"[stageB] ckpt → {ckpt_path}（phase={phase} · global epoch {global_epoch}/{epochs}）",
                flush=True,
            )

        def _on_epoch(
            epoch_index: int, train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any]
        ) -> None:
            if monitor is None:
                return
            step = phase_offset + epoch_index + 1
            train_scalars = {
                f"{phase}_{key}": value
                for key, value in train_metrics.items()
                if isinstance(value, (int, float))
            }
            epoch_keys.update(train_scalars)
            monitor.on_train_step(train_scalars, step=step)
            if val_metrics:
                monitor.on_val_step(
                    {
                        f"{phase}_{key}": value
                        for key, value in val_metrics.items()
                        if isinstance(value, (int, float))
                        and (epoch_index == 0 or key not in _VAL_CONSTANT_KEYS)
                    },
                    step=step,
                )
                val_horizon, val_slices, val_labels = _val_grouped_groups(val_metrics)
                monitor.on_val_grouped_step(
                    horizon=val_horizon, slices=val_slices, labels=val_labels, step=step
                )
            horizon_groups, slice_groups, label_groups = _train_grouped_groups(
                train_metrics, phase_dataset.label_names
            )
            monitor.on_grouped_step(
                horizon=horizon_groups, slices=slice_groups, labels=label_groups, step=step
            )
            monitor.flush(step=step)

        cfg = BCConfig(
            epochs=phase_epochs,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            lr=float(args.lr),
            loss_type=loss_type,
            traj_weight=traj_weight,
            action_weight=action_weight,
            seed=int(args.seed),
            device=device,
            max_batches=args.max_batches,
            wm_detach=True,
            freeze_prefixes=tuple(freeze_prefixes),
            phase=phase,
            history_stride=history_stride,
            train_indices=phase_train_idx,
            val_indices=val_idx,
            val_dataset=val_dataset,
            moe_enabled=bool(moe_enabled),
            load_balance_coef=float(load_balance_coef) if moe_enabled else 0.0,
            worst_flags=worst_flags,
            hard_weight=float(hard_weight),
            mild_weight=float(mild_weight),
            val_worst_flags=val_worst_flags,
            traj_aux_valid=traj_aux_valid,
            epoch_callback=_on_epoch,
            start_epoch=int(start_epoch),
            optimizer_state=optimizer_state,
            rng_state=rng_state,
            checkpoint_callback=_on_checkpoint,
        )
        result = pretrain_bc(
            model,
            phase_dataset,
            cfg,
            logger=print,
            batch_source=phase_batch_source,
        )
        if monitor is not None:
            summary = {
                f"{phase}_{key}": value
                for key, value in result.items()
                if isinstance(value, (int, float)) and f"{phase}_{key}" not in epoch_keys
            }
            if summary:
                monitor.on_train_step(summary, step=phase_offset + phase_epochs)
            monitor.flush(step=phase_offset + phase_epochs)
        return result

    # resume：ckpt 与当前相位同源 → 传优化器/RNG；跨相位边界 → 全新（原跑法本身按相位重建）
    mid_primary = 0 < resume_epoch < primary_epochs
    mid_specific = resume_epoch > primary_epochs
    # 2026-09-30：phase1-MoE 路线已由 W2/T2 闭环实验证伪（version_ledger），实验开关 STAGE_B_P1_MOE 移除；
    # phase 1 保持 MoE 关闭 + experts/router 冻结（lane U1 既有语义）。
    metrics["primary"] = _slim_phase_result(
        _run_phase(
            "primary",
            primary_epochs,
            _PRIMARY_PHASE_FREEZE,
            0,
            start_epoch=min(resume_epoch, primary_epochs),
            optimizer_state=(resume_info.get("optimizer") if mid_primary else None),
            rng_state=(resume_info.get("rng_state") if mid_primary else None),
            # lane U1 phase 1：MoE 关闭（专家/router 不参与；输出严格 = primary）
            moe_enabled=False,
            val_dataset=val_dataset,
        )
    )

    # ---- lane U1：primary 冻结（落 primary.pt）→ worst-50% 权重挖掘（全量曝光，不做硬子集）----
    tail_primary = phase_state.get("primary") or {}
    save_checkpoint(
        out_dir / "primary.pt",
        model,
        meta={"stage": "B", "phase": "primary", "epoch": primary_epochs, "moe_enabled": False},
        optimizer=tail_primary.get("optimizer"),
        epoch=primary_epochs,
        val_metrics=tail_primary.get("val_metrics"),
        rng_state=capture_rng_state(tail_primary.get("rng")),
        config_hash=config_hash,
    )
    print(f"[stageB] 冻结 primary → {out_dir / 'primary.pt'}（phase 2 参照）", flush=True)
    if worst_flags is None:
        do_mine = True if getattr(args, "mine_hard", None) is None else bool(args.mine_hard)
        if not do_mine:
            raise SystemExit(
                "[stageB] 未给 --weight-sidecar 且 --no-mine-hard → phase 2 没有 worst/mild 权重（拒绝静默）"
            )
        from pipeline.hard_mining import mine_hard_rows, write_weight_sidecar

        primary_ckpt = out_dir / "primary.pt"
        errors = row_il_errors(
            model, dataset, device=device, batch_size=batch_size, logger=print, obs_source=obs_source
        )
        mined = mine_hard_rows(
            errors["err_l1"], errors["err_weighted"], hard_frac=float(args.hard_frac)
        )
        sidecar_path = out_dir / "weight_sidecar.npz"
        write_weight_sidecar(
            sidecar_path,
            worst=mined["worst"],
            err_l1=errors["err_l1"],
            err_weighted=errors["err_weighted"],
            weight=errors["weight"],
            dataset_dir=str(args.bc_dir),
            dataset_meta=dataset.meta,
            ckpt=str(primary_ckpt),
            seed=int(args.seed),
            hard_frac=float(args.hard_frac),
            hard_weight=float(hard_weight),
            mild_weight=float(mild_weight),
            threshold=float(mined["threshold"]),
            order_rule=str(mined["order_rule"]),
            tool="pipeline.stages.run_stage_b",
        )
        worst_flags = mined["worst"].astype(np.float32)
        weight_meta = {
            "ckpt": ckpt_identity(str(primary_ckpt)),
            "worst_rows": int(mined["hard_rows"]),
            "total_rows": int(mined["total_rows"]),
            "threshold": float(mined["threshold"]),
            "hard_weight": float(hard_weight),
            "mild_weight": float(mild_weight),
        }
        print(
            f"[stageB] worst-50% 权重 → {sidecar_path}（worst={mined['hard_rows']}/{mined['total_rows']}"
            f" = {mined['hard_rows'] / max(1, mined['total_rows']):.1%}；hard_weight={hard_weight} · "
            f"mild_weight={mild_weight}；主口径=动作加权 IL 误差）",
            flush=True,
        )
        metrics["weight_sidecar"] = str(sidecar_path)
        metrics["worst_rows"] = int(mined["hard_rows"])

    if mine_only:
        # --mine-only：产出 primary.pt + weight_sidecar.npz 后停止（供审阅权重口径）
        metrics["mine_only"] = True
        metrics["worst_meta"] = dict(weight_meta)
        metrics["worst_rows"] = int(np.count_nonzero(worst_flags > 0.5))
        metrics["checkpoint"] = str(out_dir / "primary.pt")
        _write_json(out_dir / "metrics.json", metrics)
        print(
            f"[stageB] --mine-only DONE → {out_dir / 'primary.pt'} + {out_dir / 'weight_sidecar.npz'}"
            f"（worst={metrics['worst_rows']}/{worst_flags.size}）；"
            "下一步可带 --weight-sidecar 重跑 phase 2（MoE 开 + 权重 + 负载均衡）",
            flush=True,
        )
        if monitor is not None:
            monitor.close()
        return metrics

    # ---- lane U1 phase 2：specific（MoE 开 + worst/mild 权重 + Switch 式负载均衡 aux）----
    specific_dataset = dataset
    specific_train_idx = train_idx
    specific_batch_source = obs_source
    specific_worst = worst_flags
    specific_traj_valid: Optional[np.ndarray] = None  # lane U4：无 --dagger-dir → 全 1（None）
    val_worst = worst_flags if val_dataset is None else None  # legacy 切分：留出行与训练行同源
    dagger_info: Dict[str, Any] = {}
    dagger_dirs = [str(item) for item in (getattr(args, "dagger_dir", None) or []) if str(item).strip()]
    if dagger_dirs:
        (
            specific_dataset,
            specific_train_idx,
            specific_worst,
            specific_traj_valid,
            dagger_info,
        ) = _merge_dagger_rows(dataset, train_idx, worst_flags, dagger_dirs, logger=print)
        val_worst = (
            np.concatenate(
                [val_worst, np.ones(int(dagger_info["dagger_rows"]), dtype=np.float32)]
            )
            if val_worst is not None
            else None
        )
        specific_batch_source = (
            MaterializedBCDataset(specific_dataset, include_targets=True) if use_materialized else None
        )
        metrics["dagger"] = dict(dagger_info)
    metrics["specific"] = _slim_phase_result(
        _run_phase(
            "specific",
            specific_epochs,
            _SPECIFIC_PHASE_FREEZE,
            primary_epochs,
            start_epoch=max(0, resume_epoch - primary_epochs),
            optimizer_state=(resume_info.get("optimizer") if mid_specific else None),
            rng_state=(resume_info.get("rng_state") if mid_specific else None),
            # lane U1：MoE 开（输出 = primary + Σ_{i∈top2} g_i·expert_i，全场景）+ 行权重
            moe_enabled=True,
            worst_flags=specific_worst,
            val_worst_flags=val_worst,
            # lane U4：DAgger 行 traj-aux 掩码（无 --dagger-dir → None = 全 1）
            traj_aux_valid=specific_traj_valid,
            val_dataset=val_dataset,
            dataset_override=specific_dataset,
            train_idx_override=specific_train_idx,
            batch_source_override=specific_batch_source,
        )
    )
    metrics.update(
        _action_mu_stats(
            model, specific_dataset, device, batch_size=batch_size, obs_source=specific_batch_source
        )
    )
    if monitor is not None:
        monitor.close()
    if torch.device(device).type == "cuda":
        metrics["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(device) / 1e6)
    tail = phase_state.get("specific") or phase_state.get("primary") or {}
    save_checkpoint(
        out_dir / "bc.pt", model, meta=metrics, optimizer=tail.get("optimizer"), epoch=epochs,
        val_metrics=tail.get("val_metrics"), rng_state=capture_rng_state(tail.get("rng")),
        config_hash=config_hash,
    )
    save_checkpoint(
        out_dir / "final.pt", model, meta={"stage": "B", "epochs": epochs},
        optimizer=tail.get("optimizer"), epoch=epochs, val_metrics=tail.get("val_metrics"),
        rng_state=capture_rng_state(tail.get("rng")), config_hash=config_hash,
    )
    metrics["checkpoint"] = str(out_dir / "final.pt")
    metrics["worst_meta"] = dict(weight_meta)
    metrics["worst_flags_available"] = bool(worst_flags is not None)
    metrics["specific_data"] = (
        "全部训练行（worst=hard_weight / 其余=mild_weight）+ MoE(primary+top2 experts) + 负载均衡 aux"
    )
    _write_json(out_dir / "metrics.json", metrics)
    print(
        f"[stageB] DONE → {out_dir} action_mu_ds={metrics['action_mu_ds_mean']:.3f}m "
        f"（加权 {metrics['action_mu_ds_weighted_mean']:.3f}m；专家 {metrics['expert_action_ds_mean']:.3f}m）",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------------- #
# 阶段 B phase 3：迭代恢复训练（lane P3-B / P3-I）
# --------------------------------------------------------------------------- #

def _resolve_anchor_bc_dir(config: Mapping[str, Any]) -> str:
    """phase-3 5k 锚目录解析（lane P3-I；``anchor.bc_dir=auto`` 或留空时）。

    语义与 ``run.bc_dir: auto`` 一致：显式路径优先，否则 = 最新训练用
    ``datasets/BTC*_expert*``（须含 ``expert_bc.npz``；见 :func:`pipeline.run_paths.latest_dataset`）。
    """
    run_bc = str((config.get("run", {}) or {}).get("bc_dir") or "")
    if run_bc and run_bc.strip().lower() != "auto":
        return run_bc
    from pipeline.run_paths import latest_dataset

    found = latest_dataset()
    if found is None:
        raise SystemExit(
            "[stageB3] anchor.bc_dir=auto 但 datasets/ 下找不到 BTC*_expert*（含 expert_bc.npz）"
            "→ 显式给 --phase3-anchor-bc-dir 或 config stages.B.phase3.anchor.bc_dir"
        )
    return str(found)


def _merge_anchor_rows(
    dataset: BCDataset,
    train_idx: np.ndarray,
    anchor: BCDataset,
    anchor_dir: str,
    mild_weight: float,
    *,
    logger: Any = print,
) -> "tuple[BCDataset, np.ndarray, np.ndarray, Dict[str, Any]]":
    """lane P3-I：把 5k 锚数据集行并入 phase-3 训练集（窗口行在前、锚行在后）。

    - 同 schema 契约：锚必须覆盖窗口集的全部数组键（缺键 → ``SystemExit``，5k 与窗口同 schema）；
    - ``episode_id`` 累积偏移（窗口 max+1）→ 帧查表/动作链/未来目标跨集不冲突；
    - 锚行有效权重 = ``train_weight × 配平 × mild_weight``（锚段 ``train_weight`` 就地 ×
      ``mild_weight``，:func:`row_action_weights` 再乘配平；窗口行不动 = 权 1.0 口径）；
    - ``traj_aux_valid``：窗口行 0（dagger ``traj6`` 是常量外推合成值）、锚行 1（真实 traj6）。
      沿用 lane U4 掩码语义（只作用于 traj 损失项）；phase 3 ``traj_aux=0`` → 掩码仅记录在
      返回值/metrics，不参与损失（traj 只做监控）。
    """
    window_rows = int(dataset.count)
    anchor_rows = int(anchor.count)
    if anchor_rows <= 0:
        raise SystemExit(f"[stageB3] 锚数据集为空：{anchor_dir}")
    missing = [key for key in dataset.arrays if key not in anchor.arrays]
    if missing:
        raise SystemExit(
            f"[stageB3] 锚数据集 {anchor_dir} 与窗口集 schema 不一致（缺键 {missing[:6]}）→ 拒绝合并"
        )
    offset = int(np.max(np.asarray(dataset.arrays["episode_id"], dtype=np.int64))) + 1
    merged_arrays: Dict[str, np.ndarray] = {}
    for key, value in dataset.arrays.items():
        left = np.asarray(value)
        right = np.asarray(anchor.arrays[key])
        if key == "episode_id":
            right = right.astype(np.int64) + offset
        merged_arrays[key] = np.concatenate([left, right.astype(left.dtype)], axis=0)
    if "train_weight" in merged_arrays:
        merged_arrays["train_weight"][window_rows:] *= float(mild_weight)
    merged = BCDataset(merged_arrays, dict(dataset.meta))
    anchor_index = np.arange(window_rows, window_rows + anchor_rows, dtype=np.int64)
    merged_train_idx = np.concatenate([np.asarray(train_idx, dtype=np.int64), anchor_index])
    traj_aux_valid = np.concatenate(
        [np.zeros(window_rows, dtype=np.float32), np.ones(anchor_rows, dtype=np.float32)]
    )
    info: Dict[str, Any] = {
        "dir": str(anchor_dir),
        "window_rows": window_rows,
        "anchor_rows": anchor_rows,
        "merged_rows": window_rows + anchor_rows,
        "mild_weight": float(mild_weight),
        "episode_id_offset": int(offset),
        "anchor_row_base": int(window_rows),
        "train_rows": int(merged_train_idx.size),
        "window_traj_aux_masked": window_rows,
        "anchor_traj_aux_valid": anchor_rows,
    }
    logger(
        f"[stageB3] 5k 锚合并：窗口 {window_rows} + 锚 {anchor_rows}（{anchor_dir}）= "
        f"{window_rows + anchor_rows} 行；锚权 = train_weight×配平×{mild_weight:g}"
        f"（episode_id 偏移 +{offset}）；traj_aux：窗口 {window_rows} 行=0 / 锚 {anchor_rows} 行=1"
    )
    return merged, merged_train_idx, traj_aux_valid, info


def _phase3_future_fn(
    dataset: BCDataset,
    windows: FrameWindows,
    wm_valid_all: Optional[np.ndarray],
    *,
    match_future_slots: bool = True,
    match_gate_m: float = 8.0,
) -> Any:
    """构造 phase 3 未来目标闭包：t0 系对齐 + id 匹配 + presence/entry + 多步动作链。

    未来 OD/LD **必须**走 :meth:`FrameWindows.build_future`（内部 = lane A
    ``pipeline.frames.build_future`` 或本地等价实现；两者都把目标帧特征 SE(2) 对齐到
    **t0 自车系**，与 WM/rollout 的 t0 系预测语义一致），禁止直接取未来帧原始通道。
    """
    arrays = dataset.arrays

    def _future_targets(
        batch_indices: np.ndarray,
        obs_np: Mapping[str, np.ndarray],
        view: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        idx = np.asarray(batch_indices, dtype=np.int64).reshape(-1)
        wm_valid = wm_valid_all[idx] if wm_valid_all is not None else None
        future = windows.build_future(idx, wm_valid=wm_valid)
        has_identity = "od_id_fut" in future and bool(np.any(np.asarray(future["od_id_fut"]) >= 0))
        if not has_identity and match_future_slots:
            future = match_future_od_slots(
                future, obs_np["od"], obs_np["od_mask"], gate_m=float(match_gate_m)
            )
        future.setdefault("wm_valid", future["valid"])
        future["wm_valid"] = np.asarray(future["wm_valid"], dtype=np.float32) * np.asarray(
            future["valid"], dtype=np.float32
        )
        presence_target, entry_target = presence_entry_targets(future)
        future["presence_target"] = presence_target
        future["entry_target"] = entry_target
        chain, chain_valid = phase3_action_chain_targets(arrays, idx)
        future["action_chain"] = chain
        future["action_chain_valid"] = chain_valid
        return future

    return _future_targets


def run_stage_b_phase3(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """阶段 B phase 3：**迭代恢复训练**（单轮；lane P3-B，lane P3-I 配方开关）。

    - 数据 = ``--phase3 <dagger_dir>`` **单独使用**（无 worst/mild 权重、无 mining；当轮采集的
      失败窗口行）；``anchor.enabled=true``（lane P3-I，默认关）时并入 ``anchor.bc_dir``（5k）
      行，锚行权重 = ``train_weight×配平×anchor.mild_weight``，窗口行权重不变；
    - 起点 = ``--ckpt``（缺省读 config ``stages.B.phase3.init_ckpt``）；
    - **冻结配方** ``freeze``（lane P3-I，默认由 config 定）：
      ``specific_only`` = 只训 experts/router(gate)/residual_scale（复用 phase-2 冻结清单
      :data:`_SPECIFIC_PHASE_FREEZE`），WM/ego_next 无梯度 → 损失自动置 0（降级）；
      ``all`` = 全参数解冻（含 WM；旧行为）；
    - 损失 = 首步动作 + **多步动作链**（rollout ``plan`` 第 2..6 步 vs 未来专家首步标签）
      + plan head ``ego_next``（``ego_fut`` + ``wm_valid`` 尾部 mask）+ WM **OD/LD 教师强制**
      + presence/entry BCE + MoE 负载均衡 aux；``traj_*`` 只做监控（不进损失）；
    - LR 分组：base 主干 × ``lr_scale.base`` / experts+router+residual_scale × ``lr_scale.specific``；
    - 产出 ``<out>/final.pt`` + ``metrics.json`` + ``monitor/``（TB/CSV）+ 周期 ``ckpt_epoch{N}.pt``。

    ``spec_pool``/``fail_target``/``rounds``/``eval``/``guard`` 段为**编排 lane 的轮次协议**
    （采集/闭环评测/早停）；训练侧原样读取并写入 metrics.json（保证每轮口径可追溯）。
    """
    import torch

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_thread_limits(workers=1, config=config)
    device = resolve_device(args.device, config)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    stage_cfg = _stage_section(config, "B")
    p3_cfg = dict(stage_cfg.get("phase3", {}) or {})
    losses_cfg = dict(p3_cfg.get("losses", {}) or {})
    lr_scale_cfg = dict(p3_cfg.get("lr_scale", {}) or {})
    anchor_cfg = dict(p3_cfg.get("anchor", {}) or {})
    train_cfg = dict(config.get("train", {}) or {})
    dagger_dir = str(getattr(args, "phase3", "") or "")
    if not dagger_dir or not Path(dagger_dir).exists():
        raise SystemExit(f"[stageB3] --phase3 dagger 目录不存在：{dagger_dir!r}")
    round_index = max(1, int(getattr(args, "phase3_round", None) or 1))

    # ---- lane P3-I：配方开关（CLI > config；缺省 = all + 无锚 = 与旧行为一致）----
    freeze_raw = (
        args.phase3_freeze
        if getattr(args, "phase3_freeze", None) is not None
        else p3_cfg.get("freeze")
    )
    freeze_mode = str(freeze_raw or "all").strip().lower()
    if freeze_mode not in ("all", "specific_only"):
        raise SystemExit(f"[stageB3] freeze 非法：{freeze_mode!r}（all | specific_only）")
    freeze_prefixes: Tuple[str, ...] = () if freeze_mode == "all" else _SPECIFIC_PHASE_FREEZE
    frozen_loss_keys: Tuple[str, ...] = () if freeze_mode == "all" else PHASE3_WM_LOSS_KEYS
    anchor_enabled = (
        bool(args.phase3_anchor)
        if getattr(args, "phase3_anchor", None) is not None
        else bool(anchor_cfg.get("enabled", False))
    )
    anchor_bc_dir = str(
        getattr(args, "phase3_anchor_bc_dir", None) or anchor_cfg.get("bc_dir") or ""
    ).strip()
    anchor_mild_weight = float(
        args.phase3_anchor_mild_weight
        if getattr(args, "phase3_anchor_mild_weight", None) is not None
        else anchor_cfg.get("mild_weight", 0.1)
    )
    if anchor_mild_weight < 0.0:
        raise SystemExit(f"[stageB3] anchor.mild_weight 必须 ≥0：{anchor_mild_weight}")
    if anchor_enabled:
        if anchor_bc_dir.lower() in ("", "auto"):
            anchor_bc_dir = _resolve_anchor_bc_dir(config)
        if not Path(anchor_bc_dir).exists():
            raise SystemExit(
                f"[stageB3] anchor 数据集不存在：{anchor_bc_dir}"
                "（--phase3-anchor-bc-dir / config stages.B.phase3.anchor.bc_dir；auto=最新 datasets/BTC*_expert*）"
            )

    # ---- 起点权重：--ckpt 优先，否则 config stages.B.phase3.init_ckpt；缺失 = 拒绝静默随机初始化 ----
    model = build_model(_load_yaml(args.model_config))
    resume_path = _resume_path(args)
    resume_info: Dict[str, Any] = {}
    init_ckpt = ""
    if resume_path:
        if not Path(resume_path).exists():
            raise SystemExit(f"[stageB3] --resume 文件不存在：{resume_path}")
        resume_info = load_training_checkpoint(resume_path, model)
        print(
            f"[stageB3] resume {resume_path}：模型已载入（missing={len(resume_info['missing_keys'])}）",
            flush=True,
        )
    else:
        ckpt = str(args.ckpt or p3_cfg.get("init_ckpt") or "")
        if ckpt and Path(ckpt).exists():
            meta = load_checkpoint(ckpt, model)
            init_ckpt = ckpt
            print(
                f"[stageB3] 起点权重 {ckpt}（missing={len(meta.get('missing_keys', []))}）",
                flush=True,
            )
        else:
            raise SystemExit(
                f"[stageB3] 起点权重缺失：--ckpt 未给且 config stages.B.phase3.init_ckpt="
                f"{ckpt!r} 不存在（phase 3 固定从阶段 B phase 2 final 恢复，拒绝随机初始化）"
            )

    # ---- 数据：单个 dagger 目录（train 行 = 该轮采集行；val = episode 切分或显式 --val-dir）----
    dataset = BCDataset.load(dagger_dir, limit=args.limit_dataset)
    dataset_contract = validate_bc_dataset(
        dataset, dagger_dir, "B", allow_legacy=_allow_legacy_dataset(args)
    )
    arrays = dataset.arrays
    if "step" not in arrays:
        raise SystemExit("[stageB3] dagger 数据集缺少 'step'（未来目标/动作链需要精确查表）")
    val_dir_explicit = getattr(args, "val_dir", None)
    val_dataset = None
    if val_dir_explicit:
        val_path = Path(str(val_dir_explicit))
        if not val_path.exists():
            raise SystemExit(f"[stageB3] --val-dir 不存在：{val_path}")
        val_dataset = BCDataset.load(str(val_path))
        validate_bc_dataset(val_dataset, str(val_path), "B", allow_legacy=_allow_legacy_dataset(args))
        train_idx = np.arange(dataset.count, dtype=np.int64)
        val_idx = np.arange(val_dataset.count, dtype=np.int64)
        val_episodes = np.unique(val_dataset.arrays["episode_id"])
        val_source = f"dir:{val_path}"
    else:
        train_idx, val_idx, val_episodes = _episode_split(
            arrays["episode_id"], float(args.val_frac), int(args.seed)
        )
        val_source = "episode_split"
        if val_idx.size == 0:
            print("[stageB3] 警告：留出集为空（episode 过少/val_frac=0）→ 用训练帧自评", flush=True)
            val_idx = train_idx
    if train_idx.size == 0:
        raise SystemExit("[stageB3] 训练帧为空（留出比例过高）")

    # ---- lane P3-I：5k 锚行合并（默认关；val 仍只取窗口行）----
    anchor_info: Dict[str, Any] = {"enabled": False}
    traj_aux_valid: Optional[np.ndarray] = None
    if anchor_enabled:
        anchor_dataset = BCDataset.load(anchor_bc_dir, limit=args.limit_dataset)
        anchor_contract = validate_bc_dataset(
            anchor_dataset, anchor_bc_dir, "B", allow_legacy=_allow_legacy_dataset(args)
        )
        dataset, train_idx, traj_aux_valid, anchor_info = _merge_anchor_rows(
            dataset, train_idx, anchor_dataset, anchor_bc_dir, anchor_mild_weight, logger=print
        )
        anchor_info["enabled"] = True
        anchor_info["contract"] = dict(anchor_contract)
        # 掩码语义记录（U4：只作用于 traj 损失；phase 3 traj_aux=0 → 仅落 metrics/审计）
        anchor_info["traj_aux_valid_rows"] = int(traj_aux_valid.size)
        anchor_info["traj_aux_valid_sum"] = float(np.sum(traj_aux_valid))
        arrays = dataset.arrays
    windows = FrameWindows(
        arrays,
        dataset.alignments,
        episode_key="episode_id",
        step_key="step",
        keys=_bc_channel_keys(),
        step_stride=_BC_STEP_STRIDE,
    )
    wm_valid_all = np.asarray(arrays["wm_valid"], dtype=np.float32) if "wm_valid" in arrays else None

    # ---- 超参：CLI 优先 → config stages.B.phase3.* → stage B 锚点 ----
    epochs = int(args.phase3_epochs if args.phase3_epochs is not None else p3_cfg.get("epochs", 5))
    bc_train_cfg = dict(train_cfg.get("bc", {}) or {})
    batch_size = int(args.batch_size or p3_cfg.get("batch") or bc_train_cfg.get("batch_size") or DEFAULT_STAGE_BATCH_SIZE)
    micro_cfg = args.micro_batch_size if args.micro_batch_size is not None else (
        p3_cfg.get("micro") or bc_train_cfg.get("micro_batch_size")
    )
    micro_batch_size = max(1, min(batch_size, int(micro_cfg))) if micro_cfg else None
    batch_size, micro_batch_size = _apply_smoke_batch_caps(args, device, batch_size, micro_batch_size, dataset.count)
    if _smoke_cpu_guard(args, device):
        print(
            f"[stageB3] 冒烟内存保护（--limit-dataset + CPU）：上限 macro≤{_SMOKE_CPU_BATCH_CAP} · "
            f"micro≤{_SMOKE_CPU_MICRO_CAP} → 实际 macro={batch_size} micro={micro_batch_size}",
            flush=True,
        )

    def _loss_weight(key: str, default: float) -> float:
        cli = getattr(args, f"phase3_{key}_weight", None)
        return float(cli if cli is not None else losses_cfg.get(key, default))

    loss_weights = {
        "action": _loss_weight("action", 1.0),
        "action_chain": _loss_weight("action_chain", 0.2),
        "ego_next": _loss_weight("ego_next", 0.1),
        "od": _loss_weight("od", 1.0),
        "ld": _loss_weight("ld", 1.0),
        "presence": _loss_weight("presence", 0.1),
        "entry": _loss_weight("entry", 0.1),
        "traj_aux": _loss_weight("traj_aux", 0.0),
        "load_balance": float(
            args.load_balance_coef if args.load_balance_coef is not None else losses_cfg.get("load_balance", 0.01)
        ),
    }
    if loss_weights["traj_aux"] != 0.0:
        raise SystemExit(
            f"[stageB3] traj_aux={loss_weights['traj_aux']} 不支持：phase 3 的 traj 只做监控"
            "（失败窗口的 traj6 是常量外推合成值，不进损失；见设计定案）"
        )
    # lane P3-I：冻结下无梯度的上游监督项自动降级（配置值置 0；原始值留档 metrics）
    configured_loss_weights = dict(loss_weights)
    degraded_loss_weights = {
        key: float(loss_weights[key]) for key in frozen_loss_keys if float(loss_weights[key]) > 0.0
    }
    for key in frozen_loss_keys:
        loss_weights[key] = 0.0
    lr = float(args.lr)
    lr_base_scale = float(
        args.phase3_lr_base_scale if args.phase3_lr_base_scale is not None else lr_scale_cfg.get("base", 0.25)
    )
    lr_specific_scale = float(
        args.phase3_lr_specific_scale if args.phase3_lr_specific_scale is not None else lr_scale_cfg.get("specific", 0.5)
    )
    shuffle_seed_base_raw = p3_cfg.get("shuffle_seed_base")
    shuffle_seed = (
        int(shuffle_seed_base_raw) + round_index if shuffle_seed_base_raw is not None else int(args.seed)
    )
    history_stride = int(args.history_stride if args.history_stride is not None else dataset.history_stride)
    use_materialized = bool(getattr(args, "materialize", False))
    if history_stride != _BC_STEP_STRIDE:
        raise SystemExit(
            f"[stageB3] history_stride={history_stride} != {_BC_STEP_STRIDE}：动作链/未来目标按 0.5 s 策略步"
            "（step+5k）查表，禁止其他步距"
        )
    ckpt_every = _resolve_ckpt_every(args, config)
    config_hash = config_snapshot_hash(config)
    resume_epoch = _resume_epoch_of(resume_info, epochs)
    if resume_info.get("config_hash") and resume_info["config_hash"] != config_hash:
        print(
            f"[stageB3] 警告：resume ckpt 配置哈希 {resume_info['config_hash']} != 当前 {config_hash}"
            "（配置快照已变，续跑结果可能不可比）",
            flush=True,
        )
    freeze_text = (
        "全参数解冻（含 WM；freeze=all）"
        if freeze_mode == "all"
        else f"冻结配方=specific_only（只训 experts/router/residual_scale；"
        f"降级置 0：{', '.join(frozen_loss_keys)}）"
    )
    anchor_rows_log = int(anchor_info.get("anchor_rows") or 0) if anchor_enabled else 0
    anchor_text = (
        f"锚={anchor_bc_dir}（锚行={anchor_rows_log} · mild_weight={anchor_mild_weight:g}）"
        if anchor_enabled
        else "无锚（仅窗口 dagger 行）"
    )
    print(
        f"[stageB3] round={round_index} · dagger={dagger_dir}（窗口 rows={dataset.count - anchor_rows_log}）· "
        f"起点={init_ckpt or resume_path} · epochs={epochs} · {freeze_text} · {anchor_text} · "
        f"lr={lr:.2e}（base×{lr_base_scale} · specific×{lr_specific_scale}）· "
        f"shuffle_seed={shuffle_seed} · traj_aux={loss_weights['traj_aux']}（仅监控）"
        + (
            f" · 降级前权重：{degraded_loss_weights}"
            if degraded_loss_weights
            else ""
        ),
        flush=True,
    )

    # ---- 快路径：obs + 未来目标（含 LD/presence/动作链）一次性物化 ----
    obs_source: Optional[MaterializedBCDataset] = None
    future_source: Optional[_ArrayBatchSource] = None
    future_fn = _phase3_future_fn(
        dataset,
        windows,
        wm_valid_all,
        match_future_slots=bool(args.match_future_slots),
        match_gate_m=float(args.match_gate_m),
    )
    if use_materialized:
        obs_source = MaterializedBCDataset(dataset, include_targets=True)
        future_source = _materialize_future_targets(
            future_fn, obs_source, dataset.count, batch_size=batch_size, keys=_PHASE3_FUTURE_KEYS
        )
    val_future_fn = future_fn
    if val_dataset is not None:
        val_windows = FrameWindows(
            val_dataset.arrays,
            val_dataset.alignments,
            episode_key="episode_id",
            step_key="step",
            keys=_bc_channel_keys(),
            step_stride=_BC_STEP_STRIDE,
        )
        val_wm_valid = (
            np.asarray(val_dataset.arrays["wm_valid"], dtype=np.float32)
            if "wm_valid" in val_dataset.arrays
            else None
        )
        val_future_fn = _phase3_future_fn(
            val_dataset,
            val_windows,
            val_wm_valid,
            match_future_slots=bool(args.match_future_slots),
            match_gate_m=float(args.match_gate_m),
        )

    metrics: Dict[str, Any] = {
        "stage": "B",
        "kind": "planner_bc_phase3",
        "phase3": True,
        "phase3_dir": dagger_dir,
        "phase3_round": round_index,
        "phase3_rounds": int(p3_cfg.get("rounds", 5)),
        "spec_pool": str(p3_cfg.get("spec_pool", "")),
        "fail_target": (int(p3_cfg.get("fail_target", 0)) if p3_cfg.get("fail_target") is not None else None),
        "eval": dict(p3_cfg.get("eval", {}) or {}),
        "guard": dict(p3_cfg.get("guard", {}) or {}),
        "init_ckpt": init_ckpt,
        "resumed_from": (resume_path or ""),
        "dataset_schema": int(dataset.schema_version),
        "samples": int(dataset.count),
        "limit_dataset": (int(args.limit_dataset) if args.limit_dataset is not None else None),
        "train_frames": int(train_idx.size),
        "val_frames": int(val_idx.size),
        "val_episodes": int(val_episodes.size),
        "val_source": str(val_source),
        "val_dir": (str(val_dir_explicit) if val_dir_explicit else ""),
        "history_stride": history_stride,
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "grad_accum": bool(micro_batch_size is not None and micro_batch_size < batch_size),
        "lr": lr,
        "lr_scale": {"base": lr_base_scale, "specific": lr_specific_scale},
        # lane P3-I：配方开关（freeze/anchor）与降级留档
        "freeze": freeze_mode,
        "freeze_prefixes": list(freeze_prefixes),
        "frozen_loss_keys": list(frozen_loss_keys),
        "losses": dict(loss_weights),
        "losses_configured": configured_loss_weights,
        "losses_degraded": degraded_loss_weights,
        "anchor": dict(anchor_info),
        "loss_type": str(args.loss_type if args.loss_type is not None else "l2"),
        "traj_monitor_only": True,
        "wm_condition": "rollout_plan_detached",
        "materialize": bool(obs_source is not None),
        "shuffle_seed": int(shuffle_seed),
        "seed": int(args.seed),
        "ckpt_every": ckpt_every,
        "resume_epoch": resume_epoch,
        "config_hash": config_hash,
    }
    metrics.update(dataset_contract)
    metrics.update(dataset_weight_report(dataset, prefix="dataset"))
    monitor = _make_monitor(
        out_dir / "monitor",
        enabled=_monitor_enabled(args, config),
        legacy_tags=_monitor_legacy_tags(args),
    )
    if monitor is not None:
        monitor.on_train_step(
            {
                "phase3_round": float(round_index),
                "phase3_rows": float(dataset.count),
                "phase3_freeze_specific_only": float(freeze_mode == "specific_only"),
                "phase3_anchor_rows": float(anchor_info.get("anchor_rows") or 0),
                "phase3_lr_base": lr * lr_base_scale,
                "phase3_lr_specific": lr * lr_specific_scale,
            },
            step=0,
        )

    state: Dict[str, Any] = {}

    def _on_checkpoint(
        epoch_index: int,
        ckpt_model: Any,
        ckpt_optimizer: Any,
        ckpt_val_metrics: Mapping[str, Any],
        ckpt_rng: Any,
    ) -> None:
        state["optimizer"] = ckpt_optimizer
        state["val_metrics"] = dict(ckpt_val_metrics or {})
        state["rng"] = ckpt_rng
        global_epoch = epoch_index + 1
        if ckpt_every <= 0 or global_epoch % ckpt_every != 0:
            return
        ckpt_path = _periodic_ckpt_path(out_dir, global_epoch)
        save_checkpoint(
            ckpt_path,
            ckpt_model,
            meta={"stage": "B", "phase": "phase3", "round": round_index, "epoch": global_epoch, "epochs": epochs},
            optimizer=ckpt_optimizer,
            epoch=global_epoch,
            val_metrics=ckpt_val_metrics,
            rng_state=capture_rng_state(ckpt_rng),
            config_hash=config_hash,
        )
        print(f"[stageB3] ckpt → {ckpt_path}（round={round_index} · epoch {global_epoch}/{epochs}）", flush=True)

    def _on_epoch(epoch_index: int, train_metrics: Mapping[str, Any], val_metrics: Mapping[str, Any]) -> None:
        if monitor is None:
            return
        step = epoch_index + 1
        monitor.on_train_step(
            {f"phase3_{key}": value for key, value in train_metrics.items() if isinstance(value, (int, float))},
            step=step,
        )
        if val_metrics:
            monitor.on_val_step(
                {f"phase3_{key}": value for key, value in val_metrics.items() if isinstance(value, (int, float))},
                step=step,
            )
            val_horizon, val_slices, val_labels = _val_grouped_groups(val_metrics)
            monitor.on_val_grouped_step(horizon=val_horizon, slices=val_slices, labels=val_labels, step=step)
        horizon_groups, slice_groups, label_groups = _train_grouped_groups(
            train_metrics, dataset.label_names
        )
        monitor.on_grouped_step(horizon=horizon_groups, slices=slice_groups, labels=label_groups, step=step)
        monitor.flush(step=step)

    cfg = Phase3Config(
        epochs=epochs,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        lr=lr,
        loss_type=metrics["loss_type"],
        seed=int(shuffle_seed),
        device=device,
        max_batches=args.max_batches,
        freeze_mode=freeze_mode,
        freeze_prefixes=freeze_prefixes,
        train_indices=train_idx,
        val_indices=val_idx,
        val_dataset=val_dataset,
        epoch_callback=_on_epoch,
        start_epoch=resume_epoch,
        optimizer_state=(resume_info.get("optimizer") if resume_info else None),
        rng_state=(resume_info.get("rng_state") if resume_info else None),
        checkpoint_callback=_on_checkpoint,
        action_weight=loss_weights["action"],
        action_chain_weight=loss_weights["action_chain"],
        ego_next_weight=loss_weights["ego_next"],
        od_weight=loss_weights["od"],
        ld_weight=loss_weights["ld"],
        presence_weight=loss_weights["presence"],
        entry_weight=loss_weights["entry"],
        traj_aux_weight=loss_weights["traj_aux"],
        load_balance_coef=loss_weights["load_balance"],
        moe_enabled=True,
        lr_base_scale=lr_base_scale,
        lr_specific_scale=lr_specific_scale,
    )
    result = pretrain_bc_phase3(
        model,
        dataset,
        cfg,
        logger=print,
        batch_source=obs_source,
        future_source=future_source,
        future_fn=future_fn,
        val_future_fn=val_future_fn,
    )
    metrics.update(result)
    if monitor is not None:
        monitor.close()
    if torch.device(device).type == "cuda":
        metrics["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(device) / 1e6)
    tail = {k: state.get(k) for k in ("optimizer", "val_metrics", "rng")}
    save_checkpoint(
        out_dir / "final.pt",
        model,
        meta=metrics,
        optimizer=tail.get("optimizer"),
        epoch=epochs,
        val_metrics=tail.get("val_metrics"),
        rng_state=capture_rng_state(tail.get("rng")),
        config_hash=config_hash,
    )
    metrics["checkpoint"] = str(out_dir / "final.pt")
    _write_json(out_dir / "metrics.json", metrics)
    print(
        f"[stageB3] DONE round={round_index} → {out_dir / 'final.pt'}（rows={dataset.count} · "
        f"loss={metrics.get('bc_loss', float('nan')):.4f} · "
        f"od={metrics.get('bc_od_loss', float('nan')):.4f} · ld={metrics.get('bc_ld_loss', float('nan')):.4f}）",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------------- #
# 阶段 C：PPO RL（Stage-B 快照 KL 锚 + 衰减）
# --------------------------------------------------------------------------- #

def run_stage_c(args: argparse.Namespace, config: Mapping[str, Any]) -> Dict[str, Any]:
    """PPO：LqrTracker 闭环 + Stage-B 快照 KL 锚（衰减）+ 显式冻结范围（R2）+ primary lr ×0.1 + WM 全期冻结。

    P0-1（2026-09-30 修复，A-hold）：收集侧跟踪器参考 = ``repeat(a_t, 6)``——执行侧只依赖
    PPO 记账的随机变量；``plan_reference="plan"`` 仅作旧行为对照。
    P0-2（2026-09-30 修复）：router 标签在动作执行前按同帧观测取；终局 record 与 reset 分离。
    W1（2026-09-30）：``st_gnn.*`` 全期冻结（旧 ``--wm-freeze-updates`` 解冻无训练信号，已弃用；
    解冻守卫仅在 WM loss（W2）接线后可用）。
    R2/P0-6（2026-09-30）：``--trainable-scope``（config ``stages.C.trainable_scope``，默认
    ``design``）显式定义可训练范围——``design`` = allowlist（policy/value + MoE
    experts/residual_scale），其余含共享主干/primary/router/WM 全冻；"干净 PPO 基线"的
    冻结口径仅在 ``design`` 下成立，``all``（仅冻 st_gnn）为旧行为对照。
    V8r（2026-09-30，G1 §2）：router 移出 design allowlist（docs/db44fefe-system-review.md:127,240）。

    ``--critic-warmup-updates N``（config ``train.critic_warmup_updates``）：前 N 个
    update 只拟合 value 头（策略/主干冻结），之后恢复常规 PPO。
    """
    stage_cfg = _stage_section(config, "C")
    # R2/P0-6 冻结范围：CLI 优先，config stages.C.trainable_scope；默认 design（设计冻结）
    trainable_scope = str(args.trainable_scope or stage_cfg.get("trainable_scope", "design"))
    if trainable_scope not in STAGE_C_TRAINABLE_SCOPES:
        raise SystemExit(
            f"[stageC] 未知 trainable_scope={trainable_scope!r}（可选 {'/'.join(STAGE_C_TRAINABLE_SCOPES)}）"
        )
    scope_wording = (
        "design（P0-6 设计冻结：policy/value + MoE experts/residual_scale 可训；"
        "shared/encoders/primary/router/WM 冻结）"
        if trainable_scope == "design"
        else "all（旧行为：仅冻 st_gnn；全参数共享主干可训，非设计口径）"
    )
    print(
        "[stageC] ============================================================\n"
        "[stageC] P0-1 修复（A-hold）：references = repeat(a_t, 6)，执行只依赖记账动作。\n"
        "[stageC] P0-2 修复：router 标签步前同帧取 + 终局 record/reset 分离。\n"
        "[stageC] W1：WM（st_gnn）全期冻结（无 WM loss 信号；解冻被守卫禁止）。\n"
        "[stageC] V9（P2）：collect cheap path——plan_reference=repeat_action 时 rollout=False"
        "（与 update 同款）；plan 对照臂保留 rollout。\n"
        f"[stageC] R2/P0-6 trainable_scope={scope_wording}\n"
        "[stageC] 干净 PPO 基线的冻结口径以本行 trainable_scope 实际值为准。\n"
        "[stageC] 仍为实验口径：W2（WM 自监督）未接入（P2 性能优化进行中）。\n"
        "[stageC] ============================================================",
        flush=True,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = _resolve_specs(args, config)
    if not specs:
        raise SystemExit("[stages] 阶段 C 没有可用 spec")
    apply_thread_limits(workers=1, config=config)
    train_cfg = dict(config.get("train", {}) or {})
    # 固定探针批（诊断）：train.probe_batch；缺失 → 默认 datasets/BTC20260926-2343_expert5k，显式 null → 关闭
    probe_cfg = train_cfg.get("probe_batch", DEFAULT_PROBE_BATCH)
    probe_batch = None if probe_cfg is None else str(probe_cfg)
    device = resolve_device(args.device, config)
    model = build_model(_load_yaml(args.model_config))
    ckpt = args.ckpt or "runs/train/stage_b/final.pt"
    if Path(ckpt).exists():
        load_checkpoint(ckpt, model)
        print(f"[stageC] 载入阶段 B 策略快照 {ckpt}", flush=True)
    else:
        print(f"[stageC] 警告：checkpoint {ckpt} 不存在 → 从随机初始化开始", flush=True)
    primary_lr_scale = float(stage_cfg.get("primary_lr_scale", 0.1) or 0.1)
    # P0-1 A-hold：收集侧跟踪器参考口径（CLI 优先，config stages.C.plan_reference；默认 repeat_action）
    plan_reference = str(args.plan_reference or stage_cfg.get("plan_reference", "repeat_action"))
    updates = int(args.updates)
    # KL 锚 = 阶段 B 快照（冻结参考模型）；系数线性衰减（默认 0.05 → 0）
    ref_model = copy.deepcopy(model).eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    kl_initial = float(args.kl_anchor_coef)
    kl_final = float(args.kl_anchor_final_coef)
    kl_decay = bool(args.kl_anchor_decay)
    # critic warmup：前 N 个 update 只拟合 value 头（CLI 优先，否则 config train.critic_warmup_updates）
    critic_warmup_cfg = train_cfg.get("critic_warmup_updates", 0)
    critic_warmup_updates = int(
        args.critic_warmup_updates if args.critic_warmup_updates is not None else (critic_warmup_cfg or 0)
    )
    critic_warmup_updates = max(0, critic_warmup_updates)
    # W1（P1）：WM（ST-GNN）全期冻结。旧 ``--wm-freeze-updates`` 的"解冻"只是把参数重新
    # 加入优化器：PPO 更新走 rollout=False cheap path（不执行 st_gnn）+ 损失无 WM 项 ⇒ 梯度
    # 恒 None（P0 侦察 §B），故弃用旧解冻语义。
    if args.wm_freeze_updates is not None:
        print(
            "[stageC] --wm-freeze-updates 已弃用（W1）：st_gnn 全期冻结；"
            "解冻仅在 WM loss 启用时可用（当前无 WM loss → 不触发）",
            flush=True,
        )
    wm_freeze_updates = int(updates) if updates > 0 else 0  # 兼容 metrics 字段：全期冻结 ⇒ = updates
    frozen_wm = apply_freeze_prefixes(model, ("st_gnn.",))
    if not frozen_wm:
        raise SystemExit("[stageC] fail-fast：模型不含 st_gnn.* 参数，无法执行 W1 全期冻结")
    # WM 解冻守卫（W2 待做）：只有 WM loss 接线后才允许训练 st_gnn。当前无 WM 损失项，
    # 置 true 直接 fail-fast（无信号训练只烧显存/时间）。
    wm_loss_enabled = bool(stage_cfg.get("wm_loss_enabled", False))
    if wm_loss_enabled:
        raise SystemExit(
            "[stageC] fail-fast：wm_loss_enabled=true 但 WM 自监督损失尚未实现（W2）→ "
            "拒绝解冻 st_gnn"
        )
    if _wm_trainable(model):
        raise SystemExit("[stageC] fail-fast：st_gnn 冻结失败（仍有可训练参数）")
    print(
        f"[stageC] WM 冻结（W1）：st_gnn.* {len(frozen_wm)} 个参数全期 requires_grad=false"
        "（wm_trainable=false）",
        flush=True,
    )
    # R2/P0-6：design = 只保留 allowlist（policy/value + MoE specific）可训，其余全部冻结。
    # 与 build_optimizer 的 LR 分组同口径的"可训练参数组表"（前缀 → 参数量/LR）随启动打印。
    if trainable_scope == "design":
        frozen_scope = apply_trainable_allowlist(model, STAGE_C_DESIGN_PREFIXES)
        print(
            f"[stageC] 冻结范围（R2/P0-6）：design allowlist={list(STAGE_C_DESIGN_PREFIXES)}；"
            f"其余 {len(frozen_scope)} 个参数 requires_grad=false",
            flush=True,
        )
    trainable_groups = trainable_param_groups(model, float(args.lr), primary_lr_scale)
    if not trainable_groups:
        raise SystemExit("[stageC] fail-fast：可训练参数为空（scope 配置错误？）")
    for group in trainable_groups:
        print(
            f"[stageC] trainable group {group['prefix']:<28s} params={group['params']:>7d} "
            f"lr={group['lr']:.3e}",
            flush=True,
        )
    trainable_params = sum(int(group["params"]) for group in trainable_groups)

    bc_dataset = None
    if args.bc_anchor and Path(args.bc_dir).exists():
        bc_dataset = BCDataset.load(args.bc_dir, limit=args.limit_dataset)
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
        "experimental": True,
        "p0_fixes": {
            # R4：按实际 plan_reference 生成（消除 plan 臂下 p0_fixes 自称 A-hold 的自相矛盾）
            "P0-1": (
                "repeat_action(A-hold)"
                if plan_reference == "repeat_action"
                else "plan(legacy对照，未启用 A-hold)"
            ),
            "P0-2": "label_alignment(pre-step)",
            "W1": "wm_frozen",
        },
        "specs": len(specs),
        "primary_lr_scale": primary_lr_scale,
        "trainable_scope": trainable_scope,
        "trainable_params": trainable_params,
        "trainable_groups": [dict(group) for group in trainable_groups],
        "kl_anchor": {"initial": kl_initial, "final": kl_final, "decay": kl_decay, "source": str(ckpt)},
        "wm_freeze_updates": wm_freeze_updates,
        "wm_trainable": _wm_trainable(model),
        "wm_frozen_params": len(frozen_wm),
        "critic_warmup_updates": critic_warmup_updates,
        "device": device,
        "probe_batch": probe_batch,
        "plan_reference": plan_reference,
        "pool": {
            # R4：记真实池类型（LocalEnvPool / VectorPoolAdapter），不再只记请求 kind
            "kind": type(pool).__name__,
            "num_envs": int(getattr(pool, "num_envs", 1)),
            "tracker": str(getattr(pool, "tracker_kind", "kinematic")),
        },
        "tracker": str(getattr(pool, "tracker_kind", "kinematic")),
    }
    monitor = _make_monitor(
        out_dir / "monitor",
        enabled=_monitor_enabled(args, config),
        legacy_tags=_monitor_legacy_tags(args),
    )
    try:
        reward_adapter, reward_source = build_reward_adapter(logger=print)
        ppo_cfg = PPOConfig(
            lr=float(args.lr),
            epochs=int(args.ppo_epochs),
            minibatch_size=int(args.minibatch_size or (train_cfg.get("ppo", {}) or {}).get("minibatch_size") or 1024),
            kl_anchor_coef=kl_initial,
            bc_anchor_coef=float(args.bc_anchor_coef) if bc_dataset is not None else 0.0,
            primary_lr_scale=primary_lr_scale,
            critic_warmup_updates=critic_warmup_updates,
            plan_reference=plan_reference,
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
            probe_batch=probe_batch,
            logger=print,
        )
        initial = pool.reset()
        trainer.adopt_obs([record["obs"] for record in initial])
        if critic_warmup_updates > 0:
            print(
                f"[stageC] critic warmup：前 {critic_warmup_updates} 个 update 只拟合 value 头"
                "（策略/主干冻结，无 policy 更新）",
                flush=True,
            )
        history: List[Dict[str, Any]] = []
        kl_schedule: List[float] = []
        for update in range(updates):
            update_started = time.perf_counter()
            # W1 fail-fast：st_gnn 全期冻结，任何解冻迹象立刻报错（WM loss 未启用）。
            if not wm_loss_enabled and _wm_trainable(model):
                raise RuntimeError("[stageC] fail-fast：st_gnn 在训练中变为可训练（WM loss 未启用）")
            # W2（待做）：WM 自监督损失接线后才允许在此按 wm_freeze_updates 解冻（守卫见上）。
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
                f"critic_warmup={int(bool(update_metrics.get('critic_warmup', False)))} "
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
    # 资源指标（对照阶段 B vram_peak_mb）：Local 单进程另报进程 RSS（当前 + 峰值）
    try:
        import torch

        if torch.device(device).type == "cuda":
            metrics["vram_peak_mb"] = float(torch.cuda.max_memory_allocated(device) / 1e6)
    except Exception:  # noqa: BLE001 - 指标失败不阻塞产物
        pass
    try:
        import resource

        metrics["rss_peak_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        from pipeline.vector_env import process_rss_mb

        metrics["rss_mb"] = float(process_rss_mb())
    except Exception:  # noqa: BLE001
        pass
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
    parser.add_argument("--stage", choices=("A", "B", "C"), default=None,
                        help="A=WM 教师强制训练；B=planner BC（primary→specific）；"
                             "C=PPO RL（P0-1/P0-2 已修（A-hold/标签对齐）；WM 全期冻结；"
                             "冻结范围默认 design（P0-6 allowlist）；实验口径）；"
                             "--phase3 模式下可省略")
    parser.add_argument("--config", default="config/default.yaml", help="主配置（includes 合并）")
    parser.add_argument("--model-config", default=_DEFAULT_MODEL_CFG)
    parser.add_argument("--spec", default=None, help="阶段 C 训练 spec（默认取 config data.spec）")
    parser.add_argument("--geometry", default=None, help="阶段 C 按主几何标签过滤（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None, help="阶段 C spec 数上限")
    parser.add_argument("--out", default="runs/train")
    parser.add_argument("--ckpt", default=None,
                        help="A=初始化权重；B=阶段 A 产物（默认 runs/train/stage_a/final.pt）；"
                             "C=阶段 B 快照（默认 runs/train/stage_b/final.pt）")
    parser.add_argument("--ckpt-every", type=int, default=None,
                        help="A/B 周期 ckpt 间隔（epoch）：每 N 轮保存 <out>/ckpt_epoch{N:03d}.pt"
                             "（0=关；默认取 config/train.yaml train.ckpt_every=5）")
    parser.add_argument("--resume", default=None,
                        help="A/B 从周期 ckpt 恢复（模型/优化器/epoch/RNG，从 epoch+1 继续）；"
                             "无优化器状态/参数组不匹配时回退全新优化器（见日志）")
    parser.add_argument("--bc-dir", default="runs/bc_expert_full", help="阶段 A/B 的 BC 专家数据集目录")
    parser.add_argument("--limit-dataset", type=int, default=None,
                        help="BC 数据集前缀上限（冒烟用）：按**完整 episode 前缀**截断（读取阶段只解压前 M 行，"
                             "M ≤ N；首个 episode 超过 N 时保留该 episode），物化/val 切分/权重统计都基于该子集；"
                             "CPU 上同时把 macro/micro batch 收敛到 host 内存安全上限（macro≤128/micro≤8）")
    parser.add_argument("--allow-legacy-dataset", action="store_true",
                        help="放行 schema<2/缺 v2 通道（others/od_id/od_presence）的 BC 数据集。"
                             "默认硬失败（防漏设 BC_DIR 落到 v1 数据集、router 损失全程 0）；"
                             "等价环境变量 ALLOW_LEGACY_DATASET=1。放行时 metrics.json 标注 "
                             "legacy_dataset=true，训练结果可能作废")
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
    parser.add_argument("--wm-presence-coef", type=float, default=None,
                        help="阶段 A presence BCE 权重（默认取 config stages.A.world_model.presence_coef=0.1；"
                             "net 无 presence 头时自动跳过并记录 presence_available=0）")
    parser.add_argument("--wm-entry-coef", type=float, default=None,
                        help="阶段 A entry BCE 权重（默认取 config stages.A.world_model.entry_coef=0.1）")
    parser.add_argument("--wm-ego-next-coef", type=float, default=None,
                        help="阶段 A plan head ego_next 监督权重（默认取 config "
                             "stages.A.world_model.ego_next_coef=0.1；plan head/MoE 的唯一梯度来源）")
    parser.add_argument("--wm-ld-coef", type=float, default=None,
                        help="阶段 A 未来 LD 直接多步损失权重（默认取 config "
                             "stages.A.world_model.ld_coef=0.0；2026-09-30 拍板：LD 损失仅监控、不监督 LD 头）")
    # ---- 阶段 B ----
    parser.add_argument("--bc-epochs", type=int, default=None, help="阶段 B 总轮数（默认 config/stages.B.bc.epochs=10）")
    parser.add_argument("--bc-phase-split", type=float, default=None,
                        help="阶段 B primary 段占比（默认 config/stages.B.bc.primary_phase_split=0.5）")
    parser.add_argument("--action-weight", type=float, default=None, help="阶段 B 动作损失权重（默认 1.0，主项）")
    parser.add_argument("--traj-aux-weight", type=float, default=None, help="阶段 B rollout 轨迹辅助权重（默认 0.1）")
    parser.add_argument("--loss-type", choices=("l1", "l2"), default=None,
                        help="阶段 B 损失口径（默认取 config/stages.B.bc.loss_type=l2；l1 可减轻转向回归均值）")
    parser.add_argument("--history-stride", type=int, default=None,
                        help="历史/未来查表步距（env step；BC 默认 5 = 0.5 s）")
    # ---- lane U1：训练侧 val 口径 + 权重化 specific（去聚类）----
    parser.add_argument("--val-dir", type=str, default=None,
                        help="独立留出数据集目录（默认探测 train-dir 同级 *_expert500val）：训练读 train-dir "
                             "**全部行** + val-dir **全部行**；缺省回退 legacy 按 episode 比例切分并告警")
    parser.add_argument("--weight-sidecar", type=str, default=None,
                        help="worst/mild 权重 sidecar（tools/mine_hard.py 产物）；给了则跳过在线挖掘")
    parser.add_argument("--hard-frac", type=float, default=0.5, help="worst 行占比（默认 top-50%）")
    parser.add_argument("--mine-hard", action=argparse.BooleanOptionalAction, default=None,
                        help="primary 结束后在线挖掘 worst-50% 权重（默认开；--weight-sidecar 给出则跳过）")
    parser.add_argument("--hard-weight", type=float, default=None,
                        help="worst 行权重倍率（默认取 config stages.B.bc.hard_weight=1.0）")
    parser.add_argument("--mild-weight", type=float, default=None,
                        help="其余行权重倍率（默认取 config stages.B.bc.mild_weight=0.1）")
    parser.add_argument("--load-balance-coef", type=float, default=None,
                        help="MoE 负载均衡 aux α（Switch 式 α·E·Σ f_i·P_i；默认取 "
                             "config stages.B.bc.load_balance_coef=0.01）")
    parser.add_argument("--mine-only", action="store_true",
                        help="lane U1：只跑 phase 1（primary，MoE 关）+ worst 权重挖掘"
                             "（产出 primary.pt + weight_sidecar.npz）后停止；审阅后再带 --weight-sidecar 重跑")
    parser.add_argument("--dagger-dir", type=str, action="append", default=None,
                        help="DAgger 数据集目录（**可重复**：`--dagger-dir A --dagger-dir B`；phase 2 与主集行"
                             "合并训练，行权重 = 1.0；各份 episode_id 累积偏移；traj-aux 掩码 0）")
    # ---- lane P3-B：stage B phase 3（迭代恢复训练）----
    parser.add_argument("--phase3", type=str, default=None,
                        help="stage B phase 3 迭代恢复训练：当轮 DAgger 数据集目录（**单独使用**：无 5k、"
                             "无 worst/mild 权重、无 mining）；给出即启用本模式（--stage 可省），"
                             "起点权重走 --ckpt 或 config stages.B.phase3.init_ckpt")
    parser.add_argument("--phase3-round", type=int, default=1, help="迭代轮次编号（默认 1；写入 metrics/日志）")
    parser.add_argument("--phase3-epochs", type=int, default=None,
                        help="phase 3 轮数（默认取 config stages.B.phase3.epochs=5）")
    parser.add_argument("--phase3-action-weight", type=float, default=None, help="首步动作损失权重（默认 1.0）")
    parser.add_argument("--phase3-action-chain-weight", type=float, default=None,
                        help="多步动作链权重（默认取 config stages.B.phase3.losses.action_chain=0.2）")
    parser.add_argument("--phase3-ego-next-weight", type=float, default=None,
                        help="plan head ego_next 权重（默认取 config …losses.ego_next=0.1）")
    parser.add_argument("--phase3-od-weight", type=float, default=None,
                        help="WM OD 直接多步损失权重（默认 1.0）")
    parser.add_argument("--phase3-ld-weight", type=float, default=None,
                        help="WM LD 恢复监督权重（默认 1.0）")
    parser.add_argument("--phase3-presence-weight", type=float, default=None,
                        help="presence BCE 权重（默认 0.1）")
    parser.add_argument("--phase3-entry-weight", type=float, default=None,
                        help="entry BCE 权重（默认 0.1）")
    parser.add_argument("--phase3-traj-aux-weight", type=float, default=None,
                        help="轨迹辅助权重（默认 0.0：traj 只做监控，不进损失）")
    # ---- lane P3-I：配方开关（冻结 / 5k 锚）----
    parser.add_argument("--phase3-freeze", choices=("all", "specific_only"), default=None,
                        help="phase 3 冻结配方（默认取 config stages.B.phase3.freeze）："
                             "specific_only = 只训 experts/router/residual_scale（复用 phase-2 冻结清单；"
                             "WM od/ld/presence/entry 与 ego_next 无梯度 → 损失自动置 0）；"
                             "all = 全参数解冻（含 WM；旧行为）")
    parser.add_argument("--phase3-anchor", action=argparse.BooleanOptionalAction, default=None,
                        help="phase 3 5k 锚（默认取 config stages.B.phase3.anchor.enabled=false）："
                             "开启后把锚数据集行并入训练集（权 = train_weight×配平×mild_weight）")
    parser.add_argument("--phase3-anchor-bc-dir", type=str, default=None,
                        help="锚数据集目录（默认 config …anchor.bc_dir；auto = 最新 datasets/BTC*_expert*）")
    parser.add_argument("--phase3-anchor-mild-weight", type=float, default=None,
                        help="锚行权重倍率（默认取 config …anchor.mild_weight=0.1）")
    parser.add_argument("--phase3-lr-base-scale", type=float, default=None,
                        help="base 主干（encoders/mem_encoder/plan_head/st_gnn/WM/policy/value）LR 缩放"
                             "（默认取 config stages.B.phase3.lr_scale.base=0.25）")
    parser.add_argument("--phase3-lr-specific-scale", type=float, default=None,
                        help="experts/gate/residual_scale LR 缩放（默认取 config …lr_scale.specific=0.5）")
    # ---- 阶段 C ----
    parser.add_argument("--pool", choices=("auto", "vector", "local"), default="local")
    parser.add_argument("--trainable-scope", choices=STAGE_C_TRAINABLE_SCOPES, default=None,
                        help="阶段 C 可训练范围（R2/P0-6；默认取 config stages.C.trainable_scope=design）："
                             "design = 设计冻结 allowlist（policy/value + plan_head.moe.experts/"
                             "residual_scale；encoders/mem_encoder/fusion/norm/ego_next/primary/router/st_gnn 全冻）"
                             "——'干净 PPO 基线'仅在此口径成立；"
                             "all = 旧行为（仅冻 st_gnn，共享主干全 LR 可训，非设计口径）")
    parser.add_argument("--plan-reference", choices=("repeat_action", "plan"), default=None,
                        help="阶段 C 收集侧跟踪器参考口径（默认 repeat_action：6 步参考 = repeat(a_t)，"
                             "P0-1 A-hold；plan = 旧行为（首步外取 WM 规划预览），仅供对照）")
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
                        help="【已弃用（W1）】阶段 C 的 st_gnn 现为全期冻结；该参数仅触发弃用告警接收兼容")
    parser.add_argument("--critic-warmup-updates", type=int, default=None,
                        help="阶段 C 前 N 个 update 只拟合 critic（value 头；策略/主干冻结）。"
                             "默认取 config/train.yaml train.critic_warmup_updates=0（0=关闭）")
    parser.add_argument("--bc-anchor", action="store_true", help="阶段 C 启用 BC 动作锚（默认关）")
    parser.add_argument("--bc-anchor-coef", type=float, default=0.1)
    # ---- 通用 ----
    parser.add_argument("--batch-size", type=int, default=None,
                        help="A/B 默认取 config train.wm.batch_size / train.bc.batch_size（1024）")
    parser.add_argument("--materialize", action=argparse.BooleanOptionalAction, default=True,
                        help="A/B 快路径（默认开）：逐样本历史/未来目标一次性物化，训练循环只做切片 + "
                             "pin/non-blocking H2D；--no-materialize = 旧逐 batch 重建路径（等价性对照）")
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="梯度累积 micro batch（默认取 config train.wm/bc.micro_batch_size；"
                             "None/≥--batch-size = 不分片）。ST-GNN 6 步展开显存 ~32MB/样本，"
                             "12GB 卡 micro≤256/512；损失按宏 batch 口径精确缩放，宏 batch 一次更新")
    parser.add_argument("--max-batches", type=int, default=None, help="A/B 每轮批数上限（冒烟用）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--traffic-density", type=float, default=None)
    parser.add_argument("--mem-floor-mb", type=float, default=2000.0)
    parser.add_argument("--device", default=None,
                        help="默认取 config/train.yaml train.device（auto=cuda 可用则 cuda，否则 cpu）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--monitor", action=argparse.BooleanOptionalAction, default=None,
                        help="使用 pipeline.monitoring（tensorboard/CSV）；默认读 config/train.yaml::monitoring")
    parser.add_argument("--monitor-legacy-tags", action="store_true",
                        help="监控回退：按旧 tag 名/旧写入行为记录全部指标（默认关）。"
                             "默认只记 Tier-1（OD/EGO loss+KPI、router loss+KPI），其余丢弃；"
                             "清单见 docs/metrics.md")
    args = parser.parse_args(argv)
    if args.phase3 is None and args.stage is None:
        parser.error("必须给 --stage {A,B,C} 或 --phase3 <dagger_dir>（stage B phase 3 迭代恢复训练）")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    if args.phase3:
        run_stage_b_phase3(args, config)
    elif args.stage == "A":
        run_stage_a(args, config)
    elif args.stage == "B":
        run_stage_b(args, config)
    else:
        run_stage_c(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
