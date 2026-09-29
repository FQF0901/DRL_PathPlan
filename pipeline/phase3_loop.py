#!/usr/bin/env python3
"""phase 3 全自动循环编排（lane P3-C）：采集 → 训练 → 评测 ×N 轮 + 护栏 + 状态文件。

由 ``tools/train.py --phase3-loop``（仅循环）或 ``--phase3-chain``（A→B→循环全链）调用；
``tools/train.sh`` 的 ``PHASE3=1`` 走全链、``PHASE3_ONLY=1`` 直接进循环（**不新增 .sh**，
编排自包含在 repo 内）。

循环开始：**基座自评（R4）**——对 ``init_ckpt`` 用同一评测口径（spec/limit/workers）评一次 →
状态 ``base_eval``（引用 + 数值）；**不可得 = 硬错（fail-closed，不许跳过判定）**。

每轮 ``k=1..N``：

1. **采集**（subprocess ``tools/dagger_collect.py``）：学生 = **running student**
   （``r1`` = ``stages.B.phase3.init_ckpt``；``r{k}`` = 第 ``k-1`` 轮训练产物 ``final.pt``），
   在 ``spec_pool`` 上攒 ``fail_target`` 个失败窗口（``--window-fail-before <collect.window_s>``、
   ``--shuffle-seed <shuffle_seed_base+k>``、``--workers <collect.workers>``）→
   ``<datasets_root>/BTC<stamp>_phase3_dagger_r{k}``；
2. **训练**（**独立子进程** ``tools/train.py --phase3 <dir> --phase3-round k``，**起点 = running
   student**（顺序续训 R1：``--ckpt`` = 上一轮输出，``r1`` = ``init_ckpt``）；进程退出天然释放
   显存）→ ``<runs_root>/BTC<stamp>_stageB_phase3_r{k}/stage_b``；anchor 接线：
   ``stages.B.phase3.anchor.enabled=true`` 时透传 ``--phase3-anchor*``（**默认 false，行为不变**）；
3. **评测**（subprocess ``tools/test.py``，LQR，``eval.spec``/``eval.workers``）→
   ``<runs_root>/BTC<stamp>_eval500_phase3_r{k}``；
4. **护栏（R4）**：``guard.mode=base``（默认）比较 ``overall_success`` 相对**入口基座**
   （``base_eval``）：``net_vs_base = 本轮 − 基座``，``net_vs_base < −guard.stop_on_base_drop``
   （默认 0.01；确定性评测下阈值 = 实质显著性）→ 记录原因并停止；本轮指标缺失/NaN → **不判定**
   （缺失不停，与基座缺失=硬错区分）；``mode=prev`` = 旧口径（与上一轮比较，
   ``stop_on_overall_drop``/``stop_on_easy_drop`` 仅此模式使用）；``mode=off`` = 不判定
   （状态/日志显著标注）。

**keep-best（R4）**：候选 = ``{init_ckpt} ∪ {各轮产物}``，指标 ``overall_success``，平手取基座/
更早轮；循环结束（含 ``stopped_guard``）导出稳定路径 ``<loop_dir>/best/final.pt`` +
``best/metrics.json`` + ``best/manifest.json``（round/eval dir/spec hash/vs base/is_base）；
**下游消费方 = ``best/final.pt``**（清单 ``best/manifest.json``、状态 ``best``）。全部轮次均未超
基座 → 状态 ``no_gain``（**完成态、非提前停**）。

状态文件 ``<runs_root>/BTC<stamp>_phase3_loop/phase3_status.json``：每步**原子重写**
（tmp + ``os.replace``），含 loop 级状态 + 逐轮 collect/train/eval/guard 结果，供 10 分钟轮询审计；
日志 ``<runs_root>/BTC<stamp>_phase3_loop/logs/phase3_loop.log`` + 逐轮 ``round{k}_*.log``。

**OOM 自愈（lane P3-D）**：训练步捕获 ``torch.cuda.OutOfMemoryError``（或非零 rc 且日志含
``CUDA out of memory``）→ ``gc.collect()`` + ``torch.cuda.empty_cache()`` → **micro 减半整步重试**
（≤3 次；下限 64）→ 重试耗尽才失败；每次重试 ``current_step="train_retry_micro{M}"`` 落盘。

**GPU 显存守卫（lane P3-E）**：每轮训练 = **独立子进程**（进程退出天然释放显存）；训练步结束后
**无论成功/失败/重试**强制释放并记录 ``memory_reserved`` 前后对比（仍高位告警）；任何 GPU 子步骤
（采集/训练/评测）启动前查 ``torch.cuda.mem_get_info``：空闲 < ``gpu.min_free_mib``（默认 6144 MiB）
→ 释放+等待重试 ≤3 次 → 仍不足 failed（绝不带病启动）；采集/评测因 GPU OOM 失败 → 释放后重试 ≤2
（采集第 2 次重试退 ``--device cpu``）；状态文件含 ``gpu.checks`` 逐步快照（free MiB）。

**断点复用（lane P3-D）**：``PHASE3_RESUME=1`` / ``--phase3-resume`` 时每步前查最新可用产物
（采集：``BTC*_phase3_dagger_r{k}`` 且 ``report.json.counts.stored_rows>0``；训练：
``BTC*_stageB_phase3_r{k}/stage_b/final.pt``+``metrics.json``；评测：
``BTC*_eval500_phase3_r{k}/metrics.json``）→ 跳过并在日志/状态标 ``reused=<path>``。
**复用必须通过 provenance 校验（R1）**：训练侧 ``metrics.json["init_ckpt"]``、采集侧
``report.json.dagger.driver_ckpt.path``、评测侧 ``metrics.json["ckpt"]`` 必须等于**期望父节点**
（当轮 running student / 当轮训练产物）；不符/不可得 → 报错停止（绝不静默拼接）。

**错误语义**：任一步异常/非零退出/产物缺失（``expert_bc.npz`` / ``final.pt`` / ``metrics.json``）
→ 写 ``status="failed"`` + ``stop_reason``（异常类型+摘要，含 traceback 落日志）后立即返回非零，
绝不带缺产物进入下一轮；``stopped_guard`` / ``no_gain`` 语义见上。

用法::

    tools/venv-python tools/train.py --phase3-loop [--phase3-rounds N] \\
        [--phase3-runs-root runs] [--phase3-datasets-root datasets] [--limit-dataset M]
    tools/venv-python tools/train.py --phase3-chain [--phase3-only] [--phase3-rounds N]
    PHASE3=1 bash tools/train.sh            # A→B→循环 全链（一条命令全自动）
    PHASE3_ONLY=1 bash tools/train.sh       # 跳过 A/B 直接循环（上线）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 外部命令统一经 venv wrapper 启动（GL 修复；相对仓库根调用）
VENV_PY = "tools/venv-python"
STATUS_FILENAME = "phase3_status.json"
SCHEMA_VERSION = 1
#: OOM 自愈（lane P3-D）：micro 减半重试上限 / 下限 / 起点兜底（cfg.micro 缺失时按 stage B 锚点 512 起）
OOM_MAX_RETRIES = 3
OOM_MICRO_FLOOR = 64
OOM_MICRO_FALLBACK = 512
#: GPU 显存守卫（lane P3-E）：子步骤前空闲下限 / 等待重试次数与间隔 / 训练后 reserved 告警线
GPU_MIN_FREE_MIB = 6144.0
GPU_WAIT_RETRIES = 3
GPU_WAIT_SECONDS = 20.0
GPU_RESERVED_WARN_MIB = 1024.0
#: 采集/评测 GPU OOM 重试次数（采集第 2 次重试退 ``--device cpu``）
COLLECT_OOM_RETRIES = 2
EVAL_OOM_RETRIES = 2
#: R4 护栏模式：``base`` = 与循环入口基座（``base_eval``）比较（默认）；``prev`` = 与上一轮比较
#: （旧语义，``stop_on_overall_drop``/``stop_on_easy_drop`` 仅此模式生效）；``off`` = 不判定
GUARD_MODES = ("base", "prev", "off")
#: R4 基座护栏默认阈值：评测确定性（``eval_runner`` deterministic=True，动作=action_mu）下无评测
#: 噪声 → 阈值即"实质显著性"（0.01 = 1pt）；相对基座下降**超过**该值才停机
GUARD_STOP_ON_BASE_DROP = 0.01
#: 浮点比较容差：下降恰好等于阈值（如 0.49 vs 0.50 的二进制误差）不算"超过"
_GUARD_EPS = 1e-9
#: keep-best 稳定导出目录名（相对 loop_dir）：下游消费 ``best/final.pt``
BEST_DIRNAME = "best"
#: 编排层自身接管的参数：禁止经透传 EXTRA 覆盖（避免把轮次/路径/起点接错）
_RESERVED_EXTRA_ARGS = frozenset(
    {
        "--phase3", "--phase3-round", "--phase3-loop", "--phase3-chain", "--phase3-only",
        "--ckpt", "--out", "--config", "--model-config",
    }
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _opt_num(value: Any) -> Optional[float]:
    """float 化；缺失/NaN/Inf → None（护栏不可判定时拒绝误判）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _abs(path: Any) -> Path:
    """仓库根相对路径 → 绝对路径（``--phase3-loop`` 可能在任意 cwd 启动）。"""
    value = Path(str(path))
    return value if value.is_absolute() else ROOT / value


def _sha256_file(path: Any) -> str:
    """文件 sha256（缺失/不可读 → 空串；用于 provenance/导出校验，不做静默替代）。"""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _ckpt_id(path: Any) -> Dict[str, Any]:
    """权重文件标识（链根 id 等）：``{path, sha256, bytes}``（不可读 → sha256 空串）。"""
    target = Path(str(path))
    try:
        return {"path": str(target), "sha256": _sha256_file(target), "bytes": int(target.stat().st_size)}
    except OSError:
        return {"path": str(target), "sha256": "", "bytes": None}


def _same_path(left: Any, right: Any) -> bool:
    """路径等价判定（``expanduser+resolve`` 归一化；非法/空 → False）。"""
    try:
        left_path = Path(str(left)).expanduser().resolve()
        right_path = Path(str(right)).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return False
    if not str(left or "").strip() or not str(right or "").strip():
        return False
    return left_path == right_path


def _verify_train_provenance(metrics_path: Path, expected_ckpt: Any) -> "tuple[bool, str]":
    """resume 复用训练产物（R1）：``metrics.json["init_ckpt"]`` 必须 = 期望父节点（running student）。"""
    metrics = _read_json(Path(metrics_path))
    if not isinstance(metrics, Mapping):
        return False, f"metrics.json 不可读：{metrics_path}"
    recorded = metrics.get("init_ckpt")
    if not str(recorded or "").strip():
        return False, f"metrics.json 缺 init_ckpt（无法校验父节点）：{metrics_path}"
    if not _same_path(recorded, expected_ckpt):
        return False, f"训练产物父节点不符：init_ckpt={recorded!r} != 期望 {str(expected_ckpt)!r}"
    return True, ""


def _verify_collect_provenance(report_path: Path, expected_ckpt: Any) -> "tuple[bool, str]":
    """resume 复用采集目录（R1）：``report.json.dagger.driver_ckpt.path`` 必须 = 期望父节点。"""
    report = _read_json(Path(report_path))
    if not isinstance(report, Mapping):
        return False, f"report.json 不可读：{report_path}"
    dagger = report.get("dagger")
    driver = dagger.get("driver_ckpt") if isinstance(dagger, Mapping) else None
    recorded = driver.get("path") if isinstance(driver, Mapping) else None
    if not str(recorded or "").strip():
        return False, f"report.json 缺 dagger.driver_ckpt.path（无法校验 driver）：{report_path}"
    if not _same_path(recorded, expected_ckpt):
        return False, f"采集 driver 不符：driver_ckpt={recorded!r} != 期望 {str(expected_ckpt)!r}"
    return True, ""


def _verify_eval_provenance(metrics_path: Path, expected_ckpt: Any) -> "tuple[bool, str]":
    """resume 复用评测产物（R1）：``metrics.json["ckpt"]`` 必须 = 期望父节点（本轮训练产物）。"""
    metrics = _read_json(Path(metrics_path))
    if not isinstance(metrics, Mapping):
        return False, f"metrics.json 不可读：{metrics_path}"
    recorded = metrics.get("ckpt")
    if not str(recorded or "").strip():
        return False, f"metrics.json 缺 ckpt（无法校验评测对象）：{metrics_path}"
    if not _same_path(recorded, expected_ckpt):
        return False, f"评测对象不符：ckpt={recorded!r} != 期望 {str(expected_ckpt)!r}"
    return True, ""


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """原子写 JSON（tmp + ``os.replace``；与状态文件同口径）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _latest_path(paths: Sequence[Path]) -> Optional[Path]:
    """按 mtime 取最新（同 mtime 回退字典序最大，确定性）；空 → None。"""
    items = [Path(item) for item in paths if Path(item).exists()]
    if not items:
        return None
    return max(items, key=lambda item: (item.stat().st_mtime, str(item)))


def _resolve_auto_init_ckpt(runs_root: str = "runs") -> Optional[Path]:
    """``init_ckpt: auto``：最新 ``<runs_root>/BTC*_stageB*/stage_b/final.pt``（按 mtime；无 → None）。

    lane P3-G：覆盖口径 = 该 glob（含 phase3 每轮 ``BTC*_stageB_phase3_r{k}/stage_b/final.pt``），
    取最新者；调用方负责存在性校验与报错。
    """
    candidates = [
        path for path in Path(str(runs_root)).glob("BTC*_stageB*/stage_b/final.pt") if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _stage_final_ckpt(stage: str, *, config: str) -> Optional[Path]:
    """``run_config`` 解析该阶段输出目录 → ``<STAGE_DIR>/final.pt``（解析失败 → None）。

    全链模式（lane P3-G）用它取**本次链**刚训完的 B phase-2 产物；不依赖 ``_run_train_stage``
    的内部实现（测试可单独桩掉）。
    """
    from tools import run_config

    from pipeline.stages import load_config  # 延迟导入（合并 includes）

    try:
        resolved = run_config.resolve(run_config.ROOT, load_config(str(config)), stage_arg=stage)
    except SystemExit:
        return None
    return Path(str(resolved["STAGE_DIR"])) / "final.pt"


def _report_stored_rows(report: Optional[Mapping[str, Any]]) -> Optional[int]:
    """dagger ``report.json`` → ``counts.stored_rows``（缺键/非法 → None）。"""
    if not isinstance(report, Mapping):
        return None
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        return None
    try:
        return int(counts.get("stored_rows"))
    except (TypeError, ValueError):
        return None


def _dagger_rows(directory: Path) -> Optional[int]:
    """dagger 目录行数：meta ``count`` 优先，回退 report ``counts.stored_rows``。"""
    meta = _read_json(Path(directory) / "expert_bc.meta.json")
    if isinstance(meta, Mapping) and meta.get("count") is not None:
        try:
            return int(meta["count"])
        except (TypeError, ValueError):
            pass
    return _report_stored_rows(_read_json(Path(directory) / "report.json"))


def _loss_subset(metrics: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """训练 metrics.json → 状态文件里的损失摘要（缺失键跳过）。"""
    if not isinstance(metrics, Mapping):
        return None
    return {
        key: metrics.get(key)
        for key in ("bc_loss", "bc_od_loss", "bc_ld_loss", "bc_action_chain_loss")
        if metrics.get(key) is not None
    }


def _free_cuda_memory() -> None:
    """OOM 自愈：``gc.collect()`` + ``torch.cuda.empty_cache()``（无 CUDA 环境安全 no-op）。"""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - 清理失败不应打断重试
        pass


def _is_cuda_oom(exc: BaseException) -> bool:
    """异常是否 CUDA OOM（``torch.cuda.OutOfMemoryError`` 或消息含 CUDA/GPU out of memory）。"""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001 - torch 缺失/属性差异 → 退化为字符串判定
        pass
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


def _log_has_oom(log_path: Path, *, start: int = 0, tail_bytes: int = 200_000) -> bool:
    """日志 ``start`` 之后是否出现 CUDA/GPU out of memory。

    只看**本次尝试新增区域的尾部** ``tail_bytes`` 字节（子进程训练日志可达数百 KB，
    OOM traceback 在末尾；从 ``start`` 起读固定窗口会漏检长日志）。
    """
    try:
        with Path(log_path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = int(handle.tell())
            begin = max(int(start), size - int(tail_bytes))
            handle.seek(begin)
            data = handle.read().lower()
    except OSError:
        return False
    return b"out of memory" in data and (b"cuda" in data or b"gpu" in data)


def guard_decision(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    mode: str = "prev",
    base: Optional[Mapping[str, Any]] = None,
    stop_on_base_drop: float = GUARD_STOP_ON_BASE_DROP,
    stop_on_overall_drop: float = 0.05,
    stop_on_easy_drop: float = 0.10,
) -> Dict[str, Any]:
    """护栏判定（纯函数）：返回 ``mode/checked/stopped/reason/net_vs_base/overall_delta/easy_delta``。

    - ``mode="base"``（编排默认，R4）：与**入口基座** ``base``（``base_eval``，含
      ``overall_success``）比较；``net_vs_base = 本轮 − 基座``（正 = 增益），
      ``net_vs_base < −stop_on_base_drop`` → 停机；**基座缺失/不可得 → ValueError（fail-closed）**；
      本轮 ``overall_success`` 缺失/NaN → 不判定（``checked=False``；缺失不停）；
    - ``mode="prev"``：旧口径——与**上一轮**比较绝对下降；``previous is None``（r1）→ 只记录基线，
      指标缺失/NaN → 该项不判定（``delta=None``），另一项照常；
    - ``mode="off"``：不判定（调用方须在状态/日志显著标注）。
    """
    mode = str(mode or "prev").strip().lower()
    if mode not in GUARD_MODES:
        raise ValueError(f"guard.mode 非法：{mode!r}（应为 {' | '.join(GUARD_MODES)}）")
    result: Dict[str, Any] = {
        "mode": mode,
        "checked": False,
        "stopped": False,
        "reason": None,
        "net_vs_base": None,
        "base_overall": None,
        "overall_delta": None,
        "easy_delta": None,
    }
    if mode == "off":
        result["reason"] = "guard.mode=off（不判定）"
        return result
    if mode == "base":
        base_overall = _opt_num((base or {}).get("overall_success"))
        if base_overall is None:
            raise ValueError("guard.mode=base 但基座 overall_success 缺失/不可得（fail-closed）")
        result["base_overall"] = base_overall
        cur_overall = _opt_num(current.get("overall_success"))
        if cur_overall is None:
            return result
        net = cur_overall - base_overall
        result["checked"] = True
        result["net_vs_base"] = net
        if net < -float(stop_on_base_drop) - _GUARD_EPS:
            result["stopped"] = True
            result["reason"] = (
                f"overall_success 相对基座下降 {-net:.4f} > {float(stop_on_base_drop):.4f}"
                f"（基座 {base_overall:.4f} → 本轮 {cur_overall:.4f}）"
            )
        return result
    # mode == "prev"：旧的与上一轮比较口径
    if previous is None:
        return result
    result["checked"] = True
    prev_overall = _opt_num(previous.get("overall_success"))
    cur_overall = _opt_num(current.get("overall_success"))
    prev_easy = _opt_num(previous.get("easy_success"))
    cur_easy = _opt_num(current.get("easy_success"))
    if prev_overall is not None and cur_overall is not None:
        result["overall_delta"] = prev_overall - cur_overall
    if prev_easy is not None and cur_easy is not None:
        result["easy_delta"] = prev_easy - cur_easy
    if result["overall_delta"] is not None and result["overall_delta"] > float(stop_on_overall_drop):
        result["stopped"] = True
        result["reason"] = (
            f"overall_success 下降 {result['overall_delta']:.4f} > {float(stop_on_overall_drop):.4f}"
            f"（上一轮 {prev_overall:.4f} → 本轮 {cur_overall:.4f}）"
        )
    elif result["easy_delta"] is not None and result["easy_delta"] > float(stop_on_easy_drop):
        result["stopped"] = True
        result["reason"] = (
            f"easy_success 下降 {result['easy_delta']:.4f} > {float(stop_on_easy_drop):.4f}"
            f"（上一轮 {prev_easy:.4f} → 本轮 {cur_easy:.4f}）"
        )
    return result


@dataclass
class Phase3LoopConfig:
    """循环编排配置（决议顺序：CLI > 环境变量 ``PHASE3_ROUNDS`` > config ``stages.B.phase3``）。"""

    rounds: int = 5
    spec_pool: str = "env/specs/scenarios_train_5k.json"
    fail_target: int = 1000
    shuffle_seed_base: int = 0
    collect_workers: int = 16
    collect_window_s: float = 10.0
    init_ckpt: str = ""
    #: ``init_ckpt`` 来源（lane P3-G）：``cli``（--ckpt）/ ``config`` / ``auto``（最新 B final）
    #: / ``chain-B``（全链模式：**本次链刚训完的 B phase-2 final.pt**，覆盖 config 值）
    init_ckpt_source: str = "config"
    eval_spec: str = "env/specs/scenarios_eval500.json"
    eval_workers: int = 16
    stop_on_overall_drop: float = 0.05
    stop_on_easy_drop: float = 0.10
    #: R4 护栏模式（``guard.mode``）：base = 与入口基座比较（默认）；prev = 与上一轮比较；off = 不判定
    guard_mode: str = "base"
    #: R4 基座护栏阈值（``guard.stop_on_base_drop``；仅 mode=base）：确定性评测下阈值=实质显著性
    stop_on_base_drop: float = GUARD_STOP_ON_BASE_DROP
    #: 锚接线（lane P4b；``stages.B.phase3.anchor``）：透传 ``--phase3-anchor*``；默认关（行为不变）
    anchor_enabled: bool = False
    anchor_bc_dir: str = ""
    anchor_mild_weight: float = 0.1
    config: str = "config/default.yaml"
    model_config: str = "config/model.yaml"
    device: Optional[str] = None
    limit: Optional[int] = None
    runs_root: str = "runs"
    datasets_root: str = "datasets"
    stamp: Optional[str] = None
    #: 训练起始 micro batch（OOM 自愈减半的起点；None = 不显式传 --micro-batch-size）
    micro: Optional[int] = None
    #: 断点复用（``PHASE3_RESUME=1`` / ``--phase3-resume``）：已有可用产物则跳过该步
    resume: bool = False
    #: GPU 子步骤前空闲显存下限 MiB（低于 → 释放+等待重试 ≤GPU_WAIT_RETRIES；仍不足 failed）
    gpu_min_free_mib: float = GPU_MIN_FREE_MIB
    #: 透传的 phase-3 训练参数（如 ``--phase3-epochs/--batch-size``；不得含保留键）
    extra_train_args: List[str] = field(default_factory=list)

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], args: argparse.Namespace, *, extra: Sequence[str] = ()
    ) -> "Phase3LoopConfig":
        stages = dict(config.get("stages", {}) or {})
        stage_b = dict(stages.get("B", {}) or {})
        p3 = dict(stage_b.get("phase3", {}) or {})
        collect = dict(p3.get("collect", {}) or {})
        eval_cfg = dict(p3.get("eval", {}) or {})
        guard = dict(p3.get("guard", {}) or {})
        gpu_cfg = dict(p3.get("gpu", {}) or {})
        anchor_cfg = dict(p3.get("anchor", {}) or {})
        bc_cfg = dict((config.get("train", {}) or {}).get("bc", {}) or {})
        env_rounds = os.environ.get("PHASE3_ROUNDS", "").strip()
        if getattr(args, "phase3_rounds", None) is not None:
            rounds = int(args.phase3_rounds)
        elif env_rounds:
            rounds = int(env_rounds)
        else:
            rounds = int(p3.get("rounds", 5))
        if rounds < 1:
            raise ValueError(f"phase3 rounds 必须 ≥1，收到 {rounds}")
        runs_root = str(getattr(args, "phase3_runs_root", None) or cls.runs_root)
        explicit_ckpt = getattr(args, "ckpt", None)
        init_ckpt = str(explicit_ckpt or p3.get("init_ckpt") or "")
        if explicit_ckpt:
            init_ckpt_source = "cli"
        elif init_ckpt.strip().lower() == "auto":
            # lane P3-G：``init_ckpt: auto`` = 最新 ``runs/BTC*_stageB*/stage_b/final.pt``（缺失报错）
            auto = _resolve_auto_init_ckpt(runs_root)
            if auto is None:
                raise ValueError(
                    f"init_ckpt: auto 但未找到 {runs_root}/BTC*_stageB*/stage_b/final.pt"
                    "（先跑一次 Stage B 或显式给 --ckpt）"
                )
            init_ckpt, init_ckpt_source = str(auto), "auto"
        else:
            init_ckpt_source = "config" if init_ckpt else ""
        micro_raw = p3.get("micro", bc_cfg.get("micro_batch_size"))
        guard_mode = str(guard.get("mode", cls.guard_mode) or cls.guard_mode).strip().lower()
        if guard_mode not in GUARD_MODES:
            raise ValueError(f"guard.mode 非法：{guard_mode!r}（应为 {' | '.join(GUARD_MODES)}）")
        resume = bool(getattr(args, "phase3_resume", False)) or str(
            os.environ.get("PHASE3_RESUME", "")
        ).strip().lower() in ("1", "true", "yes", "on")
        return cls(
            rounds=rounds,
            spec_pool=str(getattr(args, "phase3_spec_pool", None) or p3.get("spec_pool") or cls.spec_pool),
            fail_target=int(p3.get("fail_target", cls.fail_target)),
            shuffle_seed_base=int(p3.get("shuffle_seed_base", cls.shuffle_seed_base)),
            collect_workers=int(collect.get("workers", cls.collect_workers)),
            collect_window_s=float(collect.get("window_s", cls.collect_window_s)),
            init_ckpt=init_ckpt,
            init_ckpt_source=init_ckpt_source,
            eval_spec=str(eval_cfg.get("spec") or cls.eval_spec),
            eval_workers=int(eval_cfg.get("workers", cls.eval_workers)),
            stop_on_overall_drop=float(guard.get("stop_on_overall_drop", cls.stop_on_overall_drop)),
            stop_on_easy_drop=float(guard.get("stop_on_easy_drop", cls.stop_on_easy_drop)),
            guard_mode=guard_mode,
            stop_on_base_drop=float(guard.get("stop_on_base_drop", cls.stop_on_base_drop)),
            anchor_enabled=bool(anchor_cfg.get("enabled", cls.anchor_enabled)),
            anchor_bc_dir=str(anchor_cfg.get("bc_dir") or cls.anchor_bc_dir),
            anchor_mild_weight=float(anchor_cfg.get("mild_weight", cls.anchor_mild_weight)),
            config=str(getattr(args, "config", None) or cls.config),
            model_config=str(getattr(args, "model_config", None) or cls.model_config),
            device=(str(args.device) if getattr(args, "device", None) else None),
            limit=(int(args.limit_dataset) if getattr(args, "limit_dataset", None) else None),
            runs_root=runs_root,
            datasets_root=str(getattr(args, "phase3_datasets_root", None) or cls.datasets_root),
            stamp=(str(args.phase3_stamp) if getattr(args, "phase3_stamp", None) else None),
            micro=(int(micro_raw) if micro_raw else None),
            resume=resume,
            gpu_min_free_mib=float(gpu_cfg.get("min_free_mib", cls.gpu_min_free_mib)),
            extra_train_args=[str(item) for item in extra],
        )


def train_subprocess(argv: Sequence[Any], log_path: Optional[Path] = None) -> int:
    """每轮训练 = **独立子进程**（``tools/venv-python tools/train.py ...``）。

    lane P3-E：进程退出天然释放 CUDA 显存 —— 修复 in-process 训练结束后链进程仍持
    ~10.75 GiB、饿死后续评测/采集的问题；``argv`` = :meth:`Phase3Loop._train_argv` 输出。
    """
    return run_subprocess([VENV_PY, "tools/train.py", *[str(item) for item in argv]], log_path)


def _gpu_free_mib() -> Optional[float]:
    """GPU 空闲显存 MiB（``torch.cuda.mem_get_info``）；无 CUDA / 查询失败 → None（不限制）。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return float(free) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 - 查询失败不应打断链
        return None


def _gpu_reserved_mib() -> Optional[float]:
    """当前进程 CUDA 缓存分配器 ``memory_reserved`` MiB；无 CUDA / 查询失败 → None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001
        return None


def _gpu_wait(seconds: float) -> None:
    """显存不足时的等待（独立函数便于测试注入；默认 ``time.sleep``）。"""
    time.sleep(max(0.0, float(seconds)))


def run_subprocess(argv: Sequence[Any], log_path: Optional[Path] = None) -> int:
    """运行外部命令（cwd=仓库根）：stdout/stderr 流到控制台 + ``log_path``（append）。"""
    handle = None
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [str(item) for item in argv],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            if handle is not None:
                handle.write(line)
                handle.flush()
        return int(proc.wait())
    finally:
        if handle is not None:
            handle.close()


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_status(path: Path, status: Dict[str, Any]) -> Dict[str, Any]:
    """原子写状态文件（tmp + os.replace；每步一个一致快照）。"""
    payload = dict(status)
    payload["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return payload


class Phase3Loop:
    """phase 3 轮转执行器（采集→训练→评测→护栏）；步函数可注入（测试用桩）。"""

    def __init__(
        self,
        config: Phase3LoopConfig,
        *,
        collect_fn: Optional[Callable[[Sequence[Any], Optional[Path]], int]] = None,
        train_fn: Optional[Callable[[Sequence[Any], Optional[Path]], int]] = None,
        eval_fn: Optional[Callable[[Sequence[Any], Optional[Path]], int]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.cfg = config
        self._collect_fn = collect_fn or run_subprocess
        self._train_fn = train_fn or train_subprocess
        self._eval_fn = eval_fn or run_subprocess
        self._logger = logger or print
        self._log_handle: Optional[Any] = None
        #: run() 期间的状态引用（GPU 检查/复用记录用；结束后清空）
        self._status: Optional[Dict[str, Any]] = None
        self._status_path: Optional[Path] = None

    # ------------------------------------------------------------------ 日志/状态
    def _log(self, message: str) -> None:
        line = f"[phase3] {_now()} {message}"
        self._logger(line)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    def _log_exception(self, exc: BaseException) -> None:
        """异常详情 + traceback 落日志（审计；每步异常与未捕获异常共用）。"""
        self._log(f"异常详情：{type(exc).__name__}: {exc}")
        self._log(traceback.format_exc().rstrip())

    # ------------------------------------------------------------------ 命令构造
    def _collect_argv(
        self, *, student: str, dagger_dir: Path, round_index: int, device: Optional[str] = None
    ) -> List[str]:
        cfg = self.cfg
        argv = [
            VENV_PY, "tools/dagger_collect.py",
            "--ckpt", str(student),
            "--out", str(dagger_dir),
            "--specs", str(_abs(cfg.spec_pool)),
            "--target-fails", str(int(cfg.fail_target)),
            "--shuffle-seed", str(int(cfg.shuffle_seed_base) + round_index),
            "--window-fail-before", str(float(cfg.collect_window_s)),
            "--workers", str(int(cfg.collect_workers)),
            "--config", str(_abs(cfg.config)),
            "--model-config", str(_abs(cfg.model_config)),
        ]
        effective_device = device if device is not None else cfg.device
        if effective_device:
            argv += ["--device", str(effective_device)]
        if cfg.limit:
            argv += ["--limit", str(int(cfg.limit))]
        return argv

    def _train_argv(
        self,
        *,
        dagger_dir: Path,
        train_out: Path,
        ckpt: str,
        round_index: int,
        micro: Optional[int] = None,
    ) -> List[str]:
        """phase-3 单轮训练 argv（独立子进程）。

        R1 顺序续训：``--ckpt`` = **当轮起点 = running student**（r1 = ``init_ckpt``，
        r{k} = r{k-1} 轮产物）；anchor 接线：``anchor.enabled=true`` 时透传 ``--phase3-anchor*``
        （默认 false → 不传，行为不变）。
        """
        cfg = self.cfg
        argv = [
            "--phase3", str(dagger_dir),
            "--phase3-round", str(round_index),
            "--ckpt", str(_abs(ckpt)),
            "--out", str(train_out),
            "--config", str(_abs(cfg.config)),
            "--model-config", str(_abs(cfg.model_config)),
        ]
        if cfg.anchor_enabled:
            argv += ["--phase3-anchor"]
            if str(cfg.anchor_bc_dir).strip():
                argv += ["--phase3-anchor-bc-dir", str(cfg.anchor_bc_dir)]
            argv += ["--phase3-anchor-mild-weight", f"{float(cfg.anchor_mild_weight):g}"]
        if micro is not None:
            argv += ["--micro-batch-size", str(int(micro))]
        if cfg.device:
            argv += ["--device", str(cfg.device)]
        if cfg.limit:
            argv += ["--limit-dataset", str(int(cfg.limit))]
        argv += list(cfg.extra_train_args)
        return argv

    def _eval_argv(self, *, final_ckpt: Path, eval_dir: Path) -> List[str]:
        cfg = self.cfg
        argv = [
            VENV_PY, "tools/test.py",
            "--policy", "ckpt",
            "--ckpt", str(final_ckpt),
            "--spec", str(_abs(cfg.eval_spec)),
            "--out", str(_abs(cfg.runs_root)),
            "--name", eval_dir.name,
            "--workers", str(int(cfg.eval_workers)),
            "--tracker", "lqr",
            "--config", str(_abs(cfg.config)),
        ]
        if cfg.device:
            argv += ["--device", str(cfg.device)]
        if cfg.limit:
            argv += ["--limit", str(int(cfg.limit))]
        return argv

    # ------------------------------------------------------------------ 断点复用
    def _reuse_collect(self, round_index: int) -> Optional[Path]:
        """可复用采集目录（最新）：``BTC*_phase3_dagger_r{k}`` 且 ``report.json.counts.stored_rows>0``。"""
        root = _abs(self.cfg.datasets_root)
        candidates: List[Path] = []
        for directory in root.glob(f"BTC*_phase3_dagger_r{round_index}"):
            if not directory.is_dir():
                continue
            rows = _report_stored_rows(_read_json(directory / "report.json"))
            if rows is not None and rows > 0:
                candidates.append(directory)
        return _latest_path(candidates)

    def _reuse_train(self, round_index: int) -> Optional[Path]:
        """可复用训练产物（最新）：``BTC*_stageB_phase3_r{k}/stage_b/final.pt`` + ``metrics.json``。"""
        root = _abs(self.cfg.runs_root)
        candidates = [
            final
            for final in root.glob(f"BTC*_stageB_phase3_r{round_index}/stage_b/final.pt")
            if (final.parent / "metrics.json").is_file()
        ]
        return _latest_path(candidates)

    def _reuse_eval(self, round_index: int) -> Optional[Path]:
        """可复用评测产物（最新）：``BTC*_eval500_phase3_r{k}/metrics.json``。"""
        root = _abs(self.cfg.runs_root)
        return _latest_path(
            [item for item in root.glob(f"BTC*_eval500_phase3_r{round_index}/metrics.json") if item.is_file()]
        )

    def _reuse_base_eval(self) -> Optional[Path]:
        """可复用基座评测（R4，最新）：``BTC*_eval500_phase3_base/metrics.json``。"""
        root = _abs(self.cfg.runs_root)
        candidates = [
            item for item in root.glob("BTC*_eval500_phase3_base/metrics.json") if item.is_file()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.stat().st_mtime, str(item)))

    def _base_eval_reusable(self, metrics_path: Path) -> "tuple[bool, str]":
        """resume 复用基座评测的 provenance 校验：ckpt = ``init_ckpt``、spec/limit 同口径。"""
        metrics = _read_json(Path(metrics_path))
        if not isinstance(metrics, Mapping):
            return False, "metrics.json 不可读"
        if not _same_path(metrics.get("ckpt"), self.cfg.init_ckpt):
            return False, f"ckpt={metrics.get('ckpt')!r} != init_ckpt"
        if not _same_path(metrics.get("specs_path"), _abs(self.cfg.eval_spec)):
            return False, f"spec={metrics.get('specs_path')!r} != {str(_abs(self.cfg.eval_spec))!r}"
        metric_limit = metrics.get("limit")
        if (self.cfg.limit is None) != (metric_limit is None) or (
            self.cfg.limit is not None and int(metric_limit) != int(self.cfg.limit)
        ):
            return False, f"limit={metric_limit!r} != {self.cfg.limit!r}"
        if _opt_num((metrics.get("overall") or {}).get("success_rate")) is None:
            return False, "overall.success_rate 缺失/NaN"
        return True, ""

    def _ensure_base_eval(
        self, *, base_ckpt: Path, base_dir: Path, log_dir: Path
    ) -> "tuple[Optional[Dict[str, Any]], Optional[str]]":
        """R4 基线：``resume`` 时复用已校验的基座评测，否则对 ``init_ckpt`` 自评一次。

        返回 ``(base_eval, None)`` 或 ``(None, 错误原因)``；**任何不可得 → 硬错（fail-closed）**。
        ``base_eval`` = 引用（dir/metrics）+ 数值（overall/easy）。
        """
        cfg = self.cfg
        if cfg.resume:
            reused = self._reuse_base_eval()
            if reused is not None:
                ok, detail = self._base_eval_reusable(reused)
                if not ok:
                    return None, (
                        f"resume 复用的 base_eval 校验失败：{detail}（{reused}；拒绝静默拼接；"
                        "如需重评请移除该目录或换 --phase3-runs-root）"
                    )
                summary = _eval_summary(reused)
                self._log(f"base_eval reuse={reused} overall={summary.get('overall_success')}")
                return (
                    {
                        "ckpt": str(base_ckpt),
                        "ckpt_sha256": _sha256_file(base_ckpt),
                        "dir": str(reused.parent),
                        "metrics": str(reused),
                        "overall_success": summary.get("overall_success"),
                        "easy_success": summary.get("easy_success"),
                        "reused": True,
                        "spec": str(cfg.eval_spec),
                        "workers": int(cfg.eval_workers),
                        "limit": cfg.limit,
                        "attempts": [],
                    },
                    None,
                )
        free_mib, gpu_ok = self._gpu_check(round_index=0, step="base_eval")
        if not gpu_ok:
            return None, (
                f"基座评测前 GPU 空闲不足：free={free_mib:.0f} MiB < {cfg.gpu_min_free_mib:.0f} MiB"
                f"（释放+等待 {GPU_WAIT_RETRIES} 次后仍不足）"
            )
        eval_log = Path(log_dir) / "base_eval.log"
        self._log(
            f"base_eval start ckpt={base_ckpt} dir={base_dir} "
            f"free={free_mib if free_mib is not None else 'n/a'}"
        )
        started = time.perf_counter()
        attempts: List[Dict[str, Any]] = []
        rc, oom = 1, False
        while True:
            attempt: Dict[str, Any] = {}
            log_start = eval_log.stat().st_size if eval_log.exists() else 0
            try:
                rc = int(
                    self._eval_fn(self._eval_argv(final_ckpt=base_ckpt, eval_dir=base_dir), eval_log)
                )
                oom = rc != 0 and _log_has_oom(eval_log, start=log_start)
            except Exception as exc:  # noqa: BLE001 - 基线评测异常（含 CUDA OOM）
                rc = 1
                oom = _is_cuda_oom(exc)
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                if not oom:
                    attempts.append({**attempt, "rc": rc, "oom": False})
                    self._log_exception(exc)
                    return None, f"基座评测异常 {type(exc).__name__}: {exc}（log={eval_log}）"
            attempt.update({"rc": rc, "oom": bool(oom)})
            attempts.append(attempt)
            if not oom or len(attempts) > EVAL_OOM_RETRIES:
                break
            _free_cuda_memory()
            self._log(f"base_eval OOM → 释放 → 重试（第 {len(attempts)}/{EVAL_OOM_RETRIES} 次）")
        duration = time.perf_counter() - started
        metrics_path = Path(base_dir) / "metrics.json"
        if rc != 0:
            return None, f"基座评测失败 rc={rc}（log={eval_log}）"
        if not metrics_path.is_file():
            return None, f"基座评测产物缺失：{metrics_path}"
        summary = _eval_summary(metrics_path)
        if summary.get("overall_success") is None:
            return None, f"基座评测 overall_success 缺失/NaN：{metrics_path}（fail-closed）"
        self._log(
            f"base_eval ok overall={summary.get('overall_success')} "
            f"easy={summary.get('easy_success')} ({duration:.0f}s)"
        )
        return (
            {
                "ckpt": str(base_ckpt),
                "ckpt_sha256": _sha256_file(base_ckpt),
                "dir": str(base_dir),
                "metrics": str(metrics_path),
                "overall_success": summary.get("overall_success"),
                "easy_success": summary.get("easy_success"),
                "reused": False,
                "spec": str(cfg.eval_spec),
                "workers": int(cfg.eval_workers),
                "limit": cfg.limit,
                "attempts": attempts,
            },
            None,
        )

    # ------------------------------------------------------------------ keep-best（R4）
    def _export_best(
        self,
        *,
        loop_dir: Path,
        candidates: Sequence[Mapping[str, Any]],
        base_overall: Optional[float],
    ) -> Dict[str, Any]:
        """keep-best：``{init_ckpt} ∪ {各轮产物}`` 按 ``overall_success`` 选最优并稳定导出。

        - 平手取基座/更早轮（``round`` 小者优先，基座 round=0）；
        - 稳定消费路径：``<loop_dir>/best/final.pt`` + ``best/metrics.json`` +
          ``best/manifest.json``（清单：round / eval dir / spec hash / vs base / is_base）；
        - 返回状态块（调用方写入 ``status["best"]``）。
        """
        best_dir = Path(loop_dir) / BEST_DIRNAME
        ranked = [item for item in candidates if _opt_num(item.get("overall_success")) is not None]
        if not ranked:
            raise RuntimeError("keep-best 无可用候选（基座数值缺失）")
        best = max(ranked, key=lambda item: (float(item["overall_success"]), -int(item.get("round", 0))))
        source = Path(str(best["ckpt"]))
        exported = best_dir / "final.pt"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, exported)
        source_sha = _sha256_file(source)
        exported_sha = _sha256_file(exported)
        if source_sha and exported_sha != source_sha:
            raise RuntimeError(f"keep-best 导出校验失败（sha256 不一致）：{exported} != {source}")
        metrics_src = Path(str(best.get("metrics") or ""))
        metrics_dst = best_dir / "metrics.json"
        metrics_consumer = str(best.get("metrics") or "")
        if metrics_src.is_file():
            shutil.copyfile(metrics_src, metrics_dst)
            metrics_consumer = str(metrics_dst)
        spec_path = _abs(self.cfg.eval_spec)
        best_overall = float(best["overall_success"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "phase3_keep_best",
            "generated_at": _now(),
            "metric": "overall_success",
            "tie_break": "高分优先；平手取基座（round=0）/更早轮",
            "spec": str(spec_path),
            "spec_sha256": _sha256_file(spec_path),
            "workers": int(self.cfg.eval_workers),
            "limit": self.cfg.limit,
            "base": {
                "ckpt": str(_abs(self.cfg.init_ckpt)),
                "overall_success": base_overall,
            },
            "best": {
                "round": int(best.get("round", 0)),
                "is_base": bool(best.get("is_base")),
                "ckpt_source": str(source),
                "ckpt_sha256": exported_sha or source_sha,
                "export_ckpt": str(exported),
                "eval_dir": str(best.get("eval_dir") or ""),
                "eval_metrics": metrics_consumer,
                "overall_success": best_overall,
                "easy_success": _opt_num(best.get("easy_success")),
                "vs_base": (best_overall - float(base_overall)) if base_overall is not None else None,
            },
            "candidates": [
                {
                    "round": int(item.get("round", 0)),
                    "is_base": bool(item.get("is_base")),
                    "ckpt": str(item.get("ckpt") or ""),
                    "overall_success": _opt_num(item.get("overall_success")),
                    "easy_success": _opt_num(item.get("easy_success")),
                    "eval_dir": str(item.get("eval_dir") or ""),
                    "ranked": _opt_num(item.get("overall_success")) is not None,
                    "vs_base": (
                        float(item["overall_success"]) - float(base_overall)
                        if _opt_num(item.get("overall_success")) is not None and base_overall is not None
                        else None
                    ),
                }
                for item in candidates
            ],
            "consumer": "下游消费 best/final.pt（清单 best/manifest.json；状态 phase3_status.json::best）",
        }
        _write_json_atomic(best_dir / "manifest.json", manifest)
        return {
            "dir": str(best_dir),
            "ckpt": str(exported),
            "metrics": metrics_consumer,
            "manifest": str(best_dir / "manifest.json"),
            "round": int(best.get("round", 0)),
            "is_base": bool(best.get("is_base")),
            "overall_success": best_overall,
            "easy_success": _opt_num(best.get("easy_success")),
            "vs_base": manifest["best"]["vs_base"],
            "no_gain": bool(best.get("is_base")),
        }

    # ------------------------------------------------------------------ GPU 显存守卫（lane P3-E）
    def _record_gpu_check(
        self, *, round_index: int, step: str, free_mib: Optional[float], ok: bool, waits: int
    ) -> None:
        """把一次子步骤前的显存快照写进 ``status["gpu"]["checks"]`` 并落盘。"""
        if self._status is None:
            return
        gpu = self._status.setdefault("gpu", {})
        gpu.setdefault("threshold_mib", float(self.cfg.gpu_min_free_mib))
        gpu.setdefault("checks", []).append(
            {
                "round": int(round_index),
                "step": str(step),
                "free_mib": (None if free_mib is None else round(float(free_mib), 1)),
                "ok": bool(ok),
                "waits": int(waits),
            }
        )
        if self._status_path is not None:
            _write_status(self._status_path, self._status)

    def _gpu_check(self, *, round_index: int, step: str) -> "tuple[Optional[float], bool]":
        """GPU 子步骤前置检查：free < ``gpu_min_free_mib`` → 释放+等待重试 ≤3；返回 (free, ok)。

        ``free_mib=None``（无 CUDA / 查询失败）→ 不限制（``ok=True``，不阻塞非 GPU 环境/测试）。
        仍不足 → ``ok=False``，调用方必须 failed（绝不带病启动）。
        """
        threshold = float(self.cfg.gpu_min_free_mib)
        free = _gpu_free_mib()
        waits = 0
        while free is not None and free < threshold and waits < GPU_WAIT_RETRIES:
            waits += 1
            self._log(
                f"r{round_index} {step} GPU 空闲不足：free={free:.0f} MiB < {threshold:.0f} MiB → "
                f"释放+等待 {GPU_WAIT_SECONDS:.0f}s（第 {waits}/{GPU_WAIT_RETRIES} 次）"
            )
            _free_cuda_memory()
            _gpu_wait(GPU_WAIT_SECONDS)
            free = _gpu_free_mib()
        ok = free is None or free >= threshold
        self._record_gpu_check(round_index=round_index, step=step, free_mib=free, ok=ok, waits=waits)
        if not ok:
            self._log(
                f"r{round_index} {step} GPU 空闲仍不足：free={free:.0f} MiB < {threshold:.0f} MiB"
                f"（已等待 {waits}/{GPU_WAIT_RETRIES} 次）"
            )
        return free, ok

    def _release_after_training(self, round_index: int, before: Optional[float]) -> Optional[float]:
        """训练步结束后强制释放 + 回落校验：返回释放后的 reserved MiB；仍高位则告警。"""
        _free_cuda_memory()
        after = _gpu_reserved_mib()
        if after is not None and after > GPU_RESERVED_WARN_MIB:
            self._log(
                f"警告：r{round_index} 训练后 CUDA reserved={after:.0f} MiB 仍高位"
                f"（>{GPU_RESERVED_WARN_MIB:.0f} MiB；before={before}）→ 后继 GPU 子步骤可能受挤"
            )
        return after

    # ------------------------------------------------------------------ 主循环
    def run(self) -> int:
        cfg = self.cfg
        from pipeline import run_paths

        stamp = cfg.stamp or run_paths.beijing_stamp_seconds()
        runs_root = _abs(cfg.runs_root)
        datasets_root = _abs(cfg.datasets_root)
        loop_dir = runs_root / run_paths.canonical_run_name(kind="phase3_loop", stamp=stamp)
        log_dir = loop_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = (log_dir / "phase3_loop.log").open("a", encoding="utf-8")
        status_path = loop_dir / STATUS_FILENAME
        status: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "lane": "P3-C",
            "status": "running",
            "stop_reason": None,
            "started_at": _now(),
            "updated_at": _now(),
            "rounds_total": int(cfg.rounds),
            "current_round": None,
            "current_step": None,
            "stamp": stamp,
            "loop_dir": str(loop_dir),
            "runs_root": str(runs_root),
            "datasets_root": str(datasets_root),
            "init_ckpt": str(_abs(cfg.init_ckpt)),
            "init_ckpt_source": str(cfg.init_ckpt_source or ""),
            "guard_mode": str(cfg.guard_mode),
            "config": {
                "spec_pool": str(cfg.spec_pool),
                "fail_target": int(cfg.fail_target),
                "shuffle_seed_base": int(cfg.shuffle_seed_base),
                "collect": {"workers": int(cfg.collect_workers), "window_s": float(cfg.collect_window_s)},
                "eval": {"spec": str(cfg.eval_spec), "workers": int(cfg.eval_workers)},
                "guard": {
                    "mode": str(cfg.guard_mode),
                    "stop_on_base_drop": float(cfg.stop_on_base_drop),
                    "stop_on_overall_drop": float(cfg.stop_on_overall_drop),
                    "stop_on_easy_drop": float(cfg.stop_on_easy_drop),
                },
                "anchor": {
                    "enabled": bool(cfg.anchor_enabled),
                    "bc_dir": str(cfg.anchor_bc_dir),
                    "mild_weight": float(cfg.anchor_mild_weight),
                },
                "micro": cfg.micro,
                "resume": bool(cfg.resume),
                "gpu": {"min_free_mib": float(cfg.gpu_min_free_mib)},
            },
            "gpu": {"threshold_mib": float(cfg.gpu_min_free_mib), "checks": []},
            "chain_root": None,
            "base_eval": None,
            "best": None,
            "rounds": [],
        }
        self._status = status
        self._status_path = status_path

        def _finish(code: int, state: str, reason: Optional[str] = None) -> int:
            """落盘终态（failed 保留 current_step = 失败所在步，便于审计）；句柄由 finally 关闭。"""
            status["status"] = state
            status["stop_reason"] = reason
            if state in ("completed", "stopped_guard", "no_gain"):
                status["current_step"] = "done"
            _write_status(status_path, status)
            if reason:
                self._log(f"{state}: {reason}")
            self._log(f"exit code={code} status={state} status_file={status_path}")
            return int(code)

        try:
            _write_status(status_path, status)
            self._log(
                f"start stamp={stamp} rounds={cfg.rounds} runs_root={runs_root} "
                f"status={status_path}"
            )
            self._log(
                f"cfg init_ckpt={cfg.init_ckpt} (source={cfg.init_ckpt_source or 'unknown'})"
            )
            self._log(
                f"cfg spec_pool={cfg.spec_pool} fail_target={cfg.fail_target} "
                f"collect=({cfg.collect_workers}w/{cfg.collect_window_s}s) "
                f"eval=({cfg.eval_spec} · {cfg.eval_workers}w · lqr) "
                f"guard=(mode={cfg.guard_mode} base-{cfg.stop_on_base_drop} "
                f"prev-overall-{cfg.stop_on_overall_drop}/easy-{cfg.stop_on_easy_drop}) "
                f"anchor={'on' if cfg.anchor_enabled else 'off'} "
                f"micro={cfg.micro} resume={bool(cfg.resume)} "
                f"gpu_min_free={cfg.gpu_min_free_mib:.0f}MiB"
            )
            if cfg.guard_mode == "off":
                status["guard_off"] = True
                self._log(
                    "警告：guard.mode=off → 护栏**不判定**（下降不停机）；状态已标注 guard_off=true"
                )

            init_ckpt = _abs(cfg.init_ckpt)
            if not cfg.init_ckpt or not init_ckpt.is_file():
                return _finish(
                    1, "failed",
                    f"起点权重不存在：{cfg.init_ckpt!r}（source={cfg.init_ckpt_source or 'unknown'}；"
                    "--ckpt / config stages.B.phase3.init_ckpt / chain-B / auto）",
                )
            if not _abs(cfg.spec_pool).is_file():
                return _finish(1, "failed", f"采集池不存在：{cfg.spec_pool!r}")
            if not _abs(cfg.eval_spec).is_file():
                return _finish(1, "failed", f"评测 spec 不存在：{cfg.eval_spec!r}")

            # 链根 id（R1）：循环入口权重 + sha256 + 来源（跨 resume 对齐用）
            status["chain_root"] = {**_ckpt_id(init_ckpt), "source": str(cfg.init_ckpt_source or "unknown")}
            _write_status(status_path, status)

            # ---- R4 基线：对 init_ckpt 自评一次（同 spec/limit/workers）；不可得 = 硬错 ----
            status["current_step"] = "base_eval"
            _write_status(status_path, status)
            base_dir = runs_root / run_paths.canonical_run_name(
                kind="eval500", tag="phase3_base", stamp=stamp
            )
            base_eval, base_error = self._ensure_base_eval(
                base_ckpt=init_ckpt, base_dir=base_dir, log_dir=log_dir
            )
            if base_eval is None:
                return _finish(1, "failed", f"基线评测不可得（fail-closed）：{base_error}")
            status["base_eval"] = base_eval
            status["current_step"] = "base_eval"
            _write_status(status_path, status)
            self._log(
                f"base_eval ok overall={base_eval.get('overall_success')} "
                f"easy={base_eval.get('easy_success')} dir={base_eval.get('dir')}"
                + ("（reused）" if base_eval.get("reused") else "")
            )

            # keep-best 候选：基座 + 各轮产物（平手取基座/更早轮）
            candidates: List[Dict[str, Any]] = [
                {
                    "round": 0,
                    "is_base": True,
                    "ckpt": str(init_ckpt),
                    "ckpt_sha256": str(status["chain_root"].get("sha256") or ""),
                    "eval_dir": str(base_eval.get("dir") or ""),
                    "metrics": str(base_eval.get("metrics") or ""),
                    "overall_success": base_eval.get("overall_success"),
                    "easy_success": base_eval.get("easy_success"),
                }
            ]

            student = str(init_ckpt)
            previous_eval: Optional[Dict[str, Any]] = None
            for round_index in range(1, int(cfg.rounds) + 1):
                status["current_round"] = round_index
                round_entry: Dict[str, Any] = {
                    "round": round_index,
                    "student_in": student,
                    "student_out": None,
                    "collect": {},
                    "train": {},
                    "eval": {},
                    "guard": {},
                }
                status["rounds"].append(round_entry)

                # ---- ① 采集（断点复用：最新可用 dagger 目录）----
                status["current_step"] = "collect"
                _write_status(status_path, status)
                dagger_dir = datasets_root / run_paths.canonical_run_name(
                    kind="phase3_dagger", tag=f"r{round_index}", stamp=stamp
                )
                collect_log = log_dir / f"round{round_index}_collect.log"
                reused = self._reuse_collect(round_index) if cfg.resume else None
                if reused is not None:
                    # R1 provenance：复用目录的 driver 必须 = 期望父节点（当轮 running student）
                    ok, detail = _verify_collect_provenance(reused / "report.json", student)
                    if not ok:
                        round_entry["collect"] = {"rc": None, "reused": str(reused), "out": str(reused)}
                        _write_status(status_path, status)
                        return _finish(
                            1, "failed",
                            f"r{round_index} 采集复用 provenance 校验失败：{detail}"
                            f"（--phase3-resume 拒绝静默拼接；如需重采请移除 {reused}）",
                        )
                    status["current_step"] = "collect_reuse"
                    rows = _dagger_rows(reused)
                    dagger_dir = reused
                    round_entry["collect"] = {
                        "rc": None, "reused": str(reused), "duration_s": 0.0,
                        "out": str(reused), "rows": rows, "log": None,
                    }
                    _write_status(status_path, status)
                    self._log(f"r{round_index} collect reuse={reused} rows={rows}")
                else:
                    free_mib, gpu_ok = self._gpu_check(round_index=round_index, step="collect")
                    if not gpu_ok:
                        return _finish(
                            1, "failed",
                            f"r{round_index} 采集前 GPU 空闲不足：free={free_mib:.0f} MiB < "
                            f"{cfg.gpu_min_free_mib:.0f} MiB（释放+等待 {GPU_WAIT_RETRIES} 次后仍不足）",
                        )
                    self._log(
                        f"r{round_index} collect start student={student} out={dagger_dir} "
                        f"free={free_mib if free_mib is not None else 'n/a'}"
                    )
                    started = time.perf_counter()
                    attempts: List[Dict[str, Any]] = []
                    device_override: Optional[str] = None
                    rc, oom = 1, False
                    while True:
                        attempt: Dict[str, Any] = {"device": device_override or cfg.device or "auto"}
                        log_start = collect_log.stat().st_size if collect_log.exists() else 0
                        try:
                            rc = int(
                                self._collect_fn(
                                    self._collect_argv(
                                        student=student, dagger_dir=dagger_dir,
                                        round_index=round_index, device=device_override,
                                    ),
                                    collect_log,
                                )
                            )
                            oom = rc != 0 and _log_has_oom(collect_log, start=log_start)
                        except Exception as exc:  # noqa: BLE001 - 采集异常（含 CUDA OOM）
                            rc = 1
                            oom = _is_cuda_oom(exc)
                            attempt["error"] = f"{type(exc).__name__}: {exc}"
                            if not oom:
                                attempts.append({**attempt, "rc": rc, "oom": False})
                                round_entry["collect"] = {
                                    "rc": rc, "duration_s": round(time.perf_counter() - started, 1),
                                    "out": str(dagger_dir), "rows": None, "attempts": attempts,
                                    "log": str(collect_log),
                                }
                                _write_status(status_path, status)
                                self._log_exception(exc)
                                return _finish(
                                    1, "failed", f"r{round_index} 采集异常 {type(exc).__name__}: {exc}"
                                )
                        attempt.update({"rc": rc, "oom": bool(oom)})
                        attempts.append(attempt)
                        if not oom or len(attempts) > COLLECT_OOM_RETRIES:
                            break
                        _free_cuda_memory()
                        if len(attempts) >= 2 and (device_override or cfg.device) != "cpu":
                            device_override = "cpu"  # 采集允许退 cpu 重试一次
                            status["current_step"] = "collect_retry_cpu"
                        else:
                            status["current_step"] = f"collect_retry{len(attempts)}"
                        round_entry["collect"] = {
                            "rc": None, "duration_s": None, "out": str(dagger_dir),
                            "rows": None, "attempts": attempts, "log": str(collect_log),
                        }
                        _write_status(status_path, status)
                        self._log(
                            f"r{round_index} collect OOM（device={attempts[-1]['device']}）→ 释放 → "
                            f"重试 device={device_override or cfg.device or 'auto'}"
                            f"（第 {len(attempts)}/{COLLECT_OOM_RETRIES} 次）"
                        )
                    duration = time.perf_counter() - started
                    rows = _dagger_rows(dagger_dir)
                    round_entry["collect"] = {
                        "rc": rc, "duration_s": round(duration, 1), "out": str(dagger_dir),
                        "rows": rows, "attempts": attempts, "log": str(collect_log),
                    }
                    status["current_step"] = "collect"  # 失败归因到采集步（重试标记只留在 attempts）
                    if rc != 0:
                        _write_status(status_path, status)
                        return _finish(
                            1, "failed",
                            f"r{round_index} 采集失败 rc={rc}（尝试 {[item.get('device') for item in attempts]}；"
                            f"log={collect_log}）",
                        )
                    if not (dagger_dir / "expert_bc.npz").is_file():
                        _write_status(status_path, status)
                        return _finish(1, "failed", f"r{round_index} 采集产物缺失：{dagger_dir / 'expert_bc.npz'}")
                    if not rows or int(rows) <= 0:
                        _write_status(status_path, status)
                        return _finish(
                            1, "failed",
                            f"r{round_index} 采集 0 行（rows={rows}）：该轮无失败窗口 → 停止（不静默训练空数据）",
                        )
                    self._log(f"r{round_index} collect ok rows={rows} ({duration:.0f}s)")

                # ---- ② 训练（R1 顺序续训：起点 = 当轮 running student；OOM 自愈：micro 减半整步重试）----
                status["current_step"] = "train"
                _write_status(status_path, status)
                train_out = runs_root / run_paths.canonical_run_name(
                    kind="stageB", tag=f"phase3_r{round_index}", stamp=stamp
                ) / "stage_b"
                train_log = log_dir / f"round{round_index}_train.log"
                reused = self._reuse_train(round_index) if cfg.resume else None
                if reused is not None:
                    # R1 provenance：复用训练产物的 init_ckpt 必须 = 期望父节点（当轮 running student）
                    ok, detail = _verify_train_provenance(reused.parent / "metrics.json", student)
                    if not ok:
                        round_entry["train"] = {"rc": None, "reused": str(reused), "out": str(reused.parent)}
                        _write_status(status_path, status)
                        return _finish(
                            1, "failed",
                            f"r{round_index} 训练复用 provenance 校验失败：{detail}"
                            f"（--phase3-resume 拒绝静默拼接；如需重训请移除 {reused.parent}）",
                        )
                    status["current_step"] = "train_reuse"
                    final_ckpt = reused
                    train_metrics = _read_json(reused.parent / "metrics.json")
                    round_entry["train"] = {
                        "rc": None, "reused": str(reused), "duration_s": 0.0,
                        "out": str(reused.parent), "final": str(reused),
                        "ckpt_in": str(student),
                        "metrics": (str(reused.parent / "metrics.json") if train_metrics is not None else None),
                        "losses": _loss_subset(train_metrics), "micro": cfg.micro,
                        "attempts": [], "log": None,
                    }
                    _write_status(status_path, status)
                    self._log(f"r{round_index} train reuse={reused}")
                else:
                    free_mib, gpu_ok = self._gpu_check(round_index=round_index, step="train")
                    if not gpu_ok:
                        return _finish(
                            1, "failed",
                            f"r{round_index} 训练前 GPU 空闲不足：free={free_mib:.0f} MiB < "
                            f"{cfg.gpu_min_free_mib:.0f} MiB（释放+等待 {GPU_WAIT_RETRIES} 次后仍不足）",
                        )
                    self._log(
                        f"r{round_index} train start out={train_out} ckpt={student} micro={cfg.micro} "
                        f"free={free_mib if free_mib is not None else 'n/a'}"
                    )
                    started = time.perf_counter()
                    reserved_before = _gpu_reserved_mib()
                    micro = cfg.micro
                    attempts: List[Dict[str, Any]] = []
                    retries = 0
                    exhausted = False
                    rc, oom = 1, False
                    while True:
                        attempt: Dict[str, Any] = {"micro": micro}
                        log_start = train_log.stat().st_size if train_log.exists() else 0
                        try:
                            rc = int(
                                self._train_fn(
                                    self._train_argv(
                                        dagger_dir=dagger_dir, train_out=train_out,
                                        ckpt=student, round_index=round_index, micro=micro,
                                    ),
                                    train_log,
                                )
                            )
                            oom = rc != 0 and _log_has_oom(train_log, start=log_start)
                        except Exception as exc:  # noqa: BLE001 - 训练异常（含 CUDA OOM）
                            rc = 1
                            oom = _is_cuda_oom(exc)
                            attempt["error"] = f"{type(exc).__name__}: {exc}"
                            if not oom:
                                attempts.append({**attempt, "rc": rc, "oom": False})
                                reserved_after = self._release_after_training(round_index, reserved_before)
                                round_entry["train"] = {
                                    "rc": rc, "duration_s": round(time.perf_counter() - started, 1),
                                    "out": str(train_out), "final": None, "metrics": None,
                                    "losses": None, "micro": micro, "attempts": attempts,
                                    "ckpt_in": str(student),
                                    "reserved_mib": {"before": reserved_before, "after": reserved_after},
                                    "log": str(train_log),
                                }
                                status["current_step"] = "train"
                                _write_status(status_path, status)
                                self._log_exception(exc)
                                return _finish(
                                    1, "failed",
                                    f"r{round_index} 训练异常 {type(exc).__name__}: {exc}（log={train_log}）",
                                )
                        attempt.update({"rc": rc, "oom": bool(oom)})
                        attempts.append(attempt)
                        if not oom:
                            break
                        if retries >= OOM_MAX_RETRIES or (micro is not None and micro <= OOM_MICRO_FLOOR):
                            exhausted = True
                            break
                        retries += 1
                        micro = max(
                            OOM_MICRO_FLOOR,
                            (micro if micro is not None else OOM_MICRO_FALLBACK) // 2,
                        )
                        status["current_step"] = f"train_retry_micro{micro}"
                        round_entry["train"] = {
                            "rc": None, "duration_s": None, "out": str(train_out),
                            "final": None, "metrics": None, "losses": None,
                            "micro": micro, "attempts": attempts, "ckpt_in": str(student),
                            "log": str(train_log),
                        }
                        _write_status(status_path, status)
                        self._log(
                            f"r{round_index} train OOM（micro={attempts[-1]['micro']}）→ "
                            f"empty_cache+gc → 重试 micro={micro}（第 {retries}/{OOM_MAX_RETRIES} 次）"
                        )
                        _free_cuda_memory()
                    duration = time.perf_counter() - started
                    # lane P3-E：训练步结束后**无论成功/失败/重试**强制释放 + reserved 回落校验
                    reserved_after = self._release_after_training(round_index, reserved_before)
                    final_ckpt = train_out / "final.pt"
                    train_metrics = _read_json(train_out / "metrics.json")
                    round_entry["train"] = {
                        "rc": rc, "duration_s": round(duration, 1), "out": str(train_out),
                        "final": (str(final_ckpt) if final_ckpt.is_file() else None),
                        "ckpt_in": str(student),
                        "metrics": (str(train_out / "metrics.json") if train_metrics is not None else None),
                        "losses": _loss_subset(train_metrics), "micro": attempts[-1]["micro"],
                        "attempts": attempts,
                        "reserved_mib": {"before": reserved_before, "after": reserved_after},
                        "log": str(train_log),
                    }
                    status["current_step"] = "train"  # 失败归因到训练步（重试标记只留在 attempts）
                    if exhausted:
                        _write_status(status_path, status)
                        return _finish(
                            1, "failed",
                            f"r{round_index} 训练 OOM 重试耗尽（尝试 micro={[item['micro'] for item in attempts]}；"
                            f"log={train_log}）",
                        )
                    if rc != 0:
                        _write_status(status_path, status)
                        return _finish(1, "failed", f"r{round_index} 训练失败 rc={rc}（log={train_log}）")
                    if not final_ckpt.is_file():
                        _write_status(status_path, status)
                        return _finish(1, "failed", f"r{round_index} 训练产物缺失：{final_ckpt}")
                    self._log(
                        f"r{round_index} train ok final={final_ckpt} micro={attempts[-1]['micro']} ({duration:.0f}s)"
                    )
                round_entry["student_out"] = str(final_ckpt)

                # ---- ③ 评测（断点复用：最新可用 metrics.json）----
                status["current_step"] = "eval"
                _write_status(status_path, status)
                eval_dir = runs_root / run_paths.canonical_run_name(
                    kind="eval500", tag=f"phase3_r{round_index}", stamp=stamp
                )
                eval_log = log_dir / f"round{round_index}_eval.log"
                reused = self._reuse_eval(round_index) if cfg.resume else None
                if reused is not None:
                    # R1 provenance：复用评测的 ckpt 必须 = 期望父节点（本轮训练产物）
                    ok, detail = _verify_eval_provenance(reused, final_ckpt)
                    if not ok:
                        round_entry["eval"] = {"rc": None, "reused": str(reused), "dir": str(reused.parent)}
                        _write_status(status_path, status)
                        return _finish(
                            1, "failed",
                            f"r{round_index} 评测复用 provenance 校验失败：{detail}"
                            f"（--phase3-resume 拒绝静默拼接；如需重评请移除 {reused.parent}）",
                        )
                    status["current_step"] = "eval_reuse"
                    eval_metrics_path = reused
                    eval_dir = reused.parent
                    summary = _eval_summary(reused)
                    round_entry["eval"] = {
                        "rc": None, "reused": str(reused), "duration_s": 0.0, "dir": str(eval_dir),
                        "metrics": str(reused), "overall_success": summary.get("overall_success"),
                        "easy_success": summary.get("easy_success"), "log": None,
                    }
                    _write_status(status_path, status)
                    self._log(
                        f"r{round_index} eval reuse={reused} overall={summary.get('overall_success')} "
                        f"easy={summary.get('easy_success')}"
                    )
                else:
                    free_mib, gpu_ok = self._gpu_check(round_index=round_index, step="eval")
                    if not gpu_ok:
                        return _finish(
                            1, "failed",
                            f"r{round_index} 评测前 GPU 空闲不足：free={free_mib:.0f} MiB < "
                            f"{cfg.gpu_min_free_mib:.0f} MiB（释放+等待 {GPU_WAIT_RETRIES} 次后仍不足）",
                        )
                    self._log(
                        f"r{round_index} eval start ckpt={final_ckpt} dir={eval_dir} "
                        f"free={free_mib if free_mib is not None else 'n/a'}"
                    )
                    started = time.perf_counter()
                    attempts = []
                    rc, oom = 1, False
                    while True:
                        attempt: Dict[str, Any] = {}
                        log_start = eval_log.stat().st_size if eval_log.exists() else 0
                        try:
                            rc = int(
                                self._eval_fn(
                                    self._eval_argv(final_ckpt=final_ckpt, eval_dir=eval_dir), eval_log
                                )
                            )
                            oom = rc != 0 and _log_has_oom(eval_log, start=log_start)
                        except Exception as exc:  # noqa: BLE001 - 评测异常（含 CUDA OOM）
                            rc = 1
                            oom = _is_cuda_oom(exc)
                            attempt["error"] = f"{type(exc).__name__}: {exc}"
                            if not oom:
                                attempts.append({**attempt, "rc": rc, "oom": False})
                                round_entry["eval"] = {
                                    "rc": rc, "duration_s": round(time.perf_counter() - started, 1),
                                    "dir": str(eval_dir), "metrics": None, "attempts": attempts,
                                    "log": str(eval_log),
                                }
                                _write_status(status_path, status)
                                self._log_exception(exc)
                                return _finish(
                                    1, "failed", f"r{round_index} 评测异常 {type(exc).__name__}: {exc}"
                                )
                        attempt.update({"rc": rc, "oom": bool(oom)})
                        attempts.append(attempt)
                        if not oom or len(attempts) > EVAL_OOM_RETRIES:
                            break
                        _free_cuda_memory()
                        status["current_step"] = f"eval_retry{len(attempts)}"
                        round_entry["eval"] = {
                            "rc": None, "duration_s": None, "dir": str(eval_dir),
                            "metrics": None, "attempts": attempts, "log": str(eval_log),
                        }
                        _write_status(status_path, status)
                        self._log(
                            f"r{round_index} eval OOM → 释放 → 重试（第 {len(attempts)}/{EVAL_OOM_RETRIES} 次）"
                        )
                    duration = time.perf_counter() - started
                    eval_metrics_path = eval_dir / "metrics.json"
                    summary = _eval_summary(eval_metrics_path)
                    round_entry["eval"] = {
                        "rc": rc, "duration_s": round(duration, 1), "dir": str(eval_dir),
                        "metrics": (str(eval_metrics_path) if eval_metrics_path.is_file() else None),
                        "overall_success": summary.get("overall_success"),
                        "easy_success": summary.get("easy_success"),
                        "attempts": attempts,
                        "log": str(eval_log),
                    }
                    status["current_step"] = "eval"  # 失败归因到评测步（重试标记只留在 attempts）
                    if rc != 0:
                        _write_status(status_path, status)
                        return _finish(1, "failed", f"r{round_index} 评测失败 rc={rc}（log={eval_log}）")
                    if not eval_metrics_path.is_file():
                        _write_status(status_path, status)
                        return _finish(1, "failed", f"r{round_index} 评测产物缺失：{eval_metrics_path}")
                    self._log(
                        f"r{round_index} eval ok overall={summary.get('overall_success')} "
                        f"easy={summary.get('easy_success')} ({duration:.0f}s)"
                    )

                # ---- ④ 护栏（R4：mode=base 相对入口基座；prev 相对上一轮；off 不判定）----
                status["current_step"] = "guard"
                decision = guard_decision(
                    previous_eval,
                    summary,
                    mode=cfg.guard_mode,
                    base=base_eval,
                    stop_on_base_drop=cfg.stop_on_base_drop,
                    stop_on_overall_drop=cfg.stop_on_overall_drop,
                    stop_on_easy_drop=cfg.stop_on_easy_drop,
                )
                round_entry["guard"] = decision
                if decision["mode"] == "off":
                    self._log(f"r{round_index} guard mode=off（不判定）")
                elif decision["mode"] == "base":
                    self._log(
                        f"r{round_index} guard mode=base base={decision['base_overall']} "
                        f"net_vs_base={decision['net_vs_base']} stopped={decision['stopped']}"
                    )
                elif decision["checked"]:
                    self._log(
                        f"r{round_index} guard mode=prev overall_delta={decision['overall_delta']} "
                        f"easy_delta={decision['easy_delta']} stopped={decision['stopped']}"
                    )
                else:
                    self._log(f"r{round_index} guard 基线（无上一轮，不判定）")
                # 候选留档：指标缺失（None）也入清单（不参与选择；缺失不停）
                candidates.append(
                    {
                        "round": round_index,
                        "is_base": False,
                        "ckpt": str(final_ckpt),
                        "ckpt_sha256": _sha256_file(final_ckpt),
                        "eval_dir": str(eval_dir),
                        "metrics": str(eval_metrics_path),
                        "overall_success": _opt_num(summary.get("overall_success")),
                        "easy_success": _opt_num(summary.get("easy_success")),
                    }
                )
                previous_eval = dict(summary)
                student = str(final_ckpt)
                _write_status(status_path, status)
                if decision["stopped"]:
                    best = self._export_best(
                        loop_dir=loop_dir,
                        candidates=candidates,
                        base_overall=base_eval.get("overall_success"),
                    )
                    status["best"] = best
                    return _finish(0, "stopped_guard", str(decision["reason"]))

            best = self._export_best(
                loop_dir=loop_dir,
                candidates=candidates,
                base_overall=base_eval.get("overall_success"),
            )
            status["best"] = best
            if best["no_gain"]:
                return _finish(
                    0, "no_gain",
                    f"{int(cfg.rounds)} 轮均未超过基座（best=基座 overall={best['overall_success']}；"
                    "keep-best 已导出基座 → best/final.pt）",
                )
            return _finish(0, "completed")
        except KeyboardInterrupt:
            return _finish(130, "failed", "KeyboardInterrupt: 用户中断")
        except BaseException as exc:  # noqa: BLE001 - 任何未捕获异常 → failed 落盘后非零退出
            detail = f"{type(exc).__name__}: {exc}"
            self._log(f"异常未捕获：{detail}")
            self._log(traceback.format_exc().rstrip())
            return _finish(1, "failed", detail)
        finally:
            self._status = None
            self._status_path = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None


def _eval_summary(metrics_path: Path) -> Dict[str, Any]:
    """评测 ``metrics.json`` → ``{overall_success, easy_success}``（缺键/NaN → None）。"""
    payload = _read_json(metrics_path)
    if payload is None:
        return {"overall_success": None, "easy_success": None}
    overall = _opt_num((payload.get("overall") or {}).get("success_rate"))
    easy = _opt_num(((payload.get("by_difficulty") or {}).get("easy") or {}).get("success_rate"))
    return {"overall_success": overall, "easy_success": easy}


def _phase3_only_enabled(args: argparse.Namespace) -> bool:
    """``--phase3-only`` 或环境变量 ``PHASE3_ONLY=1``（tools/train.sh 的接口）。"""
    if bool(getattr(args, "phase3_only", False)):
        return True
    return str(os.environ.get("PHASE3_ONLY", "")).strip().lower() in ("1", "true", "yes", "on")


def _stage_argv(
    resolved: Mapping[str, str],
    stage: str,
    *,
    config: str,
    device: Optional[str],
    limit: Optional[int],
    extra: Sequence[str],
) -> List[str]:
    """run_config 解析结果 → ``tools/train.py`` 单阶段 argv（与 train.sh 旧口径逐项一致）。"""
    argv = [
        "--stage", stage,
        "--config", str(config),
        "--bc-dir", str(resolved["BC_DIR"]),
        "--out", str(resolved["STAGE_DIR"]),
        "--wm-epochs", str(resolved["WM_EPOCHS"]),
        "--bc-epochs", str(resolved["BC_EPOCHS"]),
        "--ckpt-every", str(resolved["CKPT_EVERY"]),
        "--batch-size", str(resolved["BATCH_SIZE"]),
        "--micro-batch-size", str(resolved["MICRO_BATCH_SIZE"]),
    ]
    if resolved.get("TRAJ_AUX_WEIGHT"):
        argv += ["--traj-aux-weight", str(resolved["TRAJ_AUX_WEIGHT"])]
    if resolved.get("CKPT"):
        argv += ["--ckpt", str(resolved["CKPT"])]
    if resolved.get("RESUME"):
        argv += ["--resume", str(resolved["RESUME"])]
    if device:
        argv += ["--device", str(device)]
    if limit:
        argv += ["--limit-dataset", str(int(limit))]
    argv += [str(item) for item in extra]
    return argv


def _run_train_stage(
    stage: str,
    *,
    config: str,
    device: Optional[str],
    limit: Optional[int],
    extra: Sequence[str],
) -> int:
    """``run_config`` 解析 → ``tools/train.py`` 单阶段 **子进程**（输出 tee 到该阶段日志）。

    - 与旧 shell 口径逐项一致（``tools/venv-python tools/train.py --stage ...``）；
    - 独立进程 = A/B 之间无全局缓存/显存残留（旧 shell 的内存语义）；``run_config`` 与 shell
      同一出处；A 完成后解析 B 时 ``stage_a_ckpt=auto`` 会拿到刚产出的 A ``final.pt``（run 根共用）。
    """
    from tools import run_config

    from pipeline.stages import load_config  # 延迟导入（合并 includes）

    try:
        resolved = run_config.resolve(run_config.ROOT, load_config(str(config)), stage_arg=stage)
    except SystemExit as exc:  # run_config 的显式失败（无数据集等）
        print(f"[chain] stage={stage} run_config 失败", file=sys.stderr)
        code = exc.code
        return int(code) if isinstance(code, int) else 2
    log_path = Path(resolved["LOG"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[chain] stage={stage} start {_now()} work_dir={resolved['WORK_DIR']} out={resolved['STAGE_DIR']}",
        flush=True,
    )
    rc = run_subprocess(
        [
            VENV_PY,
            "tools/train.py",
            *_stage_argv(resolved, stage, config=config, device=device, limit=limit, extra=extra),
        ],
        log_path,
    )
    print(f"[chain] stage={stage} exit={rc} {_now()}", flush=True)
    return rc


def run_chain(argv: Optional[Sequence[str]] = None) -> int:
    """``tools/train.py --phase3-chain``：A→B→phase3 循环全链（``PHASE3_ONLY=1`` 跳过 A/B）。

    编排自包含：A/B 走 ``run_config`` + ``tools/train.py``（与 shell 旧口径一致），
    phase 3 循环复用 :class:`Phase3Loop`；任一步失败立即停止（非零返回）。

    **lane P3-G（全链起点语义）**：B 成功后，phase 3 的 ``init_ckpt`` 自动覆盖为**本次链**刚训完的
    B ``<run>/stage_b/final.pt``（``init_ckpt_source="chain-B"``），不再用 config 的固定旧路径；
    ``PHASE3_ONLY`` / 独立 loop 模式行为不变（仍用 config / ``--ckpt``）。
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args, extra = _parse_chain_args(argv_list)
    blocked = sorted({item for item in extra if item in _RESERVED_EXTRA_ARGS})
    if blocked:
        print(f"[phase3] 透传参数不得覆盖编排保留键：{blocked}", file=sys.stderr)
        return 2
    if not Path(args.config).is_file():
        print(f"[phase3] 配置不存在：{args.config}", file=sys.stderr)
        return 2
    from pipeline.stages import load_config  # 延迟导入（合并 includes）

    try:
        config = load_config(args.config)
        cfg = Phase3LoopConfig.from_config(config, args, extra=extra)
    except (OSError, ValueError) as exc:
        print(f"[phase3] 配置解析失败：{exc}", file=sys.stderr)
        return 2
    # 起点/池/spec 前置校验：全链模式下 A/B 很长，缺产物要**先**报错（不带病训练几小时）
    for label, path in (
        ("起点权重 init_ckpt", cfg.init_ckpt),
        ("采集池 spec_pool", cfg.spec_pool),
        ("评测 spec", cfg.eval_spec),
    ):
        if not path or not _abs(path).is_file():
            print(f"[phase3] {label} 不存在：{path!r}", file=sys.stderr)
            return 2
    try:
        if not _phase3_only_enabled(args):
            rc = _run_train_stage("A", config=args.config, device=args.device, limit=args.limit_dataset, extra=extra)
            if rc != 0:
                print(f"[chain] A 失败（rc={rc}）→ 停止，不进入 B/phase3", file=sys.stderr)
                return rc
            rc = _run_train_stage("B", config=args.config, device=args.device, limit=args.limit_dataset, extra=extra)
            if rc != 0:
                print(f"[chain] B 失败（rc={rc}）→ 停止，不进入 phase3", file=sys.stderr)
                return rc
            # lane P3-G：phase 3 起点 = **本次链**刚训完的 B phase-2 final.pt（覆盖 config 值）
            b_final = _stage_final_ckpt("B", config=args.config)
            if b_final is None or not b_final.is_file():
                print(
                    f"[chain] B 产物 final.pt 不存在：{b_final} → 停止，不进入 phase3"
                    "（检查 run_config 的 STAGE_DIR / B 是否真的写出 final.pt）",
                    file=sys.stderr,
                )
                return 2
            previous = cfg.init_ckpt
            cfg = replace(cfg, init_ckpt=str(b_final), init_ckpt_source="chain-B")
            print(
                f"[chain] phase3 init_ckpt ← 本次链 B final：{b_final}"
                f"（覆盖 {previous or '<空>'}；source=chain-B）",
                flush=True,
            )
            # 链内 B 阶段监控（2026-09-29）：phase1/phase2 完成后各做一次闭环评测（同一评测口径/worker 数）
            from pipeline import run_paths

            stamp = str(cfg.stamp or run_paths.beijing_stamp_seconds())
            cfg = replace(cfg, stamp=stamp)
            for mon_tag, mon_ckpt in (
                ("phase1", Path(b_final).parent / "primary.pt"),
                ("phase2", Path(b_final)),
            ):
                if not mon_ckpt.is_file():
                    print(f"[chain] B 阶段监控评测跳过（缺 {mon_ckpt}）", flush=True)
                    continue
                rc_mon = _chain_monitor_eval(mon_tag, Path(mon_ckpt), cfg)
                if rc_mon != 0:
                    print(f"[chain] B 监控评测 {mon_tag} 失败（rc={rc_mon}）→ 停止，不进入 phase3", file=sys.stderr)
                    return rc_mon
        return int(Phase3Loop(cfg).run())
    except KeyboardInterrupt:  # pragma: no cover - 交互中断
        print("[phase3] interrupted", file=sys.stderr)
        return 130


def _chain_monitor_eval(tag: str, ckpt: Path, cfg: "Phase3LoopConfig") -> int:
    """链内 B 阶段监控评测（phase1/phase2 各一次；与评测同口径的 ``tools/test.py`` 子进程）。"""
    from pipeline import run_paths

    name = run_paths.canonical_run_name(kind="eval500", tag=tag, stamp=str(cfg.stamp))
    argv = [
        VENV_PY, "tools/test.py",
        "--policy", "ckpt",
        "--ckpt", str(ckpt),
        "--spec", str(_abs(cfg.eval_spec)),
        "--out", str(_abs(cfg.runs_root)),
        "--name", name,
        "--workers", str(int(cfg.eval_workers)),
        "--tracker", "lqr",
        "--config", str(_abs(cfg.config)),
    ]
    if cfg.device:
        argv += ["--device", str(cfg.device)]
    if cfg.limit:
        argv += ["--limit", str(int(cfg.limit))]
    print(f"[chain] B 阶段监控评测（{tag}）→ {name}", flush=True)
    return run_subprocess(argv)


def _add_loop_flags(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """loop / chain 两种模式共用的 CLI 旗标（单一出处，避免两套解析漂移）。"""
    parser.add_argument("--config", default="config/default.yaml", help="主配置（includes 合并）")
    parser.add_argument("--model-config", default="config/model.yaml", help="模型配置（透传训练/采集）")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="起点权重（默认取 config stages.B.phase3.init_ckpt）")
    parser.add_argument("--phase3-rounds", type=int, default=None,
                        help="轮数（CLI > 环境变量 PHASE3_ROUNDS > config stages.B.phase3.rounds=5）")
    parser.add_argument("--phase3-spec-pool", default=None, help="采集池（默认 config stages.B.phase3.spec_pool）")
    parser.add_argument("--phase3-runs-root", default=None, help="runs 根（默认 runs；测试/冒烟可指到 /tmp）")
    parser.add_argument("--phase3-datasets-root", default=None, help="datasets 根（默认 datasets）")
    parser.add_argument("--phase3-stamp", default=None, help="北京戳覆盖（默认当前；测试/复现用）")
    parser.add_argument("--phase3-resume", action="store_true",
                        help="断点复用（等价环境变量 PHASE3_RESUME=1）：每步前若已有可用产物则跳过"
                             "（采集 report.counts.stored_rows>0 / 训练 final.pt+metrics.json / 评测 metrics.json）")
    parser.add_argument("--device", default=None, help="设备（透传采集/训练/评测；默认按 config）")
    parser.add_argument("--limit-dataset", type=int, default=None,
                        help="冒烟：采集/评测 --limit + 训练 --limit-dataset 同值")
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> "tuple[argparse.Namespace, List[str]]":
    parser = argparse.ArgumentParser(
        prog="tools/train.py --phase3-loop",
        description="phase 3 全自动循环（采集→训练→评测 ×N + 护栏；lane P3-C）",
        allow_abbrev=False,
    )
    parser.add_argument("--phase3-loop", action="store_true", help="启用 phase 3 循环编排")
    _add_loop_flags(parser)
    args, extra = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, extra


def _parse_chain_args(argv: Optional[Sequence[str]] = None) -> "tuple[argparse.Namespace, List[str]]":
    parser = argparse.ArgumentParser(
        prog="tools/train.py --phase3-chain",
        description="phase 3 全链（A→B→循环；PHASE3=1/PHASE3_ONLY=1 由 tools/train.sh 调用）",
        allow_abbrev=False,
    )
    parser.add_argument("--phase3-chain", action="store_true", help="启用 A→B→phase3 全链")
    parser.add_argument("--phase3-only", action="store_true",
                        help="跳过 A/B（等价环境变量 PHASE3_ONLY=1）")
    _add_loop_flags(parser)
    args, extra = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, extra


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args, extra = _parse_args(argv_list)
    blocked = sorted({item for item in extra if item in _RESERVED_EXTRA_ARGS})
    if blocked:
        print(f"[phase3] 透传参数不得覆盖编排保留键：{blocked}", file=sys.stderr)
        return 2
    if not Path(args.config).is_file():
        print(f"[phase3] 配置不存在：{args.config}", file=sys.stderr)
        return 2
    from pipeline.stages import load_config  # 延迟导入（合并 includes）

    try:
        config = load_config(args.config)
        cfg = Phase3LoopConfig.from_config(config, args, extra=extra)
    except (OSError, ValueError) as exc:
        print(f"[phase3] 配置解析失败：{exc}", file=sys.stderr)
        return 2
    try:
        return int(Phase3Loop(cfg).run())
    except KeyboardInterrupt:  # pragma: no cover - 交互中断
        print("[phase3] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
