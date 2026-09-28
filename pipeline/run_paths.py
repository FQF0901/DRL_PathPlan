"""运行目录布局的唯一来源（tools/train.py / tools/test.py / tools/run_config.py 共用；仅 stdlib）。

约定（2026-09-27 定稿；非侵入——不改 ``pipeline/stages.py``）：

- work_dir（训练，auto）= ``runs/BTC<北京戳>_<name>``；**同轮 A/B 共用一个 run 目录**：
  Stage B 的 work_dir 直接取它使用的 Stage A run（``stage_a_ckpt`` 所在目录）的 run 根。
- work_dir 内允许的文件（不凭空自创文件名）：``logs/stage_<x>.log``（每阶段唯一日志：
  launcher 行 + 训练 stdout/stderr）、``manifest.txt``、``config.snapshot.yaml``、
  ``model.snapshot.yaml``、``stage_a/``、``stage_b/``（训练由 stages.py 写在 ``--out`` 下，
  含 ckpt/monitor/metrics 等——保持原位）。
- 兼容旧平铺 run（``runs/<run>/stage_a/{final.pt,ckpt_epochNNN.pt}`` 及 ckpt 直接位于 run 根）的发现。
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_BJ = timezone(timedelta(hours=8))
_LEGACY_STAGE_RE = re.compile(r"_stage([AB])$")


def beijing_stamp(now: datetime | None = None) -> str:
    """北京戳 ``YYYYMMDD-HHMM``（datasets/ 与 runs/ 命名统一用它）。"""
    return (now or datetime.now(_BJ)).strftime("%Y%m%d-%H%M")


def new_work_dir(name: str, root: Path = ROOT, stamp: str | None = None) -> Path:
    """``runs/BTC<北京戳>_<name>``（同轮 A/B 共用；B 传 A run 的戳即可共戳）。"""
    return Path(root) / f"runs/BTC{stamp or beijing_stamp()}_{name}"


def stage_dir(work_dir: str | Path, stage: str) -> Path:
    """阶段的 ``--out`` 目录：``<work_dir>/stage_<x>``。"""
    return Path(work_dir) / f"stage_{stage.lower()}"


def work_dir_of_out(out: str | Path) -> Path:
    """``--out`` → work_dir：``<work>/stage_<x>``（或历史 ``<work>/ckpt``）时返回 ``<work>``。"""
    path = Path(out)
    return path.parent if path.name in ("ckpt", "stage_a", "stage_b") else path


def logs_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / "logs"


def stage_log(work_dir: str | Path, stage: str) -> Path:
    """每阶段唯一日志（launcher 行 + 训练 stdout/stderr 都进这里）。"""
    return logs_dir(work_dir) / f"stage_{stage.lower()}.log"


def manifest_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "manifest.txt"


def config_snapshot_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "config.snapshot.yaml"


def model_snapshot_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / "model.snapshot.yaml"


def detach_pid_path(work_dir: str | Path) -> Path:
    """分离运行（setsid+nohup）的 pid 文件（work-dir 允许清单内）。"""
    return Path(work_dir) / "detach.pid"


def prepare_layout(work_dir: str | Path, stage: str | None = None) -> dict[str, Path]:
    """建 ``logs/`` + 阶段目录并返回布局路径（幂等）。"""
    work_dir = Path(work_dir)
    logs_dir(work_dir).mkdir(parents=True, exist_ok=True)
    layout = {
        "work_dir": work_dir,
        "logs": logs_dir(work_dir),
        "manifest": manifest_path(work_dir),
        "config_snapshot": config_snapshot_path(work_dir),
        "model_snapshot": model_snapshot_path(work_dir),
        "detach_pid": detach_pid_path(work_dir),
    }
    if stage:
        layout["stage"] = stage_dir(work_dir, stage)
        layout["stage"].mkdir(parents=True, exist_ok=True)
        layout["log"] = stage_log(work_dir, stage)
    return layout


def _git(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(root), capture_output=True, text=True, timeout=5, check=False
        )
        return proc.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 取证字段尽力而为
        return "unknown"


def write_manifest(
    work_dir: str | Path,
    *,
    stage: str,
    config: str | Path | None = None,
    model_config: str | Path | None = None,
    out: str | Path | None = None,
    argv: "list[str] | tuple[str, ...]" = (),
    root: Path = ROOT,
) -> Path:
    """写 ``<work_dir>/manifest.txt`` + 配置快照（入口层取证，非侵入）。"""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fields = {
        "created_at": datetime.now(_BJ).isoformat(timespec="seconds"),
        "stage": stage,
        "git": f"{_git(root, 'rev-parse', 'HEAD')} @{_git(root, 'rev-parse', '--abbrev-ref', 'HEAD')} "
               f"dirty={len(_git(root, 'status', '--porcelain').splitlines())}",
        "config": str(config) if config else None,
        "model_config": str(model_config) if model_config else None,
        "work_dir": str(work_dir),
        "out": str(out) if out else None,
        "argv": shlex.join(str(item) for item in argv),
    }
    manifest_path(work_dir).write_text(
        "\n".join(f"{key}: {value}" for key, value in fields.items() if value is not None) + "\n",
        encoding="utf-8",
    )
    for src, dest in ((config, config_snapshot_path(work_dir)), (model_config, model_snapshot_path(work_dir))):
        if not src:
            continue
        src_path = Path(src)
        if not src_path.is_absolute():
            src_path = root / src_path
        if src_path.is_file():
            dest.write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")
    return work_dir


def extra_value(argv: "list[str] | tuple[str, ...]", key: str) -> str:
    """从 argv 里取 ``--key value`` / ``--key=value``（入口透传参数的读取用）。"""
    for index, item in enumerate(argv):
        if item == key and index + 1 < len(argv):
            return str(argv[index + 1])
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return ""


# ---------------------------------------------------------------- 发现（run 根 + stage 子目录；兼容旧平铺）
def _legacy_stage_of(name: str) -> str | None:
    """ckpt 直接落在 run 根的旧式命名 ``*_stageA|B`` → stage。"""
    match = _LEGACY_STAGE_RE.search(name)
    return match.group(1) if match else None


def run_candidates(root: Path = ROOT) -> list[tuple[Path, Path, str]]:
    """所有可发现的 stage run：``(work_dir, out_dir, stage)``，按 work_dir（≈时间戳）新→旧。"""
    entries: list[tuple[Path, Path, str]] = []
    for item in Path(root).glob("runs/*"):
        if not item.is_dir():
            continue
        found_sub = False
        for sub, stage in (("stage_a", "A"), ("stage_b", "B")):
            sub_dir = item / sub
            if sub_dir.is_dir():
                found_sub = True
                entries.append((item, sub_dir, stage))
        if not found_sub:
            legacy = _legacy_stage_of(item.name)
            if legacy:
                entries.append((item, item, legacy))
    entries.sort(key=lambda entry: str(entry[0]), reverse=True)
    return entries


def latest_final(root: Path = ROOT, stage: str = "A") -> Path | None:
    """最新 ``stage_<x>/final.pt``。"""
    for _work_dir, out_dir, entry_stage in run_candidates(root):
        final = out_dir / "final.pt"
        if entry_stage == stage.upper() and final.is_file():
            return final
    return None


def pid_alive(pid_file: str | Path) -> bool:
    """``detach.pid`` 里的进程是否仍存活（resume auto 跳过仍在运行的 run，避免双写）。"""
    try:
        pid = int(Path(pid_file).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def find_unfinished(root: Path = ROOT) -> tuple[Path, Path, str, Path] | None:
    """最新「有 ``ckpt_epoch*.pt`` 且无 ``final.pt`` 且不在运行」的 ``(work_dir, out_dir, stage, ckpt)``。"""
    for work_dir, out_dir, stage in run_candidates(root):
        if (out_dir / "final.pt").is_file() or pid_alive(detach_pid_path(work_dir)):
            continue
        ckpts = sorted(out_dir.glob("ckpt_epoch*.pt"), key=lambda item: item.name)
        if ckpts:
            return work_dir, out_dir, stage, ckpts[-1]
    return None


def latest_dataset(root: Path = ROOT) -> Path | None:
    """最新**训练用** ``datasets/BTC*_expert*``（须含 ``expert_bc.npz``）。

    排除评测集（名字含 ``val``，如 ``*_expert500val``）——否则零参跑会把评测集当训练数据。
    """
    hits = [
        d
        for d in Path(root).glob("datasets/BTC*_expert*")
        if (d / "expert_bc.npz").is_file() and "val" not in d.name.lower()
    ]
    return sorted(hits, key=lambda item: item.name)[-1] if hits else None


# ---------------------------------------------------------------- TensorBoard run 名（tools/tb.sh）
#: ``--logdir_spec`` 的固定 run 名（顺序即展示顺序）
TB_RUN_ORDER: tuple[str, ...] = ("train_stageA", "train_stageB", "eval_stageB", "eval_stageA", "eval_baseline")


def _monitor_with_events(run_dir: Path) -> Path | None:
    """``<run_dir>/monitor`` 内有 ``events.out.tfevents*`` 才收。"""
    monitor = run_dir / "monitor"
    if monitor.is_dir() and any(monitor.glob("events.out.tfevents*")):
        return monitor
    return None


def _eval_run_name(run_dir: Path) -> str:
    """评测 run 归属：``policy: baseline``（或无 ckpt）→ ``eval_baseline``；否则按 manifest 的
    ckpt 路径含 ``stage_b``/``stage_a`` 判 ``eval_stageB``/``eval_stageA``。"""
    manifest = run_dir / "manifest.txt"
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8", errors="ignore")
        if "policy: baseline" in text or "--policy baseline" in text:
            return "eval_baseline"
        if "stage_b" in text:
            return "eval_stageB"
        if "stage_a" in text:
            return "eval_stageA"
    return "eval_baseline"


def tb_spec(root: Path = ROOT) -> list[tuple[str, str]]:
    """TensorBoard ``--logdir_spec`` 键值：``[(run_name, monitor_dir)]``。

    run 名固定为 :data:`TB_RUN_ORDER`；每个名字只取**最新一个候选**（避免重名/重复）：
    ``train_stageA|B`` = 最新 ``runs/*/stage_a|b/monitor``，``eval_stageA|B`` = 最新
    ``runs/*eval*/monitor``（按 manifest 的 ckpt 判定归属）；无事件文件/无目录 → 跳过。
    """
    spec: dict[str, str] = {}
    for work_dir, out_dir, stage in run_candidates(root):  # 新→旧；同名字首个即最新
        name = f"train_stage{stage}"
        if name in spec:
            continue
        monitor = _monitor_with_events(out_dir)
        if monitor is not None:
            spec[name] = str(monitor)
    eval_dirs = sorted(
        (item for item in Path(root).glob("runs/*") if item.is_dir() and "eval" in item.name),
        key=lambda item: str(item),
        reverse=True,
    )
    for run_dir in eval_dirs:
        name = _eval_run_name(run_dir)
        if name in spec:
            continue
        monitor = _monitor_with_events(run_dir)
        if monitor is not None:
            spec[name] = str(monitor)
    return [(name, spec[name]) for name in TB_RUN_ORDER if name in spec]
