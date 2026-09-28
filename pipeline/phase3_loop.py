#!/usr/bin/env python3
"""phase 3 全自动循环编排（lane P3-C）：采集 → 训练 → 评测 ×N 轮 + 护栏 + 状态文件。

由 ``tools/train.py --phase3-loop``（仅循环）或 ``--phase3-chain``（A→B→循环全链）调用；
``tools/train.sh`` 的 ``PHASE3=1`` 走全链、``PHASE3_ONLY=1`` 直接进循环（**不新增 .sh**，
编排自包含在 repo 内）。

每轮 ``k=1..N``：

1. **采集**（subprocess ``tools/dagger_collect.py``）：学生 = 上一轮
   ``<work>/stage_b/final.pt``（``r1`` = ``stages.B.phase3.init_ckpt`` 的 phase2 产物），
   在 ``spec_pool`` 上攒 ``fail_target`` 个失败窗口（``--window-fail-before <collect.window_s>``、
   ``--shuffle-seed <shuffle_seed_base+k>``、``--workers <collect.workers>``）→
   ``<datasets_root>/BTC<stamp>_phase3_dagger_r{k}``；
2. **训练**（in-process ``tools/train.py --phase3 <dir> --phase3-round k``，**每轮起点恒为
   ``init_ckpt``**）→ ``<runs_root>/BTC<stamp>_stageB_phase3_r{k}/stage_b``；
3. **评测**（subprocess ``tools/test.py``，LQR，``eval.spec``/``eval.workers``）→
   ``<runs_root>/BTC<stamp>_eval500_phase3_r{k}``；
4. **护栏**：与**上一轮**比较 ``overall_success`` / ``easy`` 组 ``success_rate``，
   绝对下降超过 ``guard.stop_on_overall_drop`` / ``guard.stop_on_easy_drop`` → 记录原因并停止
   （不静默继续；``r1`` 无上一轮 → 只记录基线不判定）。

状态文件 ``<runs_root>/BTC<stamp>_phase3_loop/phase3_status.json``：每步**原子重写**
（tmp + ``os.replace``），含 loop 级状态 + 逐轮 collect/train/eval/guard 结果，供 10 分钟轮询审计；
日志 ``<runs_root>/BTC<stamp>_phase3_loop/logs/phase3_loop.log`` + 逐轮 ``round{k}_*.log``。

**错误语义**：任一步非零退出或产物缺失（``expert_bc.npz`` / ``final.pt`` / ``metrics.json``）
→ 记录清晰原因并把状态置 ``failed`` 后立即返回非零，绝不带缺产物进入下一轮。

用法::

    tools/venv-python tools/train.py --phase3-loop [--phase3-rounds N] \\
        [--phase3-runs-root runs] [--phase3-datasets-root datasets] [--limit-dataset M]
    tools/venv-python tools/train.py --phase3-chain [--phase3-only] [--phase3-rounds N]
    PHASE3=1 bash tools/train.sh            # A→B→循环 全链（一条命令全自动）
    PHASE3_ONLY=1 bash tools/train.sh       # 跳过 A/B 直接循环（上线）
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: 外部命令统一经 venv wrapper 启动（GL 修复；相对仓库根调用）
VENV_PY = "tools/venv-python"
STATUS_FILENAME = "phase3_status.json"
SCHEMA_VERSION = 1
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


def guard_decision(
    previous: Optional[Mapping[str, Any]],
    current: Mapping[str, Any],
    *,
    stop_on_overall_drop: float,
    stop_on_easy_drop: float,
) -> Dict[str, Any]:
    """护栏判定（纯函数）：返回 ``checked/stopped/reason/overall_delta/easy_delta``。

    - ``previous is None``（r1）→ 只记录基线，不判定；
    - 指标缺失/NaN → 该项不判定（``delta=None``），另一项照常；
    - 判定口径 = **绝对下降**（``上一轮 − 本轮 > 阈值`` 即停）。
    """
    result: Dict[str, Any] = {
        "checked": previous is not None,
        "stopped": False,
        "reason": None,
        "overall_delta": None,
        "easy_delta": None,
    }
    if previous is None:
        return result
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
    eval_spec: str = "env/specs/scenarios_eval500.json"
    eval_workers: int = 16
    stop_on_overall_drop: float = 0.05
    stop_on_easy_drop: float = 0.10
    config: str = "config/default.yaml"
    model_config: str = "config/model.yaml"
    device: Optional[str] = None
    limit: Optional[int] = None
    runs_root: str = "runs"
    datasets_root: str = "datasets"
    stamp: Optional[str] = None
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
        env_rounds = os.environ.get("PHASE3_ROUNDS", "").strip()
        if getattr(args, "phase3_rounds", None) is not None:
            rounds = int(args.phase3_rounds)
        elif env_rounds:
            rounds = int(env_rounds)
        else:
            rounds = int(p3.get("rounds", 5))
        if rounds < 1:
            raise ValueError(f"phase3 rounds 必须 ≥1，收到 {rounds}")
        init_ckpt = str(getattr(args, "ckpt", None) or p3.get("init_ckpt") or "")
        return cls(
            rounds=rounds,
            spec_pool=str(getattr(args, "phase3_spec_pool", None) or p3.get("spec_pool") or cls.spec_pool),
            fail_target=int(p3.get("fail_target", cls.fail_target)),
            shuffle_seed_base=int(p3.get("shuffle_seed_base", cls.shuffle_seed_base)),
            collect_workers=int(collect.get("workers", cls.collect_workers)),
            collect_window_s=float(collect.get("window_s", cls.collect_window_s)),
            init_ckpt=init_ckpt,
            eval_spec=str(eval_cfg.get("spec") or cls.eval_spec),
            eval_workers=int(eval_cfg.get("workers", cls.eval_workers)),
            stop_on_overall_drop=float(guard.get("stop_on_overall_drop", cls.stop_on_overall_drop)),
            stop_on_easy_drop=float(guard.get("stop_on_easy_drop", cls.stop_on_easy_drop)),
            config=str(getattr(args, "config", None) or cls.config),
            model_config=str(getattr(args, "model_config", None) or cls.model_config),
            device=(str(args.device) if getattr(args, "device", None) else None),
            limit=(int(args.limit_dataset) if getattr(args, "limit_dataset", None) else None),
            runs_root=str(getattr(args, "phase3_runs_root", None) or cls.runs_root),
            datasets_root=str(getattr(args, "phase3_datasets_root", None) or cls.datasets_root),
            stamp=(str(args.phase3_stamp) if getattr(args, "phase3_stamp", None) else None),
            extra_train_args=[str(item) for item in extra],
        )


class _Tee:
    """stdout 双写（控制台 + 日志文件）；用于 in-process 训练的输出落盘。"""

    def __init__(self, *streams: Any):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:  # noqa: BLE001 - 关闭后的 flush 不应影响训练
                pass


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


def default_train(argv: Sequence[Any], log_path: Optional[Path] = None) -> int:
    """in-process 调用 ``tools/train.py`` 的 phase-3 训练（输出 tee 到逐轮日志）。"""
    from tools.train import main as train_main

    handle = None
    stream: Any = sys.stdout
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a", encoding="utf-8")
        stream = _Tee(sys.stdout, handle)
    try:
        with contextlib.redirect_stdout(stream):
            try:
                return int(train_main([str(item) for item in argv]) or 0)
            except SystemExit as exc:  # stages 的 SystemExit（数据契约/缺文件）→ 记 rc
                code = exc.code
                if code is None:
                    return 0
                return int(code) if isinstance(code, int) else 1
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
        self._train_fn = train_fn or default_train
        self._eval_fn = eval_fn or run_subprocess
        self._logger = logger or print
        self._log_handle: Optional[Any] = None

    # ------------------------------------------------------------------ 日志/状态
    def _log(self, message: str) -> None:
        line = f"[phase3] {_now()} {message}"
        self._logger(line)
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()

    # ------------------------------------------------------------------ 命令构造
    def _collect_argv(self, *, student: str, dagger_dir: Path, round_index: int) -> List[str]:
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
        if cfg.device:
            argv += ["--device", str(cfg.device)]
        if cfg.limit:
            argv += ["--limit", str(int(cfg.limit))]
        return argv

    def _train_argv(self, *, dagger_dir: Path, train_out: Path, round_index: int) -> List[str]:
        cfg = self.cfg
        argv = [
            "--phase3", str(dagger_dir),
            "--phase3-round", str(round_index),
            "--ckpt", str(_abs(cfg.init_ckpt)),
            "--out", str(train_out),
            "--config", str(_abs(cfg.config)),
            "--model-config", str(_abs(cfg.model_config)),
        ]
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

    # ------------------------------------------------------------------ 主循环
    def run(self) -> int:
        cfg = self.cfg
        from pipeline import run_paths

        stamp = cfg.stamp or run_paths.beijing_stamp()
        runs_root = _abs(cfg.runs_root)
        datasets_root = _abs(cfg.datasets_root)
        loop_dir = runs_root / f"BTC{stamp}_phase3_loop"
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
            "config": {
                "spec_pool": str(cfg.spec_pool),
                "fail_target": int(cfg.fail_target),
                "shuffle_seed_base": int(cfg.shuffle_seed_base),
                "collect": {"workers": int(cfg.collect_workers), "window_s": float(cfg.collect_window_s)},
                "eval": {"spec": str(cfg.eval_spec), "workers": int(cfg.eval_workers)},
                "guard": {
                    "stop_on_overall_drop": float(cfg.stop_on_overall_drop),
                    "stop_on_easy_drop": float(cfg.stop_on_easy_drop),
                },
            },
            "rounds": [],
        }
        _write_status(status_path, status)
        self._log(
            f"start stamp={stamp} rounds={cfg.rounds} runs_root={runs_root} "
            f"status={status_path}"
        )
        self._log(
            f"cfg spec_pool={cfg.spec_pool} fail_target={cfg.fail_target} "
            f"collect=({cfg.collect_workers}w/{cfg.collect_window_s}s) "
            f"eval=({cfg.eval_spec} · {cfg.eval_workers}w · lqr) "
            f"guard=(overall-{cfg.stop_on_overall_drop}/easy-{cfg.stop_on_easy_drop})"
        )

        def _finish(code: int, state: str, reason: Optional[str] = None) -> int:
            status["status"] = state
            status["stop_reason"] = reason
            status["current_step"] = "done"
            _write_status(status_path, status)
            if reason:
                self._log(f"{state}: {reason}")
            self._log(f"exit code={code} status={state} status_file={status_path}")
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            return int(code)

        init_ckpt = _abs(cfg.init_ckpt)
        if not cfg.init_ckpt or not init_ckpt.is_file():
            return _finish(1, "failed", f"起点权重不存在：{cfg.init_ckpt!r}（--ckpt 或 config stages.B.phase3.init_ckpt）")
        if not _abs(cfg.spec_pool).is_file():
            return _finish(1, "failed", f"采集池不存在：{cfg.spec_pool!r}")
        if not _abs(cfg.eval_spec).is_file():
            return _finish(1, "failed", f"评测 spec 不存在：{cfg.eval_spec!r}")

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

            # ---- ① 采集 ----
            status["current_step"] = "collect"
            _write_status(status_path, status)
            dagger_dir = datasets_root / f"BTC{stamp}_phase3_dagger_r{round_index}"
            collect_log = log_dir / f"round{round_index}_collect.log"
            self._log(f"r{round_index} collect start student={student} out={dagger_dir}")
            started = time.perf_counter()
            rc = int(self._collect_fn(self._collect_argv(student=student, dagger_dir=dagger_dir, round_index=round_index), collect_log))
            duration = time.perf_counter() - started
            rows = None
            meta = _read_json(dagger_dir / "expert_bc.meta.json")
            if meta is not None:
                rows = meta.get("count")
            round_entry["collect"] = {
                "rc": rc, "duration_s": round(duration, 1), "out": str(dagger_dir),
                "rows": rows, "log": str(collect_log),
            }
            if rc != 0:
                _write_status(status_path, status)
                return _finish(1, "failed", f"r{round_index} 采集失败 rc={rc}（log={collect_log}）")
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

            # ---- ② 训练（每轮起点恒为 init_ckpt）----
            status["current_step"] = "train"
            _write_status(status_path, status)
            train_out = runs_root / f"BTC{stamp}_stageB_phase3_r{round_index}" / "stage_b"
            train_log = log_dir / f"round{round_index}_train.log"
            self._log(f"r{round_index} train start out={train_out} ckpt={init_ckpt}")
            started = time.perf_counter()
            rc = int(self._train_fn(self._train_argv(dagger_dir=dagger_dir, train_out=train_out, round_index=round_index), train_log))
            duration = time.perf_counter() - started
            final_ckpt = train_out / "final.pt"
            train_metrics = _read_json(train_out / "metrics.json")
            losses = None
            if train_metrics is not None:
                losses = {
                    key: train_metrics.get(key)
                    for key in ("bc_loss", "bc_od_loss", "bc_ld_loss", "bc_action_chain_loss")
                    if train_metrics.get(key) is not None
                }
            round_entry["train"] = {
                "rc": rc, "duration_s": round(duration, 1), "out": str(train_out),
                "final": (str(final_ckpt) if final_ckpt.is_file() else None),
                "metrics": (str(train_out / "metrics.json") if train_metrics is not None else None),
                "losses": losses, "log": str(train_log),
            }
            if rc != 0:
                _write_status(status_path, status)
                return _finish(1, "failed", f"r{round_index} 训练失败 rc={rc}（log={train_log}）")
            if not final_ckpt.is_file():
                _write_status(status_path, status)
                return _finish(1, "failed", f"r{round_index} 训练产物缺失：{final_ckpt}")
            round_entry["student_out"] = str(final_ckpt)
            self._log(f"r{round_index} train ok final={final_ckpt} ({duration:.0f}s)")

            # ---- ③ 评测 ----
            status["current_step"] = "eval"
            _write_status(status_path, status)
            eval_dir = runs_root / f"BTC{stamp}_eval500_phase3_r{round_index}"
            eval_log = log_dir / f"round{round_index}_eval.log"
            self._log(f"r{round_index} eval start ckpt={final_ckpt} dir={eval_dir}")
            started = time.perf_counter()
            rc = int(self._eval_fn(self._eval_argv(final_ckpt=final_ckpt, eval_dir=eval_dir), eval_log))
            duration = time.perf_counter() - started
            eval_metrics_path = eval_dir / "metrics.json"
            summary = _eval_summary(eval_metrics_path)
            round_entry["eval"] = {
                "rc": rc, "duration_s": round(duration, 1), "dir": str(eval_dir),
                "metrics": (str(eval_metrics_path) if eval_metrics_path.is_file() else None),
                "overall_success": summary.get("overall_success"),
                "easy_success": summary.get("easy_success"),
                "log": str(eval_log),
            }
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

            # ---- ④ 护栏（与上一轮比较）----
            status["current_step"] = "guard"
            decision = guard_decision(
                previous_eval,
                summary,
                stop_on_overall_drop=cfg.stop_on_overall_drop,
                stop_on_easy_drop=cfg.stop_on_easy_drop,
            )
            round_entry["guard"] = decision
            if decision["checked"]:
                self._log(
                    f"r{round_index} guard overall_delta={decision['overall_delta']} "
                    f"easy_delta={decision['easy_delta']} stopped={decision['stopped']}"
                )
            else:
                self._log(f"r{round_index} guard 基线（无上一轮，不判定）")
            previous_eval = dict(summary)
            student = str(final_ckpt)
            _write_status(status_path, status)
            if decision["stopped"]:
                return _finish(0, "stopped_guard", str(decision["reason"]))

        return _finish(0, "completed")


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
        return int(Phase3Loop(cfg).run())
    except KeyboardInterrupt:  # pragma: no cover - 交互中断
        print("[phase3] interrupted", file=sys.stderr)
        return 130


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
