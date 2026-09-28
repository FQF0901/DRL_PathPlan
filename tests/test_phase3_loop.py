"""phase 3 全自动循环编排（lane P3-C）回归。

覆盖（用户定稿契约）：

① 轮转与跨轮 student 传递：r{k} 采集用 r{k-1} 的 ``stage_b/final.pt``（r1 = init_ckpt），
   训练每轮起点恒为 ``init_ckpt``；产物命名 ``BTC<stamp>_{phase3_dagger,stageB_phase3,eval500_phase3}_r{k}``；
② 护栏：与上一轮比较 ``overall_success`` / ``easy`` success 绝对下降超阈 → 记录原因并停止；
③ 错误语义：任一步非零退出/产物缺失 → 立即停止（绝不带缺产物进下一轮）；
④ 状态文件字段（``phase3_status.json``：schema/status/rounds/逐轮 collect/train/eval/guard）；
⑤ 配置决议顺序 CLI > 环境变量 ``PHASE3_ROUNDS`` > config；保留键不得透传覆盖；
⑥ ``tools/train.py --phase3-loop`` 入口分派；``tools/train.sh`` 的 ``PHASE3_ONLY=1``（跳过 A/B）
   与 ``PHASE3=1``（A→B→循环）链路（fake venv-python 骨架，不启动真实采集/训练/评测）。

全部桩化：不启动真实采集/训练/评测；shell 测试只跑 fake 骨架，产物落 tmp_path。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline import phase3_loop
from pipeline.phase3_loop import Phase3Loop, Phase3LoopConfig, _parse_args, guard_decision

REPO_ROOT = Path(__file__).resolve().parents[1]


def _arg(argv, key: str) -> str:
    items = [str(item) for item in argv]
    return items[items.index(key) + 1]


def _write_pool_and_ckpt(tmp_path: Path) -> "tuple[Path, Path, Path]":
    tmp_path.mkdir(parents=True, exist_ok=True)
    init_ckpt = tmp_path / "init_phase2.pt"
    init_ckpt.write_bytes(b"fake-phase2-final")
    pool = tmp_path / "pool.json"
    pool.write_text('{"specs": []}', encoding="utf-8")
    eval_spec = tmp_path / "eval.json"
    eval_spec.write_text('{"specs": []}', encoding="utf-8")
    return init_ckpt, pool, eval_spec


class _Stubs:
    """采集/训练/评测桩：记录 argv 并生成可通过校验的产物；可注入失败/缺产物。"""

    def __init__(
        self,
        *,
        overall=(0.50, 0.50, 0.50),
        easy=(0.80, 0.80, 0.80),
        collect_rc: int = 0,
        train_rc: int = 0,
        eval_rc: int = 0,
        collect_rows: int = 42,
        collect_missing: bool = False,
        train_missing: bool = False,
        eval_missing: bool = False,
        train_plan: "list[str] | None" = None,
    ):
        self.overall = list(overall)
        self.easy = list(easy)
        self.collect_rc = int(collect_rc)
        self.train_rc = int(train_rc)
        self.eval_rc = int(eval_rc)
        self.collect_rows = int(collect_rows)
        self.collect_missing = bool(collect_missing)
        self.train_missing = bool(train_missing)
        self.eval_missing = bool(eval_missing)
        #: 逐次训练尝试的行为计划（"ok"/"oom"/"oom_log"/"rc1"/"exc:<Type>:<msg>"；不足则重复末项）
        self.train_plan = list(train_plan) if train_plan else None
        self.collect_calls: list = []
        self.train_calls: list = []
        self.eval_calls: list = []

    # ---- 步函数 ----
    def collect(self, argv, log_path) -> int:
        out = Path(_arg(argv, "--out"))
        self.collect_calls.append(
            {
                "student": _arg(argv, "--ckpt"),
                "out": out,
                "seed": int(_arg(argv, "--shuffle-seed")),
                "specs": _arg(argv, "--specs"),
                "target_fails": int(_arg(argv, "--target-fails")),
                "window": float(_arg(argv, "--window-fail-before")),
                "workers": int(_arg(argv, "--workers")),
                "log": Path(log_path) if log_path else None,
            }
        )
        if self.collect_rc == 0 and not self.collect_missing:
            out.mkdir(parents=True, exist_ok=True)
            np.savez(out / "expert_bc.npz", dummy=np.zeros(1))
            (out / "expert_bc.meta.json").write_text(
                json.dumps({"count": self.collect_rows}), encoding="utf-8"
            )
        return self.collect_rc

    def train(self, argv, log_path) -> int:
        out = Path(_arg(argv, "--out"))
        micro = int(_arg(argv, "--micro-batch-size")) if "--micro-batch-size" in [str(x) for x in argv] else None
        self.train_calls.append(
            {
                "dagger": _arg(argv, "--phase3"),
                "round": int(_arg(argv, "--phase3-round")),
                "ckpt": _arg(argv, "--ckpt"),
                "out": out,
                "micro": micro,
                "log": Path(log_path) if log_path else None,
            }
        )
        index = len(self.train_calls) - 1
        action = "ok" if not self.train_plan else self.train_plan[min(index, len(self.train_plan) - 1)]
        if action == "oom":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        if action == "oom_log":
            if log_path is not None:
                path = Path(log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write("RuntimeError: CUDA out of memory. Tried to allocate 1.00 GiB\n")
            return 1
        if action == "oom_log_alt":
            if log_path is not None:
                path = Path(log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write("RuntimeError: CUDA error: out of memory (GPU memory exhausted)\n")
            return 1
        if action.startswith("exc:"):
            _, name, message = action.split(":", 2)
            raise {"RuntimeError": RuntimeError, "ValueError": ValueError}[name](message)
        if action == "rc1":
            return 1
        if self.train_rc == 0 and not self.train_missing:
            out.mkdir(parents=True, exist_ok=True)
            (out / "final.pt").write_bytes(b"final")
            (out / "metrics.json").write_text(json.dumps({"bc_loss": 1.25}), encoding="utf-8")
        return self.train_rc

    def eval(self, argv, log_path) -> int:
        out = Path(_arg(argv, "--out"))
        name = _arg(argv, "--name")
        run_dir = out / name
        index = len(self.eval_calls)
        self.eval_calls.append(
            {
                "ckpt": _arg(argv, "--ckpt"),
                "dir": run_dir,
                "workers": int(_arg(argv, "--workers")),
                "tracker": _arg(argv, "--tracker"),
                "spec": _arg(argv, "--spec"),
                "log": Path(log_path) if log_path else None,
            }
        )
        if self.eval_rc == 0 and not self.eval_missing:
            run_dir.mkdir(parents=True, exist_ok=True)
            overall = self.overall[min(index, len(self.overall) - 1)]
            easy = self.easy[min(index, len(self.easy) - 1)]
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "overall": {"success_rate": overall},
                        "by_difficulty": {"easy": {"success_rate": easy}},
                    }
                ),
                encoding="utf-8",
            )
        return self.eval_rc


def _loop(tmp_path: Path, stubs: _Stubs, *, rounds: int = 3, **overrides) -> Phase3Loop:
    init_ckpt, pool, eval_spec = _write_pool_and_ckpt(tmp_path)
    cfg = Phase3LoopConfig(
        rounds=rounds,
        spec_pool=str(pool),
        eval_spec=str(eval_spec),
        init_ckpt=str(init_ckpt),
        runs_root=str(tmp_path / "runs"),
        datasets_root=str(tmp_path / "datasets"),
        stamp="TEST",
        fail_target=7,
        shuffle_seed_base=100,
        collect_workers=3,
        collect_window_s=5.0,
        eval_workers=2,
        **overrides,
    )
    return Phase3Loop(cfg, collect_fn=stubs.collect, train_fn=stubs.train, eval_fn=stubs.eval, logger=lambda _: None)


def _status(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "runs" / "BTCTEST_phase3_loop" / "phase3_status.json").read_text(encoding="utf-8"))


# ---------------------------------------- ① 轮转 + 跨轮 student 传递 + 命名
def test_phase3_loop_rotation_and_student_handoff(tmp_path: Path) -> None:
    stubs = _Stubs()
    rc = _loop(tmp_path, stubs, rounds=3).run()
    assert rc == 0
    assert len(stubs.collect_calls) == 3 and len(stubs.train_calls) == 3 and len(stubs.eval_calls) == 3

    init_ckpt = str(tmp_path / "init_phase2.pt")
    # 跨轮 student：r1 = init_ckpt，r{k} = r{k-1} 的 final.pt
    expected_students = [
        init_ckpt,
        str(tmp_path / "runs" / "BTCTEST_stageB_phase3_r1" / "stage_b" / "final.pt"),
        str(tmp_path / "runs" / "BTCTEST_stageB_phase3_r2" / "stage_b" / "final.pt"),
    ]
    assert [item["student"] for item in stubs.collect_calls] == expected_students
    # 训练每轮起点恒为 init_ckpt（P3-B 契约）
    assert [item["ckpt"] for item in stubs.train_calls] == [init_ckpt] * 3
    assert [item["dagger"] for item in stubs.train_calls] == [
        str(tmp_path / "datasets" / f"BTCTEST_phase3_dagger_r{k}") for k in (1, 2, 3)
    ]
    assert [str(item["out"]) for item in stubs.train_calls] == [
        str(tmp_path / "runs" / f"BTCTEST_stageB_phase3_r{k}" / "stage_b") for k in (1, 2, 3)
    ]
    # 评测用当轮 final.pt；命名/worker/tracker 口径
    assert [item["ckpt"] for item in stubs.eval_calls] == [
        str(tmp_path / "runs" / f"BTCTEST_stageB_phase3_r{k}" / "stage_b" / "final.pt") for k in (1, 2, 3)
    ]
    assert [item["dir"].name for item in stubs.eval_calls] == [
        f"BTCTEST_eval500_phase3_r{k}" for k in (1, 2, 3)
    ]
    assert all(item["workers"] == 2 and item["tracker"] == "lqr" for item in stubs.eval_calls)
    # 采集口径：shuffle seed = base + round；target-fails/window/workers 来自 config
    assert [item["seed"] for item in stubs.collect_calls] == [101, 102, 103]
    assert all(item["target_fails"] == 7 and item["window"] == 5.0 and item["workers"] == 3 for item in stubs.collect_calls)

    status = _status(tmp_path)
    assert status["status"] == "completed" and status["stop_reason"] is None
    assert status["rounds_total"] == 3 and len(status["rounds"]) == 3
    assert [item["guard"]["checked"] for item in status["rounds"]] == [False, True, True]
    assert all(item["collect"]["rows"] == 42 for item in status["rounds"])
    assert all(item["train"]["losses"]["bc_loss"] == 1.25 for item in status["rounds"])


# ------------------------------------------------ ② 护栏（overall / easy）
def test_phase3_loop_guard_stops_on_overall_drop(tmp_path: Path) -> None:
    stubs = _Stubs(overall=(0.50, 0.40, 0.50), easy=(0.8, 0.8, 0.8))
    rc = _loop(tmp_path, stubs, rounds=3).run()
    assert rc == 0, "护栏停止是预期停止（非错误退出）"
    assert len(stubs.collect_calls) == 2, "r2 超阈后不得再采集 r3"
    assert len(stubs.eval_calls) == 2
    status = _status(tmp_path)
    assert status["status"] == "stopped_guard"
    assert "overall_success" in str(status["stop_reason"])
    assert len(status["rounds"]) == 2
    assert status["rounds"][1]["guard"]["stopped"] is True
    assert status["rounds"][1]["guard"]["overall_delta"] == pytest.approx(0.10, abs=1e-6)


def test_phase3_loop_guard_stops_on_easy_drop(tmp_path: Path) -> None:
    stubs = _Stubs(overall=(0.50, 0.50, 0.50), easy=(0.80, 0.60, 0.80))
    rc = _loop(tmp_path, stubs, rounds=3).run()
    assert rc == 0
    assert len(stubs.collect_calls) == 2
    status = _status(tmp_path)
    assert status["status"] == "stopped_guard"
    assert "easy_success" in str(status["stop_reason"])
    assert status["rounds"][1]["guard"]["easy_delta"] == pytest.approx(0.20, abs=1e-6)


def test_guard_decision_missing_metrics_never_stops() -> None:
    decision = guard_decision(
        {"overall_success": None, "easy_success": float("nan")},
        {"overall_success": 0.1, "easy_success": 0.1},
        stop_on_overall_drop=0.0,
        stop_on_easy_drop=0.0,
    )
    assert decision["checked"] is True and decision["stopped"] is False
    assert decision["overall_delta"] is None and decision["easy_delta"] is None
    assert guard_decision(
        None, {"overall_success": 0.1}, stop_on_overall_drop=0.0, stop_on_easy_drop=0.0
    )["checked"] is False


# ------------------------------------------------------ ③ 错误语义
def test_phase3_loop_collect_failure_stops_immediately(tmp_path: Path) -> None:
    stubs = _Stubs(collect_rc=1)
    rc = _loop(tmp_path, stubs).run()
    assert rc == 1
    assert len(stubs.train_calls) == 0 and len(stubs.eval_calls) == 0
    status = _status(tmp_path)
    assert status["status"] == "failed" and "采集失败" in str(status["stop_reason"])


def test_phase3_loop_missing_artifacts_stop(tmp_path: Path) -> None:
    # 训练 rc=0 但 final.pt 缺失 → 立即失败，不进评测
    stubs = _Stubs(train_missing=True)
    rc = _loop(tmp_path, stubs).run()
    assert rc == 1 and len(stubs.eval_calls) == 0
    assert "训练产物缺失" in str(_status(tmp_path)["stop_reason"])

    # 评测 rc=0 但 metrics.json 缺失 → 失败
    stubs2 = _Stubs(eval_missing=True)
    rc2 = _loop(tmp_path / "second", stubs2).run()
    assert rc2 == 1
    assert "评测产物缺失" in str(_status(tmp_path / "second")["stop_reason"])


def test_phase3_loop_empty_collection_stops(tmp_path: Path) -> None:
    stubs = _Stubs(collect_rows=0)
    rc = _loop(tmp_path, stubs).run()
    assert rc == 1 and len(stubs.train_calls) == 0
    status = _status(tmp_path)
    assert status["status"] == "failed" and "0 行" in str(status["stop_reason"])


# ------------------------------------------------------ ④ 状态文件字段
def test_phase3_loop_status_file_fields(tmp_path: Path) -> None:
    stubs = _Stubs()
    assert _loop(tmp_path, stubs, rounds=1).run() == 0
    status = _status(tmp_path)
    for key in (
        "schema_version", "lane", "status", "stop_reason", "started_at", "updated_at",
        "rounds_total", "current_round", "current_step", "stamp", "loop_dir",
        "runs_root", "datasets_root", "init_ckpt", "config", "rounds",
    ):
        assert key in status, f"状态文件缺少字段 {key}"
    assert status["schema_version"] == 1 and status["lane"] == "P3-C"
    assert status["status"] == "completed" and status["current_step"] == "done"
    assert status["current_round"] == 1 and status["stamp"] == "TEST"
    assert status["config"]["collect"] == {"workers": 3, "window_s": 5.0}
    assert status["config"]["eval"] == {"spec": str(tmp_path / "eval.json"), "workers": 2}
    assert status["config"]["guard"] == {"stop_on_overall_drop": 0.05, "stop_on_easy_drop": 0.10}
    entry = status["rounds"][0]
    for key in ("round", "student_in", "student_out", "collect", "train", "eval", "guard"):
        assert key in entry, f"逐轮条目缺少字段 {key}"
    for step in ("collect", "train", "eval"):
        assert entry[step]["rc"] == 0 and "duration_s" in entry[step] and "log" in entry[step]
    assert entry["eval"]["overall_success"] == 0.50 and entry["eval"]["easy_success"] == 0.80
    # 原子写：无残留 tmp
    assert not (tmp_path / "runs" / "BTCTEST_phase3_loop" / "phase3_status.json.tmp").exists()


# --------------------------------------------- ⑤ 配置决议 + 保留键
def test_phase3_loop_config_resolution_order(tmp_path: Path, monkeypatch) -> None:
    from pipeline.stages import load_config

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "stages:\n"
        "  B:\n"
        "    phase3:\n"
        "      rounds: 5\n"
        "      spec_pool: env/specs/pool.json\n"
        "      fail_target: 123\n"
        "      shuffle_seed_base: 7\n"
        "      collect: {workers: 3, window_s: 5.0}\n"
        "      init_ckpt: runs/init.pt\n"
        "      eval: {spec: env/specs/eval.json, workers: 4}\n"
        "      guard: {stop_on_overall_drop: 0.01, stop_on_easy_drop: 0.02}\n",
        encoding="utf-8",
    )
    config = load_config(str(cfg_path))
    monkeypatch.delenv("PHASE3_ROUNDS", raising=False)
    args, extra = _parse_args(["--phase3-loop", "--config", str(cfg_path)])
    cfg = Phase3LoopConfig.from_config(config, args, extra=extra)
    assert cfg.rounds == 5, "config 兜底"
    assert cfg.fail_target == 123 and cfg.shuffle_seed_base == 7
    assert cfg.collect_workers == 3 and cfg.collect_window_s == 5.0
    assert cfg.init_ckpt == "runs/init.pt"
    assert cfg.eval_workers == 4 and cfg.stop_on_overall_drop == 0.01

    monkeypatch.setenv("PHASE3_ROUNDS", "3")
    cfg_env = Phase3LoopConfig.from_config(config, args, extra=extra)
    assert cfg_env.rounds == 3, "环境变量覆盖 config"
    args_cli, extra_cli = _parse_args(["--phase3-loop", "--config", str(cfg_path), "--phase3-rounds", "2"])
    cfg_cli = Phase3LoopConfig.from_config(config, args_cli, extra=extra_cli)
    assert cfg_cli.rounds == 2, "CLI 覆盖环境变量"

    # 透传：非保留键进 extra_train_args；保留键在 main() 被拒（不启动任何真实步骤）
    args_extra, extra_extra = _parse_args(["--phase3-loop", "--config", str(cfg_path), "--phase3-epochs", "2"])
    cfg_extra = Phase3LoopConfig.from_config(config, args_extra, extra=extra_extra)
    assert cfg_extra.extra_train_args == ["--phase3-epochs", "2"]
    rc = phase3_loop.main(["--phase3-loop", "--phase3", "x", "--config", str(cfg_path)])
    assert rc == 2, "保留键（--phase3）不得经透传覆盖编排接线"


# --------------------------------------------- ⑥ 入口分派 + 全链（Python）+ train.sh
def test_train_entry_dispatches_phase3_loop_and_chain(monkeypatch) -> None:
    from tools import train as train_entry

    loop_calls: list = []
    chain_calls: list = []
    monkeypatch.setattr(phase3_loop, "main", lambda argv: loop_calls.append(list(argv)) or 0)
    monkeypatch.setattr(phase3_loop, "run_chain", lambda argv: chain_calls.append(list(argv)) or 0)

    loop_argv = ["--phase3-loop", "--phase3-rounds", "2"]
    chain_argv = ["--phase3-chain", "--phase3-rounds", "3"]
    assert train_entry.main(loop_argv) == 0
    assert train_entry.main(chain_argv) == 0
    assert loop_calls == [loop_argv]
    assert chain_calls == [chain_argv]


def _chain_config(tmp_path: Path) -> Path:
    init_ckpt, pool, eval_spec = _write_pool_and_ckpt(tmp_path)
    cfg_path = tmp_path / "chain.yaml"
    cfg_path.write_text(
        "stages:\n"
        "  B:\n"
        "    phase3:\n"
        "      rounds: 5\n"
        f"      spec_pool: {pool}\n"
        f"      init_ckpt: {init_ckpt}\n"
        f"      eval: {{spec: {eval_spec}, workers: 2}}\n",
        encoding="utf-8",
    )
    return cfg_path


def _stub_chain(monkeypatch, stage_results=None):
    """桩掉 A/B 阶段与循环执行；返回记录列表。"""
    calls: list = []
    results = dict(stage_results or {})

    def _fake_stage(stage, **kwargs):
        calls.append(("stage", stage))
        return int(results.get(stage, 0))

    def _fake_run(self):
        calls.append(("loop", int(self.cfg.rounds)))
        return 0

    monkeypatch.setattr(phase3_loop, "_run_train_stage", _fake_stage)
    monkeypatch.setattr(phase3_loop.Phase3Loop, "run", _fake_run)
    return calls


def test_phase3_chain_orders_ab_then_loop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PHASE3_ONLY", raising=False)
    calls = _stub_chain(monkeypatch)
    cfg_path = _chain_config(tmp_path)
    rc = phase3_loop.run_chain(["--phase3-chain", "--config", str(cfg_path), "--phase3-rounds", "2"])
    assert rc == 0
    assert calls == [("stage", "A"), ("stage", "B"), ("loop", 2)], "全链必须 A→B→循环"


def test_phase3_chain_only_skips_ab(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PHASE3_ONLY", "1")
    calls = _stub_chain(monkeypatch)
    cfg_path = _chain_config(tmp_path)
    rc = phase3_loop.run_chain(["--phase3-chain", "--config", str(cfg_path)])
    assert rc == 0
    assert calls == [("loop", 5)], "PHASE3_ONLY=1 必须跳过 A/B（轮数取 config）"

    monkeypatch.delenv("PHASE3_ONLY", raising=False)
    calls_flag = _stub_chain(monkeypatch)
    rc_flag = phase3_loop.run_chain(["--phase3-chain", "--phase3-only", "--config", str(cfg_path)])
    assert rc_flag == 0 and calls_flag == [("loop", 5)], "--phase3-only 等价"


def test_phase3_chain_stops_on_stage_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PHASE3_ONLY", raising=False)
    cfg_path = _chain_config(tmp_path)

    calls_a = _stub_chain(monkeypatch, {"A": 3})
    rc_a = phase3_loop.run_chain(["--phase3-chain", "--config", str(cfg_path)])
    assert rc_a == 3 and calls_a == [("stage", "A")], "A 失败不得进入 B/循环"

    calls_b = _stub_chain(monkeypatch, {"B": 4})
    rc_b = phase3_loop.run_chain(["--phase3-chain", "--config", str(cfg_path)])
    assert rc_b == 4 and calls_b == [("stage", "A"), ("stage", "B")], "B 失败不得进入循环"


def test_phase3_chain_validates_init_ckpt_before_ab(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PHASE3_ONLY", raising=False)
    calls = _stub_chain(monkeypatch)
    cfg_path = _chain_config(tmp_path)
    missing = tmp_path / "missing_init.pt"
    rc = phase3_loop.run_chain(["--phase3-chain", "--config", str(cfg_path), "--ckpt", str(missing)])
    assert rc == 2 and calls == [], "起点缺失必须前置失败（不带病跑 A/B）"


def _fake_repo(tmp_path: Path, calls_path: Path) -> Path:
    """fake 仓库骨架：train.sh + fake venv-python（记录调用、模拟 run_config 输出）。"""
    root = tmp_path / "fake_repo"
    (root / "tools").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "tools" / "train.sh", root / "tools" / "train.sh")
    fake = root / "tools" / "venv-python"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$@" >> "${FAKE_CALLS:?FAKE_CALLS not set}"\n'
        'if [ "$1" = "tools/run_config.py" ]; then\n'
        '  stage="A"; prev=""\n'
        '  for arg in "$@"; do\n'
        '    if [ "$prev" = "--stage" ]; then stage="$arg"; fi\n'
        '    prev="$arg"\n'
        '  done\n'
        '  lower=$(printf \'%s\' "$stage" | tr \'A-Z\' \'a-z\')\n'
        '  echo "BC_DIR=runs/fake_bc"\n'
        '  echo "WORK_DIR=runs/BTC_TEST_train"\n'
        '  echo "STAGE=$stage"\n'
        '  echo "STAGE_DIR=runs/BTC_TEST_train/stage_$lower"\n'
        '  echo "LOG=runs/BTC_TEST_train/logs/stage_$lower.log"\n'
        '  echo "DETACH_PID=runs/BTC_TEST_train/detach.pid"\n'
        '  echo "WM_EPOCHS=1"\n'
        '  echo "BC_EPOCHS=1"\n'
        '  echo "CKPT_EVERY=5"\n'
        '  echo "BATCH_SIZE=8"\n'
        '  echo "MICRO_BATCH_SIZE=4"\n'
        '  echo "TRAJ_AUX_WEIGHT=0.1"\n'
        '  echo "CKPT="\n'
        '  echo "RESUME="\n'
        'fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return root


def _run_train_sh(root: Path, calls: Path, **env_overrides) -> "tuple[int, str]":
    env = {key: value for key, value in os.environ.items() if key not in ("STAGE", "RESUME", "CKPT", "WORK_DIR", "BC_DIR")}
    env.update({"FAKE_CALLS": str(calls), **env_overrides})
    proc = subprocess.run(
        ["bash", "tools/train.sh"], cwd=str(root), env=env, capture_output=True, text=True, timeout=30
    )
    return proc.returncode, proc.stdout + proc.stderr


def _wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(shutil.which("setsid") is None, reason="需要 setsid（train.sh 后台化）")
def test_run_train_stage_resolves_run_config(tmp_path: Path, monkeypatch) -> None:
    """`_run_train_stage`：run_config 解析 → tools/train.py argv（与 shell 旧口径一致）。"""
    bc_dir = tmp_path / "ds"
    bc_dir.mkdir()
    (bc_dir / "expert_bc.npz").write_bytes(b"")
    cfg_path = tmp_path / "chain.yaml"
    cfg_path.write_text(
        "run: {work_dir: auto, name: train}\n"
        "stages:\n"
        "  A: {epochs: 3, batch: 8, micro: 4}\n"
        "  B: {epochs: 5, batch: 8, micro: 4}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("BC_DIR", str(bc_dir))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("RESUME", "")  # 显式关闭 resume auto（隔离真实 runs/ 状态）
    captured: dict = {}

    def _fake_run_subprocess(argv, log_path=None):
        captured["argv"] = [str(item) for item in argv]
        captured["log"] = Path(log_path)
        return 0

    monkeypatch.setattr(phase3_loop, "run_subprocess", _fake_run_subprocess)
    rc = phase3_loop._run_train_stage(
        "A", config=str(cfg_path), device="cpu", limit=7, extra=["--phase3-epochs", "2"]
    )
    assert rc == 0
    argv = captured["argv"]
    assert argv[:2] == [phase3_loop.VENV_PY, "tools/train.py"]
    assert argv[argv.index("--stage") + 1] == "A"
    assert argv[argv.index("--bc-dir") + 1] == str(bc_dir)
    assert argv[argv.index("--out") + 1] == str(tmp_path / "work" / "stage_a")
    assert argv[argv.index("--wm-epochs") + 1] == "3"
    assert argv[argv.index("--batch-size") + 1] == "8" and argv[argv.index("--micro-batch-size") + 1] == "4"
    assert argv[argv.index("--device") + 1] == "cpu" and argv[argv.index("--limit-dataset") + 1] == "7"
    assert argv[-2:] == ["--phase3-epochs", "2"]
    assert captured["log"] == tmp_path / "work" / "logs" / "stage_a.log"
    assert captured["log"].parent.is_dir()


def test_train_sh_phase3_only_dispatches_chain(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path, tmp_path / "calls.txt")
    calls = tmp_path / "calls.txt"
    rc, out = _run_train_sh(root, calls, PHASE3_ONLY="1")
    assert rc == 0, out
    chain_log = root / "runs" / "BTC_TEST_train" / "logs" / "phase3_chain.log"
    assert _wait_for(lambda: chain_log.exists() and "[exit]" in chain_log.read_text(encoding="utf-8"))
    lines = [line for line in calls.read_text(encoding="utf-8").splitlines() if line.startswith("tools/train.py")]
    assert len(lines) == 1 and lines[0].startswith("tools/train.py --phase3-chain"), lines
    assert "--stage" not in lines[0], "PHASE3_ONLY 不得走单阶段入口（A/B 跳过由 Python 全链模式保证）"
    chain = chain_log.read_text(encoding="utf-8")
    assert "[detach]" in chain and "chain=phase3" in chain and "only=1" in chain
    assert (root / "runs" / "BTC_TEST_train" / "detach.pid").read_text(encoding="utf-8").strip().isdigit()


@pytest.mark.skipif(shutil.which("setsid") is None, reason="需要 setsid（train.sh 后台化）")
def test_train_sh_full_chain_dispatches_chain_with_rounds(tmp_path: Path) -> None:
    root = _fake_repo(tmp_path, tmp_path / "calls.txt")
    calls = tmp_path / "calls.txt"
    rc, out = _run_train_sh(root, calls, PHASE3="1", PHASE3_ROUNDS="2")
    assert rc == 0, out
    chain_log = root / "runs" / "BTC_TEST_train" / "logs" / "phase3_chain.log"
    assert _wait_for(lambda: chain_log.exists() and "[exit]" in chain_log.read_text(encoding="utf-8"))
    lines = [line for line in calls.read_text(encoding="utf-8").splitlines() if line.startswith("tools/train.py")]
    assert len(lines) == 1 and lines[0].startswith("tools/train.py --phase3-chain"), lines
    assert "--phase3-rounds 2" in lines[0], "PHASE3_ROUNDS 必须透传给全链"
    chain = chain_log.read_text(encoding="utf-8")
    assert "[detach]" in chain and "only=0" in chain and "[exit]" in chain


# ---------------------- ⑦ lane P3-D：OOM 自愈 + 断点复用 + 失败落盘
def _record_status_writes(monkeypatch) -> list:
    """记录每次 `_write_status` 的快照（用于观察重试中的 current_step 等瞬态）。"""
    snapshots: list = []
    real_write = phase3_loop._write_status

    def _record(path, payload):
        snapshots.append(json.loads(json.dumps(payload, default=str)))
        return real_write(path, payload)

    monkeypatch.setattr(phase3_loop, "_write_status", _record)
    return snapshots


def test_phase3_loop_oom_halves_micro_and_retries(tmp_path: Path, monkeypatch) -> None:
    stubs = _Stubs(train_plan=["oom", "ok"])
    freed: list = []
    monkeypatch.setattr(phase3_loop, "_free_cuda_memory", lambda: freed.append(True))
    snapshots = _record_status_writes(monkeypatch)
    rc = _loop(tmp_path, stubs, rounds=1, micro=512).run()
    assert rc == 0
    assert [item["micro"] for item in stubs.train_calls] == [512, 256], "OOM 后 micro 必须减半重试"
    assert len(freed) == 1, "重试前必须 empty_cache+gc（经 _free_cuda_memory）"
    status = _status(tmp_path)
    assert status["status"] == "completed"
    train = status["rounds"][0]["train"]
    assert train["rc"] == 0 and train["micro"] == 256 and train["final"]
    assert [item["micro"] for item in train["attempts"]] == [512, 256]
    assert [item["oom"] for item in train["attempts"]] == [True, False]
    retry_steps = [snap.get("current_step") for snap in snapshots if str(snap.get("current_step", "")).startswith("train_retry")]
    assert retry_steps == ["train_retry_micro256"], "重试期间 current_step 必须标记 train_retry_micro{M}"
    assert any(
        snap.get("status") == "running" and snap.get("current_step") == "train_retry_micro256"
        for snap in snapshots
    ), "重试必须落盘（running 状态可审计）"


def test_phase3_loop_oom_retries_exhausted_fails(tmp_path: Path, monkeypatch) -> None:
    stubs = _Stubs(train_plan=["oom"])
    monkeypatch.setattr(phase3_loop, "_free_cuda_memory", lambda: None)
    rc = _loop(tmp_path, stubs, rounds=1, micro=512).run()
    assert rc == 1 and len(stubs.eval_calls) == 0
    assert [item["micro"] for item in stubs.train_calls] == [512, 256, 128, 64], "512→256→128→64 后耗尽"
    status = _status(tmp_path)
    assert status["status"] == "failed"
    assert "OOM 重试耗尽" in str(status["stop_reason"])
    assert status["current_step"] == "train", "失败状态保留失败所在步（不再停在 running）"
    assert [item["micro"] for item in status["rounds"][0]["train"]["attempts"]] == [512, 256, 128, 64]


def test_phase3_loop_oom_detected_from_log_and_rc(tmp_path: Path, monkeypatch) -> None:
    stubs = _Stubs(train_plan=["oom_log", "ok"])
    monkeypatch.setattr(phase3_loop, "_free_cuda_memory", lambda: None)
    rc = _loop(tmp_path, stubs, rounds=1, micro=256).run()
    assert rc == 0
    assert [item["micro"] for item in stubs.train_calls] == [256, 128], "非零 rc + 日志含 OOM 也必须触发减半重试"
    status = _status(tmp_path)
    assert status["status"] == "completed" and status["rounds"][0]["train"]["micro"] == 128

    # 变体消息（"CUDA error: out of memory"）同样识别
    second = tmp_path / "second"
    stubs2 = _Stubs(train_plan=["oom_log_alt", "ok"])
    rc2 = _loop(second, stubs2, rounds=1, micro=256).run()
    assert rc2 == 0 and [item["micro"] for item in stubs2.train_calls] == [256, 128]


def test_phase3_loop_resume_reuses_collect(tmp_path: Path) -> None:
    stubs = _Stubs()
    reused = tmp_path / "datasets" / "BTC20260901-0000_phase3_dagger_r1"
    reused.mkdir(parents=True)
    (reused / "expert_bc.npz").write_bytes(b"npz")
    (reused / "expert_bc.meta.json").write_text(json.dumps({"count": 321}), encoding="utf-8")
    (reused / "report.json").write_text(
        json.dumps({"counts": {"stored_rows": 321}}), encoding="utf-8"
    )
    rc = _loop(tmp_path, stubs, rounds=1, resume=True).run()
    assert rc == 0
    assert len(stubs.collect_calls) == 0, "resume 下已有可用采集产物必须跳过"
    assert Path(stubs.train_calls[0]["dagger"]) == reused, "训练必须吃复用目录"
    status = _status(tmp_path)
    collect = status["rounds"][0]["collect"]
    assert collect["reused"] == str(reused) and collect["rows"] == 321 and collect["rc"] is None
    assert "collect reuse=" in (tmp_path / "runs" / "BTCTEST_phase3_loop" / "logs" / "phase3_loop.log").read_text(
        encoding="utf-8"
    )


def test_phase3_loop_resume_reuses_train_and_eval(tmp_path: Path) -> None:
    stubs = _Stubs()
    runs = tmp_path / "runs"
    train_dir = runs / "BTC20260901-0000_stageB_phase3_r1" / "stage_b"
    train_dir.mkdir(parents=True)
    final = train_dir / "final.pt"
    final.write_bytes(b"reused-final")
    (train_dir / "metrics.json").write_text(json.dumps({"bc_loss": 0.9}), encoding="utf-8")
    eval_dir = runs / "BTC20260901-0000_eval500_phase3_r1"
    eval_dir.mkdir(parents=True)
    (eval_dir / "metrics.json").write_text(
        json.dumps({"overall": {"success_rate": 0.61}, "by_difficulty": {"easy": {"success_rate": 0.7}}}),
        encoding="utf-8",
    )
    rc = _loop(tmp_path, stubs, rounds=1, resume=True).run()
    assert rc == 0
    assert len(stubs.collect_calls) == 1, "采集无产物 → 照常跑"
    assert len(stubs.train_calls) == 0 and len(stubs.eval_calls) == 0, "训练/评测已有产物必须跳过"
    entry = _status(tmp_path)["rounds"][0]
    assert entry["train"]["reused"] == str(final)
    assert entry["train"]["final"] == str(final) and entry["train"]["losses"] == {"bc_loss": 0.9}
    assert entry["student_out"] == str(final)
    assert entry["eval"]["reused"] == str(eval_dir / "metrics.json")
    assert entry["eval"]["overall_success"] == 0.61 and entry["eval"]["easy_success"] == 0.7


def test_phase3_loop_resume_invalid_artifacts_not_reused(tmp_path: Path) -> None:
    stubs = _Stubs()
    datasets = tmp_path / "datasets"
    # 采集：rows=0 与缺 report.json 都不可复用
    zero_dir = datasets / "BTC20260901-0000_phase3_dagger_r1"
    zero_dir.mkdir(parents=True)
    (zero_dir / "expert_bc.npz").write_bytes(b"npz")
    (zero_dir / "report.json").write_text(json.dumps({"counts": {"stored_rows": 0}}), encoding="utf-8")
    no_report = datasets / "BTC20260902-0000_phase3_dagger_r1"
    no_report.mkdir(parents=True)
    (no_report / "expert_bc.npz").write_bytes(b"npz")
    # 训练：有 final.pt 但缺 metrics.json
    train_dir = tmp_path / "runs" / "BTC20260901-0000_stageB_phase3_r1" / "stage_b"
    train_dir.mkdir(parents=True)
    (train_dir / "final.pt").write_bytes(b"stale")
    # 评测：目录在但缺 metrics.json
    (tmp_path / "runs" / "BTC20260901-0000_eval500_phase3_r1").mkdir(parents=True)

    rc = _loop(tmp_path, stubs, rounds=1, resume=True).run()
    assert rc == 0
    assert len(stubs.collect_calls) == 1, "rows=0/缺 report 不得复用"
    assert len(stubs.train_calls) == 1, "缺 metrics.json 的训练产物不得复用"
    assert len(stubs.eval_calls) == 1, "缺 metrics.json 的评测产物不得复用"
    entry = _status(tmp_path)["rounds"][0]
    for step in ("collect", "train", "eval"):
        assert "reused" not in entry[step], f"{step} 不得标记复用"


def test_phase3_loop_exception_persists_failed_status(tmp_path: Path) -> None:
    def _cfg(root: Path) -> Phase3LoopConfig:
        init_ckpt, pool, eval_spec = _write_pool_and_ckpt(root)
        return Phase3LoopConfig(
            rounds=1, spec_pool=str(pool), eval_spec=str(eval_spec), init_ckpt=str(init_ckpt),
            runs_root=str(root / "runs"), datasets_root=str(root / "datasets"), stamp="TEST",
        )

    def _boom_collect(argv, log_path):
        raise RuntimeError("boom-collect")

    loop = Phase3Loop(
        _cfg(tmp_path), collect_fn=_boom_collect,
        train_fn=lambda argv, log: 0, eval_fn=lambda argv, log: 0, logger=lambda _: None,
    )
    assert loop.run() == 1
    status = _status(tmp_path)
    assert status["status"] == "failed"
    assert "RuntimeError" in str(status["stop_reason"]) and "boom-collect" in str(status["stop_reason"])
    assert status["current_step"] == "collect"

    def _boom_train(argv, log_path):
        raise ValueError("boom-train")

    second = tmp_path / "second"
    stubs2 = _Stubs()
    loop2 = Phase3Loop(
        _cfg(second),
        collect_fn=stubs2.collect, train_fn=_boom_train, eval_fn=stubs2.eval, logger=lambda _: None,
    )
    assert loop2.run() == 1
    status2 = _status(second)
    assert status2["status"] == "failed" and status2["current_step"] == "train"
    assert "ValueError: boom-train" in str(status2["stop_reason"])
    log_text = (second / "runs" / "BTCTEST_phase3_loop" / "logs" / "phase3_loop.log").read_text(encoding="utf-8")
    assert "Traceback" in log_text, "异常必须带 traceback 落日志（便于审计）"


def test_phase3_loop_config_micro_and_resume(tmp_path: Path, monkeypatch) -> None:
    from pipeline.stages import load_config

    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        "stages:\n  B:\n    phase3:\n      rounds: 1\n      micro: 256\n", encoding="utf-8"
    )
    config = load_config(str(cfg_path))
    monkeypatch.delenv("PHASE3_RESUME", raising=False)
    args, extra = _parse_args(["--phase3-loop", "--config", str(cfg_path)])
    cfg = Phase3LoopConfig.from_config(config, args, extra=extra)
    assert cfg.micro == 256 and cfg.resume is False, "micro 取 config；resume 默认关"
    monkeypatch.setenv("PHASE3_RESUME", "1")
    assert Phase3LoopConfig.from_config(config, args, extra=extra).resume is True, "PHASE3_RESUME=1 生效"
    args_flag, extra_flag = _parse_args(["--phase3-loop", "--config", str(cfg_path), "--phase3-resume"])
    assert Phase3LoopConfig.from_config(config, args_flag, extra=extra_flag).resume is True
    # 仓库配置回归：phase3 micro = 256（P3-D 防 OOM 定案；stage A 标定上限）
    repo_cfg = load_config("config/default.yaml")
    assert repo_cfg["stages"]["B"]["phase3"]["micro"] == 256

