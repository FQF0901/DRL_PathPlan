"""P0-A 评测安全前置：ckpt 加载 fail-fast / ``--allow-partial-load`` / 记录字段。

锁定（``docs/p0_eval_compat_design.md`` §2，CPU-only，不建 env）：

- 干净加载：``_load_ckpt_model`` 三计数 = 0 通过，摘要落 ``_CKPT_LOAD_SUMMARY``；
- 部分加载（缺 1 键 + 形状 +1）：默认 fail-fast（``SystemExit(2)``，打印三计数与明细，
  不缓存模型/摘要）；
- ``--allow-partial-load``：加载通过且摘要含三计数与 missing 列表；经 ``main`` 端到端
  强制落 ``metrics.json::meta.ckpt_load``；
- 配置基线：``git show HEAD:config/model.yaml`` + ``pipeline.stages.load_config``
  （不受工作区未提交改动影响）。
"""

from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
import torch

from pipeline import eval_runner
from pipeline.eval_runner import (
    _CKPT_LOAD_SUMMARY,
    _CKPT_MODEL_CACHE,
    _load_ckpt_model,
    build_parser,
)
from pipeline.stages import build_model, load_config

_ROOT = Path(__file__).resolve().parents[1]


def _head_model_config(tmp_path: Path) -> Dict[str, Any]:
    """HEAD 版 ``config/model.yaml``（工作区可能已改）→ ``load_config`` 合并语义。"""
    proc = subprocess.run(
        ["git", "show", "HEAD:config/model.yaml"],
        cwd=str(_ROOT), capture_output=True, text=True, check=True,
    )
    path = tmp_path / "model_head.yaml"
    path.write_text(proc.stdout, encoding="utf-8")
    return load_config(str(path))


@pytest.fixture(scope="module")
def model_artifacts(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """一次构造模型 state_dict：干净 ckpt + 部分 ckpt（缺 1 键 + 1 键形状 +1）。

    同时产出一个以 **HEAD 版 model.yaml** 为 include 的临时主配置，供 ``main`` 端到端测试
    使用（工作区 ``config/model.yaml`` 可能已按排摸臂改动，不能作为本测试基线）。
    """
    tmp = tmp_path_factory.mktemp("p0a_ckpt")
    config = _head_model_config(tmp)
    state = build_model(config).state_dict()

    clean_path = tmp / "clean.pt"
    torch.save(state, clean_path)

    partial = dict(state)
    missing_key = str(next(iter(partial)))
    del partial[missing_key]
    mismatch_key = str(next(
        key for key, value in partial.items()
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.numel() > 1
    ))
    partial[mismatch_key] = torch.cat([partial[mismatch_key], partial[mismatch_key][:1]], dim=0)
    partial_path = tmp / "partial.pt"
    torch.save(partial, partial_path)

    main_config = tmp / "default_head.yaml"
    main_config.write_text(
        "includes:\n" + "".join(
            f"  - {item}\n" for item in (
                _ROOT / "config" / "env.yaml",
                tmp / "model_head.yaml",
                _ROOT / "config" / "train.yaml",
                _ROOT / "config" / "eval.yaml",
            )
        ),
        encoding="utf-8",
    )

    return {
        "config": config,
        "clean": str(clean_path),
        "partial": str(partial_path),
        "missing_key": missing_key,
        "mismatch_key": mismatch_key,
        "main_config": str(main_config),
    }


@pytest.fixture(autouse=True)
def _clear_loader_caches() -> Iterator[None]:
    """每个测试独立的 ``(ckpt, device)`` 缓存/摘要（避免跨测试命中）。"""
    _CKPT_MODEL_CACHE.clear()
    _CKPT_LOAD_SUMMARY.clear()
    yield
    _CKPT_MODEL_CACHE.clear()
    _CKPT_LOAD_SUMMARY.clear()


# --------------------------------------------------------------------------- #
# (a) 干净加载
# --------------------------------------------------------------------------- #

def test_clean_load_passes_with_zero_counts(model_artifacts: Dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    model = _load_ckpt_model(model_artifacts["clean"], model_artifacts["config"], "cpu")
    assert isinstance(model, torch.nn.Module)

    out = capsys.readouterr().out
    assert "ckpt 载入" in out  # 原日志行保留
    assert "missing=0 shape_mismatch=0 unexpected=0" in out

    summary = _CKPT_LOAD_SUMMARY[(model_artifacts["clean"], "cpu")]
    assert summary["counts"] == {"missing": 0, "unexpected": 0, "shape_mismatch": 0}
    assert summary["missing"] == [] and summary["shape_mismatch"] == [] and summary["unexpected"] == []
    assert len(summary["sha256"]) == 64  # 版本戳：ckpt sha256


# --------------------------------------------------------------------------- #
# (b) 部分加载 → 默认 fail-fast
# --------------------------------------------------------------------------- #

def test_partial_load_fail_fast_by_default(model_artifacts: Dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _load_ckpt_model(model_artifacts["partial"], model_artifacts["config"], "cpu")
    assert excinfo.value.code == 2  # 非零退出码

    out = capsys.readouterr().out
    assert "ckpt 载入" in out  # 原日志行保留
    assert "missing=1 shape_mismatch=1 unexpected=0" in out
    assert "FAIL-FAST" in out
    assert model_artifacts["missing_key"] in out  # 明细（各项前若干条）
    assert model_artifacts["mismatch_key"] in out
    # 失败不缓存（不产出任何"部分加载模型"可用状态）
    assert (model_artifacts["partial"], "cpu") not in _CKPT_MODEL_CACHE
    assert (model_artifacts["partial"], "cpu") not in _CKPT_LOAD_SUMMARY


def test_cli_flag_defaults_and_parsing() -> None:
    assert build_parser().parse_args([]).allow_partial_load is False  # 默认 fail-fast
    assert build_parser().parse_args(["--allow-partial-load"]).allow_partial_load is True


# --------------------------------------------------------------------------- #
# (c) --allow-partial-load → 放行且强制记录
# --------------------------------------------------------------------------- #

def test_allow_partial_load_passes_and_records(model_artifacts: Dict[str, Any], capsys: pytest.CaptureFixture) -> None:
    model = _load_ckpt_model(
        model_artifacts["partial"], model_artifacts["config"], "cpu", allow_partial=True
    )
    assert isinstance(model, torch.nn.Module)

    out = capsys.readouterr().out
    assert "--allow-partial-load" in out  # 日志强制记录
    assert "missing=1 shape_mismatch=1 unexpected=0" in out
    assert model_artifacts["missing_key"] in out  # missing 列表

    summary = _CKPT_LOAD_SUMMARY[(model_artifacts["partial"], "cpu")]
    assert summary["allow_partial_load"] is True
    assert summary["counts"] == {"missing": 1, "unexpected": 0, "shape_mismatch": 1}
    assert model_artifacts["missing_key"] in summary["missing"]
    assert model_artifacts["mismatch_key"] in summary["shape_mismatch"]


def _main_argv(out_root: Path, spec_file: Path, ckpt: str, main_config: str, *extra: str) -> list[str]:
    return [
        "--config", str(main_config),
        "--spec", str(spec_file),
        "--policy", "ckpt",
        "--ckpt", str(ckpt),
        "--device", "cpu",
        "--workers", "1",
        "--out", str(out_root),
        "--name", "p0a_load_failfast",
        *extra,
    ]


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str], spec_file: Path) -> int:
    """main 端到端（不建 env/不 spawn）：spec 加载与任务执行打桩。"""
    monkeypatch.setattr(eval_runner, "_require_memory", lambda *args, **kwargs: (1, {"initial_mb": 0.0}))
    monkeypatch.setattr(eval_runner, "_run_tasks", lambda tasks, workers, recycle: [])
    monkeypatch.setattr(eval_runner, "write_eval_tensorboard", lambda *args, **kwargs: False)
    import env.scenario.spec as spec_module

    monkeypatch.setattr(
        spec_module, "load_specs", lambda path: [types.SimpleNamespace(id=0, seed=0)]
    )
    return eval_runner.main(argv)


def test_main_fail_fast_before_eval_by_default(
    model_artifacts: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_file = tmp_path / "specs.json"
    spec_file.write_text("[]", encoding="utf-8")
    out_root = tmp_path / "runs"

    with pytest.raises(SystemExit) as excinfo:
        _run_main(
            monkeypatch,
            _main_argv(out_root, spec_file, model_artifacts["partial"], model_artifacts["main_config"]),
            spec_file,
        )
    assert excinfo.value.code == 2
    assert not list(out_root.glob("*/metrics.json"))  # 未产出任何正式指标


def test_main_allow_partial_records_metrics_json(
    model_artifacts: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_file = tmp_path / "specs.json"
    spec_file.write_text("[]", encoding="utf-8")
    out_root = tmp_path / "runs"

    assert _run_main(
        monkeypatch,
        _main_argv(
            out_root, spec_file, model_artifacts["partial"], model_artifacts["main_config"],
            "--allow-partial-load",
        ),
        spec_file,
    ) == 0

    metrics_files = list(out_root.glob("*/metrics.json"))
    assert len(metrics_files) == 1
    meta = json.loads(metrics_files[0].read_text(encoding="utf-8"))["meta"]
    load = meta["ckpt_load"]
    assert load["counts"] == {"missing": 1, "unexpected": 0, "shape_mismatch": 1}
    assert model_artifacts["missing_key"] in load["missing"]
    assert load["allow_partial_load"] is True
    # 版本戳（§2.2）
    assert meta["ckpt_sha256"] == load["sha256"]
    assert meta["evaluator"] == "pipeline.eval_runner"
    assert isinstance(meta["git_commit"], str) and meta["git_commit"]
    assert "--allow-partial-load" in meta["argv"]
