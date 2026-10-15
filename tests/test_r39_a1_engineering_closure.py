"""R39-A1 工程闭合回归（GPT §二-A 任务 1/2/3/4；CPU-only，不建 env/不 spawn）。

锁定面：

1. **结构化装配开关**：``--path-head {off,structured}`` 声明于 ``pipeline.eval_runner`` 与
   ``tools/test.py``（默认 off = 既有装配逐位不变）；``structured`` 时结构化 ckpt 经正式
   loader **严格装载**（已登记改名族 ``path_head.*→policy.structured_action.*`` 逐族计数），
   与"手工构造 + migration + ``strict=True``"逐位一致；
2. **ckpt migration + LoadReport**：正式加载路径接入 ``tools.legacy``（mapper + migration）；
   LoadReport = missing/unexpected/filled/rejected/renamed/dropped 计数 + 全量键列表 +
   ckpt sha256 + mapper 版本 + strict 判定；
3. **布尔 shim 无生产调用**：AST 扫描 ``net/ pipeline/ tools/ env/ config/`` +
   ``.slim/deepwork/p1_v2/``：除 fail-fast 断言测试（``tests/test_r38_p3_forward_mode.py``）
   外无任何 ``rollout=/world_model=`` 调用点；
4. **四 hash**：``_evaluation_hashes`` 原生产出 evaluation_protocol_hash / model_tree_hash /
   schema_hash / checkpoint_hash；``_record_run_meta`` 写 ``run_meta.json`` 并在既有
   ``manifest.txt`` 追加幂等 ``r39a1_hashes:`` 行；
5. **lane_anchor_xy 选项 a**：结构化前向缺键 ⇒ ``ValueError``（文案含注入指路）。
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

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


@pytest.fixture(autouse=True)
def _clear_loader_caches():
    _CKPT_MODEL_CACHE.clear()
    _CKPT_LOAD_SUMMARY.clear()
    yield
    _CKPT_MODEL_CACHE.clear()
    _CKPT_LOAD_SUMMARY.clear()


def _tiny_model_config() -> Dict[str, Any]:
    """小模型配置（CPU、快速）：直接走 ``stages.build_model`` 同一条构造路。"""
    return load_config(str(_ROOT / "config" / "model.yaml"))


def _save_structured_ckpt(tmp_path: Path, *, old_naming: bool = True) -> str:
    """产出结构化小模型 ckpt：``policy.structured_action.*``（P2b 新命名）或
    ``path_head.*``（旧命名，经 migration 改名族回填）。"""
    model = build_model(_tiny_model_config())
    model.set_path_head(True)
    model.set_path_head_ablation("structured")
    state = {key: value.clone() for key, value in model.state_dict().items()}
    if old_naming:
        state = {
            ("path_head." + key[len("policy.structured_action."):]
             if key.startswith("policy.structured_action.") else key): value
            for key, value in state.items()
        }
    path = tmp_path / ("ckpt_old_naming.pt" if old_naming else "ckpt_new_naming.pt")
    torch.save({"model": state, "meta": {"path_head_ablation": "structured"}}, path)
    return str(path)


def _structured_obs(batch: int = 2) -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    obs = {
        "ego_hist": torch.zeros(batch, 6, 1, 8),
        "ego_hist_mask": torch.ones(batch, 6, 1),
        "others_hist": torch.zeros(batch, 6, 1, 33),
        "others_hist_mask": torch.ones(batch, 6, 1),
        "od_hist": torch.zeros(batch, 6, 16, 9),
        "od_hist_mask": torch.zeros(batch, 6, 16),
        "ld_hist": torch.zeros(batch, 6, 16, 7),
        "ld_hist_mask": torch.zeros(batch, 6, 16),
        "hist_valid": torch.ones(batch, 6),
        "ego": torch.zeros(batch, 8),
    }
    s = torch.linspace(0.0, 40.0, 41)
    anchor = torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()
    obs["lane_anchor_xy"] = anchor
    for key, value in (
        ("path_left_margin", 3.0),
        ("path_right_margin", 3.0),
        ("path_static_distance", 40.0),
        ("path_dynamic_distance", 60.0),
        ("path_dynamic_ttc", 8.0),
        ("path_nav_progress", 0.2),
    ):
        obs[key] = torch.full((batch, 1), float(value))
    return obs


def _max_abs_delta(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    assert sorted(a) == sorted(b), (sorted(a), sorted(b))
    delta = 0.0
    for key in a:
        ta, tb = a[key], b[key]
        if torch.is_tensor(ta):
            delta = max(delta, float((ta.double() - tb.double()).abs().max()) if ta.numel() else 0.0)
    return delta


# --------------------------------------------------------------------------- #
# 1. CLI 声明
# --------------------------------------------------------------------------- #

def test_path_head_cli_declared_default_off() -> None:
    args = build_parser().parse_args([])
    assert args.path_head == "off"
    assert build_parser().parse_args(["--path-head", "structured"]).path_head == "structured"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--path-head", "oneshot"])
    # tools/test.py 同声明并显式转发（入口与正式 runner 同一开关）
    import tools.test as test_entry

    tools_args = test_entry.build_parser().parse_args(["--path-head", "structured"])
    assert tools_args.path_head == "structured"


def test_tools_test_entry_forwards_path_head(monkeypatch, tmp_path: Path) -> None:
    import tools.test as test_entry

    captured: dict = {}

    def _fake_eval_main(argv):
        captured["argv"] = [str(item) for item in argv]
        return 0

    monkeypatch.setattr("pipeline.eval_runner.main", _fake_eval_main)
    monkeypatch.setattr("pipeline.gl_runtime.ensure_gl_library_path", lambda **kwargs: None)
    rc = test_entry.main([
        "--config", str(_ROOT / "config/default.yaml"),
        "--policy", "baseline",
        "--out", str(tmp_path),
        "--path-head", "structured",
    ])
    assert rc == 0
    argv = captured["argv"]
    assert argv[argv.index("--path-head") + 1] == "structured"


# --------------------------------------------------------------------------- #
# 2. 结构化装载 + migration LoadReport
# --------------------------------------------------------------------------- #

def test_structured_ckpt_loads_strictly_via_formal_loader(tmp_path: Path) -> None:
    ckpt = _save_structured_ckpt(tmp_path, old_naming=True)
    config = _tiny_model_config()
    model = _load_ckpt_model(ckpt, config, "cpu", path_head="structured")
    assert model.path_head_mode() == "structured"
    summary = eval_runner._ckpt_load_summary(ckpt, "cpu", "structured")
    assert summary is not None
    report = summary["load_report"]
    counts = report["counts"]
    assert counts["missing"] == 0 and counts["unexpected"] == 0 and counts["rejected"] == 0
    assert counts["renamed"] == 18  # path_head.* → policy.structured_action.*（18 键）
    assert report["renamed_by_family"] == {"path_head.->policy.structured_action.": 18}
    assert report["strict"] is True and report["explicit_escape"] is False
    assert report["mapper"]["version"] == "r38p2b"
    sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert report["ckpt"]["sha256"] == sha and summary["sha256"] == sha
    # 结构化前向（带 anchor）可跑（policy_wm_rollout：plan 逐步组装）
    model.eval()
    with torch.no_grad():
        out = model(_structured_obs(), forward_mode="policy_wm_rollout")
    assert out["plan_from_aux"].item() == 0.0
    assert "path_ds_exec" in out or any(key.startswith("path_") for key in out)


def test_structured_load_bitwise_equals_manual_canonical_assembly(tmp_path: Path) -> None:
    """正式 loader（--path-head structured）≡ 手工装配（构造头 + migration + strict=True）逐位。"""
    from tools.legacy.checkpoint_migration import migrate_state_dict

    ckpt = _save_structured_ckpt(tmp_path, old_naming=True)
    config = _tiny_model_config()
    formal = _load_ckpt_model(ckpt, config, "cpu", path_head="structured")

    manual = build_model(dict(config))
    manual.set_path_head(True)
    manual.set_path_head_ablation("structured")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state, _report = migrate_state_dict(payload["model"], manual.state_dict().keys())
    manual.load_state_dict(state, strict=True)
    manual.eval()

    obs = _structured_obs()
    formal.eval()
    with torch.no_grad():
        out_formal = formal(obs, forward_mode="policy_wm_rollout")
        out_manual = manual(obs, forward_mode="policy_wm_rollout")
    assert _max_abs_delta(out_formal, out_manual) == 0.0


def test_formal_loader_wires_migration_primitive_strict(tmp_path: Path, monkeypatch) -> None:
    """正式 loader 的干净路径必须经 ``load_checkpoint_migrated(strict=True)``（R39-A1 接线）。"""
    import tools.legacy.checkpoint_migration as cm

    ckpt = _save_structured_ckpt(tmp_path, old_naming=True)
    calls: list[dict] = []
    real = cm.load_checkpoint_migrated

    def spy(path, model, *, strict=False, state=None):  # noqa: ANN001 - 透传
        calls.append({"strict": bool(strict), "state_given": state is not None})
        return real(path, model, strict=strict, state=state)

    monkeypatch.setattr(cm, "load_checkpoint_migrated", spy)
    _load_ckpt_model(ckpt, _tiny_model_config(), "cpu", path_head="structured")
    assert calls == [{"strict": True, "state_given": True}]
    summary = eval_runner._ckpt_load_summary(ckpt, "cpu", "structured")
    assert "load_checkpoint_migrated" in str(summary["load_report"]["loader"])


def test_new_naming_structured_ckpt_fail_fast_without_switch(tmp_path: Path, capsys) -> None:
    """未开 ``--path-head structured``：新命名结构化 ckpt 的键 = 未登记 unexpected ⇒ 默认拒绝。"""
    ckpt = _save_structured_ckpt(tmp_path, old_naming=False)
    with pytest.raises(SystemExit) as excinfo:
        _load_ckpt_model(ckpt, _tiny_model_config(), "cpu")
    assert excinfo.value.code == 2
    assert "FAIL-FAST" in capsys.readouterr().out


def test_lane_anchor_missing_fail_fast_points_to_injection(tmp_path: Path) -> None:
    """选项 a：结构化前向缺 ``lane_anchor_xy`` ⇒ ValueError（文案指路注入方式）。"""
    ckpt = _save_structured_ckpt(tmp_path, old_naming=True)
    model = _load_ckpt_model(ckpt, _tiny_model_config(), "cpu", path_head="structured")
    model.eval()
    obs = _structured_obs()
    obs.pop("lane_anchor_xy")
    with pytest.raises(ValueError, match="lane-anchor|注入"):
        with torch.no_grad():
            model(obs, forward_mode="policy_wm_rollout")


def test_legacy_import_allowlist_is_narrow() -> None:
    """R39-A1 对 R38 CI 守卫的登记例外必须保持窄（仅两个 loader 桥接模块）。"""
    from tests.test_r38_ci_guards import _ALLOWED_LEGACY_IMPORTS, _is_allowed_legacy

    assert set(_ALLOWED_LEGACY_IMPORTS) == {
        "tools.legacy.legacy_key_mapper",
        "tools.legacy.checkpoint_migration",
    }
    assert _is_allowed_legacy("tools.legacy.checkpoint_migration")
    assert _is_allowed_legacy("tools.legacy.legacy_key_mapper")
    assert not _is_allowed_legacy("tools.legacy")
    assert not _is_allowed_legacy("tools.legacy.oneshot_path")
    assert not _is_allowed_legacy("tools.diagnostics.forensics_offline")


# --------------------------------------------------------------------------- #
# 3. 布尔 shim 无生产调用（AST 全仓扫描）
# --------------------------------------------------------------------------- #

def test_no_boolean_forward_call_sites_repo_wide() -> None:
    import warnings as _warnings

    allowed = {"tests/test_r38_p3_forward_mode.py"}
    offenders: list[str] = []
    for base in ("net", "pipeline", "tools", "env", "config", ".slim/deepwork/p1_v2"):
        for path in (_ROOT / base).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            rel = str(path.relative_to(_ROOT))
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except SyntaxError as exc:  # pragma: no cover - 语法错误应单独暴露
                    offenders.append(f"{rel}: SyntaxError {exc}")
                    continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                hits = [kw.arg for kw in node.keywords if kw.arg in ("rollout", "world_model")]
                if hits and rel not in allowed:
                    offenders.append(f"{rel}:{node.lineno}: {hits}")
    assert offenders == [], f"生产调用仍有布尔 forward 开关：{offenders[:10]}"


# --------------------------------------------------------------------------- #
# 4. 四 hash + run_meta/manifest
# --------------------------------------------------------------------------- #

def _tiny_spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "specs_tiny.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "count": 2,
        "specs": [
            {"id": 1, "seed": 11, "split": "train", "difficulty": "easy"},
            {"id": 2, "seed": 12, "split": "train", "difficulty": "hard"},
        ],
    }), encoding="utf-8")
    return path


def test_evaluation_hashes_are_four_native_fields(tmp_path: Path) -> None:
    ckpt = _save_structured_ckpt(tmp_path, old_naming=True)
    spec = _tiny_spec_file(tmp_path)
    hashes = eval_runner._evaluation_hashes(
        spec_path=spec, policy="ckpt", ckpt=ckpt, tracker="exact",
        eval_reference="plan", moe_off=False, path_head="structured",
        workers=1, max_steps=10, device="cpu",
    )
    # 四 hash 原生齐全
    assert len(hashes["evaluation_protocol_hash"]) == 16
    assert len(hashes["evaluation_protocol_hash_full"]) == 64
    assert len(hashes["model_tree_hash"]) == 16
    assert hashes["schema_hash"].startswith("v")
    assert len(hashes["checkpoint_hash"]) == 64
    # protocol 记录完整（spec 集 + harness + control + model + frozen）
    protocol = hashes["protocol"]
    assert protocol["spec_set"]["n"] == 2
    assert protocol["control"] == {"tracker": "exact", "tail": "plan"}
    assert protocol["model"]["path_head"] == "structured"  # R39-A1：装配开关入 hash
    assert protocol["frozen"]["entry"]["script"] == "pipeline/eval_runner.py"
    # path_head 轴确实入 hash（同参不同装配 ⇒ hash 变化）
    hashes_off = eval_runner._evaluation_hashes(
        spec_path=spec, policy="ckpt", ckpt=ckpt, tracker="exact",
        eval_reference="plan", moe_off=False, path_head="off",
        workers=1, max_steps=10, device="cpu",
    )
    assert hashes_off["evaluation_protocol_hash"] != hashes["evaluation_protocol_hash"]
    # baseline（无 ckpt）不假造 checkpoint hash；c6 口径 control/model = not_applicable
    hashes_base = eval_runner._evaluation_hashes(
        spec_path=spec, policy="baseline", ckpt=None, tracker="exact",
        eval_reference="plan", moe_off=False, path_head="off",
        workers=1, max_steps=10, device="cpu",
    )
    assert hashes_base["checkpoint_hash"] is None
    assert hashes_base["protocol"]["control"]["tracker"].startswith("not_applicable")


def test_checkpoint_hash_matches_sha256_and_run_meta_written(tmp_path: Path) -> None:
    ckpt = _save_structured_ckpt(tmp_path, old_naming=True)
    spec = _tiny_spec_file(tmp_path)
    hashes = eval_runner._evaluation_hashes(
        spec_path=spec, policy="ckpt", ckpt=ckpt, tracker="exact",
        eval_reference="plan", moe_off=False, path_head="structured",
        workers=1, max_steps=10, device="cpu",
    )
    sha = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
    assert hashes["checkpoint_hash"] == sha
    assert hashes["checkpoint_hash16"] == sha[:16]

    out_dir = tmp_path / "run"
    out_dir.mkdir()
    (out_dir / "manifest.txt").write_text("stage: eval\nargv: x\n", encoding="utf-8")
    eval_runner._record_run_meta(out_dir, {"hashes": hashes, "ckpt_load": {"load_report": {}}})
    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["hashes"]["checkpoint_hash"] == sha
    manifest = (out_dir / "manifest.txt").read_text(encoding="utf-8")
    assert manifest.count("r39a1_hashes:") == 1
    assert sha in manifest
    # 幂等：二次写入不重复追加 manifest 行
    eval_runner._record_run_meta(out_dir, {"hashes": hashes, "ckpt_load": {"load_report": {}}})
    assert (out_dir / "manifest.txt").read_text(encoding="utf-8").count("r39a1_hashes:") == 1
