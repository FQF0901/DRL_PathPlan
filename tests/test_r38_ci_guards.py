"""R38-P3 收尾：**§八 CI 守卫**（GPT R38 §八五条 → pytest 载体）。

出处：``docs/chatgpt5p6sol_review.md`` §八 + ``docs/repo_handbook.md`` 附录 D
（可达性审计 §5 + R38 弃用清单 A/B/D/G 组的机器冻结面）。五条守卫（CPU、无 metadrive）：

1. **禁生产目录重新引用 legacy 模块/符号**：``net/``、``env/``、``pipeline/``、``tools/*.py``
   （官方 CLI）对 ``tools.legacy``/``tools.diagnostics`` 零 import；生产目录不得重新定义
   已退役符号（``set_ego_warp``/``one_shot_plan``/``_group_of_tag``/``_remap_legacy_others``/
   ``_write_legacy_grouped_tensorboard``）。
2. **禁 default config 出现 deprecated 键**：``config/**/*.yaml``（含 ``default.yaml::includes``
   链）不得出现 ``plan_anchor``/``num_anchors``/``oneshot``/``allow_oneshot``/...
   （A/B/D 组配置键）；``plan_anchors_k6.json`` 不再存在。
3. **禁正式 forward 含 oneshot/plan_override**：``net/model.py``/``net/path_head.py`` 源码级
   （AST：定义/名字/属性/关键字实参）+ 运行时（``PATH_HEAD_MODES`` 无 oneshot、
   ``set_path_head_ablation("oneshot")`` ⇒ ``ValueError``、structured 输出无 oneshot 语义键）。
4. **禁两个动作生成器同时活跃**：``action_generator_names()`` 恒单元素；structured 前向
   ``contract_*`` 全 1.0 且不产出旧头诊断键（``policy_action_mu``/``action_logstd``）。
5. **禁缺失关键字段静默填 0**：``path_dynamic_distance``/``path_dynamic_ttc`` 缺键 ⇒ fail-fast；
   ``mem_from_obs`` 旧 others-28 维 ⇒ fail-fast（指路显式适配器），不得静默零填充。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch
import yaml

from net.model import DrivingModel
from net.path_head import PATH_HEAD_MODES, long_exog_from_obs
from tests.test_net_shapes import make_obs

ROOT = Path(__file__).resolve().parents[1]
BATCH = 2

# =========================================================================== #
# ① 禁生产目录重新引用 legacy 模块/符号
# =========================================================================== #
#: 生产目录（GPT §八：禁重新引用 legacy 模块）；``tools/`` 只扫官方 CLI（顶层 *.py，
#: ``tools/legacy`` 自身与 ``tools/diagnostics`` 离线工具除外）。
_PROD_DIRS = ("net", "env", "pipeline")
_LEGACY_MODULE_PREFIXES = ("tools.legacy", "tools.diagnostics", "experiments.archive")

#: 生产目录不得重新定义/落地的退役符号（定义面；名字精确匹配，避免误伤引用/文档）
_FORBIDDEN_DEF_NAMES = frozenset(
    {
        "set_ego_warp",
        "one_shot_plan",
        "_group_of_tag",
        "_write_legacy_grouped_tensorboard",
        "_remap_legacy_others",
    }
)
#: 生产源码不得重新出现的旧布局常量**赋值**（text 级；注释/文档引用不算）
_FORBIDDEN_ASSIGNMENTS = ("OTHERS_LEGACY_DIM", "OTHERS_HEAD_LEGACY_DIM")


def _iter_prod_py():
    for directory in _PROD_DIRS:
        yield from sorted((ROOT / directory).rglob("*.py"))
    yield from sorted((ROOT / "tools").glob("*.py"))  # 官方 CLI（顶层）


def _module_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    return []


def test_guard1_no_legacy_import_in_production_dirs() -> None:
    scanned = list(_iter_prod_py())
    # 非空控制：扫描面必须真实覆盖生产目录（防 glob 失效导致空判定）
    assert len(scanned) > 20
    assert (ROOT / "net" / "model.py") in scanned
    assert (ROOT / "pipeline" / "monitoring.py") in scanned

    hits: list[str] = []
    for path in scanned:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            for module in _module_targets(node):
                if module.startswith(_LEGACY_MODULE_PREFIXES):
                    hits.append(f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', '?')} → {module}")
            # importlib / __import__ 字符串装配（反射面）
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
                if name in {"import_module", "__import__"} and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        if arg.value.startswith(_LEGACY_MODULE_PREFIXES):
                            hits.append(f"{path.relative_to(ROOT)}:{node.lineno} → {arg.value}")
    assert hits == [], f"生产目录出现 legacy 模块引用：{hits}"


def test_guard1_no_legacy_symbol_redefined_in_production() -> None:
    hits: list[str] = []
    for path in _iter_prod_py():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _FORBIDDEN_DEF_NAMES:
                hits.append(f"{path.relative_to(ROOT)}:{node.lineno} def {node.name}")
        for name in _FORBIDDEN_ASSIGNMENTS:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if line.lstrip().startswith(f"{name} ="):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno} {name} = ...")
    assert hits == [], f"生产目录重新落地退役符号：{hits}"


# =========================================================================== #
# ② 禁 default config 出现 deprecated 键
# =========================================================================== #
_DEPRECATED_CONFIG_KEYS = frozenset(
    {
        # B 组（K-anchor）
        "plan_anchor",
        "plan_anchors",
        "plan_anchors_k6",
        "num_anchors",
        "anchor_ce",
        "anchor_wta",
        "anchor_temperature",
        "anchor_hard",
        # A 组（oneshot）
        "oneshot",
        "one_shot",
        "diagnostic_oneshot",
        "allow_oneshot",
        # D 组（双实体开关）
        "path_head_ablation",
        "enable_path_head",
        # C 组（旧 warp 实验）
        "set_ego_warp",
    }
)


def _collect_mapping_keys(node, out: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            out.add(str(key))
            _collect_mapping_keys(value, out)
    elif isinstance(node, list):
        for item in node:
            _collect_mapping_keys(item, out)


def _config_files() -> list[Path]:
    default = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    files = [ROOT / "config" / "default.yaml"]
    for include in (default or {}).get("includes", []) or []:
        files.append(ROOT / str(include))
    return files


def test_guard2_key_collector_not_vacuous() -> None:
    # 负控制：收集器必须能看见嵌套键（防"扫描实现悄悄失效"导致空判定）
    found: set[str] = set()
    _collect_mapping_keys({"a": {"plan_anchor": {"enabled": False}, "b": [{"num_anchors": 6}]}}, found)
    assert {"plan_anchor", "enabled", "num_anchors"} <= found


def test_guard2_default_config_has_no_deprecated_keys() -> None:
    scanned: list[Path] = sorted((ROOT / "config").rglob("*.yaml"))
    hits: list[str] = []
    for path in scanned:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        keys: set[str] = set()
        _collect_mapping_keys(payload, keys)
        for key in sorted(keys & _DEPRECATED_CONFIG_KEYS):
            hits.append(f"{path.relative_to(ROOT)}::{key}")
    assert hits == [], f"default config 出现 deprecated 键：{hits}"
    # includes 链被真实覆盖（防 default.yaml include 改路径后扫描面失效）
    assert (ROOT / "config" / "model.yaml") in _config_files()
    assert (ROOT / "config" / "train.yaml") in _config_files()
    # B 组锚字典文件不得回归
    assert not list((ROOT / "config").rglob("plan_anchors_k6.json"))


# =========================================================================== #
# ③ 禁正式 forward 含 oneshot/plan_override
# =========================================================================== #
#: 前向实现面（``net/probes.py`` 的 ``note_flags(oneshot=..., plan_override=...)`` 是
#: **断言"路径不存在"的守卫容器 API**（D1 探针），不在本扫描面内——它的正确性由
#: ``tests/test_round38_d1_probes.py`` 覆盖）。
_FORWARD_IMPL_FILES = ("net/model.py", "net/path_head.py")
_FORBIDDEN_FORWARD_NAMES = frozenset({"one_shot", "oneshot", "one_shot_plan", "plan_override"})


def test_guard3_forward_source_has_no_oneshot_plan_override_symbols() -> None:
    hits: list[str] = []
    for relative in _FORWARD_IMPL_FILES:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _FORBIDDEN_FORWARD_NAMES:
                hits.append(f"{relative}:{node.lineno} def {node.name}")
            elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_FORWARD_NAMES:
                hits.append(f"{relative}:{node.lineno} Name {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_FORWARD_NAMES:
                hits.append(f"{relative}:{node.lineno} Attr {node.attr}")
            elif isinstance(node, ast.keyword) and node.arg in _FORBIDDEN_FORWARD_NAMES:
                hits.append(f"{relative}:{node.lineno} kwarg {node.arg}")
    assert hits == [], f"正式 forward 源面出现 oneshot/plan_override：{hits}"


def test_guard3_runtime_modes_and_ablation_entry_fail_fast() -> None:
    assert "oneshot" not in PATH_HEAD_MODES
    assert PATH_HEAD_MODES == ("policy", "structured")
    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6).eval()
    with pytest.raises(ValueError, match="oneshot"):
        model.set_path_head_ablation("oneshot")


def _structured_obs() -> dict[str, torch.Tensor]:
    obs = make_obs(batch=BATCH, seed=0)
    s = torch.linspace(0.0, 40.0, 41)
    obs["lane_anchor_xy"] = (
        torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
    )
    obs["path_dynamic_distance"] = torch.full((BATCH, ), 30.0)
    obs["path_dynamic_ttc"] = torch.full((BATCH, ), 8.0)
    return obs


def _structured_model() -> DrivingModel:
    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    model.set_path_head(True)
    return model.eval()


def test_guard3_structured_forward_output_has_no_oneshot_semantics() -> None:
    model = _structured_model()
    with torch.no_grad():
        out = model(_structured_obs())
    assert "plan" in out and "action_mu" in out
    # 逐步组装契约：plan[:,0] == action_mu（一次性钉住路径的直接否证）
    assert torch.equal(out["plan"][:, 0, :], out["action_mu"])
    assert out["plan_from_aux"].item() == 0.0
    # oneshot 语义键不得出现（结构化诊断键以 structured_ 前缀且只报"未使用"）
    assert not [key for key in out if key.lower() in {"oneshot", "plan_override", "oneshot_plan"}]


# =========================================================================== #
# ④ 禁两个动作生成器同时活跃
# =========================================================================== #
def test_guard4_single_action_generator_active() -> None:
    default_model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6).eval()
    assert default_model.action_generator_names() == ("policy", )

    structured = _structured_model()
    assert structured.action_generator_names() == ("policy.structured_action", )
    with torch.no_grad():
        out = structured(_structured_obs())
    # 运行时契约门全绿（单实例/6 调用/5 推进/辅助不进 LQR/无钉住）
    contract_keys = [key for key in out if key.startswith("contract_")]
    assert len(contract_keys) == 6, contract_keys
    for key in contract_keys:
        assert float(out[key]) == 1.0, key
    # 旧头不得并行前向（structured ⇒ 无诊断动作键）
    assert "policy_action_mu" not in out
    assert "action_logstd" not in out


# =========================================================================== #
# ⑤ 禁缺失关键字段静默填 0
# =========================================================================== #
def test_guard5_missing_dynamic_path_columns_fail_fast() -> None:
    device = torch.device("cpu")
    base = {
        "ego": torch.zeros((1, 8), dtype=torch.float32),
        "lane_semantic": torch.zeros((1, 12), dtype=torch.float32),
        "lane_semantic_mask": torch.ones(1),
        "nav": torch.zeros((1, 11), dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="禁止静默 0"):
        long_exog_from_obs(base, batch=1, device=device)
    partial = dict(base)
    partial["path_dynamic_distance"] = torch.zeros(1)
    with pytest.raises(ValueError, match="path_dynamic_ttc"):
        long_exog_from_obs(partial, batch=1, device=device)


def test_guard5_legacy_others_28_fails_fast_not_silent_zero_fill() -> None:
    from net.encoders import DEFAULT_OTHERS_DIM
    from net.mem import mem_from_obs
    from net.param_probe import make_dummy_obs

    obs = make_dummy_obs(batch=2)
    legacy_hist = torch.cat([obs["others_hist"][..., :16], obs["others_hist"][..., 21:]], dim=-1)
    legacy_now = torch.cat([obs["others"][..., :16], obs["others"][..., 21:]], dim=-1)
    with pytest.raises(ValueError, match="旧 others 布局"):
        mem_from_obs(
            {**obs, "others_hist": legacy_hist, "others": legacy_now},
            others_dim=DEFAULT_OTHERS_DIM,
            history_frames=6,
        )
    with pytest.raises(ValueError, match="tools/legacy/others_layout"):
        mem_from_obs(
            {**obs, "others_hist": legacy_hist, "others": legacy_now},
            others_dim=DEFAULT_OTHERS_DIM,
            history_frames=6,
        )
