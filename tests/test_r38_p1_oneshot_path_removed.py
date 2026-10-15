"""R38-P1：一次性六步路径已删除（`net/` 侧"该路径不存在"的 pytest 载体）。

出处：R38 弃用清单 A 组（`ARCHITECTURE_FORBIDDEN`）+ `docs/repo_handbook.md` 附录 D
§3-A / §5 守卫 #3「禁止正式 forward 含一次性钉住路径」。断言面：

1. 配置面：``set_path_head_ablation`` 写入旧模式名 ⇒ ``ValueError`` fail-fast（无 enable 开关）；
2. 枚举面：``PATH_HEAD_MODES`` 只剩 ``policy | structured``；
3. API 面：``PathExecutionHead`` / ``compose_path_plan`` / ``one_shot_plan`` / ``one_shot=`` 形参
   在 ``net`` 与 ``net.path_head`` 均不存在；``net.__all__`` 不再导出；
4. 契约门**不弱化**：``contract_deployment_does_not_use_plan_override`` 由**结构**派生 ——
   模式名回归（monkeypatch ``net.model.PATH_HEAD_MODES``）⇒ 门失败（eval/deploy ``RuntimeError``）；
5. 快照指针：fail-fast 文案指向 ``tools/legacy/oneshot_path.py``。
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

import net
import net.model as net_model
import net.path_head as net_path_head
from net.model import DrivingModel
from net.path_head import StructuredComposeLayer, StructuredPolicyHead
from tests.test_net_shapes import make_obs

ROOT = Path(__file__).resolve().parents[1]
BATCH = 2


def _structured_obs() -> dict[str, torch.Tensor]:
    obs = make_obs(batch=BATCH, seed=0)
    s = torch.linspace(0.0, 40.0, 41)
    obs["lane_anchor_xy"] = (
        torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(BATCH, -1, -1).contiguous()
    )
    obs["lane_anchor_mask"] = torch.ones(BATCH, 41)
    # R38-Q4A：t0 段 dynamic 两列缺键 = fail-fast（禁静默 0）——本测试的观测按新契约补齐
    obs["path_dynamic_distance"] = torch.full((BATCH, ), 30.0)
    obs["path_dynamic_ttc"] = torch.full((BATCH, ), 8.0)
    return obs


def _structured_model() -> DrivingModel:
    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=6)
    model.set_path_head(True)
    model.set_path_head_ablation("structured")
    return model.eval()


def test_set_path_head_ablation_rejects_removed_mode() -> None:
    """配置面 fail-fast：旧模式名 → ValueError（含快照指针），两档可用。"""
    model = _structured_model()
    with pytest.raises(ValueError, match="R38-P1 减枝删除"):
        model.set_path_head_ablation("oneshot")
    with pytest.raises(TypeError):
        model.set_path_head_ablation("oneshot", allow_oneshot=True)  # 守卫形参同步删除
    assert model.set_path_head_ablation("structured").path_head_mode() == "structured"
    assert model.set_path_head_ablation("policy").path_head_mode() == "policy"
    with pytest.raises(ValueError, match="必须是"):
        model.set_path_head_ablation("nonexistent")


def test_removed_mode_failfast_points_to_legacy_snapshot() -> None:
    """fail-fast 文案必须指向 ``tools/legacy/oneshot_path.py``（迁移可发现性）。"""
    model = _structured_model()
    with pytest.raises(ValueError) as excinfo:
        model.set_path_head_ablation("oneshot")
    assert "tools/legacy/oneshot_path.py" in str(excinfo.value)
    assert (ROOT / "tools" / "legacy" / "oneshot_path.py").is_file()


def test_mode_enum_and_api_surface_removed() -> None:
    """枚举两档 + 被删符号在 ``net`` / ``net.path_head`` 全域不存在。"""
    assert net_path_head.PATH_HEAD_MODES == ("policy", "structured")
    for name in ("PathExecutionHead", "compose_path_plan"):
        assert not hasattr(net, name), f"net.{name} 仍可导入"
        assert not hasattr(net_path_head, name), f"net.path_head.{name} 仍存在"
        assert name not in net.__all__, f"net.__all__ 仍声明 {name}"
        assert name not in net_path_head.__all__, f"net.path_head.__all__ 仍声明 {name}"
    assert not hasattr(StructuredComposeLayer, "one_shot_plan")
    assert "one_shot" not in inspect.signature(StructuredPolicyHead.forward).parameters
    assert "allow_oneshot" not in inspect.signature(DrivingModel.set_path_head_ablation).parameters
    # 源码面：path_head 模块内无一次性路径/旧别名残留
    src = (ROOT / "net" / "path_head.py").read_text(encoding="utf-8")
    for token in ("plan_override", "one_shot", "PathExecutionHead =", "def compose_path_plan"):
        assert token not in src, f"net/path_head.py 仍含 {token!r}"
    model_src = (ROOT / "net" / "model.py").read_text(encoding="utf-8")
    assert "allow_oneshot" not in model_src
    assert "compose_path_plan" not in model_src


def test_contract_gate_not_weakened_structural_derivation(monkeypatch: pytest.MonkeyPatch) -> None:
    """契约门⑤不弱化：正常态 = 1；模式名回归（结构被破坏）⇒ 0 + eval fail-fast。"""
    model = _structured_model()
    obs = _structured_obs()
    out = model(obs)
    assert float(out["contract_deployment_does_not_use_plan_override"]) == 1.0
    assert float(out["structured_plan_override_used"]) == 0.0
    assert float(out["contract_runtime_ok"]) == 1.0
    rollout_src = inspect.getsource(DrivingModel._rollout).replace("structured_plan_override_used", "")
    assert "plan_override" not in rollout_src  # 只允许契约输出键名残留（用于键集兼容）

    monkeypatch.setattr(net_model, "PATH_HEAD_MODES", ("policy", "structured", "oneshot"))
    with pytest.raises(RuntimeError, match="运行时契约断言失败"):
        model(obs)

    monkeypatch.setattr(net_model, "PATH_HEAD_MODES", ("policy", "structured"))
    out_again = model(obs)
    assert float(out_again["contract_deployment_does_not_use_plan_override"]) == 1.0


def test_no_production_import_of_legacy_snapshot() -> None:
    """CI 守卫 #1 语义：生产目录不**导入** ``tools.legacy``（AST 级；docstring 指针允许）。"""
    import ast

    offenders: list[str] = []
    for path in sorted((ROOT / "net").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "tools" or name.startswith("tools.") for name in names):
                offenders.append(f"{path.name}:{node.lineno} {names}")
    assert not offenders, offenders
