"""R38-P2b-C：legacy ``set_ego_warp``（w1/w2）退役守卫（pytest 载体）。

出处：R38 弃用清单 C 组（`LEGACY_REPRO`）+ `docs/repo_handbook.md` 附录 D
§3-C + 任务 §5.69-①（**先冻结 oracle fixture，再删实现**）。断言面：

1. 接口面：``DrivingModel.set_ego_warp`` / ``DrivingModel._ego_warp_pose_k`` / ``self.ego_warp``
   属性**不存在**；``set_ego_warp_mode`` 只接受 ``off | analytic``（w1/w2 ⇒ ValueError）；
2. 枚举面：``net.ego_warp.WARP_MODES == ("off", "analytic")``；``assemble_pose_k(mode="w1")``
   ⇒ ValueError（快照指针 = fixture + legacy 模块）；
3. **oracle fixture 完整性**：``tests/fixtures/r38_p2b_warp/`` 的逐位期望与当前 net 复算一致
   （pure_cases：off / w2→analytic / warp_pose_k 三重逐位；model_poses：off/analytic live 捕获 +
   w1/w2 fixture 复核）；
4. **legacy 快照保真**：``tools/legacy/legacy_ego_warp.py`` 纯函数与当前 net 等价函数逐位一致
   （w1/w2 复算链可审计）、``set_legacy_ego_warp`` fail-fast；
5. 生产目录不 import ``tools.legacy``（CI 守卫 #1 语义沿用）。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from net.ego_warp import WARP_MODES, assemble_pose_k, warp_pose_k
from net.model import DrivingModel
from tools.legacy.legacy_ego_warp import (
    LEGACY_WARP_MODES,
    legacy_assemble_pose_k,
    legacy_ego_warp_pose_k,
    set_legacy_ego_warp,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "r38_p2b_warp"


def _load(name: str) -> dict:
    path = FIXTURE / f"{name}.pt"
    if not path.exists():
        pytest.skip(f"fixture 缺失：{path}（生成器 .slim/deepwork/p1_v2/round38/p2b_oracle_freeze.py）")
    return torch.load(path, map_location="cpu", weights_only=False)


def _model() -> DrivingModel:
    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=2, expert_hidden=16, wm_steps=6).eval()


def test_legacy_warp_api_removed() -> None:
    model = _model()
    assert not hasattr(model, "set_ego_warp")
    assert not hasattr(model, "ego_warp")
    assert not hasattr(DrivingModel, "_ego_warp_pose_k")
    for mode in ("w1", "w2", "bogus"):
        with pytest.raises(ValueError):
            model.set_ego_warp_mode(mode)
    assert WARP_MODES == ("off", "analytic")
    assert LEGACY_WARP_MODES == ("off", "w1", "w2")


def test_assemble_pose_k_modes_are_off_analytic_only() -> None:
    ego = torch.zeros(1, 3)
    od = torch.zeros(1, 2, 3)
    vel = torch.zeros(1, 2, 2)
    ld = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="w1/w2 已退役"):
        assemble_pose_k(ego, od, vel, ld, mode="w1", dt=0.5, steps=1)
    with pytest.raises(ValueError):
        assemble_pose_k(ego, od, vel, ld, mode="w2", dt=0.5, steps=1)


def test_fixture_pure_cases_bitwise() -> None:
    fixture = _load("pure_cases")
    for name, case in fixture["cases"].items():
        off_now = assemble_pose_k(case["ego"], case["od"], case["vel"], case["ld"], mode="off", dt=0.5, steps=1)
        analytic_now = assemble_pose_k(
            case["ego"], case["od"], case["vel"], case["ld"], mode="analytic", dt=0.5, steps=1
        )
        warp_now = warp_pose_k(case["ego"], case["od"], case["ld"])
        assert torch.equal(off_now, case["assemble_off"]), name
        assert torch.equal(analytic_now, case["assemble_w2"]), name  # rename 口径 = 旧 w2 op 序
        assert torch.equal(warp_now, case["warp_pose_k"]), name


def test_fixture_model_poses_and_legacy_snapshot_bitwise() -> None:
    fixture = _load("model_poses")
    for mode in ("off", "w1", "w2"):
        entry = fixture["modes"][mode]
        ego_pose_by_step = {
            int(step): torch.cat(
                [entry["traj_xy"][:, int(step) - 1], entry["traj_theta"][:, int(step) - 1 : int(step)]], dim=-1
            )
            for step in entry["pose_k_by_step"]
        }
        for step, pose_k in entry["pose_k_by_step"].items():
            ours = legacy_assemble_pose_k(
                ego_pose_by_step[int(step)],
                entry["od_pose_t0"],
                entry["od_vel_rel"],
                entry["ld_pose_t0"],
                mode=mode,
                dt=float(entry["dt"]),
                steps=int(step),
            )
            assert torch.equal(pose_k, ours), f"{mode}/k{step}"
    # w2 ↔ analytic 重命名逐位（同一输入）
    entry = fixture["modes"]["w2"]
    step = max(int(s) for s in entry["pose_k_by_step"])
    ego_pose = torch.cat(
        [entry["traj_xy"][:, step - 1], entry["traj_theta"][:, step - 1 : step]], dim=-1
    )
    renamed = assemble_pose_k(
        ego_pose, entry["od_pose_t0"], entry["od_vel_rel"], entry["ld_pose_t0"],
        mode="analytic", dt=float(entry["dt"]), steps=step,
    )
    assert torch.equal(renamed, entry["pose_k_by_step"][step])


def test_fixture_manifest_sha256() -> None:
    import hashlib

    manifest_path = FIXTURE / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("fixture manifest 缺失")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        digest = hashlib.sha256((FIXTURE / name).read_bytes()).hexdigest()
        assert digest == entry["sha256"], f"{name} sha256 漂移"


def test_legacy_snapshot_pose_k_matches_net() -> None:
    torch.manual_seed(3)
    ego = torch.randn(2, 3)
    od = torch.randn(2, 4, 3) * 8.0
    ld = torch.randn(2, 5, 3) * 8.0
    assert torch.equal(legacy_ego_warp_pose_k(ego, od, ld), warp_pose_k(ego, od, ld))


def test_set_legacy_ego_warp_fail_fast() -> None:
    model = _model()
    for mode in ("off", "w1", "w2"):
        with pytest.raises(RuntimeError, match="已退役|w1/w2"):
            set_legacy_ego_warp(model, mode)
    with pytest.raises(ValueError):
        set_legacy_ego_warp(model, "bogus")


def test_no_production_import_of_legacy_warp() -> None:
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
