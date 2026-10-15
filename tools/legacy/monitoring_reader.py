"""R38-P3-G：legacy monitoring 读取/重放（自 ``pipeline/monitoring.py`` 迁出的旧 tag 口径）。

出处（快照 commit = ``2cfd8e8``（R38-P3 前 HEAD），行号为快照时点）：

- ``pipeline/monitoring.py:309-330`` → ``_LEGACY_HORIZON_FAMILIES`` / ``_LEGACY_STAGE_A_WM_TERMS`` /
  ``_LEGACY_STAGE_A_HEALTH`` / ``_LEGACY_BC_TERM_SUBS`` + 5 条 ``_LEGACY_*`` regex
  （本模块同名保留，供旧日志口径追溯）；
- ``pipeline/monitoring.py:333-366`` → ``_group_of_tag`` = :func:`legacy_group_of_tag`；
- ``pipeline/monitoring.py:369-391`` → ``_grouped_scalars``（旧 ``group_fn`` 分支）=
  :func:`legacy_grouped_scalars`；
- ``pipeline/monitoring.py:736-744`` → ``_write_legacy_grouped_tensorboard`` =
  :func:`write_legacy_grouped_tensorboard`（旧 ``add_scalars`` 多线族重放）。

**判定**：`LEGACY_REPRO`（R38 弃用清单 G 组：``legacy_tags`` / ``_LEGACY_*`` /
``_group_of_tag`` / ``_write_legacy_grouped_tensorboard`` 迁 ``tools/legacy/monitoring_reader.py``；
正式训练只写新 schema —— ``pipeline/monitoring.py`` 的 ``_slim_tag`` = KEEP）。

用途（GPT §七-G："如果仅用于读取旧日志"）
--------------------------------------------
- **旧 logs 读取/重分组**：:func:`read_metrics_csv` 读旧运行的长表 CSV，
  :func:`legacy_group_of_tag` / :func:`legacy_grouped_scalars` 还原旧多线族口径；
- **旧 TB 事件重放**：:func:`write_legacy_grouped_tensorboard` 把同一批 scalar 按旧多线族
  写进新的 writer（TurnBoard 侧对比图）；
- **禁止**：本模块不被 ``net/``、``env/``、``pipeline/`` 导入（CI 守卫 #1 扫描面）；
  正式 ``TrainingMonitor`` 不再接受 ``legacy_tags``（写入即 fail-fast，指路本模块）。
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping as _MappingABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LEGACY_HORIZON_FAMILIES",
    "LEGACY_STAGE_A_WM_TERMS",
    "LEGACY_STAGE_A_HEALTH",
    "LEGACY_BC_TERM_SUBS",
    "legacy_group_of_tag",
    "legacy_grouped_scalars",
    "read_metrics_csv",
    "write_legacy_grouped_tensorboard",
]

# --------------------------------------------------------------------------- #
# 旧 tag → 旧 (main_tag, sub_tag) 分组（2026-09-27 Tier-1 瘦身前的行为；
# 出处 pipeline/monitoring.py @ 2cfd8e8）
# --------------------------------------------------------------------------- #
_LEGACY_HORIZON_FAMILIES: Dict[str, str] = {
    "loss": "horizon_loss",
    "ade": "horizon_ade",
    "cv_ade": "horizon_cv_ade",
    "ego_next_loss": "horizon_ego_next",
    "traj_mse_m2": "horizon_traj_mse_m2",
    "traj_mae_m": "horizon_traj_mae_m",
}
_LEGACY_STAGE_A_WM_TERMS = ("wm_loss", "wm_loss_od", "wm_loss_ego_next", "presence_loss", "entry_loss")
_LEGACY_STAGE_A_HEALTH = ("presence_auc", "entry_auc")
_LEGACY_BC_TERM_SUBS = {"loss": "loss", "traj_loss": "traj", "action_loss": "action", "router_loss": "router"}

_LEGACY_HORIZON_GROUP_RE = re.compile(r"^horizon/(?P<group>h\d+)/(?P<metric>[^/]+)/mean$")
_LEGACY_SLICE_GROUP_RE = re.compile(r"^slice/(?P<group>[^/]+)/action_err/mean$")
_LEGACY_LABEL_GROUP_RE = re.compile(r"^label/(?P<group>[^/]+)/action_err/mean$")
_LEGACY_EXPERT_MIX_RE = re.compile(
    r"^(?P<phase>[A-Za-z][A-Za-z0-9_]*)_bc_router_expert_mix_weight_(?P<index>\d+)$"
)
_LEGACY_BC_TERM_RE = re.compile(
    r"^(?P<phase>[A-Za-z][A-Za-z0-9_]*)_bc_(?P<term>loss|traj_loss|action_loss|router_loss)$"
)
_LEGACY_GRAD_NORM_RE = re.compile(r"^grad_norm_(?P<name>.+)$")


def _finite(value: Any) -> Optional[float]:
    """转 float；NaN/Inf/非数值 → None（``pipeline.monitoring._finite`` 同口径；兼容 CSV 字符串）。"""
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def legacy_group_of_tag(tag: str) -> Optional[Tuple[str, str]]:
    """legacy：单 tag → 旧 ``(main_tag, sub_tag)``；不属于任何分组族 → None。

    （出处 ``pipeline/monitoring.py::_group_of_tag`` @ 2cfd8e8，逐行口径拷贝。）
    """
    prefix = ""
    name = tag
    if tag.startswith("val/"):
        prefix, name = "val_", tag[len("val/"):]
    elif tag.startswith("train/"):
        name = tag[len("train/"):]

    match = _LEGACY_HORIZON_GROUP_RE.match(name)
    if match:
        family = _LEGACY_HORIZON_FAMILIES.get(match.group("metric"))
        return None if family is None else (f"{prefix}{family}", match.group("group"))
    match = _LEGACY_SLICE_GROUP_RE.match(name)
    if match:
        return f"{prefix}slice_action_err", match.group("group")
    match = _LEGACY_LABEL_GROUP_RE.match(name)
    if match:
        return f"{prefix}label_action_err", match.group("group")
    match = _LEGACY_EXPERT_MIX_RE.match(name)
    if match:
        return f"{prefix}expert_mix_weight_{match.group('phase')}", f"e{match.group('index')}"
    match = _LEGACY_BC_TERM_RE.match(name)
    if match:
        return f"{prefix}bc_terms_{match.group('phase')}", _LEGACY_BC_TERM_SUBS[match.group("term")]
    if not prefix:  # Stage A 规则只在 train/ 前缀下生效（val/ 无同族）
        if name in _LEGACY_STAGE_A_WM_TERMS:
            return "wm_terms", name
        if name in _LEGACY_STAGE_A_HEALTH:
            return "health", name
        match = _LEGACY_GRAD_NORM_RE.match(name)
        if match:
            return "grad_norm", match.group("name")
    return None


def legacy_grouped_scalars(
    scalars: Mapping[str, Any], *, min_subs: int = 2
) -> Dict[str, Dict[str, float]]:
    """扁平 ``tag → value`` → ``{main_tag: {sub: value}}``（仅保留 ≥``min_subs`` 个 sub 的族）。

    （出处 ``pipeline/monitoring.py::_grouped_scalars(group_fn=_group_of_tag)`` @ 2cfd8e8。）
    """
    groups: Dict[str, Dict[str, float]] = {}
    for raw_tag, raw_value in scalars.items():
        number = _finite(raw_value)
        if number is None:
            continue
        identified = legacy_group_of_tag(str(raw_tag))
        if identified is None:
            continue
        main_tag, sub_tag = identified
        groups.setdefault(main_tag, {})[sub_tag] = number
    return {main: subs for main, subs in groups.items() if len(subs) >= min_subs}


def read_metrics_csv(path: Any) -> Dict[str, List[Tuple[Optional[int], float]]]:
    """旧 ``metrics.csv``（长表 ``step,tag,value``）→ ``{tag: [(step, value), ...]}``。

    非有限值/坏行跳过（与旧写入口径一致：``_finite`` 过滤后落盘）。``step`` 空串 → None。
    """
    series: Dict[str, List[Tuple[Optional[int], float]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tag = str(row.get("tag") or "")
            number = _finite(row.get("value"))
            if not tag or number is None:
                continue
            raw_step = str(row.get("step") or "").strip()
            step = int(raw_step) if raw_step else None
            series.setdefault(tag, []).append((step, number))
    return series


def write_legacy_grouped_tensorboard(
    writer: Any, scalars: Mapping[str, Any], *, step: int, min_subs: int = 2
) -> int:
    """legacy 重放：把 ``scalars`` 按旧族写 ``writer.add_scalars(main, subs, gstep)``。

    （出处 ``pipeline/monitoring.py::_write_legacy_grouped_tensorboard`` @ 2cfd8e8。）
    返回写入的族数；``writer=None``/无族 → 0（不 raise，供只读工具直接调用）。
    """
    if writer is None:
        return 0
    groups = legacy_grouped_scalars(scalars, min_subs=min_subs)
    for main_tag, subs in groups.items():
        writer.add_scalars(main_tag, subs, global_step=int(step))
    return len(groups)


def _self_check() -> int:
    """模块自检（CPU；无 torch/tensorboard）：旧族分组/CSV 读取/重放调用面。"""
    grouped = legacy_grouped_scalars(
        {
            "train/wm_loss": 1.0,
            "train/wm_loss_od": 2.0,
            "train/primary_bc_loss": 3.0,
            "train/primary_bc_action_loss": 4.0,
            "horizon/h1/loss/mean": 5.0,
            "horizon/h2/loss/mean": 6.0,
            "val/horizon/h1/loss/mean": 7.0,
            "val/horizon/h2/loss/mean": 8.0,
            "train/unknown_x": 1.0,
        }
    )
    assert grouped["wm_terms"] == {"wm_loss": 1.0, "wm_loss_od": 2.0}, grouped
    assert grouped["bc_terms_primary"] == {"loss": 3.0, "action": 4.0}, grouped
    assert grouped["horizon_loss"] == {"h1": 5.0, "h2": 6.0}, grouped
    assert grouped["val_horizon_loss"] == {"h1": 7.0, "h2": 8.0}, grouped
    assert legacy_group_of_tag("horizon/h1/traj_mse_m2/mean") == ("horizon_traj_mse_m2", "h1")
    assert legacy_group_of_tag("train/unknown_x") is None

    class _Writer:
        def __init__(self) -> None:
            self.calls: list = []

        def add_scalars(self, main: str, subs: dict, global_step: int) -> None:
            self.calls.append((main, dict(subs), int(global_step)))

    writer = _Writer()
    n = write_legacy_grouped_tensorboard(
        writer, {"train/wm_loss": 1.0, "train/wm_loss_od": 2.0}, step=7
    )
    assert n == 1 and writer.calls == [("wm_terms", {"wm_loss": 1.0, "wm_loss_od": 2.0}, 7)]
    print("[tools.legacy.monitoring_reader] self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
