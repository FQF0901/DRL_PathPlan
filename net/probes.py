"""R38-D1：§四 七类强制 Probes/Hooks 的统一容器（**默认全关**；开启只读）。

范围与缺项补全（对照 ``docs/chatgpt5p6sol_review.md`` §四）
----------------------------------------------------------------
- ① 状态来源（:class:`StateSourceHooks`）：每步 ``real/predicted/substituted`` +
  ego/OD/LD/nav/Memory latent（模型侧接入点：``net/model.py::encode/_rollout``）。
- ② Policy（:class:`PolicyHooks`）：backbone latent、曲率输出、纵向输出、最终 ``ds/dθ``、
  辅助输出、输入 valid mask（t0 调用 + 每步调用）。
- ③ World Model（:class:`WorldModelHooks`）：输入 action、z in/out、OD/LD 物理解码、
  presence/entry；**warp 五项 anchor 链**沿用 R37 :class:`net.ego_warp.WarpHooks`（单列，
  不重复实现）。
- ④ Memory（:class:`MemoryHooks`）：逐帧 mask、slot identity、attention 权重、历史帧贡献、
  全 mask 输出、历史顺序（``net/mem.py`` / ``net/temporal.py`` 只读接入）。
- ⑤ 六步调用链（:class:`ChainHooks`）：Policy 6 次调用 / 同一实例 / 每 ``a_k`` 读对应 ``z_k``
  （逐位 linkage 断言）/ WM 决策推进 5 次 / **ST-GNN 总调用单列** / 辅助不进 LQR /
  正式路径无 oneshot/plan_override；统一出口 = :class:`ProbeRecorder`。
- ⑥ 连续路径与控制（模块级纯函数）：六步 plan → ≤1 m 连续插值 footprint → **首次越界/
  碰撞点**（车道边界 / OD 盒）；LQR reference / 实际轨迹 / steer-throttle / 饱和由控制侧
  遥测承担（``tools/diagnostics/forensics_closed_loop.py`` +
  ``.slim/deepwork/p1_v2/round36/h3_hooks_ext.py``，见 D1 审计矩阵）。
- ⑦ 参数与梯度账本（:func:`grad_ledger` + :func:`state_dict_hash`）：参数名/模块/
  ``requires_grad``/optimizer group/grad norm/update norm/ckpt 变化；**对第 1/3/6 步 loss
  单独反向**验证梯度可达性（逐 loss 分别 ``autograd.grad``，不做三 loss 合并）。

默认关闭契约（硬性）
--------------------
``enabled=False``（默认；或容器整体 ``ProbeRecorder(enabled=False)``）时：

- 每个 ``capture*`` 首行短路返回 ``None`` ⇒ **零张量操作、不消耗 RNG、不写回输入**；
- 模型侧 ``set_probes(None)``（默认）或 ``probes.enabled=False`` ⇒ 前向逐位一致、
  输出键集一致（Gate A 自证：``.slim/deepwork/p1_v2/round38/d1_off_bitwise.py``，
  子进程双跑 HEAD/改动后 net 包 ``max|Δ|=0`` + RNG/optimizer/ckpt 不变）。

开启后只读：所有张量 ``detach().clone()`` 后存储（不改变调用方张量；反向图不受影响）。
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "ChainHooks",
    "DETACH_REGISTRY",
    "MemoryHooks",
    "PolicyHooks",
    "ProbeRecorder",
    "StateSourceHooks",
    "WorldModelHooks",
    "WorldStateHooks",
    "check_detach_registry",
    "first_boundary_breach",
    "first_od_contact",
    "grad_ledger",
    "interpolate_plan_poses",
    "scan_detaches",
    "state_dict_hash",
]

#: ① 来源标签（模型 rollout：step0 = real；k>=1 = predicted；真值替换实验 = substituted）
STATE_SOURCES = ("real", "predicted", "substituted")
#: ⑥ 连续 footprint 默认重采样间距（与 forensics ``FOOTPRINT_MAX_GAP_M`` 同口径）
FOOTPRINT_MAX_GAP_M = 1.0
#: ⑤ 期望调用次数（架构契约 §六步调用图；训练/部署一致）
EXPECTED_POLICY_CALLS = 6
EXPECTED_WM_ADVANCES = 5


def _clone(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {str(key): _clone(item) for key, item in value.items()}
    return value


def _sorted_prefixes(names: Iterable[str], depth: int = 2) -> list[str]:
    """参数名前 ``depth`` 段前缀集合（账本分组读数用）。"""
    out = set()
    for name in names:
        parts = str(name).split(".")
        out.add(".".join(parts[:depth]) if len(parts) > depth else str(name))
    return sorted(out)


def state_dict_hash(module: torch.nn.Module) -> str:
    """``state_dict`` 内容 sha256（ckpt 是否变化 / 逐位一致性证据）。"""
    digest = hashlib.sha256()
    for name in sorted(module.state_dict().keys()):
        tensor = module.state_dict()[name]
        digest.update(str(name).encode("utf-8"))
        if torch.is_tensor(tensor):
            flat = tensor.detach().to("cpu").contiguous()
            digest.update(str(tuple(flat.shape)).encode("utf-8"))
            digest.update(flat.numpy().tobytes())
        else:  # pragma: no cover - nn.Module state_dict 基本恒为 Tensor
            digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# hook 容器基类
# --------------------------------------------------------------------------- #
class _HookBase:
    """``enabled`` 短路 + 记录上限 + ``clear`` 的最小公共实现。"""

    #: snapshot 键前缀
    prefix: str = "hook"

    def __init__(self, *, enabled: bool = False, max_records: int = 0):
        self.enabled = bool(enabled)
        #: ``>0`` 时超出上限的 capture 只计数不存张量（长 rollout 内存护栏）
        self.max_records = int(max_records)
        self.records: list[dict[str, Any]] = []
        self.capture_calls = 0
        self.skipped_records = 0

    def clear(self) -> None:
        self.records.clear()
        self.capture_calls = 0
        self.skipped_records = 0

    def _append(self, record: dict[str, Any]) -> dict[str, Any] | None:
        self.capture_calls += 1
        if self.max_records > 0 and len(self.records) >= self.max_records:
            self.skipped_records += 1
            return None
        self.records.append(record)
        return record

    # ------------------------------------------------------------------ 导出
    def tensors(self) -> dict[str, Tensor]:
        """展平为 ``{prefix/...: Tensor}`` 张量字典（``torch.save`` 直接可写）。"""
        out: dict[str, Tensor] = {}
        for index, record in enumerate(self.records):
            self._flatten(out, f"{self.prefix}/r{index:02d}", record)
        return out

    @staticmethod
    def _flatten(out: dict[str, Tensor], path: str, value: Any) -> None:
        if torch.is_tensor(value):
            out[path] = value.detach().clone()
        elif isinstance(value, Mapping):
            for key, item in value.items():
                _HookBase._flatten(out, f"{path}/{key}", item)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "records": len(self.records),
            "capture_calls": int(self.capture_calls),
            "skipped_records": int(self.skipped_records),
        }


# --------------------------------------------------------------------------- #
# ① 状态来源
# --------------------------------------------------------------------------- #
class StateSourceHooks(_HookBase):
    """① 每步: ``real``/``predicted``/``substituted`` + ego/OD/LD/nav/Memory latent。"""

    prefix = "state"
    #: Memory latent 键（t0 编码期注意力聚合；逐帧细节见 :class:`MemoryHooks`）
    MEMORY_KEYS = ("ego_ctx", "od_ctx", "ld_ctx", "others_ctx")

    def capture(
        self,
        *,
        step: int,
        source: str,
        z_ego: Tensor | None = None,
        z_od: Tensor | None = None,
        z_ld: Tensor | None = None,
        nav: Tensor | None = None,
        memory: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if source not in STATE_SOURCES:
            raise ValueError(f"source 必须 ∈ {STATE_SOURCES}，收到 {source!r}")
        record = {
            "step": int(step),
            "source": str(source),
            "z_ego": _clone(z_ego),
            "z_od": _clone(z_od),
            "z_ld": _clone(z_ld),
            "nav": _clone(nav),
            "memory": None
            if memory is None
            else {key: _clone(memory.get(key)) for key in self.MEMORY_KEYS},
        }
        return self._append(record)

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base.update(
            {
                "sources": [str(record.get("source")) for record in self.records],
                "steps": [int(record.get("step", -1)) for record in self.records],
            }
        )
        return base


# --------------------------------------------------------------------------- #
# ② Policy
# --------------------------------------------------------------------------- #
class PolicyHooks(_HookBase):
    """② 结构化头：backbone latent / 曲率 / 纵向 / 最终 ds,dθ / 辅助输出 / valid mask。"""

    prefix = "policy"
    #: 归口读取的头输出键（存在才记录；见 ``net/path_head.py::StructuredPolicyHead.forward``）
    OUTPUT_KEYS = (
        "action",
        "ds_exec",
        "dtheta",
        "kappa_near",
        "kappa_mid",
        "kappa_far",
        "kappa_residual",
        "kappa_ref",
        "ds_aux",
        "aux_progress",
        "aux_plan",
        "plan_aux",
        "speed",
    )
    #: 输入 valid mask 归口键（缺省 None = 该次调用无该字段）
    MASK_KEYS = ("anchor_mask", "target_lane_valid", "long_present", "anchor_valid")

    def capture(
        self,
        *,
        step: int,
        latent: Tensor,
        outputs: Mapping[str, Tensor],
        masks: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        record: dict[str, Any] = {"step": int(step), "latent": _clone(latent)}
        for key in self.OUTPUT_KEYS:
            value = outputs.get(key)
            if torch.is_tensor(value):
                record[key] = _clone(value)
        merged_masks: dict[str, Any] = {}
        for key in self.MASK_KEYS:
            value = None
            if masks is not None and key in masks:
                value = masks[key]
            elif torch.is_tensor(outputs.get(key)):
                value = outputs[key]
            merged_masks[key] = _clone(value)
        record["masks"] = merged_masks
        return self._append(record)

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base.update(
            {
                "steps": [int(record.get("step", -1)) for record in self.records],
                "output_keys": sorted(
                    {key for record in self.records for key in record.keys() if key in self.OUTPUT_KEYS}
                ),
            }
        )
        return base


# --------------------------------------------------------------------------- #
# ③ World Model
# --------------------------------------------------------------------------- #
class WorldModelHooks(_HookBase):
    """③ 单步 WM：输入 action、z in/out、物理解码、presence/entry。

    warp 五项 anchor 链（warp 前 / 解析后 / 真实下一帧 / learned residual / 最终）由
    R37 :class:`net.ego_warp.WarpHooks` 单列承担（``DrivingModel.set_warp_hooks``）。
    """

    prefix = "wm"

    def capture(
        self,
        *,
        step: int,
        action: Tensor,
        z_in: Mapping[str, Tensor],
        z_out: Mapping[str, Tensor],
        od_pred: Tensor | None = None,
        ld_pred: Tensor | None = None,
        presence_logit: Tensor | None = None,
        entry_logit: Tensor | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        record = {
            "step": int(step),
            "action": _clone(action),
            "z_ego_in": _clone(z_in.get("z_ego")),
            "z_od_in": _clone(z_in.get("z_od")),
            "z_ld_in": _clone(z_in.get("z_ld")),
            "z_ego_out": _clone(z_out.get("z_ego")),
            "z_od_out": _clone(z_out.get("z_od")),
            "z_ld_out": _clone(z_out.get("z_ld")),
            "od_pred": _clone(od_pred),
            "ld_pred": _clone(ld_pred),
            "presence_logit": _clone(presence_logit),
            "entry_logit": _clone(entry_logit),
        }
        return self._append(record)


# --------------------------------------------------------------------------- #
# ③b R39-B1 统一 imagined WorldState / StateReadout
# --------------------------------------------------------------------------- #
class WorldStateHooks(_HookBase):
    """R39-B1：统一 ``WorldState → PolicyStateView`` 逐帧 hook（只读；**默认关**）。

    每步记录（GPT §八「状态接口 Hook」口径）：

    - 七量字段值 + **逐行 source**（``analytic/learned/held/missing``）+ present/valid mask；
    - **producer（唯一写点）** 与 source 覆盖统计；
    - WorldState 来源面：ego pose/speed、OD/LD anchor（t0 帧）与 ego 帧重表达、progress、
      route length（``inf`` = 不可推断）；
    - Imagined Memory 摘要（帧数/帧龄/来源/writeback = 0）。

    自动断言（:meth:`check`；machine-readable）：① 字段集 = ``net/world_state.POLICY_STATE_FIELDS``；
    ② 每字段 producer 唯一（跨帧不得变化）且 = ``FIELD_PRODUCERS``；③ source ∈ ``SOURCES``；
    ④ ``held`` 仅限 ``HELD_ALLOWED``；⑤ ``missing`` ⇒ ``present=0``；⑥ Imagined Memory
    ``writeback == 0``（Real Memory 无写回）。
    """

    prefix = "world_state"

    def capture(self, *, step: int, state: Any, view: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        assertions = view.get("diagnostics", {}).get("producer_assert_ok")
        record: dict[str, Any] = {
            "step": int(step),
            "fields": {name: _clone(value) for name, value in view["fields"].items()},
            "sources": {name: dict(counts) for name, counts in view["sources"].items()},
            "present": {name: _clone(value) for name, value in view["present"].items()},
            "producers": dict(view["producers"]),
            "source_codes": _clone(view.get("source_codes")),
            "producer_assert_ok": bool(assertions),
            "ego_pose": _clone(state.ego_pose),
            "ego_speed": _clone(state.ego_speed),
            "progress_s": _clone(state.progress_s),
            "route_length": _clone(state.route_length),
            "od_anchor": _clone(state.od_anchor),
            "od_valid": _clone(state.od_valid),
            "ld_anchor": _clone(state.ld_anchor),
            "ld_valid": _clone(state.ld_valid),
            "od_ego_k": _clone(view.get("diagnostics", {}).get("od_ego_k")),
            "ld_ego_k": _clone(view.get("diagnostics", {}).get("ld_ego_k")),
            "lateral_offset": _clone(view.get("diagnostics", {}).get("lateral_offset")),
            "lane_width": _clone(view.get("diagnostics", {}).get("lane_width")),
            "long_exog": _clone(view.get("long_exog")),
            "long_present": _clone(view.get("long_present")),
            "memory": None
            if getattr(state, "memory", None) is None
            else {
                "summary": state.memory.summary(),
                "writeback": int(state.memory.writeback),
            },
        }
        return self._append(record)

    def check(self) -> dict[str, Any]:
        from net.world_state import (
            FIELD_PRODUCERS,
            HELD_ALLOWED,
            POLICY_STATE_FIELDS,
            SOURCE_CODES,
            SOURCES,
        )

        violations: list[str] = []
        steps: list[int] = []
        producer_ok = True
        fields_ok = True
        source_ok = True
        held_ok = True
        missing_ok = True
        writeback_ok = True
        expected = tuple(POLICY_STATE_FIELDS)
        for record in self.records:
            step = int(record.get("step", -1))
            steps.append(step)
            fields = record.get("fields") or {}
            if tuple(fields.keys()) != expected:
                fields_ok = False
                violations.append(f"step{step}: 字段集/列序漂移 {tuple(fields.keys())}")
            producers = record.get("producers") or {}
            if producers != dict(FIELD_PRODUCERS):
                producer_ok = False
                violations.append(f"step{step}: producer 表 ≠ FIELD_PRODUCERS")
            for name, counts in (record.get("sources") or {}).items():
                for source in counts:
                    if source not in SOURCES:
                        source_ok = False
                        violations.append(f"step{step}: 字段 {name} 未知 source {source!r}")
                    if source == "held" and name not in HELD_ALLOWED:
                        held_ok = False
                        violations.append(f"step{step}: 字段 {name} 违规 held（白名单 {HELD_ALLOWED}）")
                    if source == "missing":
                        present = (record.get("present") or {}).get(name)
                        codes = record.get("source_codes")
                        if torch.is_tensor(present) and torch.is_tensor(codes) and name in expected:
                            # 逐行断言（同字段混源合法：该行 missing ⇒ 该行 present 必须 0）
                            column = codes[:, expected.index(name)] == SOURCE_CODES["missing"]
                            if column.shape[0] == present.shape[0] and bool((present[column] > 0.5).any()):
                                missing_ok = False
                                violations.append(f"step{step}: 字段 {name} source=missing 但 present>0")
                        elif torch.is_tensor(present) and bool((present > 0.5).any()):
                            missing_ok = False
                            violations.append(f"step{step}: 字段 {name} source=missing 但 present>0")
            memory = record.get("memory")
            if isinstance(memory, Mapping) and int(memory.get("writeback", 0)) != 0:
                writeback_ok = False
                violations.append(f"step{step}: Imagined Memory writeback != 0")
            if record.get("producer_assert_ok") is not True:
                producer_ok = False
                violations.append(f"step{step}: readout producer 断言未通过")
        return {
            "checks": {
                "fields_complete": bool(fields_ok),
                "producer_unique": bool(producer_ok),
                "sources_legal": bool(source_ok),
                "held_whitelist": bool(held_ok),
                "missing_implies_absent": bool(missing_ok),
                "imagined_writeback_zero": bool(writeback_ok),
            },
            "violations": violations,
            "steps": steps,
            "source_codes_legal": source_ok,
        }

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        base.update(self.check())
        base["field_producers"] = dict(self.records[0].get("producers") or {}) if self.records else {}
        return base


# --------------------------------------------------------------------------- #
# ④ Memory
# --------------------------------------------------------------------------- #
class MemoryHooks(_HookBase):
    """④ Memory：逐帧 mask / slot identity / attention 权重 / 历史帧贡献 / 全 mask 输出 / 历史顺序。"""

    prefix = "memory"
    #: 编码期归口张量（``MemBank`` 逐帧掩码与身份）
    ENCODE_KEYS = (
        "ego_mask",
        "ego_valid",
        "od_mask",
        "od_valid",
        "od_id",
        "od_presence",
        "ld_mask",
        "ld_valid",
        "others_mask",
        "others_valid",
    )
    #: 注意力记录键（``TemporalAttention`` 内部张量）
    ATTN_KEYS = ("x", "weights", "ok", "valid", "contribution", "pooled", "output", "all_masked")

    def __init__(self, *, enabled: bool = False, max_records: int = 0):
        super().__init__(enabled=enabled, max_records=max_records)
        self.attn: list[dict[str, Any]] = []

    def clear(self) -> None:
        super().clear()
        self.attn.clear()

    def capture_encode(self, mem: Any) -> dict[str, Any] | None:
        """编码期：逐帧 mask + slot identity + presence + 历史顺序（``age = T-1-pos``）。"""
        if not self.enabled:
            return None
        record: dict[str, Any] = {"frames": int(mem.frames)}
        for key in self.ENCODE_KEYS:
            value = getattr(mem, key, None)
            if torch.is_tensor(value):
                record[key] = _clone(value)
        order = torch.arange(int(mem.frames) - 1, -1, -1, dtype=torch.long)
        record["age_order"] = order  # 历史顺序：index 0 = 最老帧（age = frames-1）
        return self._append(record)

    def capture_attention(
        self,
        *,
        name: str,
        x: Tensor,
        weights: Tensor,
        ok: Tensor,
        pooled: Tensor,
        output: Tensor,
        contribution: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> dict[str, Any] | None:
        """单模态注意力：权重 / mask（ok）/ 历史帧贡献 / 全 mask 输出。

        - ``contribution`` = ``α ⊙ V(x)``（调用方 = ``net/temporal.py`` 的 ``value`` 层输出；
          即"历史帧贡献"逐帧逐槽张量）。
        - ``全 mask 输出`` = 该 (batch, slot) 在窗口内无任何有效帧（``ok`` 全 False）时的池化
          输出（= ``out`` 层的 bias；见 ``net/temporal.py`` 模块 docstring 第 19 行）。
        """
        if not self.enabled:
            return None
        all_masked = ~ok.any(dim=1)  # (B,N)：整列无有效帧
        record = {
            "name": str(name),
            "x": _clone(x),
            "weights": _clone(weights),
            "ok": _clone(ok),
            "valid": _clone(valid),
            "contribution": _clone(contribution),
            "pooled": _clone(pooled),
            "output": _clone(output),
            "all_masked": _clone(all_masked),
        }
        self.capture_calls += 1
        if self.max_records > 0 and len(self.attn) >= self.max_records:
            self.skipped_records += 1
            return None
        self.attn.append(record)
        return record

    def tensors(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for index, record in enumerate(self.records):
            self._flatten(out, f"{self.prefix}/encode{index:02d}", record)
        for index, record in enumerate(self.attn):
            path = f"{self.prefix}/attn{index:02d}/{record['name']}"
            self._flatten(out, path, {key: record[key] for key in self.ATTN_KEYS})
        return out

    def summary(self) -> dict[str, Any]:
        base = super().summary()
        attn_summary: list[dict[str, Any]] = []
        for record in self.attn:
            weights = record.get("weights")
            all_masked = record.get("all_masked")
            attn_summary.append(
                {
                    "name": str(record["name"]),
                    "weights_shape": None if weights is None else list(weights.shape),
                    "all_masked_count": None if all_masked is None else int(all_masked.sum()),
                }
            )
        base.update(
            {
                "attn_records": len(self.attn),
                "attn": attn_summary,
                "encode_records": [
                    {
                        "frames": int(record.get("frames", -1)),
                        "od_live_now": int((record["od_mask"][:, -1] > 0.5).sum())
                        if torch.is_tensor(record.get("od_mask"))
                        else None,
                    }
                    for record in self.records
                ],
            }
        )
        return base


# --------------------------------------------------------------------------- #
# ⑤ 六步调用链
# --------------------------------------------------------------------------- #
class ChainHooks(_HookBase):
    """⑤ 六步调用链：调用计数 / 同实例 / ``a_k ↔ z_k`` linkage / ST-GNN 单列 / 合规 flags。"""

    prefix = "chain"

    def __init__(
        self,
        *,
        enabled: bool = False,
        expected_policy_calls: int = EXPECTED_POLICY_CALLS,
        expected_wm_advances: int = EXPECTED_WM_ADVANCES,
        expected_st_gnn_calls: int | None = None,
    ):
        super().__init__(enabled=enabled, max_records=0)
        self.expected_policy_calls = int(expected_policy_calls)
        self.expected_wm_advances = int(expected_wm_advances)
        #: ST-GNN **总调用**期望（单列；默认 = rollout_steps，由模型 set_probes 时对齐）
        self.expected_st_gnn_calls = None if expected_st_gnn_calls is None else int(expected_st_gnn_calls)
        self.policy_steps: list[int] = []
        self.policy_instance_ids: list[int] = []
        self.policy_z: list[dict[str, Tensor | None]] = []
        self.st_gnn_total_calls = 0
        self.st_gnn_steps: list[int] = []
        self.wm_advances = 0
        self.oneshot_used: bool | None = None
        self.plan_override_used: bool | None = None
        self.plan_from_aux: float | None = None
        self.plan0_equals_action_mu: bool | None = None
        self._z_t0: dict[str, Tensor | None] | None = None
        self._z_pred: dict[str, Tensor | None] | None = None

    def clear(self) -> None:
        super().clear()
        self.policy_steps.clear()
        self.policy_instance_ids.clear()
        self.policy_z.clear()
        self.st_gnn_total_calls = 0
        self.st_gnn_steps.clear()
        self.wm_advances = 0
        self.oneshot_used = None
        self.plan_override_used = None
        self.plan_from_aux = None
        self.plan0_equals_action_mu = None
        self._z_t0 = None
        self._z_pred = None

    def note_z_t0(self, z: Mapping[str, Tensor]) -> None:
        """t0 编码状态（step0 调用读到的 z；linkage 期望值）。"""
        if not self.enabled:
            return
        self._z_t0 = {key: _clone(z.get(key)) for key in ("z_ego", "z_od", "z_ld")}

    def note_z_pred(self, z: Mapping[str, Tensor]) -> None:
        """rollout 逐步堆叠预测 latent（``z_*_pred``，``(B,T,H)``；linkage 期望值）。"""
        if not self.enabled:
            return
        self._z_pred = {key: _clone(z.get(key)) for key in ("z_ego", "z_od", "z_ld")}

    # ------------------------------------------------------------------ 采集
    def on_st_gnn_call(self, step: int) -> None:
        """ST-GNN 实际单步调用（总调用单列，含不推进决策的最后一步）。"""
        if not self.enabled:
            return
        self.st_gnn_total_calls += 1
        self.st_gnn_steps.append(int(step))

    def on_policy_call(self, *, step: int, instance: Any, z: Mapping[str, Tensor] | None = None) -> None:
        """结构化头调用（instance = ``id(self.path_head)``；z = 该次调用读到的状态 latent）。"""
        if not self.enabled:
            return
        self.policy_steps.append(int(step))
        self.policy_instance_ids.append(int(instance))
        self.policy_z.append(
            {
                "z_ego": _clone(None if z is None else z.get("z_ego")),
                "z_od": _clone(None if z is None else z.get("z_od")),
                "z_ld": _clone(None if z is None else z.get("z_ld")),
            }
        )
        self.capture_calls += 1

    def on_wm_advance(self) -> None:
        """WM 决策推进（latent 自回归一步后重算动作）。"""
        if not self.enabled:
            return
        self.wm_advances += 1

    def note_flags(
        self,
        *,
        oneshot: bool | None = None,
        plan_override: bool | None = None,
        plan_from_aux: float | None = None,
        plan0_equals_action_mu: bool | None = None,
    ) -> None:
        if not self.enabled:
            return
        if oneshot is not None:
            self.oneshot_used = bool(oneshot)
        if plan_override is not None:
            self.plan_override_used = bool(plan_override)
        if plan_from_aux is not None:
            self.plan_from_aux = float(plan_from_aux)
        if plan0_equals_action_mu is not None:
            self.plan0_equals_action_mu = bool(plan0_equals_action_mu)

    # ------------------------------------------------------------------ 断言
    def check(
        self,
        *,
        z_t0: Mapping[str, Tensor] | None = None,
        z_pred: Mapping[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        """⑤ 断言：返回 ``{"checks": {...bool}, "violations": [...], ...}``（machine-readable）。

        ``z_t0`` = t0 编码状态（step0 调用读到的 z）；``z_pred`` = rollout 逐步堆叠预测
        latent（``z_ego_pred/z_od_pred/z_ld_pred``，形状 ``(B,T,H)``）。linkage 断言：
        step0 调用 == ``z_t0``；step k（1..5）调用 == ``z_pred[:, k-1]``（逐位 ``torch.equal``）。
        缺省参数时用 :meth:`note_z_t0` / :meth:`note_z_pred` 的采集值。
        """
        if z_t0 is None:
            z_t0 = self._z_t0
        if z_pred is None:
            z_pred = self._z_pred
        checks: dict[str, bool] = {}
        checks["policy_call_count_is_six"] = (
            len(self.policy_steps) == self.expected_policy_calls
            and self.policy_steps == list(range(self.expected_policy_calls))
        )
        checks["policy_instance_ids_are_identical"] = (
            len(set(self.policy_instance_ids)) == 1 and len(self.policy_instance_ids) >= 1
        )
        checks["world_model_advances_five"] = self.wm_advances == self.expected_wm_advances
        if self.expected_st_gnn_calls is not None:
            checks["st_gnn_total_calls_matches_rollout_steps"] = (
                self.st_gnn_total_calls == self.expected_st_gnn_calls
            )
        checks["auxiliary_not_in_lqr"] = self.plan0_equals_action_mu is True
        checks["no_plan_override"] = self.plan_override_used is not True
        checks["no_oneshot_in_production"] = self.oneshot_used is not True
        linkage = self._check_z_linkage(z_t0=z_t0, z_pred=z_pred)
        if linkage is not None:
            checks["each_action_reads_corresponding_z_k"] = bool(linkage["ok"])
        violations = sorted(name for name, ok in checks.items() if not ok)
        return {
            "checks": checks,
            "violations": violations,
            "policy_calls": len(self.policy_steps),
            "policy_steps": list(self.policy_steps),
            "policy_instance_ids": list(self.policy_instance_ids),
            "wm_advances": int(self.wm_advances),
            "st_gnn_total_calls": int(self.st_gnn_total_calls),
            "st_gnn_steps": list(self.st_gnn_steps),
            "oneshot_used": self.oneshot_used,
            "plan_override_used": self.plan_override_used,
            "plan_from_aux": self.plan_from_aux,
            "plan0_equals_action_mu": self.plan0_equals_action_mu,
            "z_linkage": linkage,
        }

    def _check_z_linkage(
        self,
        *,
        z_t0: Mapping[str, Tensor] | None,
        z_pred: Mapping[str, Tensor] | None,
    ) -> dict[str, Any] | None:
        if z_t0 is None and z_pred is None:
            return None
        if not self.policy_z:
            return {"ok": False, "reason": "no policy calls recorded"}

        def _eq(record_key: str, step: int, stack: Mapping[str, Tensor] | None, at: int | None) -> dict[str, Any]:
            actual = self.policy_z[step].get(record_key)
            if actual is None:
                return {"available": False}
            if stack is None or stack.get(record_key) is None:
                return {"available": False}
            expected = stack[record_key]
            if at is not None:
                expected = expected[:, int(at)]
            if tuple(expected.shape) != tuple(actual.shape):
                return {"available": True, "equal": False, "reason": "shape mismatch"}
            return {"available": True, "equal": bool(torch.equal(expected, actual))}

        detail: dict[str, list[dict[str, Any]]] = {}
        ok = True
        for step in self.policy_steps:
            if step == 0:
                entry = {key: _eq(key, step, z_t0, None) for key in ("z_ego", "z_od", "z_ld")}
            else:
                entry = {key: _eq(key, step, z_pred, step - 1) for key in ("z_ego", "z_od", "z_ld")}
            detail[f"step{step}"] = entry
            for item in entry.values():
                if item.get("available") and item.get("equal") is not True:
                    ok = False
        return {"ok": bool(ok), "detail": detail}


# --------------------------------------------------------------------------- #
# 统一出口
# --------------------------------------------------------------------------- #
class ProbeRecorder:
    """①..⑥ 统一容器 + 机器可读出口（张量字典 + JSON 摘要；**默认全关**）。

    用法::

        probes = ProbeRecorder(enabled=True)
        model.set_probes(probes)
        out = model(obs)                      # 只读采集，不改输出契约
        probes.write(run_dir)                 # probe_tensors.pt + probe_summary.json

    ⑦（梯度账本）与 ⑥（越界/碰撞扫描）不接入前向：分别用 :func:`grad_ledger` /
    :func:`first_boundary_breach`/``first_od_contact`` 在探针脚本侧调用。
    """

    def __init__(self, *, enabled: bool = False, max_records: int = 0, state_source_label: str = "predicted"):
        self.enabled = bool(enabled)
        #: rollout 步（k>=1）的 ① 来源标签（真值替换实验由脚本改成 ``substituted``）
        self.state_source_label = str(state_source_label)
        if self.state_source_label not in STATE_SOURCES:
            raise ValueError(f"state_source_label 必须 ∈ {STATE_SOURCES}")
        self.state = StateSourceHooks(enabled=self.enabled, max_records=max_records)
        self.policy = PolicyHooks(enabled=self.enabled, max_records=max_records)
        self.wm = WorldModelHooks(enabled=self.enabled, max_records=max_records)
        self.world_state = WorldStateHooks(enabled=self.enabled, max_records=max_records)
        self.memory = MemoryHooks(enabled=self.enabled, max_records=max_records)
        self.chain = ChainHooks(enabled=self.enabled)

    # ------------------------------------------------------------------ 工具
    def set_enabled(self, enabled: bool) -> "ProbeRecorder":
        self.enabled = bool(enabled)
        for hook in (self.state, self.policy, self.wm, self.world_state, self.memory, self.chain):
            hook.enabled = self.enabled
        return self

    def clear(self) -> None:
        for hook in (self.state, self.policy, self.wm, self.world_state, self.memory, self.chain):
            hook.clear()

    def tensors(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for hook in (self.state, self.policy, self.wm, self.world_state, self.memory):
            out.update(hook.tensors())
        return out

    def summary(self, *, chain_check: dict[str, Any] | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {
            "probe_schema_version": 1,
            "enabled": bool(self.enabled),
            "state_source_label": self.state_source_label,
            "state": self.state.summary(),
            "policy": self.policy.summary(),
            "wm": self.wm.summary(),
            "world_state": self.world_state.summary(),
            "memory": self.memory.summary(),
            "chain": self.chain.check() if chain_check is None else dict(chain_check),
        }
        return out

    def write(self, out_dir: str | Path, *, stem: str = "probe",
              chain_check: dict[str, Any] | None = None) -> dict[str, str]:
        """统一出口：``<out_dir>/<stem>_tensors.pt`` + ``<out_dir>/<stem>_summary.json``。"""
        root = Path(out_dir)
        root.mkdir(parents=True, exist_ok=True)
        tensors_path = root / f"{stem}_tensors.pt"
        summary_path = root / f"{stem}_summary.json"
        torch.save(self.tensors(), tensors_path)
        summary_path.write_text(
            json.dumps(self.summary(chain_check=chain_check), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {"tensors": str(tensors_path), "summary": str(summary_path)}


# --------------------------------------------------------------------------- #
# ⑥ 连续路径与控制：六步 plan → 连续 footprint → 首次越界/碰撞点
# --------------------------------------------------------------------------- #
def interpolate_plan_poses(plan: Tensor, *, dt: float = 0.5, hz: float = 10.0) -> Tensor:
    """``plan (B,6,2)``（t0 帧 ``(ds, dθ)``）→ 连续位姿 ``(B,N,3)``（与 ``net/model.py``
    的 ``interpolate_actions`` 同口径；lazy import 避免循环依赖）。"""
    from net.model import interpolate_actions

    return interpolate_actions(plan, dt=float(dt), hz=float(hz))


def _pose_corners(x: Tensor, y: Tensor, theta: Tensor, length: float, width: float) -> Tensor:
    """位姿 → 4 角点（``(...,4,2)``；t0 帧，x 前 / y 左）。"""
    half_l, half_w = 0.5 * float(length), 0.5 * float(width)
    base = x.new_tensor([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot = torch.stack(
        [
            torch.stack([cos_t, -sin_t], dim=-1),
            torch.stack([sin_t, cos_t], dim=-1),
        ],
        dim=-2,
    )  # (...,2,2)
    corners = base.to(x.dtype).expand(*x.shape, 4, 2) @ rot.transpose(-1, -2)
    return corners + torch.stack([x, y], dim=-1).unsqueeze(-2)


def _obb_overlap(pose_a: Sequence[float], dims_a: tuple[float, float],
                 pose_b: Sequence[float], dims_b: tuple[float, float]) -> bool:
    """2D OBB 重叠（SAT；4 条分离轴）。"""
    ax, ay, ath = float(pose_a[0]), float(pose_a[1]), float(pose_a[2])
    bx, by, bth = float(pose_b[0]), float(pose_b[1]), float(pose_b[2])
    corners_a = _pose_corners(
        torch.tensor([ax]), torch.tensor([ay]), torch.tensor([ath]), dims_a[0], dims_a[1]
    )[0]
    corners_b = _pose_corners(
        torch.tensor([bx]), torch.tensor([by]), torch.tensor([bth]), dims_b[0], dims_b[1]
    )[0]
    axes = []
    for theta in (ath, ath + math.pi / 2.0, bth, bth + math.pi / 2.0):
        axes.append((math.cos(theta), math.sin(theta)))
    for ax_x, ax_y in axes:
        proj_a = [corners_a[i][0] * ax_x + corners_a[i][1] * ax_y for i in range(4)]
        proj_b = [corners_b[i][0] * ax_x + corners_b[i][1] * ax_y for i in range(4)]
        if max(proj_a) < min(proj_b) or max(proj_b) < min(proj_a):
            return False
    return True


def _project_to_polyline(px: float, py: float, poly: Tensor) -> tuple[float, float, float] | None:
    """点到折线最近投影：返回 ``(signed_lateral, s_m, tangent_angle)``（左正；段法向）。"""
    if poly is None or int(poly.shape[0]) < 2:
        return None
    pts = poly if torch.is_tensor(poly) else torch.as_tensor(poly)
    pts = pts.float()
    best: tuple[float, float, float] | None = None
    best_dist = math.inf
    s_acc = 0.0
    for i in range(int(pts.shape[0]) - 1):
        x0, y0 = float(pts[i, 0]), float(pts[i, 1])
        x1, y1 = float(pts[i + 1, 0]), float(pts[i + 1, 1])
        dx, dy = x1 - x0, y1 - y0
        seg = math.hypot(dx, dy)
        if seg <= 1e-9:
            continue
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / (seg * seg)))
        cx, cy = x0 + t * dx, y0 + t * dy
        dist = math.hypot(px - cx, py - cy)
        if dist < best_dist:
            best_dist = dist
            theta = math.atan2(dy, dx)
            # 左法向 = (−sinθ, cosθ)；signed lateral = (p − c)·n
            lateral = (px - cx) * (-math.sin(theta)) + (py - cy) * math.cos(theta)
            best = (float(lateral), float(s_acc + t * seg), float(theta))
        s_acc += seg
    return best


def first_boundary_breach(
    *,
    plan: Tensor,
    anchor_xy: Tensor,
    anchor_mask: Tensor | None = None,
    margin_left: Tensor | None = None,
    margin_right: Tensor | None = None,
    ego_length: float,
    ego_width: float,
    dt: float = 0.5,
    hz: float = 10.0,
) -> dict[str, Any]:
    """六步 plan 连续 footprint 相对车道边界的**首次越界**（t0 帧；机器可读）。

    边界口径：``W = left + right``（t0 静态路面容宽）⇒ ``|lateral_corner| > W/2`` 记为越界
    （仅目标车道中心线 + t0 margin；完整路面/静态残余障碍由控制侧遥测复核，见模块 docstring）。
    逐 batch 独立返回；缺锚/缺 margin ⇒ ``available=False``（显式 n/a，不静默 0）。
    """
    poses = interpolate_plan_poses(plan, dt=dt, hz=hz)
    batch = int(poses.shape[0])
    out: list[dict[str, Any]] = []
    for b in range(batch):
        entry: dict[str, Any] = {"available": False, "first": None, "breach_n": 0, "min_margin_m": None}
        if margin_left is None or margin_right is None:
            out.append(entry)
            continue
        poly = None
        if anchor_xy is not None and b < int(anchor_xy.shape[0]):
            poly = anchor_xy[b]
            if anchor_mask is not None:
                keep = anchor_mask[b] > 0.5
                poly = poly[keep]
        if poly is None or int(poly.shape[0]) < 2:
            out.append(entry)
            continue
        half_w = 0.5 * float(margin_left[b].reshape(-1)[0] + margin_right[b].reshape(-1)[0])
        if half_w <= 0.0:
            out.append(entry)
            continue
        entry["available"] = True
        min_margin = math.inf
        first: dict[str, Any] | None = None
        breach_n = 0
        for i in range(int(poses.shape[1])):
            x, y, theta = poses[b, i]
            corners = _pose_corners(
                x.reshape(1), y.reshape(1), theta.reshape(1), float(ego_length), float(ego_width)
            )[0]
            worst = 0.0
            worst_proj = None
            bad_here = False
            for c in range(4):
                projection = _project_to_polyline(float(corners[c, 0]), float(corners[c, 1]), poly)
                if projection is None:
                    continue
                lateral, s_m, _ = projection
                if abs(lateral) > worst:
                    worst = abs(lateral)
                    worst_proj = (s_m, float(lateral))
                if abs(lateral) > half_w:
                    bad_here = True
            margin = half_w - worst
            min_margin = min(min_margin, margin)
            if bad_here:
                breach_n += 1
                if first is None:
                    first = {
                        "index": int(i),
                        "s_m": None if worst_proj is None else float(worst_proj[0]),
                        "margin_m": float(margin),
                        "x": float(x),
                        "y": float(y),
                        "theta": float(theta),
                    }
        entry.update(
            {
                "first": first,
                "breach_n": int(breach_n),
                "min_margin_m": None if not math.isfinite(min_margin) else float(min_margin),
            }
        )
        out.append(entry)
    return {"available": any(item["available"] for item in out), "batch": out}


def first_od_contact(
    *,
    plan: Tensor,
    od_now: Tensor,
    od_live: Tensor | None = None,
    ego_length: float,
    ego_width: float,
    dt: float = 0.5,
    hz: float = 10.0,
) -> dict[str, Any]:
    """六步 plan 连续 footprint 相对 OD 盒（t0 位姿）的**首次碰撞**（t0 帧；机器可读）。

    OD 槽语义 ``[dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]``（相对速度不影响 t0 检测；
    动态推进由控制侧 TTC 遥测复核）。
    """
    poses = interpolate_plan_poses(plan, dt=dt, hz=hz)
    batch = int(poses.shape[0])
    out: list[dict[str, Any]] = []
    for b in range(batch):
        entry: dict[str, Any] = {"available": False, "first": None, "contact_n": 0}
        if od_now is None or b >= int(od_now.shape[0]):
            out.append(entry)
            continue
        entry["available"] = True
        boxes: list[tuple[int, tuple[float, float, float], tuple[float, float]]] = []
        slots = int(od_now.shape[1])
        for s in range(slots):
            if od_live is not None and b < int(od_live.shape[0]) and float(od_live[b, s]) <= 0.5:
                continue
            x = float(od_now[b, s, 0])
            y = float(od_now[b, s, 1])
            cos_t = float(od_now[b, s, 4])
            sin_t = float(od_now[b, s, 5])
            length = float(od_now[b, s, 6])
            width = float(od_now[b, s, 7])
            if length <= 0.0 or width <= 0.0:
                continue
            boxes.append((s, (x, y, math.atan2(sin_t, cos_t)), (length, width)))
        first: dict[str, Any] | None = None
        contact_n = 0
        for i in range(int(poses.shape[1])):
            x, y, theta = poses[b, i]
            pose = (float(x), float(y), float(theta))
            for slot, box_pose, dims in boxes:
                if _obb_overlap(pose, (float(ego_length), float(ego_width)), box_pose, dims):
                    contact_n += 1
                    if first is None:
                        first = {
                            "index": int(i),
                            "slot": int(slot),
                            "od_id": None,
                            "x": float(x),
                            "y": float(y),
                            "theta": float(theta),
                        }
                    break
        entry.update({"first": first, "contact_n": int(contact_n)})
        out.append(entry)
    return {"available": any(item["available"] for item in out), "batch": out}


# --------------------------------------------------------------------------- #
# ⑦ 参数与梯度账本
# --------------------------------------------------------------------------- #
def grad_ledger(
    model: torch.nn.Module,
    losses: Mapping[str, Tensor],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: bool = False,
    strict: bool = True,
) -> dict[str, Any]:
    """⑦ 参数与梯度账本：逐 loss（如第 1/3/6 步）**单独**反向 + 参数/优化器/ckpt 读数。

    - 每个 ``losses[key]`` 独立 ``torch.autograd.grad``（``retain_graph=True``）⇒ 顶层
      “每步 loss 梯度可达性”是**逐 loss** 读数，无 3-loss 合并掩盖（缺项 = 显式 None）。
    - ``optimizer`` 给出时记录 group/lr 归属；``step=True`` 时在**调用方副本**上执行一次
      ``zero_grad → Σloss.backward() → step()``，记录 grad/update norm 与 ``state_dict_hash``
      前后差异（ckpt 是否变化）。
    - 只读账本：不写回参数（``step=False``，默认）；不消费 RNG。
    - ``strict=True``（默认）：所有 step 的高斯全空（``nonzero_grad_params == 0``）⇒
      ``ValueError`` fail-fast（防"losses 与 model 不同源"的静默全 0 读账）。
    """
    params = [(name, parameter) for name, parameter in model.named_parameters()]
    names = [name for name, _ in params]
    param_tensors = [parameter for _, parameter in params]
    optimizer_groups: dict[str, Any] = {}
    if optimizer is not None:
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                for name, candidate in params:
                    if candidate is parameter:
                        optimizer_groups[name] = {
                            "group": int(group_index),
                            "lr": float(group.get("lr", 0.0)),
                        }
                        break

    ledger: dict[str, Any] = {
        "params": [
            {
                "name": name,
                "module": ".".join(name.split(".")[:2]),
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "optimizer": optimizer_groups.get(name),
            }
            for name, parameter in params
        ],
        "steps": {},
        "update": None,
        "ckpt": {"hash_before": state_dict_hash(model), "hash_after": None, "changed": None},
    }

    for key, loss in losses.items():
        if not torch.is_tensor(loss):
            raise TypeError(f"losses[{key!r}] 必须是 Tensor，收到 {type(loss).__name__}")
        if not bool(loss.requires_grad):
            raise ValueError(
                f"[D1 grad_ledger] losses[{key!r}] 无计算图（requires_grad=False）——"
                "拒绝静默全 0（detach 后不得进账本）"
            )
        grads = torch.autograd.grad(
            loss, param_tensors, retain_graph=True, allow_unused=True
        )
        per_param: dict[str, float] = {}
        for name, grad in zip(names, grads):
            if grad is None:
                continue
            norm = float(grad.detach().norm())
            if norm > 0.0:
                per_param[name] = norm
        touched = sorted(per_param.keys())
        ledger["steps"][key] = {
            "loss": float(loss.detach()),
            "requires_grad": bool(loss.requires_grad),
            "nonzero_grad_params": len(touched),
            "grad_norm_total": float(sum(value * value for value in per_param.values()) ** 0.5)
            if per_param
            else 0.0,
            "touched_prefixes": _sorted_prefixes(touched),
            "grad_norms": per_param,
        }

    if strict and all(
        int(entry["nonzero_grad_params"]) == 0 for entry in ledger["steps"].values()
    ):
        raise ValueError(
            "[D1 grad_ledger] 所有 step 的梯度全空——losses 与 model 不同源？"
            "（账本拒绝静默全 0）"
        )

    if optimizer is not None and step:
        total = None
        for loss in losses.values():
            total = loss if total is None else total + loss
        optimizer.zero_grad(set_to_none=True)
        assert total is not None
        total.backward()
        grad_norm_total = 0.0
        for parameter in param_tensors:
            if parameter.grad is not None:
                grad_norm_total += float(parameter.grad.detach().norm()) ** 2
        before = [parameter.detach().clone() for parameter in param_tensors]
        optimizer.step()
        update_norms: dict[str, float] = {}
        for name, old, parameter in zip(names, before, param_tensors):
            delta = parameter.detach() - old
            update_norms[name] = float(delta.norm())
        ledger["update"] = {
            "grad_norm_total": float(grad_norm_total ** 0.5),
            "update_norms": {name: value for name, value in update_norms.items() if value > 0.0},
            "update_norm_total": float(sum(value * value for value in update_norms.values()) ** 0.5),
        }
        ledger["ckpt"]["hash_after"] = state_dict_hash(model)
        ledger["ckpt"]["changed"] = ledger["ckpt"]["hash_after"] != ledger["ckpt"]["hash_before"]

    return ledger


# --------------------------------------------------------------------------- #
# R39-A2：生产 detach 登记表 + AST 扫描（CI 禁未登记的新增 detach）
# --------------------------------------------------------------------------- #
#: 登记表字段（逐 detach：tensor / 调用位置 file:line / stage / 预期阻断对象 / 实际可达范围）：
#: - ``tensor``：代码源文本（= AST 匹配键之一，逐字面匹配；空白已归一）；
#: - ``file`` / ``function``：相对仓库路径 + 限定函数名（AST 匹配键之二/之三）；
#: - ``line``：登记时行号（**informational**：移动不算回归，漂移记录在 check 报告）；
#: - ``stage``：所属阶段；``blocks``：**预期阻断对象**；``reach``：**实际可达范围**
#:   （``a2_reach:*`` 证据 = ``.slim/deepwork/p1_v2/round39/a2.json`` 实测；``static`` =
#:   只读/诊断语义，无参数量梯度对象）。
DETACH_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "D-WM01",
        "tensor": "z_ego_next.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1834,
        "stage": "rollout/state-chain（latent 自回归切断；v8 固定语义）",
        "blocks": "跨步链：t+1..t+6 步输出 → 本步 WM transition 与上游编码（a_{k+1} 不经 z_k 回传 WM/编码）",
        "reach": "实测（a2_reach:z_next_cross_step；hook 直取捕获张量，避开 stack 的 0-grad 假阳性）：step-2 输出对 step-1 z 的 autograd.grad = None（阻断成立；同图恒等对照 = 32.0）；z_*_pred 输出自身可达 st_gnn 本步（grad 126.3）；动作链 loss 对 st_gnn = 空集（另一面佐证）、对动作头可达（1300/25.3）",
        "evidence": "a2_reach:z_next_cross_step",
    },
    {
        "id": "D-WM02",
        "tensor": "z_od_next.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1834,
        "stage": "rollout/state-chain",
        "blocks": "同 D-WM01（OD latent 通道）",
        "reach": "同 D-WM01（三通道同一行同一语义；实测 z 通道代表 + 同表达式三通道同口径）",
        "evidence": "a2_reach:z_next_cross_step",
    },
    {
        "id": "D-WM03",
        "tensor": "z_ld_next.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1834,
        "stage": "rollout/state-chain",
        "blocks": "同 D-WM01（LD latent 通道）",
        "reach": "同 D-WM01（三通道同一行同一语义；实测 z 通道代表 + 同表达式三通道同口径）",
        "evidence": "a2_reach:z_next_cross_step",
    },
    {
        "id": "D-WM04",
        "tensor": "action.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1783,
        "stage": "rollout/wm-action（structured 梯度门 path_head_wm_action_detach=True）",
        "blocks": "WM 输入动作 a_{k-1} → 结构化头（WM 预测 loss 不经动作输入旁路回 policy 头；R36 K1 梯度门）",
        "reach": "实测 A/B（a2_reach:wm_action_detach_ab）：**当前装配（enable_action_ego=False）实测本 detach 不改变任何梯度**（wm_action 无消费者；detach True/False 头 grad 均 =0.0）⇒ 预留守卫；**反事实装配**（set_action_ego(True)+解阻末层）detach=True ⇒ 头 grad=0.0 / detach=False ⇒ 4.71（阻断语义成立、因果实证）；st_gnn 本体始终可达",
        "evidence": "a2_reach:wm_action_detach_ab",
    },
    {
        "id": "D-WM05",
        "tensor": "pose.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1739,
        "stage": "rollout/warp-ego-pose（图位姿 ego_pose_det，warp off/analytic 共用）",
        "blocks": "WM 图位姿 pose_k → 动作/位姿链（ego 累积位姿）；WM 输出不回传动作/策略链",
        "reach": "实测（a2_reach:pose_warp_detach）：pose_k（= ego_pose_det + t0 anchors + t0 速度先验）**requires_grad=False ⇒ 对任何参数无梯度**（两配置同）；同图 traj_xy 链对动作头可达（policy 23549 / structured_action 444）作对照 ⇒ 阻断来自本 detach 而非整链断",
        "evidence": "a2_reach:pose_warp_detach",
    },
    {
        "id": "D-WM06",
        "tensor": "ds.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1724,
        "stage": "rollout/geometry-progress（structured 想象步几何进度 Σds）",
        "blocks": "想象几何 progress_s → 动作 ds 链（几何重算视为确定性外源输入）",
        "reach": "实测（a2_reach:progress_detach，rollout 步头输出 loss）：对 plan/动作链 grad=None、对 st_gnn=空集（阻断成立）；可达集 = 头自身 + 本步 plan_head 链（latent 的 t0 上下文经 encoders 1.65 / mem_encoder 3.61 / semantic_wiring 0.04 可达）",
        "evidence": "a2_reach:progress_detach",
    },
    {
        "id": "D-WM07",
        "tensor": "pose.detach()",
        "file": "net/model.py",
        "function": "DrivingModel._rollout",
        "line": 1877,
        "stage": "rollout/structured-step-geometry（step_geometry 想象位姿输入）",
        "blocks": "结构化头几何（anchor/margin 重算所用想象位姿）→ 位姿链",
        "reach": "实测（a2_reach:step_geometry_detach）：同 D-WM06（同一检查的同一 loss）；几何输入（pose/progress 双 detach）为外源确定性量：对 plan=g None、st_gnn=空集",
        "evidence": "a2_reach:step_geometry_detach",
    },
    {
        "id": "D-NV01",
        "tensor": "_squeeze_pose(ego_pose_world).detach()",
        "file": "net/mem.py",
        "function": "nav_features_from_world",
        "line": 479,
        "stage": "A4/nav-rebuild（世界系 nav 重建）",
        "blocks": "nav 重建 → 世界系位姿输入（外源数据；nav 为上下文条件，不向位姿链回传）",
        "reach": "实测（a2_reach:nav_features_detach_unit）：grad(nav_feats.sum()) 对输入 pose = None；（值上）nav token 只经 encoders.nav 自身权重进入图",
        "evidence": "a2_reach:nav_features_detach_unit",
    },
    {
        "id": "D-NV02",
        "tensor": "_squeeze_route(route_world).detach()",
        "file": "net/mem.py",
        "function": "nav_features_from_world",
        "line": 480,
        "stage": "A4/nav-rebuild",
        "blocks": "nav 重建 → 世界系路线输入",
        "reach": "实测（a2_reach:nav_features_detach_unit）：grad(nav_feats.sum()) 对输入 route = None",
        "evidence": "a2_reach:nav_features_detach_unit",
    },
    {
        "id": "D-NV03",
        "tensor": "mask.detach()",
        "file": "net/mem.py",
        "function": "nav_features_from_world",
        "line": 494,
        "stage": "A4/nav-rebuild（route_world_mask 布尔化）",
        "blocks": "route 有效性掩码的梯度（布尔比较不可微）",
        "reach": "实测（a2_reach:nav_features_detach_unit）：grad 对 mask 输入 = None（掩码仅作 0/1 选择；无参数梯度对象）",
        "evidence": "a2_reach:nav_features_detach_unit",
    },
    {
        "id": "D-NV04",
        "tensor": "mem.others[:, -1, : int(nav_dim)].detach()",
        "file": "net/mem.py",
        "function": "rebuild_nav_from_world",
        "line": 574,
        "stage": "A4/nav-fallback（缺世界系键/整批不可用 ⇒ t0 冻结 nav）",
        "blocks": "回退 nav（t0 冻结上下文）→ mem.others",
        "reach": "实测（a2_reach:nav_fallback_detach_unit）：grad(fallback.sum()) 对 mem.others = None；回退口径 = 只读 t0 冻结值",
        "evidence": "a2_reach:nav_fallback_detach_unit",
    },
    {
        "id": "D-NV05",
        "tensor": "mem.others[:, -1, :NAV_DIM].detach()",
        "file": "net/model.py",
        "function": "rebuild_nav_from_world",
        "line": 439,
        "stage": "A4/nav-fallback（net.model 内联回退 helper；逐行不可用时 where 回退）",
        "blocks": "回退 nav 行 → mem.others",
        "reach": "同 D-NV04（同一语义的 model 侧内联实现；只在部分行不可用时触发）",
        "evidence": "a2_reach:nav_fallback_detach_unit",
    },
    {
        "id": "D-SD01",
        "tensor": "final.detach()",
        "file": "net/model.py",
        "function": "DrivingModel.semantic_action_decomposition",
        "line": 2241,
        "stage": "diagnostics/semantic-action-decomposition（@torch.no_grad 只读监控）",
        "blocks": "监控读出 → 计算图（防御性；方法整体 no_grad）",
        "reach": "实测（a2_reach:semantic_decomposition_nograd）：返回张量 requires_grad 全 False（纯监控，不参与任何训练目标）",
        "evidence": "a2_reach:semantic_decomposition_nograd",
    },
    {
        "id": "D-SD02",
        "tensor": "self.forward(obs, forward_mode=str(forward_mode), semantic_off=True)[\"action_mu\"].detach()",
        "file": "net/model.py",
        "function": "DrivingModel.semantic_action_decomposition",
        "line": 2246,
        "stage": "diagnostics/semantic-action-decomposition（legacy 基底前向读出）",
        "blocks": "legacy（semantic_off）基底动作 → 计算图",
        "reach": "同 D-SD01（no_grad 只读；delta = final − legacy 不参与训练目标）",
        "evidence": "a2_reach:semantic_decomposition_nograd",
    },
    {
        "id": "D-SD03",
        "tensor": "self.beta_context.detach()",
        "file": "net/model.py",
        "function": "DrivingModel.semantic_action_decomposition",
        "line": 2256,
        "stage": "diagnostics/semantic-action-decomposition（β 标量读出）",
        "blocks": "β 标量 → 计算图",
        "reach": "static：float() 标量读出（刷新到 CPU float；无梯度语义）",
        "evidence": "static",
    },
    {
        "id": "D-SD04",
        "tensor": "self.beta_token.detach()",
        "file": "net/model.py",
        "function": "DrivingModel.semantic_action_decomposition",
        "line": 2257,
        "stage": "diagnostics/semantic-action-decomposition（β 标量读出）",
        "blocks": "β 标量 → 计算图",
        "reach": "static：同 D-SD03",
        "evidence": "static",
    },
    {
        "id": "D-SM01",
        "tensor": "self.lane_encoder.normalization.detach()",
        "file": "net/semantic.py",
        "function": "SemanticWiring.normalization_constants",
        "line": 230,
        "stage": "diagnostics/semantic（固定归一化除数读出）",
        "blocks": "归一化 buffer → 计算图",
        "reach": "实测（a2_reach:semantic_norm_readout_unit）：返回 python float 元组（.detach().cpu().tolist()），无张量出口、无梯度语义",
        "evidence": "a2_reach:semantic_norm_readout_unit",
    },
    {
        "id": "D-SP01",
        "tensor": "self.alpha.detach()",
        "file": "net/spatial.py",
        "function": "LdAggregate.forward",
        "line": 234,
        "stage": "diagnostics/attn（LdAggregate 统计读数；ld_mode=attn 系才构造）",
        "blocks": "alpha 标量 → 计算图",
        "reach": "static：float() 统计读出（if self.stats_enabled；不影响前向值）",
        "evidence": "static",
    },
    {
        "id": "D-WH01",
        "tensor": "anchor_pre.detach()",
        "file": "net/ego_warp.py",
        "function": "WarpHooks.capture",
        "line": 400,
        "stage": "R37 warp hooks（默认关；只读诊断）",
        "blocks": "warp ① anchor_pre 快照 → 计算图",
        "reach": "实测（a2_reach:warp_hooks_detach_unit）：record 内张量 requires_grad 全 False（detach().clone() 只读快照）",
        "evidence": "a2_reach:warp_hooks_detach_unit",
    },
    {
        "id": "D-WH02",
        "tensor": "anchor_analytic.detach()",
        "file": "net/ego_warp.py",
        "function": "WarpHooks.capture",
        "line": 401,
        "stage": "R37 warp hooks",
        "blocks": "warp ② anchor_analytic 快照 → 计算图",
        "reach": "同 D-WH01",
        "evidence": "a2_reach:warp_hooks_detach_unit",
    },
    {
        "id": "D-WH03",
        "tensor": "anchor_actual_next.detach()",
        "file": "net/ego_warp.py",
        "function": "WarpHooks.capture",
        "line": 402,
        "stage": "R37 warp hooks",
        "blocks": "warp ③ anchor_actual 快照 → 计算图",
        "reach": "同 D-WH01",
        "evidence": "a2_reach:warp_hooks_detach_unit",
    },
    {
        "id": "D-WH04",
        "tensor": "residual_learned.detach()",
        "file": "net/ego_warp.py",
        "function": "WarpHooks.capture",
        "line": 403,
        "stage": "R37 warp hooks",
        "blocks": "warp ④ learned residual 快照 → 计算图（接口留给 WM residual 训练，本 detach 只影响快照）",
        "reach": "同 D-WH01",
        "evidence": "a2_reach:warp_hooks_detach_unit",
    },
    {
        "id": "D-WH05",
        "tensor": "final.detach()",
        "file": "net/ego_warp.py",
        "function": "WarpHooks.capture",
        "line": 404,
        "stage": "R37 warp hooks",
        "blocks": "warp ⑤ anchor_final 快照 → 计算图",
        "reach": "同 D-WH01",
        "evidence": "a2_reach:warp_hooks_detach_unit",
    },
    {
        "id": "D-PR01",
        "tensor": "value.detach()",
        "file": "net/probes.py",
        "function": "_clone",
        "line": 79,
        "stage": "R38-D1 probes 内部（统一出口快照；默认关/只读）",
        "blocks": "探针快照 → 被观测图（只读副本）",
        "reach": "实测（a2_reach:probe_internal_detach_unit）：micro 检查 requires_grad(输入)=True ⇒ 快照 False；探针默认全关，不触碰生产图",
        "evidence": "a2_reach:probe_internal_detach_unit",
    },
    {
        "id": "D-PR02",
        "tensor": "tensor.detach()",
        "file": "net/probes.py",
        "function": "state_dict_hash",
        "line": 101,
        "stage": "R38-D1 probes 内部（ckpt 内容 sha256）",
        "blocks": "状态字典读出 → 计算图",
        "reach": "static：hash 只读（.detach().cpu().numpy().tobytes()）",
        "evidence": "static",
    },
    {
        "id": "D-PR03",
        "tensor": "value.detach()",
        "file": "net/probes.py",
        "function": "_HookBase._flatten",
        "line": 150,
        "stage": "R38-D1 probes 内部（hook 张量扁平化落盘）",
        "blocks": "hook 记录张量 → 计算图",
        "reach": "同 D-PR01（只读快照）",
        "evidence": "a2_reach:probe_internal_detach_unit",
    },
    {
        "id": "D-PR04",
        "tensor": "grad.detach()",
        "file": "net/probes.py",
        "function": "grad_ledger",
        "line": 1012,
        "stage": "R39-D1 probes 内部（逐 loss autograd.grad 读数）",
        "blocks": "梯度读数 → 计算图（读数即终点）",
        "reach": "static：norm 读数（float），只读账本",
        "evidence": "static",
    },
    {
        "id": "D-PR05",
        "tensor": "loss.detach()",
        "file": "net/probes.py",
        "function": "grad_ledger",
        "line": 1017,
        "stage": "R38-D1 probes 内部（loss 值读数）",
        "blocks": "loss 标量 → 计算图",
        "reach": "static：float 读数",
        "evidence": "static",
    },
    {
        "id": "D-PR06",
        "tensor": "parameter.grad.detach()",
        "file": "net/probes.py",
        "function": "grad_ledger",
        "line": 1045,
        "stage": "R38-D1 probes 内部（update step 后 grad norm 读数）",
        "blocks": "优化器梯度 → 计算图",
        "reach": "static：norm 读数",
        "evidence": "static",
    },
    {
        "id": "D-PR07",
        "tensor": "parameter.detach()",
        "file": "net/probes.py",
        "function": "grad_ledger",
        "line": 1046,
        "stage": "R38-D1 probes 内部（step 前参数快照）",
        "blocks": "参数快照 → 计算图",
        "reach": "static：副本快照（update norm = ‖Δθ‖）",
        "evidence": "static",
    },
    {
        "id": "D-PR08",
        "tensor": "parameter.detach()",
        "file": "net/probes.py",
        "function": "grad_ledger",
        "line": 1050,
        "stage": "R38-D1 probes 内部（step 后 Δθ 读数）",
        "blocks": "参数差 → 计算图",
        "reach": "static：update norm 读数",
        "evidence": "static",
    },
    {
        "id": "D-WS01",
        "tensor": "value.detach()",
        "file": "net/world_state.py",
        "function": "_detach",
        "line": 143,
        "stage": "R39-B1 unified-world-state（想象态非可微边界；单一 detach 写点）",
        "blocks": "已执行动作 ds/dθ、累积位姿 pose、WM 残差 ΔA 与 latent → 跨步想象链/策略头几何输入",
        "reach": "static：`_detach` 只被 `WorldState` 构造与 `analytic_update` 调用（`world_state_api=off` 默认时整条路径不执行）；语义 = 想象世界状态是「已执行动作 + WM 残差」的确定性函数 —— 残差学习走 `net/world_state.residual_loss` one-step 真实下一状态监督（不穿越跨步链），策略头几何输入与 HEAD 的 `step_geometry` 同为外源确定性量",
        "evidence": "static",
    },
    {
        "id": "D-B201",
        "tensor": "value.detach()",
        "file": "net/world_residual.py",
        "function": "save_residual_head",
        "line": 315,
        "stage": "R39-B2 残差头 ckpt 序列化（one-step 训练产物落盘；**非**状态链/学习图边界）",
        "blocks": "头参数 → 磁盘 state_dict",
        "reach": "static：仅 `save_residual_head` 的序列化路径调用（训练 lane 落盘时一次；默认路径/前向不执行）；训练/评测的梯度边界仍为 `world_state.residual_loss` 语义（one-step 真实下一状态监督，D-WS01 不变）",
        "evidence": "static",
    },
)

#: 登记表键 = (file, function, tensor)；`line` 仅为登记时行号（漂移不判失败，记入报告）
DETACH_REGISTRY_FIELDS: tuple[str, ...] = (
    "id",
    "tensor",
    "file",
    "function",
    "line",
    "stage",
    "blocks",
    "reach",
    "evidence",
)


def _detach_qualname_scan(source: str, filename: str) -> list[dict[str, Any]]:
    """单个 Python 源文件的 ``.detach()`` 调用点（AST；含限定函数名与源文本）。"""
    tree = ast.parse(source, filename=filename)
    occurrences: list[dict[str, Any]] = []

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "detach"
                and not child.args
                and not child.keywords
            ):
                text = ast.get_source_segment(source, child) or ""
                occurrences.append(
                    {
                        "file": filename,
                        "function": scope or "<module>",
                        "tensor": " ".join(text.split()),
                        "line": int(child.lineno),
                    }
                )
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, f"{scope}.{child.name}" if scope else child.name)
            else:
                visit(child, scope)

    visit(tree, "")
    return occurrences


def scan_detaches(root: str | Path | None = None, *, net_dir: str = "net") -> list[dict[str, Any]]:
    """扫描 ``<root>/<net_dir>/**/*.py`` 的 ``.detach()`` 调用（AST 口径；忽略注释/字符串）。

    返回按 ``(file, line, tensor)`` 排序的调用点列表；``file`` = 相对 root 的 POSIX 路径。
    """
    base = Path(__file__).resolve().parents[1] if root is None else Path(root)
    target = base / net_dir
    if not target.is_dir():
        raise FileNotFoundError(f"scan_detaches: 目录不存在 {target}")
    occurrences: list[dict[str, Any]] = []
    for path in sorted(target.rglob("*.py")):
        relative = path.relative_to(base).as_posix()
        source = path.read_text(encoding="utf-8")
        occurrences.extend(_detach_qualname_scan(source, relative))
    return sorted(occurrences, key=lambda row: (row["file"], row["line"], row["tensor"]))


def check_detach_registry(
    root: str | Path | None = None, *,
    registry: Sequence[Mapping[str, Any]] | None = None,
    net_dir: str = "net",
) -> dict[str, Any]:
    """CI 守卫：``net/**`` 全部 ``.detach()`` 必须逐项登记（未登记/失配 ⇒ ``ok=False``）。

    - ``unregistered``：代码中存在但登记表缺失的调用点（**新增未登记 detach** ⇒ fail）；
    - ``stale``：登记表存在但代码中已不存在（反向陈旧 ⇒ fail，防账本腐烂）；
    - ``duplicate_registry``：登记表键重复；``invalid_entries``：字段缺失/为空；
    - ``line_drift``：键匹配但行号漂移（informational，不判失败）。
    """
    entries = tuple(DETACH_REGISTRY if registry is None else registry)
    occurrences = scan_detaches(root, net_dir=net_dir)

    def keys(rows: Iterable[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
        return [(str(row["file"]), str(row["function"]), str(row["tensor"])) for row in rows]

    code_keys = keys(occurrences)
    registry_keys = keys(entries)
    code_count: dict[tuple[str, str, str], int] = {}
    for key in code_keys:
        code_count[key] = code_count.get(key, 0) + 1
    registry_count: dict[tuple[str, str, str], int] = {}
    for key in registry_keys:
        registry_count[key] = registry_count.get(key, 0) + 1

    # 多集差：同一 (file, function, tensor) 在代码里出现多次（如同一行 3 个 z_*_next）时，
    # 登记表也必须逐次登记；超出/不足都判 fail（未登记新增 / 账本腐烂）。
    unregistered_counter = dict(registry_count)
    unregistered: list[dict[str, Any]] = []
    for row in occurrences:
        key = (row["file"], row["function"], row["tensor"])
        if unregistered_counter.get(key, 0) <= 0:
            unregistered.append(row)
        else:
            unregistered_counter[key] -= 1
    stale_counter = dict(code_count)
    stale: list[dict[str, Any]] = []
    for entry in entries:
        key = (str(entry["file"]), str(entry["function"]), str(entry["tensor"]))
        if stale_counter.get(key, 0) <= 0:
            stale.append(entry)
        else:
            stale_counter[key] -= 1
    invalid_entries = [
        {"id": entry.get("id"), "missing": [field for field in DETACH_REGISTRY_FIELDS if not entry.get(field)]}
        for entry in entries
        if any(not entry.get(field) for field in DETACH_REGISTRY_FIELDS)
    ]
    registered_lines: dict[tuple[str, str, str], list[int]] = {}
    for entry in entries:
        key = (str(entry["file"]), str(entry["function"]), str(entry["tensor"]))
        registered_lines.setdefault(key, []).append(int(entry["line"]))
    actual_lines: dict[tuple[str, str, str], list[int]] = {}
    for row in occurrences:
        key = (row["file"], row["function"], row["tensor"])
        actual_lines.setdefault(key, []).append(int(row["line"]))
    line_drift = [
        {
            "file": key[0],
            "function": key[1],
            "tensor": key[2],
            "registered_lines": sorted(registered_lines[key]),
            "actual_lines": sorted(actual_lines[key]),
        }
        for key in sorted(registered_lines)
        if key in actual_lines and sorted(registered_lines[key]) != sorted(actual_lines[key])
    ]
    return {
        "ok": not unregistered and not stale and not invalid_entries,
        "n_code_detaches": len(occurrences),
        "n_registry_entries": len(entries),
        "unregistered": unregistered,
        "stale": stale,
        "invalid_entries": invalid_entries,
        "line_drift": line_drift,
    }
