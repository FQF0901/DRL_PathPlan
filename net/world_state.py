"""R39-B1：统一 imagined **WorldState → PolicyStateView**（默认关；零训练接口）。

规格出处
--------
- `docs/chatgpt5p6sol_review.md` 主方案 ``647f909`` §二 R39-B（九步循环：Policy 读
  ``PolicyStateView_k`` → 动作 → 解析运动学 → LD ego-motion → OD 对象先验 + ego-motion →
  WM 学习残差 → **统一 readout 重算全部物理量** → Imagined Memory → 同一 Policy 决策）；
  补充裁决 ``fe5de55``（臂 B0–B4 与「B2 必须优于 B1」）；
- ``docs/deepseekv4p1_argue.md`` 当期快照 §1/§2（A3 教训：纯自洽 speed 正反馈发散
  5.77→6.33 m/s ⇒ 阻尼 + 上限；残差复用 ``st_gnn`` 解码头 = 零新增参数）。

统一公式（每步）::

    A_{k+1} = AnalyticUpdate(A_k, a_k) + ΔA_θ(S_k, a_k)

- ``AnalyticUpdate`` = 本模块 :func:`analytic_update`（运动学是**解析层**唯一职责，
  见 ``AGENT.md`` 守则 4）；
- ``ΔA_θ`` = **可选**学习残差（``analytic_residual`` 档）；本 lane 复用 ``st_gnn`` 现有
  物理解码头（``od_pred``/``ld_pred``）的输出减去其自身先验 = 零新增参数；训练接口 =
  :func:`residual_loss`（one-step；监督目标 = **真实下一状态**；禁自产状态回灌）。

禁止（硬约束；本模块结构性排除）
--------------------------------
- 评测脚本注入未来真值：``WorldState`` 只由 t0 观测 + 已执行动作 + WM 残差构造，
  无任何「未来帧」入参；
- t0 字段静默 hold：七量逐字段带 ``source ∈ {analytic, learned, held, missing}``，
  ``held`` 只允许 :data:`HELD_ALLOWED`（route progress 在 route 长度不可推断时）
  且显式标注；
- 缺失静默填 0：缺失 = 值 0 **∧** ``present=0`` **∧** ``source="missing"``（与
  ``env/obs/path_field.py`` 的 ``<key>__present`` 口径一致，显式可判）；
- 外部提供未来状态：不存在此类入参。

WorldState（t0 帧单一表达）
---------------------------
- ego：``ego_pose (B,3)``（t0 帧累积位姿）+ ``ego_speed (B,)``（阻尼解析速度）；
- OD：``od_anchor (B,S,5)`` = ``[dx,dy,vx,vy,heading]``（t0 帧；``vx/vy`` = t0 相对速度，
  对象先验 = **世界系速度 hold** ⇒ 位置按 ``dt·(v_ref+vx, vy)`` 推进；与 ``st_gnn``
  物理解码同空间 ⇒ ΔA 可直接相加）+ ``od_feat (B,S,9)``（L/W 等静态属性 carry）+
  ``od_latent (B,S,H)`` + ``od_valid (B,S)``；
- LD：``ld_anchor (B,L,4)`` = ``[dx,dy,heading,curvature]``（t0 帧；车道线静止 ⇒ 只随
  ΔA 变化）+ ``ld_latent/ld_valid``；另有目标车道折线 ``lane_anchor/lane_mask``（策略
  几何，t0 静态地图）；
- nav/route：``nav_progress``（route_completion）+ 推断 ``route_length``（``inf`` =
  不可推断 ⇒ held）；
- memory：:class:`ImaginedMemory`（rollout 开始复制 t0 状态，**写回 Real Memory = 0**，
  rollout 结束 ``destroy()``）。

ego-motion 补偿（第 4/5 步）在执行口径上是 **readout 期的确定性重表达**
（``R(−θ_k)(p_t0 − t_k)``）；解析层内部保持 t0 帧单表达 ⇒ 无逐步重表达漂移、残差与
``st_gnn`` 解码同帧（见 :func:`readout`）。

默认关 / 逐位
-------------
本模块**不**被默认路径 import 之外触碰：``DrivingModel.set_world_state_api("off")``
（默认）时 ``_rollout`` 不构造 WorldState、不调用本模块任何函数 ⇒ 前向逐位不变
（证据 = ``.slim/deepwork/p1_v2/round39/b1_bitwise.json``，d1 双跑范式）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

__all__ = [
    "FIELD_PRODUCERS",
    "HELD_ALLOWED",
    "IMAGINED_MEMORY_MAX_FRAMES",
    "POLICY_STATE_FIELDS",
    "SOURCE_CODES",
    "SOURCES",
    "WORLD_STATE_API_MODES",
    "ImaginedMemory",
    "ImaginedMemoryFrame",
    "WorldState",
    "analytic_update",
    "od_decode_prior",
    "readout",
    "residual_loss",
    "residual_targets",
]

#: 开关取值（``DrivingModel.set_world_state_api``；默认 ``off`` ⇒ 逐位不变）
WORLD_STATE_API_MODES: tuple[str, ...] = ("off", "analytic", "analytic_residual")

#: 逐字段来源标签（readout 输出；机器可读）
SOURCES: tuple[str, ...] = ("analytic", "learned", "held", "missing")
#: 来源 → 编码（hook 张量栈用；顺序 = :data:`SOURCES`）
SOURCE_CODES: dict[str, int] = {name: index for index, name in enumerate(SOURCES)}

#: 统一 readout 的物理量字段（列序 = ``net/path_head.LONG_EXOG_FEATURE_NAMES``；7 量 8 列，
#: 「left/right margin」为一组两列）
POLICY_STATE_FIELDS: tuple[str, ...] = (
    "speed",
    "left_margin",
    "right_margin",
    "static_distance",
    "dynamic_distance",
    "dynamic_ttc",
    "nav_progress",
    "distance_to_target_lane_end",
)

#: 允许 ``source="held"`` 的字段白名单（**仅限物理上该保持的量**）：route progress 在
#: route 长度不可从 t0 观测推断时（``distance_to_target_lane_end`` 缺 present 或
#: ``route_completion == 1``）保持上一值并显式标注。其余字段出现 ``held`` ⇒ readout 断言失败。
HELD_ALLOWED: tuple[str, ...] = ("nav_progress",)

#: 字段 → **唯一 producer**（readout 断言：每字段唯一写点；跨帧不得变化）。
#: 断言语义 = 「同字段唯一 producer」：一个字段只由一个写点产出，来源差异只体现在 source。
FIELD_PRODUCERS: dict[str, str] = {
    "speed": "net.world_state:analytic_update.speed",
    "left_margin": "net.world_state:readout.lane_geometry",
    "right_margin": "net.world_state:readout.lane_geometry",
    "static_distance": "net.world_state:readout.static_progress",
    "dynamic_distance": "net.world_state:readout.od_kinematics",
    "dynamic_ttc": "net.world_state:readout.od_kinematics",
    "nav_progress": "net.world_state:analytic_update.route_progress",
    "distance_to_target_lane_end": "net.world_state:readout.lane_progress",
}

#: Imagined Memory 环形帧数上限（t0 复制帧 + rollout 逐步预测帧；超出 ⇒ 只计数不存张量）
IMAGINED_MEMORY_MAX_FRAMES = 8

#: readout 车道几何口径（R38-A4 ``fixed``：foot 搜索排除 ``mask=0`` 段 + margin 数据集符号
#: ``left = W/2 + lat_k``；出处 = A3 §3.2/§3.3 行级证据 + A4 落地。**只管想象力几何**，
#: 不触碰 ``step_geometry`` 的进程级默认）。
READOUT_LANE_GEOMETRY_MODE = "fixed"

#: TTC 接近判据 / cap（= ``env.obs.long_exog`` 同源常量）
TTC_EPS = 1e-3
TTC_CAP_S = 60.0


def _detach(value: Tensor | None) -> Tensor | None:
    """想象世界状态的非可微边界（**单一 detach 写点**；A2 detach 登记表登记项）。

    语义：WorldState 是「已执行动作 + WM 残差」的**确定性状态**；学习发生在残差头
    （``residual_loss`` one-step 监督），不穿越跨步想象链 ⇒ 所有入状态张量在此 detach。
    """
    return None if value is None else value.detach()


def _align(value: Tensor, ref: Tensor) -> Tensor:
    """dtype/device 对齐（float 化；调用方保证形状）。"""
    return value.to(dtype=ref.dtype, device=ref.device).float()


# --------------------------------------------------------------------------- #
# Imagined Memory（与 Real Memory 严格隔离）
# --------------------------------------------------------------------------- #
@dataclass
class ImaginedMemoryFrame:
    """想象记忆单帧（rollout 开始复制 t0 状态；之后逐步压入预测状态）。"""

    #: 步号（0 = t0 复制帧，k≥1 = 第 k 步预测帧）
    step: int
    #: ``real`` = 由真实 t0 编码复制；``predicted`` = 想象步产出
    source: str
    ego_pose: Tensor
    ego_speed: Tensor
    od_anchor: Tensor
    od_valid: Tensor
    ld_anchor: Tensor
    ld_valid: Tensor
    lane_anchor: Tensor
    od_latent: Tensor | None = None
    ld_latent: Tensor | None = None


@dataclass
class ImaginedMemory:
    """想象记忆（**只读副本**；绝不写回 Real Memory —— 结构性保证，见下）。

    - ``writeback`` 恒 0：本容器只持有 ``detach().clone()`` 副本，不持有 ``MemBank``/
      ``encoded`` 的任何引用 ⇒ 无写回路径（GPT §二.2「绝不写回 Real Memory」）；
    - ``destroy()``：rollout 结束销毁（``frames`` 清空、``destroyed=True``，计数保留）；
    - ``frames/shift`` 计数与帧龄（``age``）供 hook 落盘。
    """

    batch: int
    max_frames: int = IMAGINED_MEMORY_MAX_FRAMES
    store_latents: bool = True
    frames: list[ImaginedMemoryFrame] = field(default_factory=list)
    #: 写回 Real Memory 次数（**硬不变量 = 0**；只读断言用）
    writeback: int = 0
    destroyed: bool = False
    skipped: int = 0

    def push(self, frame: ImaginedMemoryFrame) -> ImaginedMemoryFrame:
        if self.destroyed:
            raise RuntimeError("[world_state] ImaginedMemory 已销毁：rollout 结束后不得再写入")
        if frame.source not in ("real", "predicted"):
            raise ValueError(f"[world_state] 未知帧来源 {frame.source!r}（real|predicted）")
        if len(self.frames) >= int(self.max_frames):
            self.skipped += 1
            return frame
        self.frames.append(frame)
        return frame

    def clone_frame(self, *, step: int, source: str, **tensors: Tensor | None) -> ImaginedMemoryFrame:
        """按帧字段克隆压入（``detach().clone()`` 副本；latents 可选）。"""
        payload = {name: _detach(value) for name, value in tensors.items()}
        payload = {
            name: None if value is None else value.clone() for name, value in payload.items()
        }
        latents = {}
        for name in ("od_latent", "ld_latent"):
            value = payload.pop(name, None)
            latents[name] = value if self.store_latents else None
        frame = ImaginedMemoryFrame(step=int(step), source=str(source), **payload, **latents)
        return self.push(frame)

    def destroy(self) -> "ImaginedMemory":
        """rollout 结束销毁（幂等）。"""
        self.frames.clear()
        self.destroyed = True
        return self

    def summary(self) -> dict[str, Any]:
        return {
            "frames": len(self.frames),
            "max_frames": int(self.max_frames),
            "writeback": int(self.writeback),
            "destroyed": bool(self.destroyed),
            "skipped": int(self.skipped),
            "steps": [int(frame.step) for frame in self.frames],
            "sources": [str(frame.source) for frame in self.frames],
        }


# --------------------------------------------------------------------------- #
# WorldState
# --------------------------------------------------------------------------- #
@dataclass
class WorldState:
    """统一想象世界状态（t0 帧单表达；字段语义见模块 docstring）。

    构造用 :meth:`t0`（t0 真实编码复制；**不**接受任何未来帧）。``analytic_update`` 就地
    推进（返回同一对象）。
    """

    # ---- ego
    ego_pose: Tensor
    ego_speed: Tensor
    #: t0 自车速度（对象世界系速度的参考系 = t0 frame x̂；不随想象步变化）
    ego_speed_ref: Tensor
    # ---- OD（t0 帧解码空间 [dx,dy,vx,vy,heading]）
    od_anchor: Tensor
    od_feat: Tensor
    od_latent: Tensor
    od_valid: Tensor
    # ---- LD（t0 帧解码空间 [dx,dy,heading,curvature]）
    ld_anchor: Tensor
    ld_latent: Tensor
    ld_valid: Tensor
    # ---- 车道几何（t0 静态地图：目标车道折线 + 8 列外源几何模板/present）
    lane_anchor: Tensor
    lane_mask: Tensor
    lane_target_valid: Tensor
    t0_exog: Tensor
    t0_present: Tensor
    # ---- nav/route
    nav_progress: Tensor
    nav_progress_present: Tensor
    route_length: Tensor
    route_trackable: Tensor
    # ---- bookkeeping
    progress_s: Tensor
    dt: float
    speed_alpha: float
    speed_max: float
    step: int = 0
    memory: ImaginedMemory | None = None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def t0(
        cls,
        *,
        ego_speed: Tensor,
        od_anchor: Tensor,
        od_feat: Tensor,
        od_latent: Tensor,
        od_valid: Tensor,
        ld_anchor: Tensor,
        ld_latent: Tensor,
        ld_valid: Tensor,
        lane_anchor: Tensor,
        lane_mask: Tensor | None,
        lane_target_valid: Tensor,
        t0_exog: Tensor,
        t0_present: Tensor,
        dt: float = 0.5,
        speed_alpha: float = 1.0,
        speed_max: float = 20.0,
        memory_max_frames: int = IMAGINED_MEMORY_MAX_FRAMES,
        store_latents: bool = True,
    ) -> "WorldState":
        """t0 状态（**只由真实 t0 编码 + t0 静态地图构造**）；自动创建 Imagined Memory 并
        复制 t0 帧（source=``real``）。"""
        ref = od_anchor
        speed = _align(ego_speed.reshape(-1), ref)
        batch = int(speed.shape[0])
        if lane_mask is None:
            lane_mask = torch.ones(
                (int(lane_anchor.shape[0]), int(lane_anchor.shape[1])),
                dtype=ref.dtype,
                device=ref.device,
            )
        exog = _align(t0_exog, ref)
        present = _align(t0_present, ref)
        progress = torch.clamp(exog[:, 6], 0.0, 1.0)
        end = exog[:, 7]
        remaining_frac = (1.0 - progress)
        trackable = (
            (present[:, 6] > 0.5) & (present[:, 7] > 0.5) & (remaining_frac > 1e-3) & (end > 0.0)
        )
        route_length = torch.where(
            trackable, end / remaining_frac.clamp_min(1e-3), torch.full_like(end, float("inf"))
        )
        state = cls(
            ego_pose=torch.zeros((batch, 3), dtype=ref.dtype, device=ref.device),
            ego_speed=speed.clone(),
            ego_speed_ref=speed.clone(),
            od_anchor=od_anchor.clone(),
            od_feat=od_feat.clone(),
            od_latent=od_latent,
            od_valid=od_valid.clone(),
            ld_anchor=ld_anchor.clone(),
            ld_latent=ld_latent,
            ld_valid=ld_valid.clone(),
            lane_anchor=lane_anchor.clone(),
            lane_mask=lane_mask.clone(),
            lane_target_valid=lane_target_valid.clone(),
            t0_exog=exog.clone(),
            t0_present=present.clone(),
            nav_progress=progress.clone(),
            nav_progress_present=present[:, 6].clone(),
            route_length=route_length,
            route_trackable=trackable.float(),
            progress_s=torch.zeros((batch, ), dtype=ref.dtype, device=ref.device),
            dt=float(dt),
            speed_alpha=float(speed_alpha),
            speed_max=float(speed_max),
        )
        memory = ImaginedMemory(batch=batch, max_frames=int(memory_max_frames),
                                store_latents=bool(store_latents))
        memory.clone_frame(
            step=0,
            source="real",
            ego_pose=state.ego_pose,
            ego_speed=state.ego_speed,
            od_anchor=state.od_anchor,
            od_valid=state.od_valid,
            ld_anchor=state.ld_anchor,
            ld_valid=state.ld_valid,
            lane_anchor=state.lane_anchor,
            od_latent=state.od_latent,
            ld_latent=state.ld_latent,
        )
        state.memory = memory
        return state


# --------------------------------------------------------------------------- #
# 解析更新（第 3–6 步）
# --------------------------------------------------------------------------- #
def analytic_update(
    state: WorldState,
    *,
    ds: Tensor,
    dtheta: Tensor,
    pose: Tensor | None = None,
    z_od: Tensor | None = None,
    z_ld: Tensor | None = None,
    od_residual: Tensor | None = None,
    ld_residual: Tensor | None = None,
    step: int | None = None,
    speed_alpha: float | None = None,
) -> WorldState:
    """解析更新（就地）+ 可选学习残差：``A_{k+1} = AnalyticUpdate(A_k, a_k) + ΔA_θ``。

    - **ego 运动学**：位姿 = 调用方累积位姿（``pose``；缺省 = 由 ``state.ego_pose`` 用
      ``net.model.arc_step/compose_pose`` 自行推进——同一实现出处）；**speed 带阻尼**：
      ``v' = clamp(v + α·(ds/dt − v), 0, v_max)``（α = ``speed_alpha``，缺省 = 状态设定，
      构造默认 1.0。A3 教训：α=1 纯自洽 speed 正反馈发散 5.77→6.33 m/s ⇒ α 为暴露参数，
      B2 训练目标之一）；
    - **OD 对象先验**：世界系速度 hold ⇒ ``p += dt·(v_ref+vx, vy)``（t0 帧推进）；
      ``ΔA_θ``（``od_residual (B,S,5)``，t0 帧解码空间）直接相加（掩码外严格 0）；
    - **LD**：t0 帧静止（ego-motion 补偿在 :func:`readout` 的重表达里，见模块 docstring）；
      ``ld_residual (B,L,4)`` 直接相加；
    - **route**：``nav_progress += ds / route_length``（route 长度不可推断 ⇒ held，见
      :data:`HELD_ALLOWED`）；
    - **latent/记忆**：``z_od/z_ld`` 覆写状态 latent（``z_ego`` 为策略侧 latent，不入
      WorldState 物量）；``memory.clone_frame`` 压入预测帧。

    所有入状态张量经 :func:`_detach`（想象态非可微边界；残差学习走
    :func:`residual_loss` one-step 监督，不穿越跨步链）。
    """
    if not isinstance(state, WorldState):
        raise TypeError(f"[world_state] state 必须是 WorldState，收到 {type(state).__name__}")
    ref = state.ego_pose
    ds = _align(_detach(ds).reshape(-1), ref)
    dtheta = _align(_detach(dtheta).reshape(-1), ref)
    batch = int(ref.shape[0])
    if int(ds.shape[0]) != batch or int(dtheta.shape[0]) != batch:
        raise ValueError(
            f"[world_state] ds/dtheta B 应为 {batch}，收到 {int(ds.shape[0])}/{int(dtheta.shape[0])}"
        )

    # (3) ego：位姿 + 阻尼 speed
    if pose is None:
        from net.model import arc_step, compose_pose  # lazy import：避免循环依赖

        dx, dy = arc_step(ds, dtheta)
        pose = compose_pose(state.ego_pose, dx, dy, dtheta)
    pose = _align(_detach(pose), ref)
    if tuple(pose.shape) != (batch, 3):
        raise ValueError(f"[world_state] pose 形状应为 {(batch, 3)}，收到 {tuple(pose.shape)}")
    state.ego_pose = pose

    alpha = float(state.speed_alpha if speed_alpha is None else speed_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"[world_state] speed_alpha 必须 ∈ [0,1]，收到 {alpha}")
    target_speed = ds / max(float(state.dt), 1e-6)
    speed = state.ego_speed + alpha * (target_speed - state.ego_speed)
    state.ego_speed = speed.clamp(0.0, float(state.speed_max))

    # (5) OD：对象先验（世界系速度 hold；t0 帧推进）+ 学习残差
    world_vx = state.ego_speed_ref.unsqueeze(-1) + state.od_anchor[..., 2]
    world_vy = state.od_anchor[..., 3]
    step_dt = float(state.dt)
    od_anchor = state.od_anchor.clone()
    od_anchor[..., 0] = od_anchor[..., 0] + step_dt * world_vx
    od_anchor[..., 1] = od_anchor[..., 1] + step_dt * world_vy
    if od_residual is not None:
        residual = _align(_detach(od_residual), od_anchor)
        if tuple(residual.shape) != tuple(od_anchor.shape):
            raise ValueError(
                f"[world_state] od_residual 形状应为 {tuple(od_anchor.shape)}，收到 {tuple(residual.shape)}"
            )
        od_anchor = od_anchor + residual
    state.od_anchor = od_anchor * state.od_valid.unsqueeze(-1)

    # (4) LD：t0 帧静止；学习残差直接相加（ego-motion 补偿 = readout 重表达）
    if ld_residual is not None:
        residual = _align(_detach(ld_residual), state.ld_anchor)
        if tuple(residual.shape) != tuple(state.ld_anchor.shape):
            raise ValueError(
                f"[world_state] ld_residual 形状应为 {tuple(state.ld_anchor.shape)}，收到 {tuple(residual.shape)}"
            )
        state.ld_anchor = (state.ld_anchor + residual) * state.ld_valid.unsqueeze(-1)

    # (7 的一部分) route progress：可推断 ⇒ 解析推进；否则 held（显式标注）
    advance = ds / state.route_length.clamp_min(1e-6)
    trackable = state.route_trackable > 0.5
    state.nav_progress = torch.where(
        trackable,
        (state.nav_progress + advance).clamp(0.0, 1.0),
        state.nav_progress,
    )
    state.progress_s = state.progress_s + ds

    # latent + 记忆（z_ego 是策略侧 latent，不属于 WorldState 物量；不摄入状态）    if z_od is not None:
        state.od_latent = _detach(z_od)
    if z_ld is not None:
        state.ld_latent = _detach(z_ld)
    if step is not None:
        state.step = int(step)
    if state.memory is not None:
        state.memory.clone_frame(
            step=state.step,
            source="predicted",
            ego_pose=state.ego_pose,
            ego_speed=state.ego_speed,
            od_anchor=state.od_anchor,
            od_valid=state.od_valid,
            ld_anchor=state.ld_anchor,
            ld_valid=state.ld_valid,
            lane_anchor=state.lane_anchor,
            od_latent=state.od_latent,
            ld_latent=state.ld_latent,
        )
    return state


def od_decode_prior(od_anchor_t0: Tensor, *, step: int, dt: float) -> Tensor:
    """``st_gnn`` 物理解码先验（``od_predict`` 的 prior 项；**公式镜像** = ``net/st_gnn.py``
    forward 内联式，供 ``analytic_residual`` 计算 ``ΔA_θ = od_pred − prior``）。

    ``[dx + vx·(k·dt), dy + vy·(k·dt), vx, vy, heading]``（t0 帧；k = ``step``）。
    """
    offset = float(step) * float(dt)
    anchor = od_anchor_t0
    return torch.stack(
        [
            anchor[..., 0] + anchor[..., 2] * offset,
            anchor[..., 1] + anchor[..., 3] * offset,
            anchor[..., 2],
            anchor[..., 3],
            anchor[..., 4],
        ],
        dim=-1,
    )


# --------------------------------------------------------------------------- #
# 统一 readout（第 7 步）：七量重算 + source/present/producer
# --------------------------------------------------------------------------- #
def _field_rows(
    value: Tensor,
    *,
    source: str | Tensor,
    present: Tensor,
    code: Tensor,
    field: str,
) -> Tensor:
    """逐行写一列（producer 唯一性 + source 合法性 + held 白名单断言）。"""
    if isinstance(source, str):
        if source not in SOURCES:
            raise ValueError(f"[world_state] 未知 source={source!r}")
        code = torch.full_like(code, SOURCE_CODES[source])
    else:
        source = source.to(dtype=torch.long, device=code.device).reshape(-1)
        if tuple(source.shape) != tuple(code.shape):
            raise ValueError(
                f"[world_state] 字段 {field!r} source 形状应为 {tuple(code.shape)}，"
                f"收到 {tuple(source.shape)}"
            )
        if bool(((source < 0) | (source >= len(SOURCES))).any()):
            raise ValueError(f"[world_state] 字段 {field!r} source 编码越界（0..{len(SOURCES) - 1}）")
        code = source.clone()
    held_code = SOURCE_CODES["held"]
    if bool((code == held_code).any()) and field not in HELD_ALLOWED:
        raise AssertionError(
            f"[world_state] 字段 {field!r} 出现 source='held'，但不在 HELD_ALLOWED={HELD_ALLOWED}"
        )
    if tuple(value.shape) != tuple(present.shape):
        raise ValueError(
            f"[world_state] 字段 {field!r} 值/掩码形状不一致：{tuple(value.shape)} vs {tuple(present.shape)}"
        )
    missing = present <= 0.5
    if bool(((value != 0.0) & missing).any()):
        raise AssertionError(
            f"[world_state] 字段 {field!r} present=0 但值非 0（缺失静默填值禁止）"
        )
    return code


def _lane_geometry(state: WorldState) -> dict[str, Tensor]:
    """第 4 步（policy 视角）：目标车道锚 ego-motion 补偿 + margin/静态距/末端距重算。

    委托 ``net.path_head.step_geometry``（``fixed`` 口径；A3/A4 证据：foot 排除 ``mask=0``
    段 + margin 数据集符号 ``left = W/2 + lat_k``）；动态列（speed/dyn/ttc/nav）被
    :func:`readout` 覆盖为统一 readout 产物（不沿用 hold）。
    """
    from net.path_head import step_geometry  # lazy：避免 import 环

    return step_geometry(
        state.lane_anchor,
        state.lane_mask,
        state.t0_exog,
        state.t0_present,
        state.ego_pose,
        state.progress_s,
        geometry_mode=READOUT_LANE_GEOMETRY_MODE,
    )


def _od_kinematics(state: WorldState) -> dict[str, Tensor]:
    """第 5 步：OD 对象先验（t0 帧）→ ego_k 帧重表达 → 动态距离/TTC（公式 = ``env.obs.long_exog``）。

    相对速度换算：``v_rel(k) = R(−θ_k)·(v_ref·x̂ + v_t0) − [v_k, 0]``（对象世界系速度 hold；
    自车加速 ⇒ ``vx_k`` 更负 ⇒ 接近更快，见 A3 ``kin_b`` 同式）。
    """
    anchor = state.od_anchor
    feat = state.od_feat
    valid = state.od_valid
    ref_dtype = anchor.dtype
    pose = state.ego_pose
    speed = state.ego_speed

    length = _align(feat[..., 6], anchor)
    width = _align(feat[..., 7], anchor)
    theta = pose[:, 2]
    cos_t = torch.cos(theta).unsqueeze(-1)
    sin_t = torch.sin(theta).unsqueeze(-1)

    world_vx = state.ego_speed_ref.unsqueeze(-1) + anchor[..., 2]
    world_vy = anchor[..., 3]
    px = anchor[..., 0] - pose[:, 0:1]
    py = anchor[..., 1] - pose[:, 1:2]
    dx_k = cos_t * px + sin_t * py
    dy_k = -sin_t * px + cos_t * py
    vx_k = cos_t * world_vx + sin_t * world_vy - speed.unsqueeze(-1)
    vy_k = -sin_t * world_vx + cos_t * world_vy
    heading_k = anchor[..., 4] - theta.unsqueeze(-1)
    cth_k = torch.cos(heading_k)
    sth_k = torch.sin(heading_k)

    finite = (
        torch.isfinite(dx_k) & torch.isfinite(dy_k) & torch.isfinite(vx_k) & torch.isfinite(vy_k)
        & torch.isfinite(length) & torch.isfinite(width)
    )
    live = (valid > 0.5) & finite & (length > 0.0) & (width > 0.0)
    u = -(dx_k * cth_k + dy_k * sth_k)
    v = dx_k * sth_k - dy_k * cth_k
    ex = (u.abs() - 0.5 * length).clamp_min(0.0)
    ey = (v.abs() - 0.5 * width).clamp_min(0.0)
    slot_dist = torch.where(live, torch.hypot(ex, ey), torch.full_like(ex, float("inf")))
    inf = torch.full_like(slot_dist, float("inf"))
    slot_ttc = torch.where(
        live & (dx_k > 0.0) & (vx_k < -TTC_EPS), dx_k / (-vx_k).clamp_min(1e-9), inf
    )
    n_live = live.sum(dim=-1)
    has_live = n_live > 0
    dyn_distance = torch.where(
        has_live, slot_dist.min(dim=-1).values, torch.zeros_like(speed)
    )
    dyn_ttc = torch.minimum(
        slot_ttc.min(dim=-1).values, torch.full_like(speed, float(TTC_CAP_S))
    )
    return {
        "dx": dx_k.to(ref_dtype),
        "dy": dy_k.to(ref_dtype),
        "vx": vx_k.to(ref_dtype),
        "vy": vy_k.to(ref_dtype),
        "cos": cth_k.to(ref_dtype),
        "sin": sth_k.to(ref_dtype),
        "live": live,
        "slot_dist": slot_dist,
        "slot_ttc": slot_ttc,
        "n_live": n_live.to(ref_dtype),
        "has_live": has_live,
        "dynamic_distance": dyn_distance,
        "dynamic_ttc": dyn_ttc,
    }


def readout(state: WorldState) -> dict[str, Any]:
    """``PolicyStateView_k = readout(WorldState_k)``：**七量统一重算**（第 7 步）。

    返回（头输入契约 = 现结构化头 forward 形参）：
    ``anchor_xy/anchor_mask/long_exog/long_present/target_lane_valid`` +
    逐字段 ``fields``（值）/``present``/``sources``（覆盖统计）/``source_codes (B,F)``/
    ``producers`` + ``diagnostics``（OD/LD ego 帧重表达张量、lane 几何）。

    断言（每次调用）：① 字段集 = :data:`POLICY_STATE_FIELDS` 且 producer = 唯一写点；
    ② ``held`` 仅限 :data:`HELD_ALLOWED`；③ ``missing`` ⇒ 值 0 ∧ present=0。
    """
    from net.path_head import LONG_EXOG_FEATURE_NAMES  # lazy

    if not isinstance(state, WorldState):
        raise TypeError(f"[world_state] state 必须是 WorldState，收到 {type(state).__name__}")
    ref = state.ego_speed
    batch = int(ref.shape[0])

    geometry = _lane_geometry(state)
    exog = geometry["long_exog"].clone()
    od = _od_kinematics(state)

    values: dict[str, Tensor] = {}
    present: dict[str, Tensor] = {}
    codes: dict[str, Tensor] = {}
    sources: dict[str, str | Tensor] = {}

    def _codes(kind: str, length: int) -> Tensor:
        return torch.full((length, ), SOURCE_CODES[kind], dtype=torch.long, device=ref.device)

    def _row_codes(mask_analytic: Tensor, *, absent: str = "missing") -> Tensor:
        return torch.where(
            mask_analytic,
            torch.full_like(ref, SOURCE_CODES["analytic"]).long(),
            torch.full_like(ref, SOURCE_CODES[absent]).long(),
        )

    zeros = torch.zeros_like(ref)
    ones = torch.ones_like(ref)

    # speed ← analytic（阻尼状态）
    values["speed"] = state.ego_speed
    present["speed"] = ones
    sources["speed"] = "analytic"

    # left/right margin ← analytic（lane 几何重算；present = t0 margin 可用性）
    margin_present = _align(state.t0_present[:, 1] * state.t0_present[:, 2], ref)
    values["left_margin"] = _align(exog[:, 1], ref) * margin_present
    values["right_margin"] = _align(exog[:, 2], ref) * margin_present
    present["left_margin"] = margin_present
    present["right_margin"] = margin_present
    sources["left_margin"] = _row_codes(margin_present > 0.5)
    sources["right_margin"] = _row_codes(margin_present > 0.5)

    # static clearance / distance-to-lane-end ← analytic（t0 静态距 − 路径进度；缺 present ⇒ missing）
    static_present = _align(state.t0_present[:, 3], ref)
    end_present = _align(state.t0_present[:, 7], ref)
    values["static_distance"] = _align(exog[:, 3], ref) * static_present
    present["static_distance"] = static_present
    sources["static_distance"] = _row_codes(static_present > 0.5)
    values["distance_to_target_lane_end"] = _align(exog[:, 7], ref) * end_present
    present["distance_to_target_lane_end"] = end_present
    sources["distance_to_target_lane_end"] = _row_codes(end_present > 0.5)

    # dynamic distance / TTC ← analytic（OD 对象先验 + ego-motion 补偿）
    # 无 live 槽 ⇒ 值 0 + present=0 + source=missing（env/obs/path_field.py「显式主态」同口径）
    dyn_present = od["has_live"].to(ref.dtype)
    values["dynamic_distance"] = _align(od["dynamic_distance"], ref) * dyn_present
    present["dynamic_distance"] = dyn_present
    sources["dynamic_distance"] = _row_codes(od["has_live"])
    values["dynamic_ttc"] = _align(od["dynamic_ttc"], ref)
    present["dynamic_ttc"] = ones
    sources["dynamic_ttc"] = "analytic"

    # route progress ← analytic（route 长度可推断）/ held（不可推断，显式标注）
    values["nav_progress"] = state.nav_progress
    present["nav_progress"] = _align(state.nav_progress_present, ref)
    sources["nav_progress"] = _row_codes(state.route_trackable > 0.5, absent="held")

    # 列序归一（= POLICY_STATE_FIELDS；hook/readout dict 顺序稳定）。
    # 防御性断言：字段集必须恰好覆盖（producer 唯一性 = FIELD_PRODUCERS 逐字段查表）。
    try:
        values = {name: values[name] for name in POLICY_STATE_FIELDS}
        present = {name: present[name] for name in POLICY_STATE_FIELDS}
        sources = {name: sources[name] for name in POLICY_STATE_FIELDS}
    except KeyError as error:
        raise AssertionError(
            f"[world_state] readout 字段集 ≠ POLICY_STATE_FIELDS（缺 {error.args[0]!r}）"
        ) from error
    fields = tuple(values.keys())

    code_matrix = torch.zeros((batch, len(POLICY_STATE_FIELDS)), dtype=torch.long,
                              device=ref.device)
    coverage: dict[str, dict[str, int]] = {}
    for index, name in enumerate(POLICY_STATE_FIELDS):
        decision = _field_rows(
            values[name],
            source=sources[name],
            present=present[name],
            code=code_matrix[:, index],
            field=name,
        )
        code_matrix[:, index] = decision
        counts: dict[str, int] = {}
        for code_value, code_name in enumerate(SOURCES):
            count = int((decision == code_value).sum())
            if count:
                counts[code_name] = count
        coverage[name] = counts

    long_exog = torch.stack([values[name] for name in POLICY_STATE_FIELDS], dim=-1)
    long_present = torch.stack([present[name] for name in POLICY_STATE_FIELDS], dim=-1)
    for index, name in enumerate(LONG_EXOG_FEATURE_NAMES):
        if name != POLICY_STATE_FIELDS[index]:
            raise AssertionError(
                f"[world_state] 列序与 LONG_EXOG_FEATURE_NAMES 不一致：{name!r} != "
                f"{POLICY_STATE_FIELDS[index]!r}"
            )

    # LD 槽 ego-motion 补偿（第 4 步；t0 帧 → ego_k 帧的重表达，供 hook/B2 残差目标）
    theta = state.ego_pose[:, 2]
    cos_t = torch.cos(theta).unsqueeze(-1)
    sin_t = torch.sin(theta).unsqueeze(-1)
    ld_px = state.ld_anchor[..., 0] - state.ego_pose[:, 0:1]
    ld_py = state.ld_anchor[..., 1] - state.ego_pose[:, 1:2]
    ld_ego = torch.stack(
        [
            cos_t * ld_px + sin_t * ld_py,
            -sin_t * ld_px + cos_t * ld_py,
            state.ld_anchor[..., 2] - theta.unsqueeze(-1),
            state.ld_anchor[..., 3],
        ],
        dim=-1,
    )
    od_ego = torch.stack([od["dx"], od["dy"], od["vx"], od["vy"], od["cos"], od["sin"]], dim=-1)

    return {
        "anchor_xy": geometry["anchor_xy"],
        "anchor_mask": geometry["anchor_mask"],
        "long_exog": long_exog,
        "long_present": long_present,
        "target_lane_valid": state.lane_target_valid,
        "fields": values,
        "present": present,
        "sources": coverage,
        "source_codes": code_matrix,
        "producers": dict(FIELD_PRODUCERS),
        "diagnostics": {
            "lateral_offset": geometry["lateral_offset"],
            "lane_width": geometry["lane_width"],
            "progress_s": state.progress_s,
            "route_length": state.route_length,
            "route_trackable": state.route_trackable,
            "speed": state.ego_speed,
            "speed_ref": state.ego_speed_ref,
            "od_ego_k": od_ego,
            "od_live": od["live"],
            "n_live": od["n_live"],
            "od_slot_dist": od["slot_dist"],
            "od_slot_ttc": od["slot_ttc"],
            "ld_ego_k": ld_ego,
            "ld_valid": state.ld_valid,
            "step": state.step,
            "producer_assert_ok": True,
        },
    }


# --------------------------------------------------------------------------- #
# 训练接口（**只留接口；本 lane 不训练**）：one-step 真值监督
# --------------------------------------------------------------------------- #
def residual_targets(
    analytic_next: Tensor,
    real_next: Tensor,
    valid: Tensor | None = None,
) -> Tensor:
    """残差监督目标 = ``真实下一状态 − AnalyticUpdate(A_k, a_k)``（one-step）。

    数据口径 = A1 harness 的**连续行**（同一 episode 连续 7 决策步；行 t+k 的真实
    encode 状态即第 k 步真值）。**禁止**用自产状态回灌：``real_next`` 必须来自真实
    下一帧编码，``analytic_next`` 必须来自 :func:`analytic_update` 的前向输出
    （本函数不校验来源，只定义接口；训练侧由 harness 采样保证）。
    """
    if tuple(analytic_next.shape) != tuple(real_next.shape):
        raise ValueError(
            f"[world_state] residual_targets 形状不一致：{tuple(analytic_next.shape)} vs "
            f"{tuple(real_next.shape)}"
        )
    target = real_next - analytic_next
    if valid is not None:
        target = target * valid.unsqueeze(-1).to(target.dtype)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("[world_state] residual_targets 出现非有限值（真值/解析输出检查）")
    return target


def residual_loss(
    residual: Tensor,
    analytic_next: Tensor,
    real_next: Tensor,
    valid: Tensor | None = None,
    *,
    mode: str = "smooth_l1",
    beta: float = 1.0,
) -> dict[str, Tensor]:
    """``L = loss(AnalyticUpdate + ΔA_θ, 真实下一状态)``（one-step；**接口**，本 lane 不训练）。

    - ``residual`` = 残差头输出 ``ΔA_θ(S_k, a_k)``（可训练张量；B2 复用 ``st_gnn``
      ``od_pred − prior``/``ld_pred − prior``，零新增参数）；
    - ``real_next`` = **真实下一状态**（连续行；禁自产状态回灌）；
    - ``valid`` = 槽位掩码（无效槽不计入；无效槽输出恒 0 是编码器空间不变量）。
    """
    if mode not in ("smooth_l1", "mse", "l1"):
        raise ValueError(f"[world_state] 未知 loss mode={mode!r}")
    if residual.shape != analytic_next.shape:
        raise ValueError(
            f"[world_state] residual_loss 形状不一致：{tuple(residual.shape)} vs "
            f"{tuple(analytic_next.shape)}"
        )
    pred = analytic_next + residual
    target = residual_targets(analytic_next, real_next, valid)
    diff = pred - real_next
    if mode == "mse":
        per = diff * diff
    elif mode == "l1":
        per = diff.abs()
    else:
        per = torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff), beta=float(beta),
                                                 reduction="none")
    if valid is not None:
        weight = valid.unsqueeze(-1).to(per.dtype)
        loss = (per * weight).sum() / weight.sum().clamp_min(1.0)
    else:
        loss = per.mean()
    return {"loss": loss, "target": target, "pred": pred}
