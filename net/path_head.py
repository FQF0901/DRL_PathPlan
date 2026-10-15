"""Round-36 lane K1：``StructuredPolicyHead`` —— **唯一动作生成器**的结构化形式。

架构修正出处
------------
``docs/chatgpt5p6sol_review.md`` §八/§九（GPT ``ac78f76``）+ ``docs/architecture_contract.md``
（本 lane 交付的架构契约，含六步调用图 / 辅助与主边界 / 训练部署同构声明 / 门定义）。

H1 的原型头（``PathExecutionHead``）一次调用给出 6 步 plan 并在 rollout 里**钉住**尾段——等价于
第二个六步轨迹生成器、绕过了「决策 → WM 推进一步 → 同一头重新决策」（R38-P1 减枝已把该
一次性路径整体删除并迁 ``tools/legacy/oneshot_path.py``）。本模块的 ``StructuredPolicyHead``：

1. **每步调用**：同一实例、同一参数，用**当前** ``z_k`` 产出 ``a_k = (ds_k, dθ_k = κ_near_k·ds_k)``；
   ``st_gnn`` 推进一步得到 ``z_{k+1}`` 后**同一头**重算下一步（6 次调用 = 6 个动作 a0..a5）；
   6 步 ``plan`` 由逐步动作**组装**供 LQR/环境/评测（``plan[:,0]==action_mu`` 契约不变）。
2. **内部结构**：``lateral`` 分支（锚 κ_ref + 残差 → ``κ_near/mid/far``）+ ``longitudinal``
   分支（``[latent, 几何(κ_near, 外源几何)]`` → 当前 ``ds``）+ ``compose`` 确定性几何层
   （``a=(ds, κ_near·ds)``，动作界截断；无参数）。
3. **辅助输出**：``κ_mid/κ_far`` 与未来 progress/ds 仅在每步附带产出（意图/监督），
   **不进动作、不进 LQR**（``longitudinal`` 只吃 ``κ_near``；见契约门「辅助输出不出现在
   LQR 输入」）。
4. **imagined 步几何**：k≥1 的车道锚/边界余量/静态距/目标车道末端距由 **t0 静态地图 +
   想象位姿**确定性重算（:func:`step_geometry`）；动态距离/TTC/nav/speed 保持当前步
   （动态 TTC 的 WM ``od_pred`` 变体未接入，见契约 caveats）。
5. **数据驱动初始化/限幅**（替代 H1 的固定 3.5 m / ``tanh × 0.1``）：
   - ``ds_init = clamp(v_current·dt, bounds)``（:func:`ds_init_from_speed`；bounds 由专家 ds
     P10/P90 标定，P50 校验 rel 0.3%）；
   - 残差限幅 = ``clamp(tanh(raw)·κ_p95(v), ±κ_p99(v))``，``κ_max(v) = min(分桶表(v),
     a_y_budget/v²)``（:func:`kappa_limit_by_speed`；标定脚本
     ``.slim/deepwork/p1_v2/round36/k1_calibrate_head_init.py``，产物
     ``runs/round36/k1/calibration.json``）。数据不足 → 常量 fallback：``ds_init=3.5``、
     ``κ_limit=0.1``（:data:`DS_INIT_FALLBACK` / :data:`KAPPA_LIMIT_FALLBACK`，显式标注）。

消融脚手架（默认关；``DrivingModel.set_path_head_ablation``）
-------------------------------------------------------------
- ``policy``：① 旧 ``PolicyHead`` + WM rollout（= 默认关路径的行为）；
- ``structured``（**默认**，生产口径）：② 结构化头 + WM 逐步 rollout。

③ H1 一次性 6 步诊断臂（头内一次性开关 + 尾段钉住）**已删除**（R38-P1 减枝）；
``DrivingModel.set_path_head_ablation`` 写入该模式名即 ``ValueError`` fail-fast，
历史复现快照见 ``tools/legacy/oneshot_path.py``。

默认关 / 逐位兼容
-----------------
- 本模块**不被** :class:`net.model.DrivingModel` 默认构造（``set_path_head(True)`` 惰性构造，
  构造期保存/恢复 CPU RNG ⇒ 不扰动既有初始化抽样序列）；默认关时模型键集/前向逐位不变。
- ``forward`` 只在模型启用结构化头时被调用；纯函数（几何/组合/标定）无全局状态、不消耗 RNG。
- R38-A4 想象步几何口径开关（:func:`set_step_geometry`；默认 ``production`` ⇒ 逐位不变）：
  ``fixed`` = 两条确定性几何缺陷修复（foot 排除 ``mask=0`` 段；margin 符号 = 数据集口径），
  仅影响 ``step_geometry``；证据与验收见 ``.slim/deepwork/p1_v2/round38/a4_report.md``。

输入契约（启用时；``obs`` 键）
------------------------------
- ``lane_anchor_xy (B,P,2)``（必需；容忍 ``(B,1,P,2)``）——目标车道中心线折线（ego t0 系）；
- ``lane_anchor_mask (B,P)``（可选；默认全 1）；
- ``target_lane_valid (B,)``（可选；缺省 = 锚有效；``lane_semantic`` 列 6 兜底）；
- ``path_left_margin/path_right_margin/path_static_distance/path_dynamic_distance/
  path_dynamic_ttc/path_nav_progress/path_distance_to_target_lane_end (B,)``（可选，缺省 0）；
  当前速度取 ``obs['ego']`` 第 0 维（v, m/s）；``lane_semantic`` 存在时 margin/d2end/valid
  回退其列 4/5/11/6。缺键 → 0 + presence 掩码（默认 0 不静默，H3 hook 审计）。

训练/冻结集（H1 manifest 同步；K1 沿用）
----------------------------------------
- **可训**：``path_head.`` + ``plan_head.fusion.2.``（:data:`TRAIN_ALLOWLIST_PREFIXES`）；
- **冻结**：WM(``st_gnn.``)/``encoders.``/``mem_encoder.``/旧 ``policy.``/``value.``/
  ``plan_head.moe.`` 等（LQR 不在模型内）。

Caveats（K1 口径）
------------------
- 曲率空间锚定（起点不搬到锚线上）：横向偏移/航向误差只作特征；偏移修正由残差头学习。
- ``dθ`` 截断 ``[−0.6,0.6]``（policy 动作界）与 ``ds`` 截断 ``[0,10]``：掩码随输出返回。
- 残差 P95/P99 用 ``|κ_expert|`` **代理**（DP-6 锚残差未就绪时的 fallback，产物里标注）；
  高速桶（>12 m/s）无数据 ⇒ 最近桶 + ``a_y`` 外推。
- 想象步的动态列（dynamic_distance/ttc）与 nav/speed **保持 t0/当前步**；静态距/末端距按
  **路径进度**收缩（近似：不走 lane 坐标系还原）；静态距/末端距的 lane 级精确重算待 H2/H3。
- R36 L-A 后：``structured`` 模式**默认不再前向旧 ``PolicyHead``**（诊断经 ``set_policy_diag(True)``/``set_path_head_trace(True)`` 恢复）；``action_logstd`` 不再随结构化输出提供（RL 未接线；接线时必须统一均值/方差来源）。
- R38-P1：一次性 6 步（H1）路径已删除（唯一动作源守则）⇒ 本模块只导出逐步口径
  （``compose_step_action`` + ``aux_preview``）与几何/标定件。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

from net.encoders import H
from net.mem import squeeze_batch_singletons
from net.policy import ACTION_HIGH, ACTION_LOW
from net.spatial import wrap_angle

__all__ = [
    "ANCHOR_FEATURE_NAMES",
    "ANCHOR_FEATURE_NORMALIZATION",
    "DS_HIGH",
    "DS_INIT_BOUNDS",
    "DS_INIT_FALLBACK",
    "DTHETA_HIGH",
    "DTHETA_LOW",
    "DT",
    "EXPERT_DS_P10",
    "EXPERT_DS_P50",
    "EXPERT_DS_P90",
    "KAPPA_AY_BUDGET_P95",
    "KAPPA_AY_BUDGET_P99",
    "KAPPA_CTRL_FRACTIONS",
    "KAPPA_CTRL_S",
    "KAPPA_LIMIT_FALLBACK",
    "KAPPA_RESIDUAL_P95_BY_SPEED",
    "KAPPA_RESIDUAL_P99_BY_SPEED",
    "KAPPA_LIMIT_SPEED_NODES",
    "KAPPA_SPEED_BUCKETS",
    "LAT_HORIZON_M",
    "LONG_EXOG_FEATURE_NAMES",
    "LONG_EXOG_NORMALIZATION",
    "LONG_EXOG_OBS_KEYS",
    "LONG_HEAD_FEATURE_NAMES",
    "LONG_HEAD_NORMALIZATION",
    "NUM_STEPS",
    "PATH_HEAD_DEFAULT_MODE",
    "PATH_HEAD_MODES",
    "STEP_GEOMETRY_DEFAULT_MODE",
    "STEP_GEOMETRY_MODES",
    "TRAIN_ALLOWLIST_PREFIXES",
    "PathGeometryEncoder",
    "StructuredComposeLayer",
    "StructuredPolicyHead",
    "anchor_from_obs",
    "anchor_reference",
    "compose_step_action",
    "ds_init_from_speed",
    "get_step_geometry",
    "kappa_at",
    "kappa_limit_by_speed",
    "load_head_calibration",
    "long_exog_from_obs",
    "replay_plan",
    "set_step_geometry",
    "step_geometry",
    "target_lane_valid_from_obs",
]

_EPS = 1e-6
#: rollout / 计划步数（与 WM/LQR 契约一致）
NUM_STEPS = 6
#: 单步时长（s）；``ds`` 与速度换算 ``v = ds/dt``
DT = 0.5
#: 横向曲率路径的固定前瞻长度（m）：≈ 3 s × 11.7 m/s；控制站位与之挂钩
LAT_HORIZON_M = 35.0
#: 曲率控制点站位 = ``fractions × horizon`` → (5.25, 17.5, 29.75) m
KAPPA_CTRL_FRACTIONS: tuple[float, float, float] = (0.15, 0.50, 0.85)
KAPPA_CTRL_S: tuple[float, float, float] = tuple(
    round(fraction * LAT_HORIZON_M, 6) for fraction in KAPPA_CTRL_FRACTIONS
)
#: 动作上界（与 :data:`net.policy.ACTION_HIGH` 同口径；ds ∈ [0,10] m/步、dθ ∈ [−0.6,0.6] rad/步）
DS_HIGH = float(ACTION_HIGH[0])
DTHETA_LOW = float(ACTION_LOW[1])
DTHETA_HIGH = float(ACTION_HIGH[1])

# ---------------------------------------------------------------------- 数据驱动标定
# 出处：``.slim/deepwork/p1_v2/round36/k1_calibrate_head_init.py`` →
# ``runs/round36/k1/calibration.json``（源 = DP-6 同一专家 npz：expert_bc.npz，
# 360,497 行 / 265,801 可用训练行；列 action=(N,6,2) 的 (ds,dθ)、ego[:,0,0]=v）。
#: 速度分桶下边界（与标定脚本 SPEED_BUCKETS 一致；文档/审计用）
KAPPA_SPEED_BUCKETS: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0)
#: κ 限幅插值节点 = 各桶专家速度**中位数**（m/s；>12 m/s 桶无数据 ⇒ 不设节点，
#: 代码按端点常数延伸保持最近节点值 = 保守 fallback）
KAPPA_LIMIT_SPEED_NODES: tuple[float, ...] = (1.942971, 5.092408, 8.330247, 11.1)
#: 残差/曲率限幅 P95（对齐 :data:`KAPPA_LIMIT_SPEED_NODES`；1/m）。**fallback proxy**：
#: 专家 residual 需 DP-6 锚标签，未就绪 ⇒ 用 ``|κ_expert|``（含锚线曲率，偏保守）。
KAPPA_RESIDUAL_P95_BY_SPEED: tuple[float, ...] = (
    0.156869,
    0.100262,
    0.041499,
    0.008506,
)
#: 残差硬限幅 P99（对齐节点；1/m）
KAPPA_RESIDUAL_P99_BY_SPEED: tuple[float, ...] = (
    0.213718,
    0.141979,
    0.083002,
    0.009982,
)
#: 高速外推的横向加速度上界（m/s²）= 各节点 ``max(p·v_median²)``（P95 / P99 两档）
KAPPA_AY_BUDGET_P95 = 2.879741
KAPPA_AY_BUDGET_P99 = 5.759759
#: 标定不可用时的常量 fallback（= H1 旧口径 ``tanh × 0.1``；**显式标注**，勿作生产默认）
KAPPA_LIMIT_FALLBACK = 0.1
#: 专家 ds 分位（step0，可用行；m/步）——供校验/审计
EXPERT_DS_P10 = 1.443026
EXPERT_DS_P50 = 4.143237
EXPERT_DS_P90 = 4.166698
#: ``ds_init = clamp(v_current·dt, bounds)`` 的 bounds = [max(0.5, 0.5·P10), min(10, 1.5·P90)]
DS_INIT_BOUNDS: tuple[float, float] = (0.721513, 6.250048)
#: 速度缺失/标定不可用时的常量 fallback（H1 旧口径）
DS_INIT_FALLBACK = 3.5

#: 横向头输入的锚几何特征（顺序即列序；归一化由 :data:`ANCHOR_FEATURE_NORMALIZATION` 固定）
ANCHOR_FEATURE_NAMES: tuple[str, ...] = (
    "lateral_offset",
    "heading_error",
    "kappa_ref_near",
    "kappa_ref_mid",
    "kappa_ref_far",
    "anchor_total_s",
    "anchor_valid",
    "target_lane_valid",
)
#: 锚几何特征固定归一化除数（入口口径，与 ``env.obs.lane_semantic`` 同范式）
ANCHOR_FEATURE_NORMALIZATION: tuple[float, ...] = (
    5.0,
    math.pi,
    0.1,
    0.1,
    0.1,
    100.0,
    1.0,
    1.0,
)

#: 纵向头**外源**几何特征（顺序即列序；当前速度取 ``obs['ego']`` 第 0 维）
LONG_EXOG_FEATURE_NAMES: tuple[str, ...] = (
    "speed",
    "left_margin",
    "right_margin",
    "static_distance",
    "dynamic_distance",
    "dynamic_ttc",
    "nav_progress",
    "distance_to_target_lane_end",
)
#: 外源几何固定归一化除数
LONG_EXOG_NORMALIZATION: tuple[float, ...] = (10.0, 25.0, 25.0, 50.0, 50.0, 10.0, 1.0, 200.0)
#: 外源几何的 obs 键（顺序同 :data:`LONG_EXOG_FEATURE_NAMES`；``None`` = 由 ``obs['ego']`` 求速度）
LONG_EXOG_OBS_KEYS: tuple[str | None, ...] = (
    None,
    "path_left_margin",
    "path_right_margin",
    "path_static_distance",
    "path_dynamic_distance",
    "path_dynamic_ttc",
    "path_nav_progress",
    "path_distance_to_target_lane_end",
)

#: 纵向头完整输入 = 当前步曲率 ``κ_near``(1) + 外源几何(8)。
#: **只吃 κ_near**：``κ_mid/κ_far`` 是辅助输出，不进动作（架构契约门之一）。
LONG_HEAD_FEATURE_NAMES: tuple[str, ...] = ("kappa_near",) + LONG_EXOG_FEATURE_NAMES
LONG_HEAD_NORMALIZATION: tuple[float, ...] = (0.1,) + LONG_EXOG_NORMALIZATION

#: K1 训练 allowlist（与 ``pipeline.trainer.apply_trainable_allowlist`` 同口径）：
#: - ``path_head.`` = 结构化头的横向/纵向分支 + 几何编码（``compose`` 无参数）；
#: - ``plan_head.fusion.2.`` = PolicyBackbone 末层（供给 latent）；其余全冻。
TRAIN_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "path_head.",
    "plan_head.fusion.2.",
)

#: 消融脚手架模式（默认 ``structured``；见模块 docstring「消融脚手架」）。
#: R38-P1 减枝：H1 一次性 6 步诊断臂已删除 ⇒ 枚举只剩两档（写入旧模式名 = fail-fast）。
PATH_HEAD_MODES: tuple[str, ...] = ("policy", "structured")
PATH_HEAD_DEFAULT_MODE = "structured"

#: R38-A4：想象步几何（:func:`step_geometry`）口径开关（进程级；**默认 ``production`` = 逐位不变**）：
#: - ``production``：产线口径（重锚 foot 搜索不看 ``anchor_mask``；margin 符号 ``left=W/2−lat_k``）；
#: - ``fixed``：两条确定性几何缺陷修复（出处 = R38-A3 逐位隔离 + R38-A4 落地）：
#:   ① 重锚 foot 搜索排除 ``anchor_mask=0`` 段（零填充锚的"原点簇 + 桥接段"把 foot 吸到
#:   padding ⇒ 锚几何整段零化；链上 65.5% 想象步命中 ⇒ 修后 0%）；
#:   ② margin 符号改数据集口径 ``left=W/2+lat_k``、``right=W/2−lat_k``（产线把 row 的
#:   t0 left/right 互换；出处 = I1 ``_long_exog_row`` 公式 + A3 行级实测）。
#: 证据（foot_pad 67.9%→0 / ``valid_agree`` 0.302→0.881 / κ_ref MAE 0.098→0.027 /
#: c1,c2 对齐 MAE 3.617/3.690→2.928/3.028）见 ``.slim/deepwork/p1_v2/round38/a3_report.md``
#: §2.3/§2.4 与 ``a4_report.md``。非参数开关，不进 state_dict/ckpt。
STEP_GEOMETRY_MODES: tuple[str, ...] = ("production", "fixed")
STEP_GEOMETRY_DEFAULT_MODE = "production"
_STEP_GEOMETRY_MODE = STEP_GEOMETRY_DEFAULT_MODE


def set_step_geometry(mode: str = STEP_GEOMETRY_DEFAULT_MODE) -> str:
    """设置想象步几何口径（进程级；返回旧值）；未知取值 ``ValueError``（fail-fast）。

    默认 ``production`` ⇒ :func:`step_geometry` 逐位 = 历史产线口径；``fixed`` = R38-A4
    两条确定性几何修复（foot 排除 ``mask=0`` 段 + 数据集 margin 符号）。纯开关，无 RNG/
    参数副作用；``step_geometry(..., geometry_mode=...)`` 可就地覆盖（不读本全局）。
    """
    global _STEP_GEOMETRY_MODE
    if mode not in STEP_GEOMETRY_MODES:
        raise ValueError(f"未知 step_geometry 口径 {mode!r}（{' | '.join(STEP_GEOMETRY_MODES)}）")
    previous = _STEP_GEOMETRY_MODE
    _STEP_GEOMETRY_MODE = mode
    return previous


def get_step_geometry() -> str:
    """当前想象步几何口径（``production`` / ``fixed``；见 :func:`set_step_geometry`）。"""
    return _STEP_GEOMETRY_MODE


_GEOM_EMB_DIM = 64


# ---------------------------------------------------------------------- 标定（数据驱动）
def _piecewise_linear(x: Tensor, xs: tuple[float, ...], ys: tuple[float, ...]) -> Tensor:
    """分块线性插值（节点 = 桶下边界；``x`` 超出端点 ⇒ 端点值常数延伸）。"""
    if len(xs) != len(ys) or len(xs) < 1:
        raise ValueError(f"标定表长度非法：{len(xs)}/{len(ys)}")
    if len(xs) == 1:
        return torch.full_like(x, float(ys[0]))
    node_x = torch.tensor(xs, dtype=x.dtype, device=x.device)
    node_y = torch.tensor(ys, dtype=x.dtype, device=x.device)
    clamped = torch.clamp(x, min=float(xs[0]), max=float(xs[-1]))
    out = torch.full_like(clamped, float(ys[0]))
    for index in range(len(xs) - 1):
        x0, x1 = float(xs[index]), float(xs[index + 1])
        y0, y1 = float(ys[index]), float(ys[index + 1])
        span = max(x1 - x0, _EPS)
        t = ((clamped - x0) / span).clamp(0.0, 1.0)
        segment = y0 + (y1 - y0) * t
        out = torch.where(clamped >= x0, segment, out)
    return torch.where(clamped >= float(xs[-1]), torch.full_like(clamped, float(ys[-1])), out)


def kappa_limit_by_speed(
    speed: Tensor,
    *,
    breakpoints: tuple[float, ...] = KAPPA_LIMIT_SPEED_NODES,
    p95: tuple[float, ...] = KAPPA_RESIDUAL_P95_BY_SPEED,
    p99: tuple[float, ...] = KAPPA_RESIDUAL_P99_BY_SPEED,
    ay_p95: float = KAPPA_AY_BUDGET_P95,
    ay_p99: float = KAPPA_AY_BUDGET_P99,
    fallback: float = KAPPA_LIMIT_FALLBACK,
) -> tuple[Tensor, Tensor]:
    """速度条件化曲率限幅 ``(scale_p95, hard_p99)``（1/m；均为 ``(B,)``）。

    ``scale_p95`` = ``tanh`` 残差的缩放（= 分桶 P95 与 ``a_y`` 外推的较小者）；
    ``hard_p99`` = 残差硬限幅（= 分桶 P99 与 ``a_y`` 外推的较小者，且 ≥ ``scale_p95``）。
    **数据不足 fallback**：``p95``/``p99`` 为空 ⇒ 两侧都返回常量 :data:`KAPPA_LIMIT_FALLBACK`
    （= H1 旧口径，调用方/产物需标注）。
    """
    v = speed.float().reshape(-1)
    if not p95 or not p99:
        constant = torch.full_like(v, float(fallback))
        return constant, constant.clone()
    if len(p95) != len(breakpoints) or len(p99) != len(breakpoints):
        raise ValueError(
            f"标定表长度必须 = len(breakpoints)={len(breakpoints)}，收到 {len(p95)}/{len(p99)}"
        )
    scale = _piecewise_linear(v, breakpoints, p95)
    hard = _piecewise_linear(v, breakpoints, p99)
    safe_v = v.clamp_min(1e-3)
    ay_scale = float(ay_p95) / safe_v.pow(2)
    ay_hard = float(ay_p99) / safe_v.pow(2)
    scale = torch.minimum(scale, ay_scale)
    hard = torch.maximum(torch.minimum(hard, ay_hard), scale)
    return scale, hard


def ds_init_from_speed(
    speed: Tensor,
    *,
    bounds: tuple[float, float] | None = DS_INIT_BOUNDS,
    dt: float = DT,
    fallback: float = DS_INIT_FALLBACK,
) -> Tensor:
    """``ds_init = clamp(v_current·dt, bounds)``（m/步）；``bounds=None`` ⇒ 常量 fallback。

    标定：专家 ds（step0 可用行）P10/P50/P90 = 1.443/4.143/4.167 m，``v·dt`` P50 = 4.156 m
    （rel 0.3%）⇒ 数据驱动校验通过（``runs/round36/k1/calibration.json::ds_init_validation``）。
    """
    v = speed.float().reshape(-1)
    if bounds is None:
        return torch.full_like(v, float(fallback))
    low, high = (float(bounds[0]), float(bounds[1]))
    if not 0.0 <= low < high <= DS_HIGH:
        raise ValueError(f"ds_init bounds 非法：{bounds}（要求 0 ≤ lo < hi ≤ {DS_HIGH}）")
    return (v * float(dt)).clamp(low, high)


def load_head_calibration(path: str | Path) -> dict:
    """读 ``runs/round36/k1/calibration.json`` → 头构造参数（显式覆盖默认常量）。

    返回 ``{"speed_breakpoints", "kappa_p95_by_speed", "kappa_p99_by_speed",
    "kappa_ay_budget_p95", "kappa_ay_budget_p99", "ds_init_bounds"}``（缺字段报错，
    防静默漂移）；缺失文件 ⇒ ``FileNotFoundError``（调用方决定 fallback）。
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = (
        "kappa_limit_speed_nodes",
        "kappa_p95_by_speed",
        "kappa_p99_by_speed",
        "kappa_ay_budget_p95",
        "kappa_ay_budget_p99",
        "ds_init_bounds",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"标定文件缺字段：{missing}（{path}）")
    return {
        "speed_breakpoints": tuple(float(v) for v in payload["kappa_limit_speed_nodes"]),
        "kappa_p95_by_speed": tuple(float(v) for v in payload["kappa_p95_by_speed"]),
        "kappa_p99_by_speed": tuple(float(v) for v in payload["kappa_p99_by_speed"]),
        "kappa_ay_budget_p95": float(payload["kappa_ay_budget_p95"]),
        "kappa_ay_budget_p99": float(payload["kappa_ay_budget_p99"]),
        "ds_init_bounds": tuple(float(v) for v in payload["ds_init_bounds"]),
    }


# ---------------------------------------------------------------------- 几何纯函数
def _interp_nodes(s_nodes: Tensor, values: Tensor, s_query: Tensor) -> Tensor:
    """逐行线性插值（常数延伸）：``s_nodes (B,P)`` 非降、``values (B,P)``、``s_query (B,Q)``。

    ``s_query`` 先按行 clamp 到 ``[s_nodes[0], s_nodes[-1]]``（超出端点 = 端点值常数延伸）。
    P 很小（锚折线 ≤ 64 点），逐区间掩码实现，避免对 torch 版本的依赖。
    """
    batch, points = int(s_nodes.shape[0]), int(s_nodes.shape[1])
    lo = s_nodes[:, :1]
    hi = s_nodes[:, -1:]
    s = torch.clamp(s_query, min=lo, max=hi)
    out = torch.zeros_like(s_query)
    prev = s_nodes[:, :1]
    for index in range(points - 1):
        nxt = s_nodes[:, index + 1 : index + 2]
        span = (nxt - prev).clamp_min(_EPS)
        t = ((s - prev) / span).clamp(0.0, 1.0)
        inside = (s >= prev) & (s <= nxt)
        segment = values[:, index : index + 1] * (1.0 - t) + values[:, index + 1 : index + 2] * t
        out = torch.where(inside, segment, out)
        prev = nxt
    out = torch.where(s >= hi, values[:, -1:], out)
    return out


def anchor_reference(
    anchor_xy: Tensor,
    anchor_mask: Tensor | None = None,
    *,
    horizon: float = LAT_HORIZON_M,
) -> dict[str, Tensor]:
    """目标车道中心线折线 → 锚参考量（纯函数；无 NaN、无 RNG）。

    Args:
        anchor_xy: ``(B,P,2)`` 车道中心线采样点（ego 当前帧：x 前向 / y 左向；点序沿车道前进；
            首点 ≈ 自车在车道上的投影点，允许横向偏移）。想象步 k 由 :func:`step_geometry`
            把 t0 锚重表达进 ego-k 帧后再传入。
        anchor_mask: ``(B,P)`` 1=有效点（可选；默认全 1）。有效点 <3 或总长 <1 m 的行
            ``valid=0``（``kappa_ref`` 归零 = 直锚兜底；**不报错**，由调用方决定是否 fail-fast）。
        horizon: 控制站位前瞻（m）；``s_ctrl = KAPPA_CTRL_FRACTIONS × horizon``（固定）。

    Returns:
        dict（全部 (B,·) / float32）：
        - ``kappa_ref (B,3)``：三控制站位的锚曲率（顶点中心差分 + 线性插值 + 端点常数延伸）；
        - ``s_ctrl (B,3)``：控制站位 [m]（常数，供组合层积分用）；
        - ``total_s (B,)``、``valid (B,)``、``kappa_vertex (B,P)``（诊断）、``s_nodes (B,P)``；
        - ``lateral_offset (B,)`` = ``-anchor_xy[:,0,1]``（自车相对锚线横向偏移；正 = 自车偏左）；
          ``heading_error (B,)`` = ``-θ_anchor(0)``（正 = 自车航向偏左）。
          无效行上上述量全部置 0（``valid`` 门控）。
    """
    if not torch.is_tensor(anchor_xy):
        raise TypeError(f"anchor_xy 必须是 Tensor，收到 {type(anchor_xy).__name__}")
    xy = anchor_xy.float()
    if xy.ndim != 3 or int(xy.shape[-1]) != 2:
        raise ValueError(f"anchor_xy 形状应为 (B,P,2)，收到 {tuple(xy.shape)}")
    batch, points, _ = (int(size) for size in xy.shape)
    if points < 2:
        raise ValueError(f"anchor_xy 至少需要 2 个点，收到 P={points}")
    device, dtype = xy.device, xy.dtype
    if anchor_mask is None:
        mask = torch.ones((batch, points), dtype=dtype, device=device)
    else:
        mask = anchor_mask.float()
        if mask.ndim != 2 or tuple(mask.shape) != (batch, points):
            raise ValueError(f"anchor_mask 形状应为 {(batch, points)}，收到 {tuple(mask.shape)}")

    segment = xy[:, 1:] - xy[:, :-1]
    segment_len = segment.norm(dim=-1)
    segment_valid = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
    segment_len = torch.where(segment_valid, segment_len, torch.zeros_like(segment_len))
    s_nodes = torch.cat(
        [torch.zeros((batch, 1), dtype=dtype, device=device), torch.cumsum(segment_len, dim=1)],
        dim=1,
    )
    theta_segment = torch.atan2(segment[..., 1], segment[..., 0])
    if points >= 3:
        pair_len = 0.5 * (segment_len[:, 1:] + segment_len[:, :-1])
        pair_valid = segment_valid[:, 1:] & segment_valid[:, :-1] & (pair_len > _EPS)
        pair_dtheta = wrap_angle(theta_segment[:, 1:] - theta_segment[:, :-1])
        kappa_inner = torch.where(
            pair_valid, pair_dtheta / pair_len.clamp_min(_EPS), torch.zeros_like(pair_len)
        )
        kappa_vertex = torch.cat(
            [kappa_inner[:, :1], kappa_inner, kappa_inner[:, -1:]], dim=1
        )
    else:
        kappa_vertex = torch.zeros_like(s_nodes)

    total_s = s_nodes[:, -1]
    valid = ((mask.sum(dim=1) >= 3) & (total_s >= 1.0)).to(dtype)
    s_ctrl = (
        torch.as_tensor(KAPPA_CTRL_FRACTIONS, dtype=dtype, device=device)
        .mul(float(horizon))
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    kappa_ref = _interp_nodes(s_nodes, kappa_vertex, s_ctrl) * valid.unsqueeze(-1)
    lateral_offset = -xy[:, 0, 1] * valid
    # 航向误差取**首个有效段**（想象步重锚后可能有零长前缀段；全无效 ⇒ 0）
    segment_valid_int = segment_valid.to(torch.int64)
    first_idx = segment_valid_int.argmax(dim=1)
    has_valid = segment_valid_int.any(dim=1)
    theta_first = theta_segment.gather(1, first_idx.unsqueeze(1)).squeeze(1)
    heading_error = -torch.where(has_valid, theta_first, torch.zeros_like(theta_first)) * valid
    return {
        "kappa_ref": kappa_ref,
        "s_ctrl": s_ctrl,
        "total_s": total_s * valid,
        "valid": valid,
        "kappa_vertex": kappa_vertex,
        "s_nodes": s_nodes,
        "lateral_offset": lateral_offset,
        "heading_error": heading_error,
    }


def kappa_at(kappa_ctrl: Tensor, s_ctrl: Tensor, s_query: Tensor) -> Tensor:
    """分段线性 κ(s)：``s<s0`` 取 κ0、``s>s2`` 取 κ2（常数延伸）。

    ``kappa_ctrl (B,3)``、``s_ctrl (B,3)``（升序）、``s_query`` 任意尾维（钳制到 [s0,s2] 后求值）。
    """
    s = torch.clamp(s_query, min=s_ctrl[..., 0:1], max=s_ctrl[..., 2:3])
    k0, k1, k2 = kappa_ctrl.unbind(-1)
    s0, s1, s2 = s_ctrl.unbind(-1)
    t01 = ((s - s0.unsqueeze(-1)) / (s1 - s0).clamp_min(_EPS).unsqueeze(-1)).clamp(0.0, 1.0)
    t12 = ((s - s1.unsqueeze(-1)) / (s2 - s1).clamp_min(_EPS).unsqueeze(-1)).clamp(0.0, 1.0)
    kappa01 = k0.unsqueeze(-1) + (k1 - k0).unsqueeze(-1) * t01
    kappa12 = k1.unsqueeze(-1) + (k2 - k1).unsqueeze(-1) * t12
    return torch.where(s <= s1.unsqueeze(-1), kappa01, kappa12)


def replay_plan(plan: Tensor, theta0: Tensor | None = None) -> dict[str, Tensor]:
    """``(B,K,2)`` 的 ``(ds,dθ)`` plan → 位姿链（段内恒曲率圆弧）。

    ``net.model.arc_step``/``compose_pose`` 是运动学唯一实现出处（函数内导入避免循环依赖）。
    返回 ``traj_xy (B,K,2)``、``traj_theta (B,K)``（t0 自车系；``theta0`` 缺省 0 = 自车航向）。
    """
    from net.model import arc_step, compose_pose

    if plan.ndim != 3 or int(plan.shape[-1]) != 2:
        raise ValueError(f"plan 形状应为 (B,K,2)，收到 {tuple(plan.shape)}")
    batch, steps = int(plan.shape[0]), int(plan.shape[1])
    dtype, device = plan.dtype, plan.device
    pose = torch.zeros((batch, 3), dtype=dtype, device=device)
    if theta0 is not None:
        if torch.is_tensor(theta0):
            pose[:, 2] = theta0.reshape(batch)
        else:
            pose[:, 2] = float(theta0)
    xy: list[Tensor] = []
    theta: list[Tensor] = []
    for index in range(steps):
        ds, dtheta = plan[:, index, 0], plan[:, index, 1]
        dx, dy = arc_step(ds, dtheta)
        pose = compose_pose(pose, dx, dy, dtheta)
        xy.append(pose[:, :2])
        theta.append(pose[:, 2])
    return {"traj_xy": torch.stack(xy, dim=1), "traj_theta": torch.stack(theta, dim=1)}


def compose_step_action(ds: Tensor, kappa_near: Tensor, *, dt: float = DT) -> dict[str, Tensor]:
    """**确定性单步几何**：``(ds, κ_near)`` → 执行动作 ``a=(ds, dθ=κ_near·ds)``（含动作界截断）。

    架构口径（GPT ``ac78f76`` §八.4）：``a_k=(ds_k, dθ_k=κ_near_k·ds_k)`` —— 每步只执行
    ``κ_near/ds``；``κ_mid/κ_far`` 与未来 progress 不进这里。截断 = ``ds∈[0,10]``、
    ``dθ∈[−0.6,0.6]``（policy 动作界）；掩码随输出返回。
    """
    if ds.ndim != 1 or kappa_near.ndim != 1:
        raise ValueError(
            f"ds/kappa_near 形状应为 (B,)，收到 {tuple(ds.shape)}/{tuple(kappa_near.shape)}"
        )
    if int(ds.shape[0]) != int(kappa_near.shape[0]):
        raise ValueError(f"ds B={int(ds.shape[0])} != kappa_near B={int(kappa_near.shape[0])}")
    ds_raw = ds.float()
    kappa = kappa_near.float()
    finite = torch.isfinite(ds_raw) & torch.isfinite(kappa)
    ds_clamped = torch.where(finite, ds_raw.clamp(0.0, DS_HIGH), torch.zeros_like(ds_raw))
    ds_clamped_mask = (ds_raw < 0.0) | (ds_raw > DS_HIGH) | ~torch.isfinite(ds_raw)
    dtheta_raw = kappa * ds_clamped
    dtheta = dtheta_raw.clamp(DTHETA_LOW, DTHETA_HIGH)
    dtheta_clamped_mask = (dtheta_raw < DTHETA_LOW) | (dtheta_raw > DTHETA_HIGH)
    kappa_seg = torch.where(ds_clamped > 1e-3, dtheta / ds_clamped.clamp_min(_EPS), torch.zeros_like(dtheta))
    kappa_seg_raw = torch.where(
        ds_clamped > 1e-3, dtheta_raw / ds_clamped.clamp_min(_EPS), torch.zeros_like(dtheta_raw)
    )
    return {
        "action": torch.stack([ds_clamped, dtheta], dim=-1),
        "ds": ds_clamped,
        "dtheta": dtheta,
        "speed": ds_clamped / float(dt),
        "kappa_seg": kappa_seg,
        "kappa_seg_raw": kappa_seg_raw,
        "ds_clamped": ds_clamped_mask,
        "dtheta_clamped": dtheta_clamped_mask,
    }


# ---------------------------------------------------------------------- 想象步几何
def step_geometry(
    anchor_xy: Tensor,
    anchor_mask: Tensor | None,
    long_exog: Tensor,
    long_present: Tensor,
    pose: Tensor,
    progress_s: Tensor,
    *,
    dynamic_mode: str = "hold",
    geometry_mode: str | None = None,
) -> dict[str, Tensor]:
    """t0 静态地图 + **想象位姿** → 想象步的静态几何（确定性重算；无 RNG）。

    口径（架构契约 §imagined 步几何，GPT ``ac78f76`` §八）：

    - **车道锚**：t0 锚折线（ego t0 系）刚体逆变换进 ego-``pose`` 帧
      （``p_k = R(−θ)·(p_t0 − t)``），再把自车在 lane 线上的**垂足点**重锚为新首点
      （零长前缀段由 ``anchor_reference`` 的 pair_valid/首个有效段规则排除）⇒ 横向偏移/
      航向误差/``κ_ref``/控制站位前瞻均以想象位姿为基准；
    - **边界余量**：``W = left_t0 + right_t0``（路面容宽，t0 静态地图）+ 新横向偏移
      ``lat_k = −anchor_k[0,1]``（自车相对车道中心线，左正）⇒ ``left_k = clamp(W/2 − lat_k, 0)``、
      ``right_k = clamp(W/2 + lat_k, 0)``（``production`` 口径；与
      ``_same_direction_road_surface_margins`` 同源：t0 恒等）；t0 缺 margin ⇒ 保持 0/present=0；
    - **静态距 / 目标车道末端距**：``clamp(d_t0 − progress_s, 0)``（障碍/lane 末端沿路径的
      位置 = t0 静态地图已编码 ⇒ 按累计路径进度收缩；不走 lane 坐标精确还原，见 caveats）；
    - **动态距离/TTC/nav_progress/speed**：``dynamic_mode="hold"`` = 保持当前步（t0）值
      （允许分支；WM ``od_pred`` 变体未接入）。

    R38-A4 几何口径开关（``geometry_mode``；**默认 = 进程级** :func:`get_step_geometry`，
    缺省 ``production`` ⇒ 逐位 = 历史产线口径）：
    ``fixed`` = 两条确定性几何缺陷修复（证据 = R38-A3 逐位隔离 / R38-A4 落地）：

    ① 重锚 foot 搜索排除 ``anchor_mask=0`` 段 —— obs 锚为零填充（padding = ``(0,0)``）时，
    padding 点与末有效点构成的"桥接段"穿过自车 ⇒ foot 被吸到 padding，锚几何整段零化；
    ② margin 符号 = **数据集口径** ``left = W/2 + lat_k``、``right = W/2 − lat_k``
    （``production`` 把 row 的 t0 left/right 互换；出处 = I1 ``_long_exog_row`` + A3 行级实测）。

    Args:
        anchor_xy: ``(B,P,2)`` t0 锚折线（ego t0 系）。
        anchor_mask: ``(B,P)`` 或 None。
        long_exog: ``(B,8)`` t0 外源几何（列序 = :data:`LONG_EXOG_FEATURE_NAMES`）。
        long_present: ``(B,8)`` t0 可用掩码。
        pose: ``(B,3)`` 想象位姿（ego t0 系；调用方通常传 ``detach``：几何是外源确定性输入）。
        progress_s: ``(B,)`` 累计路径进度（= Σ 已执行 ds）。
        dynamic_mode: 目前仅支持 ``hold``（显式 fail-fast，防静默变体）。
        geometry_mode: ``None``（默认）= 读进程级口径（:func:`set_step_geometry`）；显式
            ``production``/``fixed`` 覆盖（未知取值 fail-fast）。

    Returns:
        dict：``anchor_xy/anchor_mask/long_exog/long_present``（想象步输入）+ 诊断
        ``lateral_offset/lane_width/progress_s/foot_index/dynamic_mode``。
    """
    if dynamic_mode != "hold":
        raise ValueError(f"dynamic_mode 仅支持 'hold'（当前），收到 {dynamic_mode!r}")
    mode = get_step_geometry() if geometry_mode is None else str(geometry_mode)
    if mode not in STEP_GEOMETRY_MODES:
        raise ValueError(
            f"未知 geometry_mode={mode!r}（{' | '.join(STEP_GEOMETRY_MODES)}；None = 进程级口径）"
        )
    if not torch.is_tensor(anchor_xy):
        raise TypeError(f"anchor_xy 必须是 Tensor，收到 {type(anchor_xy).__name__}")
    xy = anchor_xy.float()
    if xy.ndim != 3 or int(xy.shape[-1]) != 2:
        raise ValueError(f"anchor_xy 形状应为 (B,P,2)，收到 {tuple(xy.shape)}")
    batch = int(xy.shape[0])
    exog = long_exog.float()
    present = long_present.float()
    if tuple(exog.shape) != (batch, len(LONG_EXOG_FEATURE_NAMES)):
        raise ValueError(
            f"long_exog 形状应为 {(batch, len(LONG_EXOG_FEATURE_NAMES))}，收到 {tuple(exog.shape)}"
        )
    if tuple(present.shape) != tuple(exog.shape):
        raise ValueError(f"long_present 形状应同 long_exog，收到 {tuple(present.shape)}")
    if not torch.is_tensor(pose) or pose.ndim != 2 or tuple(pose.shape) != (batch, 3):
        raise ValueError(f"pose 形状应为 {(batch, 3)}，收到 {tuple(getattr(pose, 'shape', ()))}")
    if not torch.is_tensor(progress_s) or progress_s.reshape(batch).shape[0] != batch:
        raise ValueError(f"progress_s 形状应为 {(batch,)}，收到 {tuple(getattr(progress_s, 'shape', ()))}")
    pose = pose.float()
    progress = progress_s.float().reshape(batch)
    mask = None if anchor_mask is None else anchor_mask.float()

    theta = pose[:, 2]
    cos_t, sin_t = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
    dx = xy[..., 0] - pose[:, 0:1]
    dy = xy[..., 1] - pose[:, 1:2]
    xy_k = torch.stack([cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy], dim=-1)

    # **重锚**：把自车在 lane 线上的垂足点作为新首点（点 0..foot 全置为 foot；零长前缀段
    # 对 κ/航向无效，由 :func:`anchor_reference` 的 pair_valid/首个有效段规则排除）⇒
    # 步骤 k 的 lateral_offset/heading_error/控制站位前瞻都以**想象位姿**为基准。
    segment_vec = xy_k[:, 1:] - xy_k[:, :-1]
    denom = (segment_vec * segment_vec).sum(dim=-1).clamp_min(_EPS)
    param = -(xy_k[:, :-1] * segment_vec).sum(dim=-1) / denom
    param = param.clamp(0.0, 1.0)
    foot = xy_k[:, :-1] + segment_vec * param.unsqueeze(-1)
    foot_d2 = (foot * foot).sum(dim=-1)
    if mode == "fixed" and mask is not None:
        # R38-A4 ①：foot 搜索排除 anchor_mask=0 段（零填充锚的"原点簇 + 桥接段"伪影；
        # production 不看 mask ⇒ 67.9% 想象步 foot 落 padding、锚几何整段零化）
        segment_valid = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
        foot_d2 = torch.where(segment_valid, foot_d2, torch.full_like(foot_d2, float("inf")))
    foot_index = foot_d2.argmin(dim=-1)
    keep = (
        torch.arange(int(xy_k.shape[1]), device=xy_k.device).unsqueeze(0)
        > foot_index.unsqueeze(1)
    )
    xy_k = torch.where(keep.unsqueeze(-1), xy_k, foot.gather(1, foot_index.reshape(batch, 1, 1).expand(-1, 1, 2)).expand_as(xy_k))

    lat_k = -xy_k[..., 0, 1]
    left0, right0 = exog[:, 1], exog[:, 2]
    margins_present = (present[:, 1] > 0.5) & (present[:, 2] > 0.5)
    lane_width = left0 + right0
    zero = torch.zeros_like(lat_k)

    exog_k = exog.clone()
    if mode == "fixed":
        # R38-A4 ②：数据集口径（I1 _long_exog_row + A3 行级实测）：left = W/2 + lat_k、
        # right = W/2 − lat_k（production 反号 ⇒ 把 row 的 t0 left/right 互换）
        exog_k[:, 1] = torch.where(margins_present, (0.5 * lane_width + lat_k).clamp_min(0.0), zero)
        exog_k[:, 2] = torch.where(margins_present, (0.5 * lane_width - lat_k).clamp_min(0.0), zero)
    else:
        # lat_right = −lat_k（MetaDrive 口径）；t0 恒等：left_0 = W/2 − lat_0（见 docstring）
        exog_k[:, 1] = torch.where(margins_present, (0.5 * lane_width - lat_k).clamp_min(0.0), zero)
        exog_k[:, 2] = torch.where(margins_present, (0.5 * lane_width + lat_k).clamp_min(0.0), zero)
    for column in (3, 7):  # static_distance / distance_to_target_lane_end：按进度收缩
        has = present[:, column] > 0.5
        exog_k[:, column] = torch.where(has, (exog[:, column] - progress).clamp_min(0.0), zero)
    # 0 speed / 4 dynamic_distance / 5 dynamic_ttc / 6 nav_progress：hold（见口径）
    return {
        "anchor_xy": xy_k,
        "anchor_mask": mask,
        "long_exog": exog_k,
        "long_present": present.clone(),
        "lateral_offset": lat_k,
        "lane_width": lane_width,
        "progress_s": progress,
        "foot_index": foot_index,
        "dynamic_mode": dynamic_mode,
    }


# ---------------------------------------------------------------------- obs → 输入
def anchor_from_obs(obs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor | None]:
    """``obs`` → ``(lane_anchor_xy (B,P,2), lane_anchor_mask (B,P)|None)``；缺键 fail-fast。"""
    if not isinstance(obs, Mapping):
        raise TypeError(f"obs 必须是映射，收到 {type(obs).__name__}")
    value = obs.get("lane_anchor_xy")
    if not torch.is_tensor(value):
        raise ValueError(
            "path head 启用时 obs 必须提供 'lane_anchor_xy'（目标车道中心线折线，(B,P,2)，"
            "ego t0 系；首点 ≈ 自车在车道上的投影点）"
        )
    xy = squeeze_batch_singletons(value.float(), 3, "lane_anchor_xy")
    if int(xy.shape[-1]) != 2:
        raise ValueError(f"lane_anchor_xy 形状应为 (B,P,2)，收到 {tuple(xy.shape)}")
    mask_value = obs.get("lane_anchor_mask")
    mask: Tensor | None = None
    if torch.is_tensor(mask_value):
        mask = squeeze_batch_singletons(mask_value.float(), 2, "lane_anchor_mask")
        if tuple(mask.shape) != tuple(xy.shape[:2]):
            raise ValueError(
                f"lane_anchor_mask 形状应为 {tuple(xy.shape[:2])}，收到 {tuple(mask.shape)}"
            )
    return xy, mask


def target_lane_valid_from_obs(
    obs: Mapping[str, Tensor], *, batch: int, device: torch.device | None = None
) -> Tensor | None:
    """``obs['target_lane_valid']``（或 ``lane_semantic`` 列 6）→ ``(B,)``；都缺 → ``None``。

    ``None`` 时 :class:`StructuredPolicyHead` 以锚有效位兜底（``anchor_valid``）。
    """
    value = obs.get("target_lane_valid")
    if torch.is_tensor(value):
        vector = squeeze_batch_singletons(value.float(), 1, "target_lane_valid")
        if int(vector.shape[0]) != batch:
            raise ValueError(f"target_lane_valid B={int(vector.shape[0])} != {batch}")
        return vector.to(device=device) if device is not None else vector
    semantic = obs.get("lane_semantic")
    if torch.is_tensor(semantic):
        sem = squeeze_batch_singletons(semantic.float(), 2, "lane_semantic")
        if int(sem.shape[0]) == batch and int(sem.shape[1]) >= 12:
            column = sem[:, 6]
            return column.to(device=device) if device is not None else column
    return None


def _vector_from_obs(
    obs: Mapping[str, Tensor],
    key: str,
    *,
    batch: int,
    device: torch.device,
    default: float = 0.0,
) -> tuple[Tensor, float]:
    """可选 ``(B,)``（或 ``(B,1)``）标量通道 → ``(values (B,), present)``。"""
    value = obs.get(key)
    if not torch.is_tensor(value):
        return torch.full((batch,), float(default), dtype=torch.float32, device=device), 0.0
    vector = squeeze_batch_singletons(value.float(), 1, key)
    if int(vector.shape[0]) != batch:
        raise ValueError(f"obs[{key!r}] B={int(vector.shape[0])} != {batch}")
    return vector.to(device=device), 1.0


def long_exog_from_obs(
    obs: Mapping[str, Tensor], *, batch: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``obs`` → 纵向头外源几何 ``(long_exog (B,8), present (B,8))``（缺键默认 0，记录掩码）。

    列序 = :data:`LONG_EXOG_FEATURE_NAMES`。速度优先取 ``obs['ego']`` 第 0 维（v, m/s），
    否则取 ``path_speed``；margin/d2end/target 有效位在缺键时回退 ``lane_semantic``
    （列 4/5/11/6，semantic 模式数据）；仍缺 → 0。
    """
    names = LONG_EXOG_FEATURE_NAMES
    values: list[Tensor] = []
    present = torch.zeros((batch, len(names)), dtype=torch.float32, device=device)

    semantic = obs.get("lane_semantic")
    semantic_ok = torch.is_tensor(semantic)
    semantic_batch_ok = False
    if semantic_ok:
        sem = squeeze_batch_singletons(semantic.float(), 2, "lane_semantic")
        semantic_batch_ok = sem.ndim == 2 and int(sem.shape[0]) == batch and int(sem.shape[1]) >= 12
        semantic = sem if semantic_batch_ok else None

    def _push(index: int, vector: Tensor, has: float) -> None:
        values.append(vector)
        present[:, index] = float(has)

    # 0) speed：obs['ego'] 第 0 维（当前帧；容忍 (B,8)/(B,1,8)/(B,T,8)）
    ego = obs.get("ego")
    speed_ok = False
    if torch.is_tensor(ego):
        ego_t = ego.float()
        while ego_t.ndim > 2 and int(ego_t.shape[1]) == 1:
            ego_t = ego_t.squeeze(1)
        if ego_t.ndim == 3:
            ego_t = ego_t[:, -1, :]
        if ego_t.ndim == 2 and tuple(ego_t.shape) == (batch, 8):
            _push(0, ego_t[:, 0].to(device=device), 1.0)
            speed_ok = True
    if not speed_ok:
        value, has = _vector_from_obs(obs, "path_speed", batch=batch, device=device)
        _push(0, value, has)

    # 1) left_margin / 2) right_margin
    left, left_has = _vector_from_obs(obs, str(LONG_EXOG_OBS_KEYS[1]), batch=batch, device=device)
    if left_has == 0.0 and semantic_batch_ok:
        left, left_has = semantic[:, 4].to(device=device), 1.0
    _push(1, left, left_has)
    right, right_has = _vector_from_obs(obs, str(LONG_EXOG_OBS_KEYS[2]), batch=batch, device=device)
    if right_has == 0.0 and semantic_batch_ok:
        right, right_has = semantic[:, 5].to(device=device), 1.0
    _push(2, right, right_has)

    # 3) static / 4) dynamic / 5) ttc
    for index in (3, 4, 5):
        value, has = _vector_from_obs(
            obs, str(LONG_EXOG_OBS_KEYS[index]), batch=batch, device=device
        )
        _push(index, value, has)

    # 6) nav_progress：显式键 → obs['nav'] 末维（route_completion）→ 0
    progress, progress_has = _vector_from_obs(
        obs, str(LONG_EXOG_OBS_KEYS[6]), batch=batch, device=device
    )
    if progress_has == 0.0:
        nav = obs.get("nav")
        if torch.is_tensor(nav):
            nav_t = nav.float()
            while nav_t.ndim > 2 and int(nav_t.shape[1]) == 1:
                nav_t = nav_t.squeeze(1)
            if nav_t.ndim == 2 and int(nav_t.shape[0]) == batch and int(nav_t.shape[1]) >= 11:
                progress, progress_has = nav_t[:, -1].to(device=device), 1.0
    _push(6, progress, progress_has)

    # 7) distance_to_target_lane_end
    distance, distance_has = _vector_from_obs(
        obs, str(LONG_EXOG_OBS_KEYS[7]), batch=batch, device=device
    )
    if distance_has == 0.0 and semantic_batch_ok:
        distance, distance_has = semantic[:, 11].to(device=device), 1.0
    _push(7, distance, distance_has)

    return torch.stack(values, dim=-1).to(device=device, dtype=torch.float32), present


# ---------------------------------------------------------------------- 模块
class PathGeometryEncoder(nn.Module):
    """几何特征编码层：raw 几何 → 隐向量（固定归一化 buffer + 最小 2 层 MLP）。"""

    def __init__(self, in_dim: int, out_dim: int = _GEOM_EMB_DIM, normalization: tuple[float, ...] | None = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if normalization is None:
            normalization = (1.0,) * self.in_dim
        if len(normalization) != self.in_dim:
            raise ValueError(f"normalization 长度 {len(normalization)} != in_dim {self.in_dim}")
        self.register_buffer(
            "normalization", torch.tensor(normalization, dtype=torch.float32), persistent=True
        )
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, feats: Tensor) -> Tensor:
        if feats.ndim != 2 or int(feats.shape[-1]) != self.in_dim:
            raise ValueError(f"几何特征形状应为 (B,{self.in_dim})，收到 {tuple(feats.shape)}")
        normalized = feats / self.normalization.clamp_min(_EPS)
        return self.net(normalized)


class StructuredComposeLayer(nn.Module):
    """**确定性几何层**（无参数）：``lateral``/``longitudinal`` 分支 → 动作 / 辅助预览。

    - :meth:`step_action`：``(ds, κ_near)`` → 执行动作 ``(ds, κ_near·ds)``（生产逐步口径）；
    - :meth:`aux_preview`：未来 5 步 ``(ds_aux, κ_near)`` → 辅助 plan（**仅供参考/监督**）。

    R38-P1：H1 一次性 6 步方法已删除并迁 ``tools/legacy/oneshot_path.py``。
    """

    def __init__(self, *, dt: float = DT, horizon: float = LAT_HORIZON_M):
        super().__init__()
        self.dt = float(dt)
        self.horizon = float(horizon)

    def step_action(self, ds: Tensor, kappa_near: Tensor) -> dict[str, Tensor]:
        return compose_step_action(ds, kappa_near, dt=self.dt)

    def aux_preview(self, ds_aux: Tensor, kappa_near: Tensor) -> dict[str, Tensor]:
        """辅助预览（B,K−1）：逐步 ``(ds_aux_k, κ_near·ds_aux_k)`` + 累计 progress。"""
        if ds_aux.ndim != 2 or kappa_near.ndim != 1:
            raise ValueError(
                f"ds_aux 形状应为 (B,K−1)、kappa_near 为 (B,)，收到 {tuple(ds_aux.shape)}/"
                f"{tuple(kappa_near.shape)}"
            )
        if int(ds_aux.shape[0]) != int(kappa_near.shape[0]):
            raise ValueError(f"ds_aux B={int(ds_aux.shape[0])} != kappa_near B={int(kappa_near.shape[0])}")
        dtheta = kappa_near.unsqueeze(-1) * ds_aux
        plan = torch.stack([ds_aux, dtheta], dim=-1)
        progress = torch.cumsum(ds_aux, dim=-1)
        return {"plan": plan, "progress_s": progress}

class StructuredPolicyHead(nn.Module):
    """唯一动作生成器的结构化形式：``lateral`` + ``longitudinal`` + ``compose``。

    **每次调用只产出当前步动作** ``action=(ds_exec, κ_near·ds_exec)``（供 WM 单步推进 + LQR）；
    ``κ_mid/κ_far``、未来 ds/progress 为辅助输出（不进动作）。初始化口径：

    - ``lateral`` 末层零初始化 ⇒ 初始 ``κ(s) = κ_ref(s)``（= 锚中心线；直锚 ⇒ 直行）；
    - ``longitudinal`` 末层零初始化 ⇒ 初始 ``ds = clamp(v_current·dt, bounds)``
      （数据驱动，见 :func:`ds_init_from_speed`；``ds_init`` 显式传入时用常量，供对照/测试）；
    - 残差限幅 = ``clamp(tanh(raw)·κ_p95(v), ±κ_p99(v))``（速度条件化；标定见模块 docstring）。
    """

    def __init__(
        self,
        hidden: int = H,
        *,
        geom_emb: int = _GEOM_EMB_DIM,
        horizon: float = LAT_HORIZON_M,
        num_steps: int = NUM_STEPS,
        dt: float = DT,
        ds_init: float | None = None,
        ds_bounds: tuple[float, float] | None = DS_INIT_BOUNDS,
        kappa_calibration: Mapping[str, object] | None = None,
    ):
        super().__init__()
        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError(f"hidden 必须为正，收到 {hidden}")
        if int(num_steps) != NUM_STEPS:
            raise ValueError(
                f"num_steps 必须 = {NUM_STEPS}（LQR/rollout 6 步契约），收到 {num_steps}"
            )
        if ds_init is not None and not 0.0 < float(ds_init) < DS_HIGH:
            raise ValueError(f"ds_init 必须在 (0,{DS_HIGH})，收到 {ds_init}")
        self.hidden = hidden
        self.horizon = float(horizon)
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.ds_init_override = None if ds_init is None else float(ds_init)
        # 标定（数据驱动；不可用 → 常量 fallback，显式标注）
        calibration = dict(kappa_calibration or {})
        self.speed_breakpoints: tuple[float, ...] = tuple(
            calibration.get("speed_breakpoints", KAPPA_LIMIT_SPEED_NODES)
        )
        self.kappa_p95_by_speed: tuple[float, ...] = tuple(
            calibration.get("kappa_p95_by_speed", KAPPA_RESIDUAL_P95_BY_SPEED)
        )
        self.kappa_p99_by_speed: tuple[float, ...] = tuple(
            calibration.get("kappa_p99_by_speed", KAPPA_RESIDUAL_P99_BY_SPEED)
        )
        self.kappa_ay_budget_p95 = float(
            calibration.get("kappa_ay_budget_p95", KAPPA_AY_BUDGET_P95)
        )
        self.kappa_ay_budget_p99 = float(
            calibration.get("kappa_ay_budget_p99", KAPPA_AY_BUDGET_P99)
        )
        self.ds_bounds = None if ds_bounds is None else (float(ds_bounds[0]), float(ds_bounds[1]))

        self.lateral_geom = PathGeometryEncoder(
            len(ANCHOR_FEATURE_NAMES), geom_emb, ANCHOR_FEATURE_NORMALIZATION
        )
        self.longitudinal_geom = PathGeometryEncoder(
            len(LONG_HEAD_FEATURE_NAMES), geom_emb, LONG_HEAD_NORMALIZATION
        )
        self.lateral = nn.Sequential(
            nn.Linear(hidden + int(geom_emb), hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.lateral[-1].weight)
        nn.init.zeros_(self.lateral[-1].bias)
        self.longitudinal = nn.Sequential(
            nn.Linear(hidden + int(geom_emb), max(8, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden // 2), self.num_steps),
        )
        nn.init.zeros_(self.longitudinal[-1].weight)
        nn.init.zeros_(self.longitudinal[-1].bias)
        self.compose = StructuredComposeLayer(dt=self.dt, horizon=self.horizon)

    # ------------------------------------------------------------------ 工具
    def parameter_names(self) -> tuple[str, ...]:
        """本模块参数/buffer 名（供 ckpt allowlist/冻结声明核对）。"""
        names = [name for name, _ in self.named_parameters()]
        names += [name for name, _ in self.named_buffers()]
        return tuple(sorted(names))

    def step_limits(self, speed: Tensor) -> dict[str, Tensor]:
        """速度条件化 ``(κ_p95 缩放, κ_p99 硬限, ds_init)``（诊断/门证据用）。"""
        scale, hard = kappa_limit_by_speed(
            speed,
            breakpoints=self.speed_breakpoints,
            p95=self.kappa_p95_by_speed,
            p99=self.kappa_p99_by_speed,
            ay_p95=self.kappa_ay_budget_p95,
            ay_p99=self.kappa_ay_budget_p99,
        )
        if self.ds_init_override is not None:
            init = torch.full_like(speed.float().reshape(-1), self.ds_init_override)
        else:
            init = ds_init_from_speed(speed, bounds=self.ds_bounds, dt=self.dt)
        return {"kappa_p95": scale, "kappa_p99": hard, "ds_init": init}

    # ------------------------------------------------------------------ 前向
    def forward(
        self,
        latent: Tensor,
        *,
        anchor_xy: Tensor,
        anchor_mask: Tensor | None = None,
        long_exog: Tensor | None = None,
        target_lane_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``latent (B,H)`` + 锚折线 + 外源几何 → **当前步**动作 + 辅助输出（见模块 docstring）。

        返回键（全部 Tensor；附加步进诊断 ``structured_*`` 由模型层组装）：
        动作/主输出 ``action/action_mu/ds/ds_exec/ds_aux/ds_init/kappa_ctrl/kappa_residual/
        kappa_ref/kappa_near/kappa_mid/kappa_far/kappa_limit/kappa_hard/kappa_seg/
        kappa_seg_raw/ds_clamped/dtheta_clamped``；辅助 ``aux_progress/aux_plan/plan_aux/
        progress_aux``；几何 ``s_ctrl/anchor_lateral_offset/anchor_heading_error/anchor_total_s/
        anchor_valid/target_lane_valid/anchor_features/long_features``。
        **每次调用只产当前步动作**（R38-P1 后不存在一次性 6 步口径；H1 快照见
        ``tools/legacy/oneshot_path.py``）。重复调用（同输入）逐位一致（无 RNG/全局状态）。
        """
        if latent.ndim != 2 or int(latent.shape[-1]) != self.hidden:
            raise ValueError(f"latent 形状应为 (B,{self.hidden})，收到 {tuple(latent.shape)}")
        batch = int(latent.shape[0])
        anchor_xy = anchor_xy.to(device=latent.device)
        if anchor_mask is not None:
            anchor_mask = anchor_mask.to(device=latent.device)
        anchor = anchor_reference(anchor_xy, anchor_mask, horizon=self.horizon)
        if int(anchor["valid"].shape[0]) != batch:
            raise ValueError(
                f"lane_anchor_xy B={int(anchor['valid'].shape[0])} != latent B={batch}"
            )
        if long_exog is None:
            long_exog = torch.zeros(
                (batch, len(LONG_EXOG_FEATURE_NAMES)), dtype=latent.dtype, device=latent.device
            )
        else:
            long_exog = long_exog.float()
            if tuple(long_exog.shape) != (batch, len(LONG_EXOG_FEATURE_NAMES)):
                raise ValueError(
                    f"long_exog 形状应为 {(batch, len(LONG_EXOG_FEATURE_NAMES))}，"
                    f"收到 {tuple(long_exog.shape)}"
                )
            long_exog = long_exog.to(device=latent.device)
        if target_lane_valid is None:
            target_lane_valid = anchor["valid"]
        else:
            tlv = squeeze_batch_singletons(target_lane_valid.float(), 1, "target_lane_valid")
            if int(tlv.shape[0]) != batch:
                raise ValueError(f"target_lane_valid B={int(tlv.shape[0])} != latent B={batch}")
            target_lane_valid = tlv.to(device=latent.device)
        long_exog = long_exog.to(dtype=latent.dtype)

        # ---- lateral 分支：锚 κ_ref + 速度条件化残差 → κ_near/mid/far
        anchor_features = torch.stack(
            [
                anchor["lateral_offset"],
                anchor["heading_error"],
                anchor["kappa_ref"][:, 0],
                anchor["kappa_ref"][:, 1],
                anchor["kappa_ref"][:, 2],
                anchor["total_s"],
                anchor["valid"],
                target_lane_valid,
            ],
            dim=-1,
        ).to(dtype=latent.dtype)
        lateral_input = torch.cat([latent, self.lateral_geom(anchor_features)], dim=-1)
        limits = self.step_limits(long_exog[:, 0])
        residual_raw = torch.tanh(self.lateral(lateral_input))
        residual = residual_raw * limits["kappa_p95"].unsqueeze(-1)
        residual = torch.clamp(
            residual, -limits["kappa_p99"].unsqueeze(-1), limits["kappa_p99"].unsqueeze(-1)
        )
        kappa_ctrl = anchor["kappa_ref"].to(dtype=latent.dtype) + residual
        kappa_near, kappa_mid, kappa_far = kappa_ctrl.unbind(-1)

        # ---- longitudinal 分支：只吃 κ_near + 外源几何（κ_mid/κ_far 不进动作）
        long_features = torch.cat([kappa_near.unsqueeze(-1), long_exog], dim=-1)
        long_input = torch.cat([latent, self.longitudinal_geom(long_features)], dim=-1)
        ds_raw = self.longitudinal(long_input)
        init_ratio = (limits["ds_init"] / DS_HIGH).clamp(1e-6, 1.0 - 1e-6)
        ds = DS_HIGH * torch.sigmoid(ds_raw + torch.log(init_ratio / (1.0 - init_ratio)).unsqueeze(-1))
        ds_exec, ds_aux = ds[:, 0], ds[:, 1:]

        # ---- compose 确定性几何层：执行动作 + 辅助预览
        step = self.compose.step_action(ds_exec, kappa_near)
        preview = self.compose.aux_preview(ds_aux, kappa_near)
        plan_aux = torch.cat([step["action"].unsqueeze(1), preview["plan"]], dim=1)
        progress_aux = torch.cat(
            [step["ds"].unsqueeze(-1), step["ds"].unsqueeze(-1) + preview["progress_s"]], dim=-1
        )
        output: dict[str, Tensor] = {
            "action": step["action"],
            "action_mu": step["action"],
            "ds": ds,
            "ds_exec": step["ds"],
            "ds_aux": ds_aux,
            "dtheta": step["dtheta"],
            "ds_init": limits["ds_init"],
            "kappa_ctrl": kappa_ctrl,
            "kappa_residual": residual,
            "kappa_ref": anchor["kappa_ref"],
            "kappa_near": kappa_near,
            "kappa_mid": kappa_mid,
            "kappa_far": kappa_far,
            "kappa_limit": limits["kappa_p95"],
            "kappa_hard": limits["kappa_p99"],
            "kappa_seg": step["kappa_seg"],
            "kappa_seg_raw": step["kappa_seg_raw"],
            "ds_clamped": step["ds_clamped"],
            "dtheta_clamped": step["dtheta_clamped"],
            "speed": step["speed"],
            "aux_progress": progress_aux[:, -1],
            "aux_plan": preview["plan"],
            "progress_aux": progress_aux,
            "plan_aux": plan_aux,
            "s_ctrl": anchor["s_ctrl"],
            "anchor_lateral_offset": anchor["lateral_offset"],
            "anchor_heading_error": anchor["heading_error"],
            "anchor_total_s": anchor["total_s"],
            "anchor_valid": anchor["valid"],
            "target_lane_valid": target_lane_valid,
            "anchor_features": anchor_features,
            "long_features": long_features,
        }
        return output
