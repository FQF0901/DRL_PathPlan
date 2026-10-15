"""R39-B2：**readout 列残差头** ``ΔA_θ(S_k, a_k)``（小头；默认不构造 ⇒ 默认路径逐位不变）。

规格出处
--------
- GPT R39 ``647f909`` §二：``A_{k+1} = AnalyticUpdate(A_k, a_k) + ΔA_θ(S_k, a_k)``；臂
  **B2 = analytic + learned residual**，且 **B2 必须优于 B1**（``fe5de55``）；
- ``.slim/deepwork/p1_v2/round39/b1_report.md`` §5：B1 阴性对照的两条主残留 =
  ① **speed 自洽正反馈**（``v_k = ds_k/dt`` 把 head 的 ds 链推到 5.57→6.32 m/s，中高速过发）
  ② 现成 ``st_gnn`` 解码残差的先验（kin_a 相对速度 hold）与 B1 解析 base（kin_b 世界系速度
  hold + ego-motion 补偿）**帧错配** ⇒ ``B1_residual`` 11.797/2.951 更差；
- 训练接口 = :func:`net.world_state.residual_loss`（one-step；监督目标 = **实际下一状态**；
  **禁自产状态回灌**）。

为什么选「新增小残差头」（B1 §5.2 的路线 b）而不是「复用 ``st_gnn`` 解码头」（路线 a）
------------------------------------------------------------------------------
1. **门的主杠杆是 speed 列**：结构化头的纵向分支
   ``ds = DS_HIGH·σ(ds_raw + logit(clamp(v·dt)/DS_HIGH))``，其中 ``v`` = 几何输入
   ``long_exog[:, 0]``（speed 列）。B1 的过发 = speed 列的 ds/dt 自洽反馈。``st_gnn``
   解码头只有 OD/LD anchor 空间输出（5/4 维，**无 ego/speed 通道**）⇒ 结构上无法进 speed 列；
2. 复用 ``st_gnn`` 还要求把其内部先验从 kin_a 改成 kin_b（或训练目标错基）——正是
   ``B1_residual`` 记录的错配来源，且触碰 WM 训练语义（``st_gnn`` 为共享模块）；
3. 专用小头把训练面收敛为**唯一可训模块**：optimizer allowlist = 本头全部参数，其余全冻；
   参数量 ≤ :data:`HEAD_PARAM_CAP`（20k），在 +2% 预算余量（30,211）内（``net/param_probe.py``
   同源基线 1,510,570 → 上限 1,540,781）。

设计（训练与评测**同一实现**）
------------------------------
- **输入特征**（每次调用；全部为当前世界状态的确定性函数，无未来帧）：
  ``[prev_exog(8)/scale, prev_present(8), (ds,dθ)/scale, base_exog(4)/scale, base_present(4)]``
  其中 ``prev_*`` = ``readout(S_k)``（当前状态列）、``base_*`` = ``readout(AnalyticUpdate(S_k,a_k))``
  的四个目标列（``speed/static_distance/dynamic_distance/dynamic_ttc``）；
- **输出** ``ΔA_θ (B,4)``（物理单位：m/s, m, m, s）；``caps·tanh(raw/caps)`` 有界 ⇒
  末层零初始化时严格 ``Δ = 0``（⇒ 挂载未训练头时 B2 与 B1 **逐位一致**，自证 S-B2-0）；
- **应用**（:func:`apply_readout_residual` + ``net.world_state.readout(..., learned=...)``）：
  逐列在 ``base present=1`` 的行上加残差，字段 ``source`` 由 ``analytic`` 变为 ``learned``；
  ``present=0``（missing）行不加（缺失不静默填值）。

训练数据口径（``b2_train.py`` 落地；本模块只定义特征/目标列契约）
------------------------------------------------------------------
- 对（row t → row t+5）真实相邻行：``prev`` = 行 t 真实 encode 的 ``readout``（恒等位姿），
  ``a_t`` = 生产结构化头在行 t 的动作 ``(ds,dθ)``；
  ``base = readout(AnalyticUpdate(WorldState.t0(行 t), a_t))``（**同基**：kin_b + 阻尼 speed
  α=1.0）；目标 = **行 t+5 的真实列** ``long_exog/long_present``（同列序/单位）⇒
  ``Δ* = real_next − base``；loss 掩码 = ``base_present ∧ real_present``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from net.world_state import POLICY_STATE_FIELDS, TTC_CAP_S

__all__ = [
    "ACTION_SCALE",
    "BASE_EXOG_SCALE",
    "COLUMN_LOSS_WEIGHTS",
    "DELTA_CAPS",
    "HEAD_PARAM_CAP",
    "PREV_EXOG_SCALE",
    "READOUT_RESIDUAL_FIELDS",
    "READOUT_RESIDUAL_INDICES",
    "TOTAL_PARAM_BASELINE",
    "TOTAL_PARAM_LIMIT",
    "TARGET_LOSS_SCALE",
    "TTC_EVENT_TAU_S",
    "ReadoutResidualHead",
    "apply_readout_residual",
    "check_param_budget",
    "column_residual_loss",
    "column_residual_loss_aligned",
    "load_residual_head",
    "readout_residual_features",
    "residual_head_param_count",
    "save_residual_head",
]

#: 学习残差覆盖的可学列（readout 七量中的可学列；单位/口径 = ``b1_matrix`` 同列）
READOUT_RESIDUAL_FIELDS: tuple[str, ...] = (
    "speed",
    "static_distance",
    "dynamic_distance",
    "dynamic_ttc",
)
#: 上述列在 ``POLICY_STATE_FIELDS``（= ``net.path_head.LONG_EXOG_FEATURE_NAMES``）中的列号
READOUT_RESIDUAL_INDICES: tuple[int, ...] = tuple(
    POLICY_STATE_FIELDS.index(name) for name in READOUT_RESIDUAL_FIELDS
)
#: prev/base 列的固定归一化尺度（**数据无关常量**；训练/评测同一实现 ⇒ 无口径漂移）
PREV_EXOG_SCALE: tuple[float, ...] = (10.0, 10.0, 10.0, 50.0, 50.0, 60.0, 1.0, 100.0)
BASE_EXOG_SCALE: tuple[float, ...] = (10.0, 50.0, 50.0, 60.0)
ACTION_SCALE: tuple[float, float] = (8.0, 1.2)
#: 输出上界（有界残差：m/s, m, m, s；tanh 饱和 ⇒ 评测期无失控外推）
DELTA_CAPS: tuple[float, ...] = (4.0, 30.0, 100.0, 60.0)
#: 训练损失逐列归一化常数（列尺度差异 ⇒ 均衡梯度；≈ 1.5× one-step 基线 MAE 量级）
TARGET_LOSS_SCALE: tuple[float, ...] = (0.5, 0.5, 0.5, 10.0)
#: B2b 逐列损失权重（speed = 门主杠杆 ⇒ 加权；ttc 走事件式，见
#: :func:`column_residual_loss_aligned`；预注册 ``.slim/.../round39/b2b_prereg.md`` §3）
COLUMN_LOSS_WEIGHTS: tuple[float, ...] = (2.0, 0.5, 1.0, 1.0)
#: B2b TTC 事件式监督的软判据尺度（``σ((cap − pred)/τ)``；秒）
TTC_EVENT_TAU_S = 10.0
#: 本头参数上限（GPT R39 §5.83 参数门：总量 ≤ 基线 ×1.02 ⇒ 余量 30,211；留出裕度定 20k）
HEAD_PARAM_CAP = 20_000
#: 总量基线/上限（= ``net/param_probe.TOTAL_BASELINES["structured"]`` × ``TOTAL_LIMIT``）
TOTAL_PARAM_BASELINE = 1_510_570
TOTAL_PARAM_LIMIT = 1.02


class ReadoutResidualHead(nn.Module):
    """readout 列残差 MLP（26 → H → H → 4；末层零初始化 ⇒ 初始严格零残差）。"""

    #: 输入维数 = 8(prev 列) + 8(prev present) + 2(动作) + 4(base 列) + 4(base present)
    INPUT_DIM = 26
    #: 默认隐藏宽度（参数量见 :meth:`param_count`；H=64 时 6,148 ≤ 20k）
    DEFAULT_HIDDEN = 64

    def __init__(self, hidden: int = DEFAULT_HIDDEN) -> None:
        super().__init__()
        if int(hidden) <= 0:
            raise ValueError(f"[world_residual] hidden 必须 > 0，收到 {hidden}")
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, len(READOUT_RESIDUAL_FIELDS)),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.register_buffer("caps", torch.tensor(DELTA_CAPS, dtype=torch.float32))
        self.register_buffer(
            "prev_scale", torch.tensor(PREV_EXOG_SCALE, dtype=torch.float32)
        )
        self.register_buffer("base_scale", torch.tensor(BASE_EXOG_SCALE, dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(ACTION_SCALE, dtype=torch.float32))

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(
        self,
        *,
        prev_exog: Tensor,
        prev_present: Tensor,
        action: Tensor,
        base_exog: Tensor,
        base_present: Tensor,
    ) -> Tensor:
        """→ ``ΔA_θ (B,4)``（speed/static/dynamic/ttc；有界）。

        形状契约：``prev_exog (B,8)``、``prev_present (B,8)``、``action (B,2)``（ds,dθ）、
        ``base_exog (B,4)``、``base_present (B,4)``。
        """
        if prev_exog.ndim != 2 or int(prev_exog.shape[-1]) != len(POLICY_STATE_FIELDS):
            raise ValueError(
                f"[world_residual] prev_exog 形状应为 (B,{len(POLICY_STATE_FIELDS)})，"
                f"收到 {tuple(prev_exog.shape)}"
            )
        batch = int(prev_exog.shape[0])
        for name, value, width in (
            ("prev_present", prev_present, len(POLICY_STATE_FIELDS)),
            ("base_exog", base_exog, len(READOUT_RESIDUAL_FIELDS)),
            ("base_present", base_present, len(READOUT_RESIDUAL_FIELDS)),
            ("action", action, 2),
        ):
            if value.ndim != 2 or int(value.shape[0]) != batch or int(value.shape[-1]) != width:
                raise ValueError(
                    f"[world_residual] {name} 形状应为 {(batch, width)}，收到 {tuple(value.shape)}"
                )
        dtype = self.net[0].weight.dtype
        device = self.net[0].weight.device
        prev_scale = self.prev_scale.to(dtype)
        base_scale = self.base_scale.to(dtype)
        action_scale = self.action_scale.to(dtype)
        caps = self.caps.to(dtype)
        features = torch.cat(
            [
                prev_exog.to(dtype=dtype, device=device) / prev_scale,
                prev_present.to(dtype=dtype, device=device).clamp(0.0, 1.0),
                action.to(dtype=dtype, device=device) / action_scale,
                base_exog.to(dtype=dtype, device=device) / base_scale,
                base_present.to(dtype=dtype, device=device).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        raw = self.net(features)
        return caps * torch.tanh(raw / caps)


def readout_residual_features(
    *,
    prev_view: Mapping[str, Any],
    action: Tensor,
    base_view: Mapping[str, Any],
) -> dict[str, Tensor]:
    """``readout`` 视图 + 执行动作 → 残差头关键字输入（训练/评测**唯一**特征构造实现）。"""
    prev_exog = prev_view["long_exog"]
    prev_present = prev_view["long_present"]
    base_exog = base_view["long_exog"][:, list(READOUT_RESIDUAL_INDICES)]
    base_present = base_view["long_present"][:, list(READOUT_RESIDUAL_INDICES)]
    return {
        "prev_exog": prev_exog,
        "prev_present": prev_present,
        "action": action,
        "base_exog": base_exog,
        "base_present": base_present,
    }


def apply_readout_residual(*, base_view: Mapping[str, Any], delta: Tensor) -> dict[str, Tensor]:
    """残差 → ``readout(..., learned=...)`` 载荷：逐列掩码 = ``base present``（missing 不加）。"""
    if delta.ndim != 2 or int(delta.shape[-1]) != len(READOUT_RESIDUAL_FIELDS):
        raise ValueError(
            f"[world_residual] delta 形状应为 (B,{len(READOUT_RESIDUAL_FIELDS)})，"
            f"收到 {tuple(delta.shape)}"
        )
    if not bool(torch.isfinite(delta).all()):
        raise ValueError("[world_residual] delta 出现非有限值")
    present = torch.stack(
        [base_view["present"][name] for name in READOUT_RESIDUAL_FIELDS], dim=-1
    )
    return {"delta": delta, "mask": (present > 0.5).to(delta.dtype)}


def column_residual_loss(
    delta: Tensor,
    *,
    base_exog: Tensor,
    base_present: Tensor,
    real_exog: Tensor,
    real_present: Tensor,
    mode: str = "smooth_l1",
    beta: float = 1.0,
    scale: "Tensor | Sequence[float] | None" = None,
) -> dict[str, Tensor]:
    """readout 列残差 one-step 监督损失（**= :func:`net.world_state.residual_loss` 同语义**）：
    ``L = loss(base + ΔA_θ, 实际下一状态)``，掩码 = ``base_present ∧ real_present``。

    - ``base_exog/real_exog``：``(B,4)`` = 目标列 ``speed/static_distance/dynamic_distance/
      dynamic_ttc``（物理单位；``real`` 必须来自**真实下一状态**，禁自产状态回灌）；
    - ``delta``：头输出 ``(B,4)``（同列序）；``scale``（可选）= 逐列损失归一化常数
      （列尺度不同 ⇒ 均衡各列梯度；``None`` = 物理单位损失）；
    - 返回逐列损失/掩码/目标（供训练账本）。
    """
    if mode not in ("smooth_l1", "mse", "l1"):
        raise ValueError(f"[world_residual] 未知 loss mode={mode!r}")
    for name, value in (
        ("delta", delta), ("base_exog", base_exog), ("base_present", base_present),
        ("real_exog", real_exog), ("real_present", real_present),
    ):
        if tuple(value.shape) != tuple(delta.shape):
            raise ValueError(
                f"[world_residual] {name} 形状应与 delta 一致：{tuple(value.shape)} vs "
                f"{tuple(delta.shape)}"
            )
    target = real_exog - base_exog
    diff = base_exog + delta - real_exog
    if scale is not None:
        scale_t = torch.as_tensor(scale, dtype=diff.dtype, device=diff.device).reshape(-1)
        if int(scale_t.numel()) != int(diff.shape[-1]):
            raise ValueError(
                f"[world_residual] scale 长度应为 {int(diff.shape[-1])}，收到 {int(scale_t.numel())}"
            )
        diff = diff / scale_t.clamp_min(1e-6)
    if mode == "mse":
        per = diff * diff
    elif mode == "l1":
        per = diff.abs()
    else:
        per = torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), beta=float(beta), reduction="none"
        )
    mask = ((base_present > 0.5) & (real_present > 0.5)).to(per.dtype)
    weight = mask.sum(dim=0)
    per_column = (per * mask).sum(dim=0) / weight.clamp_min(1.0)
    active = weight > 0
    loss = per_column[active].mean() if bool(active.any()) else per.sum() * 0.0
    return {"loss": loss, "per_column": per_column, "mask": mask, "target": target * mask}


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """``values*mask`` 的元素均值（全空 ⇒ 0；``mask`` ∈ {0,1} 浮点）。"""
    weight = mask.sum()
    return (values * mask).sum() / weight.clamp_min(1.0)


def column_residual_loss_aligned(
    delta: Tensor,
    *,
    base_exog: Tensor,
    base_present: Tensor,
    real_exog: Tensor,
    real_present: Tensor,
    speed_target: Tensor | None = None,
    mode: str = "smooth_l1",
    beta: float = 1.0,
    scale: "Tensor | Sequence[float] | None" = None,
    column_weights: "Sequence[float] | None" = None,
    ttc_event_weight: float = 0.5,
    ttc_regression_weight: float = 0.5,
    ttc_tau_s: float = TTC_EVENT_TAU_S,
) -> dict[str, Tensor]:
    """R39-B2b 对齐损失（预注册 §3）：逐列加权 + speed 目标可替换 + **TTC 事件式**。

    与 :func:`column_residual_loss` 的差异（单测覆盖；旧函数语义不动）：

    - ``speed_target``（``(B,)``；``None`` = 用 ``real_exog[:,0]``）：speed 列目标可改为
      **ds/dt 口径**（本步位移/dt）⇒ 残差对"步距语义"负责，而非复读 speed 列；
    - 逐列权重 ``column_weights``（默认 :data:`COLUMN_LOSS_WEIGHTS`，speed 加权）；
    - TTC 列（index 3）改为事件式：``event = [real_ttc < cap]``；
      ``ℓ_ttc = w_e·BCE(σ((cap − pred)/τ), event) + w_r·(reg_event + reg_cap)/2``，
      ``reg_event`` = 冲突行回归（``scale`` 归一），``reg_cap`` = 无冲突行"推至 cap"
      （``min(pred, cap)`` 与 cap 的 smooth_l1）——有界、分档，规避重尾回归。
    - 掩码/返回键与 :func:`column_residual_loss` 同义（``loss/per_column/mask/target``）+
      ``ttc_event``/``n_ttc_event``/``n_ttc_noevent`` 诊断。
    """
    if mode not in ("smooth_l1", "mse", "l1"):
        raise ValueError(f"[world_residual] 未知 loss mode={mode!r}")
    for name, value in (
        ("delta", delta), ("base_exog", base_exog), ("base_present", base_present),
        ("real_exog", real_exog), ("real_present", real_present),
    ):
        if tuple(value.shape) != tuple(delta.shape):
            raise ValueError(
                f"[world_residual] {name} 形状应与 delta 一致：{tuple(value.shape)} vs "
                f"{tuple(delta.shape)}"
            )
    width = int(delta.shape[-1])
    if speed_target is not None and tuple(speed_target.shape) != (int(delta.shape[0]),):
        raise ValueError(
            f"[world_residual] speed_target 形状应为 {(int(delta.shape[0]),)}，"
            f"收到 {tuple(speed_target.shape)}"
        )
    weights = torch.as_tensor(
        COLUMN_LOSS_WEIGHTS if column_weights is None else column_weights,
        dtype=delta.dtype, device=delta.device,
    ).reshape(-1)
    if int(weights.numel()) != width:
        raise ValueError(f"[world_residual] column_weights 长度应为 {width}")
    scale_t = None
    if scale is not None:
        scale_t = torch.as_tensor(scale, dtype=delta.dtype, device=delta.device).reshape(-1)
        if int(scale_t.numel()) != width:
            raise ValueError(f"[world_residual] scale 长度应为 {width}")

    target = real_exog.clone()
    if speed_target is not None:
        target = torch.cat([speed_target.reshape(-1, 1), target[:, 1:]], dim=-1)
    diff = base_exog + delta - target
    if scale_t is not None:
        diff = diff / scale_t.clamp_min(1e-6)
    if mode == "mse":
        per = diff * diff
    elif mode == "l1":
        per = diff.abs()
    else:
        per = torch.nn.functional.smooth_l1_loss(
            diff, torch.zeros_like(diff), beta=float(beta), reduction="none"
        )
    mask = ((base_present > 0.5) & (real_present > 0.5)).to(per.dtype)
    per_column = (per * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1.0)

    # ---- TTC 列（index 3）：事件式（冲突存在性 BCE + 分档回归/cap 行为）----
    ttc_index = len(READOUT_RESIDUAL_FIELDS) - 1
    cap = float(TTC_CAP_S)
    pred_ttc = base_exog[:, ttc_index] + delta[:, ttc_index]
    real_ttc = real_exog[:, ttc_index]
    active = mask[:, ttc_index] > 0.5
    event = (real_ttc < cap - 1e-3) & active
    logits = (cap - pred_ttc) / max(float(ttc_tau_s), 1e-6)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, event.to(pred_ttc.dtype), reduction="none"
    )
    bce_mean = _masked_mean(bce, active.to(bce.dtype))
    scale_ttc = (
        scale_t[ttc_index] if scale_t is not None else torch.ones_like(pred_ttc)
    ).clamp_min(1e-6)
    reg_event = torch.nn.functional.smooth_l1_loss(
        (pred_ttc - real_ttc) / scale_ttc, torch.zeros_like(pred_ttc),
        beta=float(beta), reduction="none",
    )
    reg_cap = torch.nn.functional.smooth_l1_loss(
        (pred_ttc.clamp(max=cap) - cap) / scale_ttc, torch.zeros_like(pred_ttc),
        beta=float(beta), reduction="none",
    )
    reg_event_mean = _masked_mean(reg_event, event.to(reg_event.dtype))
    reg_cap_mean = _masked_mean(reg_cap, ((~event) & active).to(reg_cap.dtype))
    per_column = per_column.clone()
    per_column[ttc_index] = float(ttc_event_weight) * bce_mean + float(
        ttc_regression_weight
    ) * 0.5 * (reg_event_mean + reg_cap_mean)

    active_columns = mask.sum(dim=0) > 0
    weight_sum = (weights * active_columns.to(weights.dtype)).sum()
    loss = (
        (per_column * weights * active_columns.to(weights.dtype)).sum() / weight_sum
        if bool((weight_sum > 0).item())
        else per.sum() * 0.0
    )
    return {
        "loss": loss,
        "per_column": per_column,
        "mask": mask,
        "target": target * mask,
        "ttc_event": event.to(pred_ttc.dtype),
        "n_ttc_event": int(event.sum().item()),
        "n_ttc_noevent": int(((~event) & active).sum().item()),
    }


def residual_head_param_count(head: nn.Module) -> int:
    """头参数量（含 buffer 不计；只统计可训参数）。"""
    return int(sum(p.numel() for p in head.parameters()))


def check_param_budget(
    head: nn.Module,
    *,
    model_total: int | None = None,
    baseline: int = TOTAL_PARAM_BASELINE,
    limit: float = TOTAL_PARAM_LIMIT,
    head_cap: int = HEAD_PARAM_CAP,
) -> dict[str, int]:
    """参数门断言（``net/param_probe.py`` 同源基线）：head ≤ cap ∧（可选）总量 ≤ 基线×1.02。"""
    head_params = residual_head_param_count(head)
    if head_params > int(head_cap):
        raise AssertionError(f"[world_residual] 残差头参数 {head_params} > 上限 {head_cap}")
    report = {"head_params": head_params, "head_cap": int(head_cap), "baseline": int(baseline)}
    if model_total is not None:
        total = int(model_total)
        cap = int(baseline * float(limit))
        if total > cap:
            raise AssertionError(
                f"[world_residual] 模型总量 {total} > 基线 {baseline} × {limit} = {cap}"
            )
        report.update({"model_total": total, "model_total_cap": cap})
    return report


def save_residual_head(
    path: str | Path, head: nn.Module, *, meta: Mapping[str, Any] | None = None
) -> Path:
    """残差头 ckpt（独立文件：**不**动 base ckpt 键面/摘要；meta = 口径与训练账本）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "meta": dict(meta or {}),
            "format": "r39-b2-readout-residual-head/v1",
        },
        str(target),
    )
    return target


def load_residual_head(
    path: str | Path, *, device: str | torch.device | None = None
) -> tuple[ReadoutResidualHead, dict[str, Any]]:
    """装载残差头 ckpt → ``(head, meta)``（结构/口径 fail-fast）。"""
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != "r39-b2-readout-residual-head/v1":
        raise ValueError(f"[world_residual] {path!r} 不是 r39-b2 残差头 ckpt（format 缺失/不符）")
    meta = dict(payload.get("meta") or {})
    hidden = int(meta.get("hidden", ReadoutResidualHead.DEFAULT_HIDDEN))
    head = ReadoutResidualHead(hidden=hidden)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    if device is not None:
        head.to(device)
    return head, meta
