"""mem-bank：4 个 per-modality 记忆的输入契约、拷贝隔离与编码聚合（v2 架构）。

输入契约（obs，B 维在前，float32；键名解析见 :func:`mem_from_obs`）
------------------------------------------------------------------
============== =============================== ==============================
模态            规范键（env schema v2）          形状 / 语义
============== =============================== ==============================
ego            ``ego_hist`` / ``ego_hist_mask`` ``(B,6,8)/(B,6)``（env 原始
                                               ``(B,6,1,8)/(B,6,1)`` 自动挤压）；
                                               末 2 维 = 上一策略步 (ds,dθ)（§8.4）
OD             ``od_hist`` / ``od_hist_mask``   ``(B,6,16,9)/(B,6,16)``；槽位 =
                                                track id（跨帧稳定）
               ``od_id_hist``                   ``(B,6,16)`` int64 轨道身份（-1=空槽）
               ``od_presence_hist``             ``(B,6,16)`` 对象本帧在盒内被观测
LD             ``ld_hist`` / ``ld_hist_mask``   ``(B,6,16,7)``（只作输入）
others         ``others_hist`` / ``others_hist_mask`` ``(B,6,28)/(B,6)``；28 =
                                               nav(11)+speed_limit(1)+signal(4)+
                                               road_class one-hot(12)
帧级有效性      ``hist_valid``                    ``(B,6)``（warmup 补位帧为 0）
============== =============================== ==============================

当前帧键（回退/上下文用，规范存在）：``od_id (B,16) int64``、``od_presence (B,16)``、
``ego/od/ld/others/nav/signal``（可选）。

兼容与回退（历史伴随数组缺失时；不引入第二套键名）
-------------------------------------------------
- ``ego_hist``/``others_hist`` 缺失时用当前帧 ``ego``/``others`` 复制 6 帧（mask 由
  ``hist_valid`` 门控）；
- ``od_id_hist`` 缺失时用当前 ``od_id`` 广播（再缺则槽位下标占位）；
- ``od_presence_hist`` 缺失时用当前 ``od_presence`` 广播（再缺则 ``od_hist_mask × hist_valid``）；
- 单槽历史通道 ``(B,6,1,F)``（env builder 的原始形状）自动挤压为 ``(B,6,F)``。

时间轴约定：**index 0 = 最老，index -1 = 当前帧**（与 ``env/obs/memory.py`` 一致）。

溯源与只读约束
--------------
真 mem 的 provenance 只在 env：net 只读（编码/池化），**绝不写回**。``MemBank``
持有 obs 张量的引用；rollout 必须先 :meth:`MemBank.clone` 出 4 份副本再
:meth:`MemBank.shift_*`（切片 + ``cat`` 生成新张量，不会原地改写副本或真 mem）。

时序聚合（规格第 2 条）
----------------------
- OD / Ego / Others：各自用 :class:`net.temporal.TemporalAttention` 对 mem 做
  注意力（6 帧 + ``hist_valid`` + 槽位掩码）；
- **LD 不做时序**：直接用当前帧（``ld_hist[:, -1]``）。理由与前提见
  :meth:`MemEncoder.encode` 的 docstring。

世界系 nav 重建（v3 / A4，供 rollout 与教师强制调用）
----------------------------------------------------
旧缺陷：rollout 全程复用 t0 的 ``nav_token``/``signal_token``，且 ``others`` mem 的前
``NAV_DIM`` 维（nav 子向量）同样 t0 冻结（P0-3）。v3 obs 新增 ``ego_world``/``route_world``
（见 ``env/obs/world.py``），本模块提供：

- :func:`nav_features_from_world`：给定世界系位姿 + 世界系路线折线，重算 nav 特征
  （2 checkpoint 自车系坐标 + 命令 one-hot + route_completion），与 ``env.obs.nav`` 同口径；
- :func:`rebuild_nav_from_world`（**固定签名，纯函数**）：返回
  ``(nav_feats (B,11), nav_mask (B,1))`` 供 ``embed_nav`` 生成 token，**不修改 mem**；
- :func:`sync_others_nav_dims`（**纯函数**）：``mem = sync_others_nav_dims(mem, nav_feats)``
  返回 others 当前帧 nav 子向量被替换后的**新** MemBank（赋值式替换，供 teacher forcing /
  rollout 反向；原地写会触发 autograd inplace 报错）；
- :func:`advance_pose_world`：世界系位姿按一步 ``(ds,dθ)`` 推进（与
  ``net.model.arc_step``/``compose_pose`` 同口径，供 rollout/教师强制的位姿链复用）。

回退（旧 ckpt/旧数据缺键）：``route_world``/``ego_world`` 缺失或路线不可用时，
:func:`rebuild_nav_from_world` 返回 mem 里现存的 nav 子向量（= 旧行为：t0 冻结），
并**一次性告警**。所有几何输入在函数内 ``detach``：nav 是上下文条件，不把梯度引回
位姿链/路线（与 t0 nav 来自 obs 常量一致）。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Mapping

import torch
from torch import Tensor, nn

from net.encoders import (
    EGO_MEM_DIM,
    H,
    LD_MEM_DIM,
    NAV_DIM,
    OD_MEM_DIM,
    FrameEncoding,
    ObsEncoders,
)
from net.temporal import TemporalAttention

#: 命令判定阈值（与 ``env.obs.nav.TURN_EPS_RAD`` 一致：两段航向差 < 10° 视为直行）
_TURN_EPS_RAD = math.radians(10.0)
#: 一次性回退告警（key -> 已告警）
_NAV_FALLBACK_WARNED: set[str] = set()


def _warn_nav_fallback(key: str, message: str) -> None:
    if key not in _NAV_FALLBACK_WARNED:
        _NAV_FALLBACK_WARNED.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

#: 每个模态的历史键（env schema v2 规范键；不引入第二套别名）
_HISTORY_KEYS: dict[str, tuple[str, ...]] = {
    "ego": ("ego_hist", ),
    "od": ("od_hist", ),
    "ld": ("ld_hist", ),
    "others": ("others_hist", ),
}
#: 单槽模态（历史形状 (B,T,1,F)，需要挤压 dim=2；od/ld 带槽位维不在此列）


def _require_float(tensor: object, name: str) -> Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"obs[{name!r}] 必须是 Tensor，收到 {type(tensor).__name__}")
    if tensor.dtype != torch.float32:
        raise ValueError(f"obs[{name!r}] 必须是 float32，收到 {tensor.dtype}")
    return tensor


def _check_shape(tensor: Tensor, name: str, tail: tuple[int, ...], batch: int) -> None:
    if tuple(tensor.shape[1:]) != tail or int(tensor.shape[0]) != batch:
        expect = ",".join(str(v) for v in tail)
        raise ValueError(f"obs[{name!r}] 形状应为 (B,{expect})，收到 {tuple(tensor.shape)}")


def _pick_key(obs: Mapping[str, object], names: tuple[str, ...], *, required: bool, what: str) -> str | None:
    for name in names:
        if name in obs:
            return name
    if required:
        raise ValueError(f"obs 缺少必需通道 {names[-1]!r}（{what}）")
    return None


def squeeze_batch_singletons(tensor: Tensor, target_ndim: int, name: str) -> Tensor:
    """挤压 batch 维之后的冗余单例维（``(B,1,1,F) → (B,F)``、``(B,1,1) → (B,1)``）。

    单槽通道在 env builder 里的逐帧形状是 ``(1,F)``/``(1,)``；不同组装路径可能保留 0~2 层
    单例维（trainer 挤压、buffer 堆叠、直接 batching）。这里统一折叠到 ``target_ndim``，
    避免把"多包了一层 1"误判为契约错误。
    """
    while tensor.ndim > target_ndim and int(tensor.shape[1]) == 1:
        tensor = tensor.squeeze(1)
    if tensor.ndim != target_ndim:
        raise ValueError(f"obs[{name!r}] 形状应为 {target_ndim} 维（B 维在前），收到 {tuple(tensor.shape)}")
    return tensor


def _squeeze_dim(tensor: Tensor, name: str, dim: int) -> Tensor:
    """单槽通道的冗余单例维（env builder 原始形状 ``(B,T,1,F)``/``(B,T,1)``）→ 挤压。

    ``dim`` 为该单例维的下标（历史特征=2，单槽当前帧=1）；不是单例则原样返回。
    """
    if tensor.ndim == dim:
        return tensor
    if tensor.ndim < dim:
        raise ValueError(f"obs[{name!r}] 维数不足（收到 {tuple(tensor.shape)}）")
    if int(tensor.shape[dim]) == 1:
        return tensor.squeeze(dim)
    return tensor


@dataclass
class MemBank:
    """4 个 per-modality 记忆 + 帧级 validity（内部按模态维护，便于 rollout 滑动）。

    形状：``ego (B,T,8)``、``od (B,T,S,9)``、``ld (B,T,L,7)``、``others (B,T,F_o)``；
    对应的 ``*_mask`` 为槽位/帧掩码，``*_valid`` 为帧级 validity（``hist_valid`` 的
    模态副本，合成帧挤入时置 1）。
    """

    ego: Tensor
    ego_mask: Tensor
    ego_valid: Tensor
    od: Tensor
    od_mask: Tensor
    od_valid: Tensor
    od_id: Tensor
    od_presence: Tensor
    ld: Tensor
    ld_mask: Tensor
    ld_valid: Tensor
    others: Tensor
    others_mask: Tensor
    others_valid: Tensor
    frames: int

    @property
    def batch(self) -> int:
        return int(self.ego.shape[0])

    def clone(self) -> "MemBank":
        """深拷贝 4 份 mem（rollout 的隔离副本；真 mem 绝不写回）。"""
        return MemBank(
            ego=self.ego.clone(),
            ego_mask=self.ego_mask.clone(),
            ego_valid=self.ego_valid.clone(),
            od=self.od.clone(),
            od_mask=self.od_mask.clone(),
            od_valid=self.od_valid.clone(),
            od_id=self.od_id.clone(),
            od_presence=self.od_presence.clone(),
            ld=self.ld.clone(),
            ld_mask=self.ld_mask.clone(),
            ld_valid=self.ld_valid.clone(),
            others=self.others.clone(),
            others_mask=self.others_mask.clone(),
            others_valid=self.others_valid.clone(),
            frames=int(self.frames),
        )

    # ---------------------------------------------------------------- 滑动窗口
    def _shift(self, tensor: Tensor, value: Tensor, name: str, tail: tuple[int, ...]) -> Tensor:
        if tuple(value.shape) != (self.batch, *tail):
            expect = ",".join(str(v) for v in tail)
            raise ValueError(f"{name} 形状应为 (B,{expect})，收到 {tuple(value.shape)}")
        return torch.cat([tensor[:, 1:], value.unsqueeze(1)], dim=1)

    def shift_ego(self, frame: Tensor, mask: Tensor | None = None) -> None:
        """挤入最新 ego 帧并弹出最老帧（只在 rollout 副本上调用）。"""
        ones = torch.ones((self.batch, ), dtype=frame.dtype, device=frame.device)
        self.ego = self._shift(self.ego, frame, "ego_frame", (EGO_MEM_DIM, ))
        self.ego_mask = self._shift(self.ego_mask, ones if mask is None else mask, "ego_mask", ())
        self.ego_valid = self._shift(self.ego_valid, ones, "ego_valid", ())

    def shift_od_ld(
        self,
        od_frame: Tensor,
        od_mask: Tensor,
        od_id: Tensor,
        od_presence: Tensor,
        ld_frame: Tensor,
        ld_mask: Tensor,
    ) -> None:
        """挤入最新 OD/LD 帧并弹出各自最老帧（只在 rollout 副本上调用）。"""
        ones = torch.ones((self.batch, ), dtype=od_frame.dtype, device=od_frame.device)
        slots = int(self.od.shape[2])
        self.od = self._shift(self.od, od_frame, "od_frame", (slots, OD_MEM_DIM))
        self.od_mask = self._shift(self.od_mask, od_mask, "od_mask", (slots, ))
        self.od_id = self._shift(self.od_id, od_id.long(), "od_id", (slots, ))
        self.od_presence = self._shift(self.od_presence, od_presence, "od_presence", (slots, ))
        self.od_valid = self._shift(self.od_valid, ones, "od_valid", ())
        self.ld = self._shift(self.ld, ld_frame, "ld_frame", (int(self.ld.shape[2]), LD_MEM_DIM))
        self.ld_mask = self._shift(self.ld_mask, ld_mask, "ld_mask", (int(self.ld.shape[2]), ))
        self.ld_valid = self._shift(self.ld_valid, ones, "ld_valid", ())


def mem_from_obs(obs: Mapping[str, Tensor], *, others_dim: int, history_frames: int) -> MemBank:
    """校验 obs 并组装 :class:`MemBank`（真 mem，只读）。

    键名解析与缺失回退见模块 docstring 的"兼容与回退"；任何形状/dtype 不符立即报错
    （避免静默错位）。
    """
    if not isinstance(obs, Mapping):
        raise TypeError(f"obs 必须是映射，收到 {type(obs).__name__}")

    keys: dict[str, str | None] = {}
    for name, candidates in _HISTORY_KEYS.items():
        keys[name] = _pick_key(obs, candidates, required=name in ("od", "ld"), what=f"{name} mem")
    if keys["od"] is None or keys["ld"] is None:
        raise ValueError("obs 必须提供 OD/LD 的 6 帧历史（od_hist/ld_hist，env schema v2）")
    if "hist_valid" not in obs:
        raise ValueError("obs 缺少必需通道 'hist_valid'（帧级 validity，warmup 补位帧为 0）")

    hist_valid = _require_float(obs["hist_valid"], "hist_valid")
    if hist_valid.ndim != 2 or int(hist_valid.shape[1]) != history_frames:
        raise ValueError(f"obs['hist_valid'] 形状应为 (B,{history_frames})，收到 {tuple(hist_valid.shape)}")
    batch = int(hist_valid.shape[0])
    frames = int(history_frames)

    # ---------------------------------------------------------------- OD / LD
    od = _require_float(obs[keys["od"]], str(keys["od"]))
    if od.ndim != 4:
        raise ValueError(f"obs[{keys['od']!r}] 形状应为 (B,{frames},16,{OD_MEM_DIM})，收到 {tuple(od.shape)}")
    _check_shape(od, keys["od"], (frames, int(od.shape[2]), OD_MEM_DIM), batch)  # type: ignore[index]
    od_mask_key = f"{keys['od']}_mask"  # type: ignore[index]
    if od_mask_key not in obs:
        raise ValueError(f"obs 缺少必需通道 {od_mask_key!r}")
    od_mask = _require_float(obs[od_mask_key], od_mask_key)
    _check_shape(od_mask, od_mask_key, (frames, int(od.shape[2])), batch)

    ld = _require_float(obs[keys["ld"]], str(keys["ld"]))
    if ld.ndim != 4:
        raise ValueError(f"obs[{keys['ld']!r}] 形状应为 (B,{frames},16,{LD_MEM_DIM})，收到 {tuple(ld.shape)}")
    _check_shape(ld, keys["ld"], (frames, int(ld.shape[2]), LD_MEM_DIM), batch)  # type: ignore[index]
    ld_mask_key = f"{keys['ld']}_mask"  # type: ignore[index]
    if ld_mask_key not in obs:
        raise ValueError(f"obs 缺少必需通道 {ld_mask_key!r}")
    ld_mask = _require_float(obs[ld_mask_key], ld_mask_key)
    _check_shape(ld_mask, ld_mask_key, (frames, int(ld.shape[2])), batch)

    slots = int(od.shape[2])

    # OD 身份（伴随数组，缺失时回退到当前帧 / 槽位下标）
    id_key = "od_id_hist" if "od_id_hist" in obs else None
    if id_key is not None:
        od_id = obs[id_key]
        if not torch.is_tensor(od_id):
            raise TypeError(f"obs[{id_key!r}] 必须是 Tensor，收到 {type(od_id).__name__}")
        if od_id.ndim != 3 or tuple(od_id.shape) != (batch, frames, slots):
            raise ValueError(f"obs[{id_key!r}] 形状应为 (B,{frames},{slots})，收到 {tuple(od_id.shape)}")
        od_id = od_id.long()
    elif "od_id" in obs:
        current_id = obs["od_id"]
        if not torch.is_tensor(current_id) or tuple(current_id.shape) != (batch, slots):
            raise ValueError(f"obs['od_id'] 形状应为 (B,{slots})，收到 {tuple(getattr(current_id, 'shape', ()))}")
        od_id = current_id.long().unsqueeze(1).expand(batch, frames, slots).clone()
    else:
        index = torch.arange(slots, dtype=torch.long, device=od.device)
        od_id = index.view(1, 1, -1).expand(batch, frames, slots).clone()

    # OD presence（对象本帧在盒内被观测）
    presence_key = "od_presence_hist" if "od_presence_hist" in obs else None
    if presence_key is not None:
        od_presence = _require_float(obs[presence_key], presence_key)
        _check_shape(od_presence, presence_key, (frames, slots), batch)
    elif "od_presence" in obs:
        current = obs["od_presence"]
        if not torch.is_tensor(current) or tuple(current.shape) != (batch, slots):
            raise ValueError(f"obs['od_presence'] 形状应为 (B,{slots})，收到 {tuple(getattr(current, 'shape', ()))}")
        od_presence = current.float().unsqueeze(1).expand(batch, frames, slots).clone()
    else:
        od_presence = od_mask * hist_valid.unsqueeze(-1)

    # ---------------------------------------------------------------- ego / others
    ego_key = keys["ego"]
    if ego_key is not None:
        ego = _require_float(obs[ego_key], ego_key)
        ego = _squeeze_dim(ego, ego_key, 2)
        _check_shape(ego, ego_key, (frames, EGO_MEM_DIM), batch)
        ego_mask_key = f"{ego_key}_mask"
        if ego_mask_key in obs:
            ego_mask = _require_float(obs[ego_mask_key], ego_mask_key)
            ego_mask = _squeeze_dim(ego_mask, ego_mask_key, 2)
            _check_shape(ego_mask, ego_mask_key, (frames, ), batch)
        else:
            ego_mask = torch.ones((batch, frames), dtype=torch.float32, device=ego.device)
    else:
        # 回退：当前帧 ego 复制 6 帧（旧契约没有 ego 历史）
        ego_now = obs.get("ego")
        if not torch.is_tensor(ego_now):
            raise ValueError("obs 缺少 'ego_hist'，且无 'ego' 当前帧可回退")
        ego_now = squeeze_batch_singletons(ego_now.float(), 2, "ego")
        _check_shape(ego_now, "ego", (EGO_MEM_DIM, ), batch)
        ego = ego_now.unsqueeze(1).expand(batch, frames, EGO_MEM_DIM)
        ego_mask = torch.ones((batch, frames), dtype=torch.float32, device=ego.device)

    others_key = keys["others"]
    if others_key is not None:
        others = _require_float(obs[others_key], others_key)
        others = _squeeze_dim(others, others_key, 2)
        if int(others.shape[-1]) != int(others_dim):
            raise ValueError(
                f"obs[{others_key!r}] 特征维 {int(others.shape[-1])} != others_dim {int(others_dim)}"
                f"（env schema v2 = nav(11)+speed_limit(1)+signal(4)+road_class(12) = 28；"
                f"如 env 改版请用 DrivingModel(others_dim=F_o) 构造）"
            )
        _check_shape(others, others_key, (frames, int(others_dim)), batch)
        others_mask_key = f"{others_key}_mask"
        if others_mask_key in obs:
            others_mask = _require_float(obs[others_mask_key], others_mask_key)
            others_mask = _squeeze_dim(others_mask, others_mask_key, 2)
            _check_shape(others_mask, others_mask_key, (frames, ), batch)
        else:
            others_mask = torch.ones((batch, frames), dtype=torch.float32, device=others.device)
    else:
        # 回退：当前帧 others 复制 6 帧；连当前帧都没有 → 全 0 + 无效
        others_now = obs.get("others")
        if torch.is_tensor(others_now):
            others_now = squeeze_batch_singletons(others_now.float(), 2, "others")
            _check_shape(others_now, "others", (int(others_dim), ), batch)
            others = others_now.unsqueeze(1).expand(batch, frames, int(others_dim))
            others_mask = torch.ones((batch, frames), dtype=torch.float32, device=others.device)
        else:
            others = torch.zeros(
                (batch, frames, int(others_dim)), dtype=torch.float32, device=od.device
            )
            others_mask = torch.zeros((batch, frames), dtype=torch.float32, device=od.device)

    return MemBank(
        ego=ego,
        ego_mask=ego_mask,
        ego_valid=hist_valid,
        od=od,
        od_mask=od_mask,
        od_valid=hist_valid,
        od_id=od_id,
        od_presence=od_presence,
        ld=ld,
        ld_mask=ld_mask,
        ld_valid=hist_valid,
        others=others,
        others_mask=others_mask,
        others_valid=hist_valid,
        frames=frames,
    )


# ---------------------------------------------------------------------- 世界系 nav 重建
def world_state_from_obs(obs: Mapping[str, object]) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    """从 obs 提取 ``(ego_world, route_world, route_world_mask)``；缺键返回 ``(None, None, None)``。"""
    if not isinstance(obs, Mapping):
        return None, None, None
    ego_world = obs.get("ego_world")
    route_world = obs.get("route_world")
    if not torch.is_tensor(ego_world) or not torch.is_tensor(route_world):
        return None, None, None
    route_mask = obs.get("route_world_mask")
    return ego_world, route_world, route_mask if torch.is_tensor(route_mask) else None


def _squeeze_pose(pose: Tensor) -> Tensor:
    if pose.ndim == 3 and int(pose.shape[1]) == 1:
        pose = pose[:, 0]
    if pose.ndim != 2 or int(pose.shape[1]) < 3:
        raise ValueError(f"ego_pose_world 形状应为 (B,3)（或 (B,1,3)），收到 {tuple(pose.shape)}")
    return pose


def _squeeze_route(route: Tensor) -> Tensor:
    if route.ndim == 4 and int(route.shape[1]) == 1:
        route = route[:, 0]
    if route.ndim != 3 or int(route.shape[-1]) != 2:
        raise ValueError(f"route_world 形状应为 (B,M,2)（或 (B,1,M,2)），收到 {tuple(route.shape)}")
    return route


def nav_features_from_world(
    ego_pose_world: Tensor,
    route_world: Tensor,
    route_world_mask: Tensor | None = None,
    *,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """按世界系位姿重投影世界系路线 → nav 特征 ``(B,NAV_DIM)`` + 有效掩码 ``(B,1)``。

    与 ``env.obs.nav.NavChannel`` 同口径：

    - checkpoint1/2 = 自车**前方**第 1/2 个路线顶点（顶点 = 路段终点，见
      ``env.obs.world.route_world_polyline``）在自车系下的 (x,y)（x 前 / y 左，不裁剪距离）；
    - 命令 = 当前所在路段弦向 vs 下一路段弦向（|Δ|<10° → forward，否则按叉积判 left/right，
      与 :func:`env.obs.nav.navigation_command` 同一逻辑）；
    - route_completion = 自车在折线上的投影弧长 / 折线总弧长（弦长近似；直线路段与上游
      ``travelled/total`` 逐位一致，弯道有 <1% 量级的弧-弦差）。

    输入一律 ``detach``（nav 是上下文条件，不向位姿链回传梯度）。``route_world_mask``
    缺省时视全部点为真实顶点；含无效点的行返回 ``mask=0``。
    """
    pose = _squeeze_pose(ego_pose_world).detach()
    route = _squeeze_route(route_world).detach()
    batch, points = int(route.shape[0]), int(route.shape[1])
    if int(pose.shape[0]) != batch:
        raise ValueError(f"ego_pose_world batch {int(pose.shape[0])} != route_world batch {batch}")
    if route_world_mask is None:
        valid = torch.ones((batch, points), dtype=torch.bool, device=route.device)
    else:
        mask = route_world_mask
        if mask.ndim == 3 and int(mask.shape[1]) == 1:
            mask = mask[:, 0]
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch, -1)
        if tuple(mask.shape) != (batch, points):
            raise ValueError(f"route_world_mask 形状应为 (B,M)=({batch},{points})，收到 {tuple(mask.shape)}")
        valid = mask.detach() > 0.5

    feats = torch.zeros((batch, NAV_DIM), dtype=route.dtype, device=route.device)
    nav_mask = torch.zeros((batch, 1), dtype=route.dtype, device=route.device)
    if points < 2:
        return feats, nav_mask

    seg = route[:, 1:] - route[:, :-1]  # (B,S,2)
    seg_len2 = (seg * seg).sum(dim=-1)
    seg_ok = valid[:, 1:] & valid[:, :-1] & (seg_len2 > eps)
    has_seg = seg_ok.any(dim=-1)  # (B,)
    if not bool(has_seg.any()):
        return feats, nav_mask

    rel = pose[:, None, :2] - route[:, :-1]  # (B,S,2)
    t = ((rel * seg).sum(dim=-1) / seg_len2.clamp(min=eps)).clamp(0.0, 1.0)
    proj = route[:, :-1] + t.unsqueeze(-1) * seg
    dist2 = ((proj - pose[:, None, :2]) ** 2).sum(dim=-1)
    inf = torch.full_like(dist2, float("inf"))
    best = torch.where(seg_ok, dist2, inf).argmin(dim=-1)  # (B,)

    seg_len = seg_len2.clamp(min=0.0).sqrt()
    cum = torch.cumsum(torch.where(seg_ok, seg_len, torch.zeros_like(seg_len)), dim=-1)  # (B,S)
    cum_v = torch.cat([torch.zeros((batch, 1), dtype=route.dtype, device=route.device), cum], dim=1)  # (B,M)
    gather_best = best.unsqueeze(1)
    s_ego = cum_v.gather(1, gather_best).squeeze(1) + t.gather(1, gather_best).squeeze(1) * seg_len.gather(
        1, gather_best
    ).squeeze(1)
    total = cum_v[:, -1]
    route_completion = (s_ego / total.clamp(min=eps)).clamp(0.0, 1.0)

    # 下一个 checkpoint = 弧长严格领先自车投影的第一个顶点（顶点 0 的弧长为 0）
    ahead = (cum_v <= (s_ego.unsqueeze(1) + eps)).sum(dim=-1)  # (B,)
    index1 = ahead.clamp(min=1, max=points - 1)
    index2 = (index1 + 1).clamp(max=points - 1)
    ck1 = route.gather(1, index1.view(batch, 1, 1).expand(batch, 1, 2)).squeeze(1)
    ck2 = route.gather(1, index2.view(batch, 1, 1).expand(batch, 1, 2)).squeeze(1)

    seg_dir = seg / seg_len.unsqueeze(-1).clamp(min=eps)
    dir0 = seg_dir.gather(1, gather_best.unsqueeze(-1).expand(batch, 1, 2)).squeeze(1)
    h0 = torch.atan2(dir0[:, 1], dir0[:, 0])
    seg_index = index1.clamp(max=max(int(seg.shape[1]) - 1, 0))
    dir1 = seg_dir.gather(1, seg_index.view(batch, 1, 1).expand(batch, 1, 2)).squeeze(1)
    h1 = torch.atan2(dir1[:, 1], dir1[:, 0])
    h1 = torch.where(seg_ok.gather(1, seg_index.unsqueeze(1)).squeeze(1), h1, h0)
    delta = torch.atan2(torch.sin(h0 - h1), torch.cos(h0 - h1))
    forward = delta.abs() < _TURN_EPS_RAD
    turning_left = torch.sin(h1 - h0) > 0.0
    command = torch.where(forward, torch.zeros_like(index1), torch.where(turning_left, torch.ones_like(index1), torch.full_like(index1, 2)))

    cos_t, sin_t = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
    for slot, ckpt in enumerate((ck1, ck2)):
        delta_xy = ckpt - pose[:, :2]
        feats[:, 2 * slot] = cos_t * delta_xy[:, 0] + sin_t * delta_xy[:, 1]
        feats[:, 2 * slot + 1] = -sin_t * delta_xy[:, 0] + cos_t * delta_xy[:, 1]
    feats.scatter_(1, (4 + command).unsqueeze(1), 1.0)
    feats[:, 4 + 6] = route_completion
    nav_mask[:, 0] = has_seg.to(route.dtype)
    return feats, nav_mask


def rebuild_nav_from_world(
    mem: MemBank,
    ego_pose_world: Tensor | None,
    route_world: Tensor | None,
    route_world_mask: Tensor | None = None,
    *,
    nav_dim: int = NAV_DIM,
) -> tuple[Tensor, Tensor]:
    """**固定签名 helper**：按世界系位姿重建 nav，返回 ``(nav_feats (B,NAV_DIM), nav_mask (B,1))``。

    调用方用 ``model.encoders.embed_nav(nav_feats.unsqueeze(1), nav_mask)`` 得 nav token；
    需要把重建值写回 ``mem.others`` 时用 :func:`sync_others_nav_dims`（**纯函数**，返回新
    MemBank）——本函数**绝不原地改写** ``mem``（原地写会让此前 encode 的 autograd 图报
    "modified by an inplace operation"，见 teacher forcing backward 回归测试）。

    向后兼容：``route_world``/``ego_world`` 缺失（旧 ckpt/旧数据）或整批路线不可用 →
    返回 mem 里现存的 nav 子向量（旧行为：t0 冻结）+ 一次性 ``RuntimeWarning``；
    逐行路线不可用时该行回退到旧值（mask=0）。
    """
    fallback_feats = mem.others[:, -1, : int(nav_dim)].detach()
    fallback_mask = torch.ones((mem.batch, 1), dtype=mem.others.dtype, device=mem.others.device)
    if ego_pose_world is None or route_world is None:
        _warn_nav_fallback(
            "rebuild_nav_from_world",
            "obs 缺少 ego_world/route_world（旧数据/旧 ckpt）→ nav 回退 t0 冻结（A4 重建未启用）",
        )
        return fallback_feats, fallback_mask
    if not torch.is_tensor(ego_pose_world) or not torch.is_tensor(route_world):
        raise TypeError("ego_pose_world / route_world 必须是 torch.Tensor 或 None")
    feats, mask = nav_features_from_world(ego_pose_world, route_world, route_world_mask)
    valid_rows = mask.reshape(-1) > 0.5
    if not bool(valid_rows.all()):
        if not bool(valid_rows.any()):
            _warn_nav_fallback(
                "rebuild_nav_from_world",
                "route_world 不可用（无真实顶点/折线退化）→ nav 回退 t0 冻结（A4 重建未启用）",
            )
        else:
            _warn_nav_fallback(
                "rebuild_nav_from_world_partial",
                "route_world 仅部分 batch 行可用 → 不可用行回退 t0 nav（A4 逐行回退）",
            )
        feats = torch.where(
            valid_rows.unsqueeze(1), feats, fallback_feats.to(device=feats.device, dtype=feats.dtype)
        )
    return feats, mask


def sync_others_nav_dims(mem: MemBank, nav_features: Tensor, *, nav_dim: int = NAV_DIM) -> MemBank:
    """others-nav 维同步 helper（**纯函数**）：返回 others 当前帧 nav 子向量被替换后的新 MemBank。

    与 :func:`rebuild_nav_from_world` 配套：``mem = sync_others_nav_dims(mem, nav_feats)``。
    在 ``mem.others.clone()`` 上赋值（clone 未参与任何既有 autograd 图，安全），原 ``mem``
    对象及其张量不被修改——这是 teacher forcing / rollout 可反向的必要条件。
    """
    if nav_features.ndim != 2 or int(nav_features.shape[-1]) != int(nav_dim):
        raise ValueError(f"nav_features 形状应为 (B,{int(nav_dim)})，收到 {tuple(nav_features.shape)}")
    others = mem.others.clone()
    others[:, -1, : int(nav_dim)] = nav_features.to(device=others.device, dtype=others.dtype)
    return replace(mem, others=others)


def advance_pose_world(pose_world: Tensor, ds: Tensor, dtheta: Tensor, *, eps: float = 1e-6) -> Tensor:
    """世界系位姿按一步策略动作 ``(ds,dθ)`` 推进（与 ``net.model.arc_step``+``compose_pose`` 同口径）。

    ``(ds,dθ)`` 为**当前自车系**下的圆弧位移（恒曲率，y 左向）；返回 ``(B,3)`` 世界位姿。
    rollout / 教师强制用同一函数累积位姿链，保证"同一步进语义"。
    """
    if pose_world.ndim == 3 and int(pose_world.shape[1]) == 1:
        pose_world = pose_world[:, 0]
    if pose_world.ndim != 2 or int(pose_world.shape[1]) != 3:
        raise ValueError(f"pose_world 形状应为 (B,3)，收到 {tuple(pose_world.shape)}")
    small = dtheta.abs() < eps
    safe = torch.where(small, torch.ones_like(dtheta), dtheta)
    radius = ds / safe
    dx = torch.where(small, ds, radius * torch.sin(dtheta))
    dy = torch.where(small, torch.zeros_like(ds), radius * (1.0 - torch.cos(dtheta)))
    theta = pose_world[:, 2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    new_theta = torch.atan2(torch.sin(theta + dtheta), torch.cos(theta + dtheta))
    return torch.stack(
        [pose_world[:, 0] + cos_t * dx - sin_t * dy, pose_world[:, 1] + sin_t * dx + cos_t * dy, new_theta],
        dim=-1,
    )


@dataclass
class EncodedMem:
    """编码后的 mem：4 个模态的聚合 + 当前帧（供 plan head / ST-GNN 消费）。"""

    #: ``(B,H)`` ego 六帧注意力聚合
    ego_ctx: Tensor
    #: ``(B,S,H)`` OD 六帧逐槽注意力聚合
    od_ctx: Tensor
    #: ``(B,L,H)`` LD 当前帧编码（不做时序）
    ld_ctx: Tensor
    #: ``(B,H)`` others 六帧注意力聚合
    others_ctx: Tensor
    #: ``(B,S)`` OD 当前帧 live 掩码（mask & presence）
    od_live: Tensor
    #: ``(B,L)`` LD 当前帧掩码
    ld_live: Tensor
    #: ``(B,8)`` 当前帧 ego 原始特征
    ego_now: Tensor
    #: ``(B,S,9)`` 当前帧 OD 原始特征（rollout 合成帧的静态属性 carry）
    od_now: Tensor
    #: ``(B,L,7)`` 当前帧 LD 原始特征（静态属性 carry）
    ld_now: Tensor
    #: ``(B,S)`` 当前帧 OD presence
    od_presence_now: Tensor
    #: ``(B,S)`` 当前帧 OD id（long）
    od_id_now: Tensor
    #: 当前帧节点（`[ego, OD, LD]`，含 pose），供 ST-GNN 构图
    frame: FrameEncoding


class MemEncoder(nn.Module):
    """把 :class:`MemBank` 编码为 :class:`EncodedMem`（OD/Ego/Others 注意力，LD 当前帧）。"""

    def __init__(self, hidden: int = H):
        super().__init__()
        self.hidden = int(hidden)
        self.ego_attn = TemporalAttention(hidden)
        self.od_attn = TemporalAttention(hidden)
        self.others_attn = TemporalAttention(hidden)

    def encode(self, encoders: ObsEncoders, mem: MemBank) -> EncodedMem:
        """编码 mem。

        LD 不做时序（规格第 2 条）：直接取当前帧 ``ld_hist[:, -1]`` 的嵌入。
        理由与前提：**当 LD 帧按下标对齐到各自自车系且对齐正确时，LD 历史相对当前帧
        近似恒等**——同一批世界系车道点在每个时刻的"自车系表达"只差一个已知的
        SE(2) 变换，而车道几何本身不随时间变化；因此历史帧几乎不提供额外信息
        （反而引入 ego 运动补偿误差）。前提是 env 的对齐/槽位排序一致（同一槽位
        对应同一条车道线）；若对齐不可靠，需要改为对 LD 也做时序聚合。
        """
        od_h = encoders.embed_od(mem.od, mem.od_mask, ids=mem.od_id)
        ld_h = encoders.embed_ld(mem.ld, mem.ld_mask)
        ego_h = encoders.embed_ego(mem.ego, mem.ego_mask).unsqueeze(2)  # (B,T,1,H)
        others_h = encoders.embed_others(mem.others, mem.others_mask).unsqueeze(2)

        ego_ctx = self.ego_attn(ego_h, None, mem.ego_valid)[:, 0]
        od_ctx = self.od_attn(od_h, mem.od_mask, mem.od_valid)
        ld_ctx = ld_h[:, -1]  # LD：不做时序，直接用当前帧
        others_ctx = self.others_attn(others_h, None, mem.others_valid)[:, 0]

        od_live = (mem.od_mask[:, -1] > 0.5) & (mem.od_presence[:, -1] > 0.5)
        ld_live = mem.ld_mask[:, -1] > 0.5
        ego_now = mem.ego[:, -1]
        od_now = mem.od[:, -1]
        ld_now = mem.ld[:, -1]
        zeros = torch.zeros((mem.batch, 1, 3), dtype=ego_now.dtype, device=ego_now.device)
        frame = FrameEncoding(
            nodes=torch.cat([ego_h[:, -1], od_h[:, -1], ld_h[:, -1]], dim=1),
            node_mask=torch.cat(
                [
                    torch.ones((mem.batch, 1), dtype=ego_now.dtype, device=ego_now.device),
                    od_live.to(ego_now.dtype),
                    ld_live.to(ego_now.dtype),
                ],
                dim=1,
            ),
            type_ids=torch.cat(
                [
                    torch.zeros((mem.batch, 1), dtype=torch.long, device=ego_now.device),
                    torch.full((mem.batch, int(od_now.shape[1])), 1, dtype=torch.long, device=ego_now.device),
                    torch.full((mem.batch, int(ld_now.shape[1])), 2, dtype=torch.long, device=ego_now.device),
                ],
                dim=1,
            ),
            pose=torch.cat(
                [
                    zeros,
                    encoders.od_pose(od_now, od_live.to(ego_now.dtype)),
                    encoders.ld_pose(ld_now, ld_live.to(ego_now.dtype)),
                ],
                dim=1,
            ),
            ego_feat=ego_now,
            od_feat=od_now,
            od_mask=od_live.to(ego_now.dtype),
            ld_feat=ld_now,
            ld_mask=ld_live.to(ego_now.dtype),
        )
        return EncodedMem(
            ego_ctx=ego_ctx,
            od_ctx=od_ctx,
            ld_ctx=ld_ctx,
            others_ctx=others_ctx,
            od_live=od_live.to(ego_now.dtype),
            ld_live=ld_live.to(ego_now.dtype),
            ego_now=ego_now,
            od_now=od_now,
            ld_now=ld_now,
            od_presence_now=mem.od_presence[:, -1],
            od_id_now=mem.od_id[:, -1].long(),
            frame=frame,
        )
