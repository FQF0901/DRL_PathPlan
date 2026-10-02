"""观测编码器：把原始通道特征投影到隐藏维 H，并组装空间图节点。

与 ``env/obs`` 的接口约定（形状/语义以该目录为准）
---------------------------------------------------
- 自车系 x 前向 / y 左向；
- OD 9 维：``[dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]``（vx/vy 为相对速度）；
- LD 7 维：``[dx, dy, heading_rel, curvature, speed_limit, left_line_type_id, right_line_type_id]``；
- ego 8 维：``[v, a_long, a_lat, yaw_rate, steer, curvature, reserved0, reserved1]``。
  **最后 2 维 reserved 承载上一策略步 (ds, dθ)**（p2-contract §8.4）：本模块按普通特征消费，
  不做任何特殊处理；采集侧负责写入，训练侧由此获得动作历史；
- others ``F_o`` 维：**规范上下文向量** = nav(11) + speed_limit(1) + signal(4) + static(5)
  + road_class one-hot(K)（env schema v4，``env/obs/others.py`` / ``env/obs/static.py`` 定义
  语义；net 只做投影，默认 F_o=33）。旧布局（schema v2/v3，28 维）由 ``net.mem.mem_from_obs``
  零填充 static 段后消费（一次性告警）；
- nav 11 维：2 个 checkpoint（自车系点）+ 6 命令 one-hot + route_completion；
- signal 4 维：绿/黄/红/未知 one-hot（本项目恒为未知占位）。

v2（mem-bank）说明：编码器权重在 **mem 帧 / rollout 合成帧 / 当前帧** 之间共享——同一个
``embed_od/embed_ld/embed_ego/embed_others`` 既编码 6 帧 mem（``(B,T,S,F)``），也编码
rollout 里重新拼出的单帧；``embed_od`` 还接受可选 ``od_id (B,T,S)`` 做轨道身份桶嵌入
（``id % ID_BUCKETS``，桶化以避免未知 id 词表；env 只需给出稳定非负 id）。

节点顺序固定为 ``[ego, od_0..od_{K-1}, ld_0..ld_{K-1}]``（默认 K=16，共 33 个空间节点）；
nav/signal/others 作为全局上下文 token 单独返回，不参与空间消息传递。

无效槽位（mask=0）的嵌入被显式清零，保证下游对填充值不敏感。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

#: 主干隐藏维（config/model.yaml: hidden_dim）
H = 96
EGO_DIM = 8
OD_DIM = 9
LD_DIM = 7
NAV_DIM = 11
SIGNAL_DIM = 4
OD_SLOTS = 16
LD_SLOTS = 16
HISTORY_FRAMES = 6

#: mem-bank 形状（v2 契约）：``*_mem`` 时间轴 T=HISTORY_FRAMES，最后一帧 = 当前帧。
EGO_MEM_DIM = EGO_DIM
OD_MEM_DIM = OD_DIM
LD_MEM_DIM = LD_DIM
#: ``others_mem`` 的特征维 F_o（env schema v4：
#: ``nav(11)+speed_limit(1)+signal(4)+static(5)+road_class(K=12)`` = 33）。
#: 语义由 ``env/obs/others.py`` / ``env/obs/static.py`` 定义；net 只做投影，
#: 构造 ``DrivingModel(others_dim=...)`` 可覆盖。
DEFAULT_OTHERS_DIM = 33
#: 旧 others 布局（env schema v2/v3）：``nav(11)+speed_limit(1)+signal(4)+road_class(12)`` = 28；
#: ``net.mem.mem_from_obs`` 对旧数据做"零填充 static 段"的重排（一次性告警）。
OTHERS_LEGACY_DIM = 28
#: 旧布局中 static 段之前的公共前缀维数（nav + speed_limit + signal）
OTHERS_HEAD_LEGACY_DIM = 16
#: v4 新增 static 段维数（present + gap_norm + 相对车道 one-hot(3)）
OTHERS_STATIC_DIM = 5
#: OD 轨道 id 的桶数（``id % ID_BUCKETS`` 查嵌入；避免未知 id 词表爆炸）。
ID_BUCKETS = 64

#: 节点类型 id（用于 type embedding；顺序与 FrameEncoding.nodes 一致）
TYPE_EGO = 0
TYPE_OD = 1
TYPE_LD = 2
TYPE_NAV = 3
TYPE_SIGNAL = 4
NUM_NODE_TYPES = 5


@dataclass
class FrameEncoding:
    """单帧（当前观测帧）的编码结果；N = 1 + od_slots + ld_slots。"""

    #: ``(B, N, H)`` 节点嵌入（ego / OD / LD），无效槽位为 0
    nodes: Tensor
    #: ``(B, N)`` 节点有效掩码
    node_mask: Tensor
    #: ``(B, N)`` 节点类型 id（long）
    type_ids: Tensor
    #: ``(B, N, 3)`` 节点自车系位姿 ``(x, y, θ)``，ego 恒为 (0,0,0)，无效槽位为 0
    pose: Tensor
    #: ``(B, 8)`` ego 原始特征（供世界模型/ego 条件化使用）
    ego_feat: Tensor
    #: ``(B, K, 9)`` OD 原始特征
    od_feat: Tensor
    #: ``(B, K)`` OD 掩码
    od_mask: Tensor
    #: ``(B, L, 7)`` LD 原始特征
    ld_feat: Tensor
    #: ``(B, L)`` LD 掩码
    ld_mask: Tensor


class ObsEncoders(nn.Module):
    """各通道共享结构的线性投影 + 类型嵌入 + LayerNorm。

    编码器权重在"mem 帧 / 历史帧 / rollout 合成帧 / 当前帧"之间共享（OD/LD 用同一投影），
    保证时序注意力与 rollout 重编码看到的特征空间一致。
    """

    def __init__(
        self,
        hidden: int = H,
        od_slots: int = OD_SLOTS,
        ld_slots: int = LD_SLOTS,
        others_dim: int = DEFAULT_OTHERS_DIM,
        id_buckets: int = ID_BUCKETS,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.others_dim = int(others_dim)
        self.id_buckets = int(id_buckets)
        self.ego = nn.Linear(EGO_DIM, hidden)
        self.od = nn.Linear(OD_DIM, hidden)
        self.ld = nn.Linear(LD_DIM, hidden)
        self.others = nn.Linear(self.others_dim, hidden)
        self.nav = nn.Linear(NAV_DIM, hidden)
        self.signal = nn.Linear(SIGNAL_DIM, hidden)
        self.type_embed = nn.Embedding(NUM_NODE_TYPES, hidden)
        self.id_embed = nn.Embedding(self.id_buckets, hidden)
        self.norm = nn.LayerNorm(hidden)

    # ---------------------------------------------------------------- 单通道嵌入
    def _embed(
        self,
        linear: nn.Linear,
        feat: Tensor,
        mask: Tensor,
        type_id: int,
        extra: Tensor | None = None,
    ) -> Tensor:
        """``feat (..., F)`` + ``mask (...,)`` -> ``(..., H)``；无效位置严格为 0。"""
        mask = mask.reshape(feat.shape[:-1])
        h = linear(feat)
        if feat.ndim == 2:
            # 2 维输入（(B,F)）：类型嵌入取 (H,)，避免 (H,)+(B,1,H) 的意外广播
            type_bias = self.type_embed.weight[int(type_id)]
        else:
            ids = torch.full(feat.shape[:-1], int(type_id), dtype=torch.long, device=feat.device)
            type_bias = self.type_embed(ids)
        h = h + type_bias
        if extra is not None:
            h = h + extra
        return self.norm(h) * mask.unsqueeze(-1)

    def embed_ego(self, feat: Tensor, mask: Tensor) -> Tensor:
        return self._embed(self.ego, feat, mask, TYPE_EGO)

    def embed_od(self, feat: Tensor, mask: Tensor, ids: Tensor | None = None) -> Tensor:
        """OD 嵌入；``ids (...,)`` 为轨道身份（任意整数/浮点，内部按桶查表）。"""
        extra = None
        if ids is not None:
            buckets = torch.remainder(ids.long(), self.id_buckets)
            extra = self.id_embed(buckets)
        return self._embed(self.od, feat, mask, TYPE_OD, extra=extra)

    def embed_ld(self, feat: Tensor, mask: Tensor) -> Tensor:
        return self._embed(self.ld, feat, mask, TYPE_LD)

    def embed_others(self, feat: Tensor, mask: Tensor) -> Tensor:
        return self._embed(self.others, feat, mask, TYPE_OD)

    def embed_nav(self, feat: Tensor, mask: Tensor) -> Tensor:
        return self._embed(self.nav, feat, mask, TYPE_NAV)

    def embed_signal(self, feat: Tensor, mask: Tensor) -> Tensor:
        return self._embed(self.signal, feat, mask, TYPE_SIGNAL)

    # ---------------------------------------------------------------- 位姿提取
    @staticmethod
    def od_pose(feat: Tensor, mask: Tensor) -> Tensor:
        """OD 位姿：位置取 ``dx,dy``，朝向取 ``atan2(sinθ, cosθ)``（自车系）。

        空槽位（``cos=sin=0``）会让 ``atan2(0,0)`` 的反向产生 NaN，并被 ``mask`` 乘法
        以 ``0 × NaN = NaN`` 的方式污染梯度（trainer 线曾为此加临时补丁）。这里先判退化：
        退化槽位用 ``(cos=1, sin=0)``（前向仍被 ``mask`` 乘为 0，与旧实现一致），反向有限。
        """
        cos_t = feat[..., 4]
        sin_t = feat[..., 5]
        norm = torch.sqrt(cos_t * cos_t + sin_t * sin_t)
        safe = norm > 1e-6
        cos_safe = torch.where(safe, cos_t, torch.ones_like(cos_t))
        sin_safe = torch.where(safe, sin_t, torch.zeros_like(sin_t))
        heading = torch.atan2(sin_safe, cos_safe).unsqueeze(-1)
        return torch.cat([feat[..., 0:2], heading], dim=-1) * mask.unsqueeze(-1)

    @staticmethod
    def ld_pose(feat: Tensor, mask: Tensor) -> Tensor:
        """LD 位姿：位置取 ``dx,dy``，朝向取 ``heading_rel``（自车系）。"""
        return torch.cat([feat[..., 0:2], feat[..., 2:3]], dim=-1) * mask.unsqueeze(-1)

    # ---------------------------------------------------------------- 历史帧
    # 说明：v2 mem-bank 直接用 ``embed_od/embed_ld/embed_ego/embed_others`` 编码整段 6 帧
    # mem（见 net.mem.MemEncoder）；旧 v1 的 ``embed_history_frame``/``encode_history``
    # 入口在 net 重构后已无调用方（pipeline 侧自行组装 ``*_hist`` 数组），已移除。
