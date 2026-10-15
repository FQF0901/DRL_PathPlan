"""R21-3A 语义接线：``lane_semantic`` → ``target_lane_token`` + role-aware 分组读出。

契约（round-23 §4；要点与取档见 ``docs/repo_handbook.md`` 附录 D）：

- **入口固定归一化**：12 维原始特征逐元素除以
  :data:`env.obs.lane_semantic.LANE_SEMANTIC_NORMALIZATION`（常数入 schema/模型快照 +
  逐位单测）；不得在训练脚本里散落。
- **target_lane_token**：``lane_semantic → 归一化 → 2 层 MLP → (B,H)``；
  ``lane_semantic_mask=0 ∨ target_lane_valid=0 ⇒ token 严格 0``（MLP 后乘掩码，非"近似 0"）。
  **第一版不进 MemBank**（当前语义来自当前真实 pose/route，作为当前条件；temporal 留待
  独立消融）。
- **LD 逐槽增广**：``z_ld_slot + role_embed + route_associated_embed + offset_s 编码 +
  target-group relation``；``ld_group`` **只做 equality/embedding，禁作数值大小**；
  无效槽增广后仍严格 0（乘 ld_live）。
- **显式分组池化（禁自由注意力）**：target / continuation / left_adjacent / right_adjacent /
  route / other 六组 masked mean（空组 ⇒ 严格零向量）；
  ``ld_context = MLP([target_lane_token, target_pool, continuation_pool, left_pool, right_pool,
  route_pool, other_pool])``。
- 统一消费：``net.model.DrivingModel.build_geometry_context`` 是唯一入口（PlanHead t0 +
  rollout 1–6 + policy/value head tokens）；semantic 模式下 ``plan_step`` 不允许回退
  ``masked_mean``（缺 context 直接报错）。

符号/编码：role 0..4（见 ``env.obs.lane_semantic.ROLE_*``），无效槽 role=-1 → embedding
index 5；group relation 4 类（0=target 槽、1=同 group、2=异 group、3=未知/无 target）。

R38-P3 来源标签（GPT §七-F；只标注，不改行为）
------------------------------------------------
``target_lane_token`` / ``lane_semantic`` 来源 = **客观 lane topology**
（:data:`TARGET_LANE_SOURCE_TAGS` 中的 ``road_level_navigation`` + ``deployment_available``）；
训练/评测教师注入的 target（``lane_anchor_xy`` 注入、train-side target-lane dropout）为
``teacher_target_lane``（teacher-only；部署不可用）——长期移除生产部署对唯一 target lane
的依赖，课程训练与诊断路径暂不删除。完整字段映射 = ``env.obs.lane_semantic.TARGET_LANE_FIELD_SOURCE_TAGS``。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from env.obs.lane_semantic import (
    LANE_SEMANTIC_DIM,
    LANE_SEMANTIC_NORMALIZATION,
    ROLE_LEFT_ADJACENT,
    ROLE_RIGHT_ADJACENT,
    ROLE_TARGET,
    ROLE_TARGET_CONTINUATION,
    TARGET_LANE_FIELD_SOURCE_TAGS,
    TARGET_LANE_SOURCE_TAGS,
)

__all__ = [
    "LaneSemanticEncoder",
    "SemanticWiring",
    "NUM_ROLES",
    "ROLE_INVALID_INDEX",
    "NUM_GROUP_RELATIONS",
    "GROUP_REL_TARGET",
    "GROUP_REL_SAME",
    "GROUP_REL_OTHER",
    "GROUP_REL_UNKNOWN",
    "SEMANTIC_AUX_TARGETS",
    "TARGET_LANE_SOURCE_TAGS",
    "TARGET_LANE_TOKEN_SOURCE_TAGS",
]

#: R38-P3（GPT §七-F）来源标签：``target_lane_token`` = 客观 lane topology 的 net 侧编码
#: （来自 ``lane_semantic``；部署期 env 自身可产出）。标签枚举/字段映射单一出处 =
#: ``env.obs.lane_semantic.TARGET_LANE_*``；本模块只做转出（behavior-free）。
TARGET_LANE_TOKEN_SOURCE_TAGS: tuple[str, ...] = TARGET_LANE_FIELD_SOURCE_TAGS["target_lane_token"]

#: role 取值 0..4（5 类）+ 无效槽 index 5
NUM_ROLES = 5
ROLE_INVALID_INDEX = NUM_ROLES
#: group relation 编码（ld_group 只做 equality）
GROUP_REL_TARGET = 0
GROUP_REL_SAME = 1
GROUP_REL_OTHER = 2
GROUP_REL_UNKNOWN = 3
NUM_GROUP_RELATIONS = 4

#: 3B 语义 aux 的监督目标名（仅这些；margin/topology **不监督**，round-23 §6B）
SEMANTIC_AUX_TARGETS: tuple[str, ...] = ("lat", "heading", "curv_near", "curv_far", "lane_change")


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """``(B,S,H)`` 按 ``(B,S)`` 掩码求均值；全无效行返回**严格 0**（空组 ⇒ 零向量）。"""
    return (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


class LaneSemanticEncoder(nn.Module):
    """``lane_semantic (B,12)`` → ``target_lane_token (B,H)``（固定归一化 + 2 层 MLP）。

    ``normalization`` 为持久 buffer（随 ckpt 保存，供模型快照/复现核对）；前向
    ``token = MLP(feats / normalization) * valid``（valid=0 → **严格 0**）。
    """

    def __init__(self, hidden: int, in_dim: int = LANE_SEMANTIC_DIM):
        super().__init__()
        self.hidden = int(hidden)
        self.in_dim = int(in_dim)
        self.register_buffer(
            "normalization",
            torch.tensor(LANE_SEMANTIC_NORMALIZATION, dtype=torch.float32),
            persistent=True,
        )
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )

    def forward(self, feats: Tensor, valid: Tensor) -> Tensor:
        """``feats (B,12)`` / ``valid (B,1)``（1=有效）→ ``(B,H)``；valid=0 严格 0。"""
        if feats.ndim != 2 or int(feats.shape[-1]) != self.in_dim:
            raise ValueError(f"lane_semantic 应为 (B,{self.in_dim})，收到 {tuple(feats.shape)}")
        if valid.ndim != 2 or int(valid.shape[-1]) != 1:
            raise ValueError(f"lane_semantic_mask 应为 (B,1)，收到 {tuple(valid.shape)}")
        normalized = feats / self.normalization.to(dtype=feats.dtype)
        return self.net(normalized) * valid.to(dtype=feats.dtype)


class SemanticWiring(nn.Module):
    """LD 逐槽增广 + 显式分组池化读出（见模块 docstring 契约）。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.lane_encoder = LaneSemanticEncoder(self.hidden)
        self.role_embed = nn.Embedding(NUM_ROLES + 1, self.hidden)
        self.route_embed = nn.Embedding(2, self.hidden)
        self.offset_encoder = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.group_relation_embed = nn.Embedding(NUM_GROUP_RELATIONS, self.hidden)
        self.readout = nn.Sequential(
            nn.Linear(7 * self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        # 稳定初始化：增广项初始接近 0（aug ≈ z_ld），offset 编码末层零初始化。
        for embedding in (self.role_embed, self.route_embed, self.group_relation_embed):
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.zeros_(self.offset_encoder[-1].weight)
        nn.init.zeros_(self.offset_encoder[-1].bias)

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        z_ld: Tensor,
        ld_live: Tensor,
        *,
        lane_semantic: Tensor,
        lane_semantic_mask: Tensor,
        role: Tensor,
        offset_s: Tensor,
        route_assoc: Tensor,
        group: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """返回 ``(ld_context (B,H), target_lane_token (B,H))``。

        输入（均在 current frame；形状见 :meth:`net.model.DrivingModel._semantic_inputs_from_obs`）：

        - ``z_ld (B,L,H)``：LD 逐槽嵌入（rollout 逐步传入当步 latent）；
        - ``ld_live (B,L)``：LD 槽掩码；
        - ``lane_semantic (B,12)`` / ``lane_semantic_mask (B,1)``；
        - ``role (B,L) int64``（-1=无效）、``offset_s (B,L)``、``route_assoc (B,L)``、
          ``group (B,L) int64``（-1=无效；只做 equality）。
        """
        batch, slots = int(z_ld.shape[0]), int(z_ld.shape[1])
        valid_sem = (lane_semantic_mask > 0.5) & (lane_semantic[:, 6:7] > 0.5)
        target_token = self.lane_encoder(lane_semantic, valid_sem.to(lane_semantic.dtype))

        live = ld_live > 0.5
        role_idx = torch.where(
            role >= 0, role, torch.full_like(role, ROLE_INVALID_INDEX)
        ).clamp(0, ROLE_INVALID_INDEX)
        route_idx = (route_assoc > 0.5).to(torch.int64).clamp(0, 1)
        relation = self._group_relation(role, group, live)

        aug = (
            z_ld
            + self.role_embed(role_idx)
            + self.route_embed(route_idx)
            + self.offset_encoder(offset_s.unsqueeze(-1))
            + self.group_relation_embed(relation)
        )
        aug = aug * ld_live.unsqueeze(-1)  # 无效槽严格 0

        target_mask = live & (role == ROLE_TARGET)
        continuation_mask = live & (role == ROLE_TARGET_CONTINUATION)
        left_mask = live & (role == ROLE_LEFT_ADJACENT)
        right_mask = live & (role == ROLE_RIGHT_ADJACENT)
        route_mask = live & (route_assoc > 0.5)
        classified = target_mask | continuation_mask | left_mask | right_mask
        other_mask = live & ~classified

        pools = [
            _masked_mean(aug, target_mask.to(aug.dtype)),
            _masked_mean(aug, continuation_mask.to(aug.dtype)),
            _masked_mean(aug, left_mask.to(aug.dtype)),
            _masked_mean(aug, right_mask.to(aug.dtype)),
            _masked_mean(aug, route_mask.to(aug.dtype)),
            _masked_mean(aug, other_mask.to(aug.dtype)),
        ]
        context = self.readout(torch.cat([target_token] + pools, dim=-1))
        return context, target_token

    @staticmethod
    def _group_relation(role: Tensor, group: Tensor, live: Tensor) -> Tensor:
        """group relation（0=target 槽 / 1=同 group / 2=异 group / 3=未知）：只做 equality。"""
        target_mask = live & (role == ROLE_TARGET)
        has_target = target_mask.any(dim=1, keepdim=True)
        target_group = torch.where(
            target_mask, group, torch.full_like(group, -1)
        ).max(dim=1, keepdim=True).values
        same_group = has_target & (group == target_group) & (group >= 0)
        relation = torch.full_like(group, GROUP_REL_UNKNOWN)
        relation = torch.where(same_group & target_mask, torch.full_like(group, GROUP_REL_TARGET), relation)
        relation = torch.where(same_group & ~target_mask, torch.full_like(group, GROUP_REL_SAME), relation)
        relation = torch.where(
            has_target & live & ~same_group, torch.full_like(group, GROUP_REL_OTHER), relation
        )
        return relation.clamp(0, NUM_GROUP_RELATIONS - 1)

    # ---------------------------------------------------------------- 诊断
    def normalization_constants(self) -> tuple[float, ...]:
        """入口归一化常数（模型快照/诊断用；与 ``env.obs.lane_semantic`` 同源）。"""
        return tuple(float(v) for v in self.lane_encoder.normalization.detach().cpu().tolist())

    def parameter_names(self) -> tuple[str, ...]:
        """本模块参数/buffer 名（旧 ckpt fail-fast 白名单用）。"""
        names = [name for name, _ in self.named_parameters()]
        names += [name for name, _ in self.named_buffers()]
        return tuple(sorted(names))
