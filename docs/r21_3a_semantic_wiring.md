# R21-3A 语义接线契约（net 侧；默认关、baseline 逐位不变）

状态：R21-3A 实现契约（2026-10-13）；实现出处 `net/semantic.py` +
`net/model.py`（`set_ld_mode("semantic")` / `build_geometry_context` / `plan_step`）。
依据：`docs/chatgpt5p6sol_review.md` round-23 §4/§6A；`docs/deepseekv4p1_argue.md` §5.15。

## 1. 模式开关

- `ld_mode` 新增取值 **`semantic`**（默认仍 `baseline`，逐位不变；既有
  `baseline/no_chain/set_attn/attn/attn_no_chain` 语义不动）。
- `semantic` 模式**惰性构造** `net.semantic.SemanticWiring`（旧 ckpt 键集不变）；
  LD-LD 链与 LD 节点 MP 保持 baseline 口径。
- **旧 schema/dataset/ckpt fail-fast**：`semantic` 模式前向缺
  `lane_semantic/lane_semantic_mask/ld_lane_role/ld_offset_s_norm/ld_route_associated/ld_group`
  任一键 → `ValueError`；旧 ckpt 用 `strict=True` 加载 → `RuntimeError`（
  `strict=False` 时 missing 必须 ⊆ `semantic_wiring.*` 白名单，否则拒绝）。

## 2. target_lane_token

```
lane_semantic (B,12) → ÷LANE_SEMANTIC_NORMALIZATION → 2 层 MLP (12→H→H, GELU) → (B,H)
mask=0 ∨ target_lane_valid=0 ⇒ token 严格 0（MLP 后乘掩码）
```

- 归一化常数单一出处 `env.obs.lane_semantic.LANE_SEMANTIC_NORMALIZATION`
  （schema `lane_semantic_layout.normalization` 同源；编码器持久 buffer 随 ckpt 保存）。
- **第一版不进 MemBank**：语义 = 当前真实 pose/route 的条件（t0 静态）；rollout 每步用
  当步 `z_ld` + 同一 t0 语义条件重建 context。temporal 平滑留待独立消融。

## 3. LD 逐槽增广与显式分组池化（禁自由注意力）

```
aug_slot = z_ld_slot
         + role_embed[role]              # 0..4，无效槽 index 5
         + route_embed[route_assoc]      # 0/1
         + offset_encoder(offset_s_norm) # 2 层 MLP（末层零初始化 → 稳定起步）
         + group_relation_embed[rel]     # 0=target 槽 / 1=同 group / 2=异 group / 3=未知
aug_slot *= ld_live                    # 无效槽严格 0
```

- **`ld_group` 只做 equality**（group relation 4 类）；不得作为数值大小使用。
- 显式分组池化（masked mean；**空组 ⇒ 严格零向量**）：
  `target / continuation / left_adjacent / right_adjacent / route / other`（other = live 且
  不属于前四类）。
- `ld_context = MLP([target_lane_token, target_pool, continuation_pool, left_pool,
  right_pool, route_pool, other_pool])`（7H→H→H）。

## 4. 统一消费（`build_geometry_context`）

- **唯一入口** `DrivingModel.build_geometry_context(z_ld, ld_live, z_ego, nav_token,
  others_ctx, *, semantic=None) -> (ld_context, target_lane_token|None)`：
  - 非 semantic：`(aggregate_ld(...), None)`（baseline/attn 路径逐位不变、调用次数不变）；
  - semantic：必须提供 semantic 输入，否则 `RuntimeError`（**禁 `masked_mean` 隐藏回退**）。
- 消费位置：
  - **PlanHead t0**（`encode` → `plan_step(geometry_context=...)`）；
  - **rollout 1–6**（`_rollout` 每步用当步 latent 重建 context 后传入）；
  - **policy/value head tokens**（`_head_tokens`：`ld_pool` 槽替换为 `ld_context`，
    semantic 模式追加 `target_lane_token` 令牌 → T=40）。
- `plan_step` 在 semantic 模式缺 `geometry_context` → `RuntimeError`（代码断言，见单测
  `test_no_hidden_mean_fallback`）。
- WM/教师强制外部 `plan_step` 调用点（`pipeline/stages.py` / `pipeline/trainer.py`）在
  semantic 模式下同样 fail-fast——S0/S1 正式训练前必须显式接线（未批准，不预改）。

## 5. 3A 契约单测（`tests/test_r21_3a_semantic_wiring.py`，16 项）

| # | 契约 | 断言 |
|---|---|---|
| 1 | invalid → token=0 | mask=0 / valid=0 行 token 严格 0（`torch.equal`） |
| 2 | 归一化逐位 | 常量三元一致（env 常量 / schema layout / 编码器 buffer）+ 手算除值逐位 |
| 3 | role/group permutation 一致性 | 槽序置换 context/token 不变（≤1e-6） |
| 4 | target-role 改变 | target pool/context 变化 |
| 5 | PlanHead t0 + rollout 每步 | 扰动 lane_semantic ⇒ latent 变、6 步 plan 全变、action 变 |
| 6 | policy/value 一致 | 同一令牌集合 ⇒ action 与 value 同步响应 |
| 7 | 旧 schema/ckpt fail-fast | 缺键 `ValueError`；strict `RuntimeError`；missing ⊆ 白名单 |
| 8 | 无 hidden mean fallback | `aggregate_ld` 置炸弹仍可前向；缺 context 的 `plan_step` 报错 |
| 9 | 全灭 fwd/bwd 有限 | 全 mask=0 → 输出/梯度有限 |
| 10 | route/ego 切换仅契约允许处 | token 只由 lane_semantic+mask/valid 决定；role 变化不动 token |
| + | 改名一致性 | `distance_to_target_lane_end` / `same_direction_road_surface_margin`；旧名不在 schema |
| + | group 仅 equality | 保 equality 的 group 数值重标定不改 context；改 equality 结构才变 |

## 6. 边界（本轮不做的）

- 不训练主模型（3B 仅冻结主干 canary）；不改 MoE/WM/ST-GNN/policy/value 结构；
- 不新增 attention 层/容量；不加 semantic 之外的损失（3B aux 仅 lat/heading/curv/lane-change）；
- margin 与 distance_to_target_lane_end **不作监督真值**。
