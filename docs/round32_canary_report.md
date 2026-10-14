# Round-32 canary 总报告（`GO_EGO_ACTION_CONDITIONING_CANARY`；C-A expert-only）

> 依据：`docs/deepseekv4p1_argue.md` §5.37/§5.39（GPT `d86e91a`/`cde2e51` 裁决）；用户令"开工 canary"。
> 纪律：零训练（除 canary 本体的 4 臂小训练）、零其他生产改动；统一口径 K1e2+MoE-off；4 臂 = AC0(base) / AC1 / P-only / AC1-P。
> 产物：`runs/round32/canary/*`（ckpt、Gate 0/1–2/3/4 报告）；脚本 `.slim/deepwork/p1_v2/round32/*`。

## 0. 生产改动与 Gate 0（兼容性）

- 改动（+103/−3）：`net/st_gnn.py`（`action_ego` 惰性残差 + `alpha_ego` 标量，末层零初始化；`forward(action=)`）、`net/model.py`（`_rollout` 传 `a_{k-1}`；`ego_phys` 惰性 + `set_action_ego/set_ego_phys`）、`pipeline/stages.py`（`_wm_predictions` 传 `raw_action`；5 个冻结清单增 `ego_phys.`）。
- **Gate 0 七项全 PASS**：默认模型 = HEAD 键集（168、0 新增）；旧 ckpt strict 载入 0/0/0；关闭/α=0 前向逐位；RNG 不变；梯度只流新参数；**OD/LD 不广播**；baseline↔semantic z-path 逐位等价；全仓 966 tests passed。

## 1. Gate 1（Action Fidelity；`semantic_v1_independent` N=200）

| 指标 | base | AC1 | P-only | AC1-P | 门 |
|---|---|---|---|---|---|
| Action Swap top-1 | 0.101 | **0.284** | 0.101 | 0.236 | ≥0.35 ✗（均未达；AC1 最接近） |
| Action Swap top-3 | 0.635 | **0.917** | 0.635 | 0.849 | ≥0.85 ✓(AC1) / 临界(AC1-P) |
| matched 严格更优率 | 0.099 | 0.284 | 0.099 | 0.236 | — |
| zero−matched（E_all，>0=zero 更差） | −0.001 | **+0.186** | −0.001 | +0.044 | ✓ |
| FD dθ/ds 比值（mean，≤10×） | 38.74 | 7.34 | 38.74 | **2.74** | ✓(AC1/AC1-P) |
| ds-progress 单调（WM probe，≥0.90） | 0.743 | 0.202 | 0.743 | **0.946** | ✓(AC1-P) |

⇒ **动作辨识结构生效**：AC1 排名最强（top-3 ✓）、AC1-P 敏感度/单调最强；top-1 未达 0.35。

## 2. Gate 2（物理语义；独立池全量）

| 指标 | base | AC1 | AC1-P | 门 |
|---|---|---|---|---|
| ΔR²_WM state prog / speed（>0） | +0.012/+0.006 | +0.015/+0.019 | **+0.023/+0.029** | ✓ |
| lane 保留（ld_slots lat/head，降≤5pp） | 0.456/0.415 | **0.521/0.601** | 0.475/0.493 | ✓（均改善） |
| ΔR²_action state prog（>0） | +0.0006 | 625*（off-manifold 伪大） | +0.330 | ✓ |
| 泄漏 od+ld_pool prog（增≤10%） | — | +4.3% | **+0.5%** | ✓ |
| 泄漏 od_pool prog（单通道） | — | **+66%** | +41% | ✗（单通道超） |

（* AC1 的 shuffled 分支远离流形 ⇒ R²(shuffled) 巨负、Δ 伪大；判读看符号与 AC1-P 的量级。）

## 3. Gate 3（部署接口；50-ep 同协议录制 vs base `r32_K1e2_moeoff`；raw 口径）

| 指标（h3–5 均值 / h5） | base | AC1 | AC1-P | 门 |
|---|---|---|---|---|
| E_if ds（−30%） | 1.734 | **2.097（+20.9%）✗** | 1.795（+3.5%）✗ | ✗ |
| E_action ds（−25%） | 1.619 | 1.897（+17.2%）✗ | 1.679（+3.7%）✗ | ✗ |
| overshoot ds@h5（−40%） | 1.660 | **1.933（+16.5%）✗** | 1.695（+2.1%）✗ | ✗ |
| E_real ds（退化≤5%） | 0.343 | 0.351（+2.4%）✓ | **0.312（−8.9%）✓** | ✓ |

⇒ **Gate 3 全 FAIL**：AC1 的接口/overshoot **反向恶化**（动作敏感性放大了 rollout 自反馈漂移）；AC1-P 近中性、真态误差改善。

## 4. Gate 4（calib50 闭环；LQR+plan；MoE-off）

| 臂 | succ | off | coll | tmo | rc | 逐 episode（翻正/翻负） |
|---|---|---|---|---|---|---|
| base | 0.560 | 0.140 | 0.080 | 11 | 0.862 | — |
| **AC1** | **0.600 ✓** | 0.120 ✓ | **0.180 ✗** | 5 ✓ | 0.867 | 3/1 |
| P-only | 0.560 | 0.140 | 0.080 | 11 | 0.862 | 0/0（≡base 对照） |
| **AC1-P** | 0.540 ✗ | 0.160 ✗ | 0.020 ✓ | 14 ✗ | 0.842 | 2/3 |

门：succ ≥0.58、off ≤0.14、coll ≤0.08、tmo 不劣。⇒ **AC1 3/4（succ 首破 0.58：0.600，+4pp）但碰撞翻倍（0.08→0.18）**；AC1-P 未过（succ 降）。

## 5. 结论与裁决建议（待用户/CG）

1. **结构方向有效（fidelity 级）**：动作条件化可学习（Gate 1/2 多指标改善，AC1-P 的 FD 2.74×/单调 0.946 为证）。
2. **部署门未过**：Gate 3 全 FAIL（AC1 恶化、AC1-P 中性）；Gate 4 无臂全过。**不自动进入 C-B/60-40**（按预注册纪律）。
3. **AC1 现象（重点）**：succ +4pp 首破 0.58、tmo 减半——但 **coll 0.08→0.18**、overshoot +16.5% ⇒ "更灵敏的 ds 条件化"在部署链中放大自反馈漂移/激进性；需归因（逐 episode 路型/速度、plan_ds 剖面、碰撞场景）。
4. **下一步候选（供裁）**：①AC1 碰撞/overshoot 归因（只读；逐 episode + 剖面）；②分支正则化/限幅（如 ds 残差幅值约束、α 上限、仅 dθ 或 ds 冻结对照）；③C-B 前提重议（当前未过门）；④AC2（相对场景条件化）后置；⑤回滚选项：全部默认关、零风险（生产改动不启用即逐位不变）。
5. **P-only 机制注记**：单独训练动力学 ≡ base（仅读出）；其价值经 AC1-P 联合训练体现（phys 损失经 `z_ego_next` 回传塑形 action 分支）。
