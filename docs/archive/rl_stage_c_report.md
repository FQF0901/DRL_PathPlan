# Stage C（RL）夜间排模报告 · 2026-09-30（01:00 → 09:00）

> 交付对象：用户晨间 review（"各版本改动点 + 效果 + 最终方案"）。
> 边界：本报告只覆盖 **Stage C（PPO RL）** 的启动与排模；A/B 链路锁版见 `docs/LOCKS.md`。
> 复核链：G1（P1 修复）→ G2（P2 优化）→ G3（P3 结果/方案）均由独立 oracle 复核。

## 0. 一句话

**P1 修复后，RL 管线的"账实一致与数值零漂移"（V1–V12 守卫）通过；P2 排模把吞吐提到 3.3×（**仅 repeat_action cheap path**；plan 臂仍走旧路径、不按 3.3× 估时）；P3 的 6 个配方臂在 **last-only 协议下的末点**显著低于 init（net −53…−160），A4 的"least-bad"seed 不稳 ⇒ 现配方不可采纳，**RL 基座维持 E-β′（0.436）**。两个未消融混杂要求结论限定：①last-only（无中间点）；②**V1 后训练执行语义（repeat_action）与评测执行语义（6 步 plan 预览）不一致**（`eval_runner.py:878-891`）。"训练奖励与闭环 KPI 错配"为**强候选**（见 §4）。G3 指定的最小下一步（周期 ckpt / 语义救援臂 R1 / 奖励消融）已在夜间执行（§7 如实汇报）。**

## 1. 锁版与开工（P0）

- tags：`stageA-lock-v1` / `stageB-lock-v1` / `rl-baseline-start` @ `12aab99` + `docs/LOCKS.md`（`a0136e6`）。
- RL 起点：`runs/_refs_rlbase/e_beta_prime/final.pt`（0.436；干净集 0.446）；回退 L2（0.328）。
- 运行机制：长任务 setsid 脱离；20min×8h 监督（`/tmp/opencode/rl_supervise.sh`）；官方库只读。

## 2. P1 · P0 阻塞修复（全部 commit；守卫=定向测试 + no-op 探针 + e1 口径）

| 版本 | commit | 改动点 | 效果/验收 |
|---|---|---|---|
| V1 | `2ef8880` | **P0-1 A-hold**：LQR 参考只由被记账的采样动作决定（`references=repeat(a_t,6)`；plan 臂保留对照） | 执行-记账一致；`test_stage_c_effective_action` 4 用例 |
| V2 | `1246e8b` | **WM W1**：`st_gnn` 全期冻结 + fail-fast + `wm_trainable` 指标；`--wm-freeze-updates` 弃用 | 33 张量逐位不变；测试 2 用例 |
| V3 | `04c0f7c` | **P0-2 对齐**：标签步前同帧（Local `labels_now`/Vector `labels_at_step_start`）+ 终局 record/reset 分离 | 测试 7 用例 |
| V4 | `7c4138b` | 协议/资源指标（pool/tracker/plan_reference/vram/rss） | 冒烟字段核对 |
| **R7** | `cc9fc4f` | **（G1 发现的阻塞）collect/update 观测不一致**：update 侧 history 缺 `ego/others` 两通道 → ratio 首轮≠1；恢复四通道 | 红→绿：`approx_kl` 0.00358→≈0；真实池 no-op `0.0` |
| R2 | `477068a` | `trainable_scope: design|all`（默认 **design**=allowlist `policy/value/experts/residual`；含 router 冻结，G1 §2） | 组表+指标；scope 测试 |
| R4 | `ee86748` | `p0_fixes` 按实际生成 / `pool.kind` 真实类型 / probe 默认存在路径 | 冒烟核对 |
| 收尾 | — | 冒烟（独立目录×3 口径）；`--ppo-epochs 1` 首 update `1.54e-06`；全套 **367 passed** | G1 重审=通过 |

## 3. P2 · 性能排模（3.3×；**仅 repeat_action cheap path**，plan 臂不适用）

| 版本 | commit | 改动点 | H=128 读数（wall / steps/s） | 守卫 |
|---|---|---|---|---|
| 基线 | — | （V4 代码） | 86.2s / 30.8 | P2 基线画像（`rl_perf_baseline.md`） |
| V8r | `ad06a47` | router 出 design allowlist（合同落定） | — | no-op 0.0 |
| **V9** | `57bc013` | **collect cheap-path**（`repeat_action` 臂 `rollout=False`；plan 臂保留旧路径） | **32.1s / 100.0（−62.8%）** | cheap 等价测试 + 逐位 KL |
| V10 | `15bc8eb` | 对齐路径去 post-step 标签双算（Local/Vector） | 32.0s / — | 标签语义测试 |
| V11 | `aa615fb` | `trim_memory_every=4` | 30.9s / — | RSS 不回归 |
| **V12** | `1f58589` | 每 update 整批组装一次 + fancy indexing | **30.3s / 103.0（H256: 107.7）** | 切片一致性测试；全套 **369 passed** |

- 零漂移证据：五版本 `--ppo-epochs 1` 同 seed 3-update 的 `approx_kl` 序列**逐位相同**。
- 剩余瓶颈：单 env 物理步 ≈70%（Vector 复活需先修 tracker 降级语义，未做）；KL 锚 <0.1%；探针 ~2.5ms/update。
- G2 复核 = **通过**（口径复算一致；RSS 峰 2226MB 属启动瞬态；`--pool vector` 不得用于 RL 结论）。

## 4. P3 · 版本实验（预声明协议；A1–A6）

协议：**last-only**（stage C 仅保存 `final.pt`；无周期 ckpt/keep-best——已声明）；双评测（clean500 主 / eval500 辅，内容哈希已记录）；pin：design / repeat_action / local-1-lqr / H256·updates200·seed0 / probe-on / trim=4 / critic_warmup=0。

| 臂 | init | 变量 | clean500 | eval500 | net（vs init） |
|---|---|---|---|---|---|
| A1 | E-β′ | lr 3e-4，KL 0.05→0 | 0.256 | 0.244 | −95 / −96 |
| A2 | L2 | 同 A1 | 0.098 | 0.122 | −110 / −103 |
| A3 | E-β′ | lr 1e-4，KL 0.05→0 | 0.266 | 0.260 | −90 / −88 |
| A4 | E-β′ | lr 3e-4，KL **floor 0.01** | 0.322 | 0.330 | −62 / −53（least-bad） |
| A5 | E-β′ | lr 3e-4，**KL 0** | 0.150 | 0.124 | −148 / −156 |
| A6 | E-β′ | =A4，**seed 11** | 0.128 | 0.116 | −159 / −160（**不复现**） |

**判定（按 G3 §4 限定）**：① **末点**（last-only，无中间点）在两集全部显著低于 init（z 6.1–12.6）；"200 更新过程性摧毁"不可知、不成立；② 无臂可采纳（A4 seed 0 least-bad −62/−53；seed 11 不复现 −159/−160）；③ 仅 **KL 轴**剂量-反应可判（无锚 → 归零 → floor 单调改善）；lr 3e-4 vs 1e-4 在噪声内；**不给逐臂排名**；④ "训练奖励与闭环 KPI 错配"= **强候选**：证据（`speed_ratio` 占每步回报主体；仓库合成序"蠕动 58 < 碰撞 105 < 正常 252"；Stage C v2 跨代同签名）+ 两个未消融混杂（last-only；训练/评测参考语义不一致）；且**评测分布内总回报同样下降**（主因提前终止）⇒ 更准确的表述：训练分布内目标改善、评测分布内目标与 KPI 同降（**迁移失败 + 目标错配的联合**）；⑤ 无周期 ckpt/keep-best ⇒ 中段最优点不可知、不可保留（keep-best 需先补代码）。

## 5. 复核链（G1–G3）

- **G1**：初评"需修正"（发现 R7 阻塞）→ 修补（R7/R2/R4）→ 重审 **通过**。
- **G2**：P2 优化 **通过**（3.3× 可信、零漂移、无行为回归；附记账项）。
- **G3**：**通过（条件通过）** —— 12 格配对独立复算一致；新发现**训练/评测执行语义不一致**（V1 后训练用 repeat_action，评测用 6 步 plan 参考）为未消融混杂；要求措辞限定（本版已按 G3 §4 表执行）；最小下一步 = 周期 ckpt / 语义救援臂 R1 / 奖励消融。

### 5.1 P4 前置执行结果（C1–C4；按 G3 最小下一步；2026-09-30 05:41–06:42）

- **C1 周期 ckpt（commit `17efc2f`）**：`--ckpt-every` / `stages.C.ckpt_every=25`（0=关）；冒烟 `ckpt_u001/u002` 落盘 + metrics 记录；新增 4 用例 ⇒ **373 passed**。**奖励钩子（commit `93164ea`）**：`stages.C.reward` 透传 + `--reward-term-weight NAME=W`；新增 4 用例 ⇒ **377 passed**。
- **C2（H3：周期 ckpt 版 A4 复跑）**：`ckpt_u200.pt` 与 P3-A4 `final.pt` **逐位相同**（同参同轨迹控制 ✓）；四点 clean500：

| update | 25 | 50 | 100 | 200 |
|---|---|---|---|---|
| succ | 0.272 | 0.260 | 0.264 | 0.322 |
| net vs init | **−87** | −93 | −91 | −62 |

**预注册判定**：首个保存点（u25）即 −87、无任何近 init 的中间点；最优点=末点（浅 U 形）⇒ **H3（"last-only 藏住更优中段/选型机制缺失"）被否定**；本质 = **早期内在崩溃（首 25 更新内）**。
- **C3（H1：语义救援臂 R1 = `--plan-reference plan` + 恒定 KL 0.01 + 100 upd）**：clean500 **0.356（net −45，z 5.50）**；预注册阈值（≥−20 / ≤−50）**两侧均未命中**（差 5 条）⇒ **训练/评测语义为显著贡献因素**（同预算对照 C2-u100 −91 → −45，**+46**），但不足以解释全部损伤。
- **C4（H2：`speed_ratio=0` 消融，200 upd 同 A4 可比）**：clean500 **0.252（net −97）**，比 A4（−62）**更差**、off_road 0.676 ⇒ **不支持**"speed_ratio 主导的奖励错配"（拔除效率项后更倾向出界）；单 seed 消融，如实标注。

数据：`/tmp/opencode/rl_p4prep_{c2,c3,c4,summary}.json`；日志 `rl_p4prep_*.log`。

## 6. 运行与合规

- 20min 监督：01:00–06:00 全程（每轮快照：git/进程/GPU/MEM/SWAP/日志）。
- 未改官方库；未新增 repo `.sh`（工具链在 `/tmp/opencode/`）；工作区截至 P3 收尾 **clean**。
- 测试：350 → **377**（+27 新守卫；含 C1/C4 后）；冒烟/评测全部 `errors=0`。

## 7. 最终方案（G3 输入 + P4 前置结果定稿）

**三个混杂已被逐一实验裁定**：
- **训练/评测执行语义不一致（H1）**：显著贡献（+46）但**非全部**（C3）；
- **last-only / 选型缺失（H3）**：**否定**——损伤在首个 25-update 保存点即完整，不存在可保留的近 init 中间点（C2）；
- **speed_ratio 奖励错配（H2）**：**不支持**（C4 拔除后 −97 更差）。

**最终方案（按优先序）**：
1. **口径统一（工程必修，最先做）**：训练与评测的 tracker 参考语义必须统一——C3 表明训练切到 plan 语义可挽回近半损伤，但评测侧注释警告"单动作参考退化跟踪"⇒ **建议两侧统一到 plan 语义（或共同改用 repeat_action 并重标定基线），并在统一口径下重跑头部对照（init 两点 + A4 一臂）**；成本 ≈1h。
2. **"不损伤底线"短预算实验**：损伤出现在 ≤25 upd ⇒ 用**密集 ckpt（every 5）+ 预算 25–50 upd + 强 KL floor（0.05 恒定）**搜寻"不崩且可测"的预算窗；周期 ckpt 已接线（`17efc2f`），keep-best 导出可在此窗上自然落地。成本 ≈40min。
3. **奖励侧系统诊断（第三优先）**：H2 单点消融不成立 ⇒ 项权重扫描/回报尺度/优势归一化（`--reward-term-weight` 钩子已就绪 `93164ea`，零代码扩展）；**在口径统一后再做**，避免在错配口径上归因。
4. **不变项**：**RL 基座维持 E-β′（0.436）**；stage C 代码（P1 修复 + P2 3.3× + 周期 ckpt/奖励钩子）已就绪，可作为后续 RL 的干净起点。

## 8. 附录

- 文档：`docs/rl_stage_c_experiments.md`（P3 协议+结果）、`/tmp/opencode/rl_recon_p0.md`（侦察）、`rl_perf_baseline.md`、`rl_perf_optimized.md`。
- 数据：`/tmp/opencode/rl_p3_a1..a6.json`、`rl_p3_*.log`、`runs/BTC20260930-*_stage_c_p3_a*`、`runs/BTC20260930-*_eval500_p3a*`。
- commits 一览：`git log --oneline rl-baseline-start..HEAD`（含 C1 `17efc2f` 周期 ckpt、C4 钩子 `93164ea`、P4 前置）。
- P4 前置数据：`/tmp/opencode/rl_p4prep_summary.json`；运行目录 `runs/BTC20260930-05{43,51,17}_*`（a4ckpt / holdplan / nospeed）。
