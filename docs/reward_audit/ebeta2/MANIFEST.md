# E-β″（v6 基座）奖励审计产物入库（tracked；P4 前置-C）

> **用途**：P4 奖励口径（剖面 C + rc 各档终局值）的 E-β″ 定稿产物。**不覆盖** P2/E-β′ 产物
> （`docs/reward_audit/` 根目录）；上游规格 `docs/v6_program_prereg.md` §7.1（六条）、
> `docs/rl_reward_v5.md` §7。

## 产物清单（逐位一致：入库副本 = 原始 `runs/reward_audit_ebeta2/` 文件）

| tracked 路径 | 原始 `runs/` 路径 | sha256 |
|---|---|---|
| `docs/reward_audit/ebeta2/reward_audit.md` | `runs/reward_audit_ebeta2/report/reward_audit.md` | `446fba1fec47d3da646fd2edc472f6a68af46c90f95053454170c596f17f8be9` |
| `docs/reward_audit/ebeta2/reward_audit.json` | `runs/reward_audit_ebeta2/report/reward_audit.json` | `d851fe4db9efc0f46ff22aaeff3fe9de21eddbe8e946ea7055157cf197856e9b` |
| `docs/reward_audit/ebeta2/reward_audit_swap.json` | `runs/reward_audit_ebeta2/report_swap/reward_audit.json（A/B 互换方向 B，逐位一致）` | `bfcd1b249bbf29ed55e90d49af92bf4726389d141227a8539a7f1f2ba702fd62` |
| `docs/reward_audit/ebeta2/ab_comparison.json` | `runs/reward_audit_ebeta2/report_ab_comparison.json` | `731e91b34855b606508e2c9ebda0300e10bd998c8d47f5ab381adde15fcda589` |
| `docs/reward_audit/ebeta2/ab_comparison.md` | `runs/reward_audit_ebeta2/report_ab_comparison.md` | `194622b752c876e69f7d519b1c0c4bc469c27eea677746d4f228988feb4e0924` |
| `docs/reward_audit/ebeta2/config_draft_rc1.yaml` | `runs/reward_audit_ebeta2/report/config_draft_rc1.yaml` | `0a1d397e1f7dcb72d778e635bb9acefe1eeb535771eb15e7ab012b24f29e339b` |
| `docs/reward_audit/ebeta2/config_draft_rc3.yaml` | `runs/reward_audit_ebeta2/report/config_draft_rc3.yaml` | `738c91f7a5f20f568ab9405547337175b422e7e03439aa0dd91f10587514ec10` |
| `docs/reward_audit/ebeta2/config_draft_rc10.yaml` | `runs/reward_audit_ebeta2/report/config_draft_rc10.yaml` | `6b32b264ef1bcfcba67a8262dc8860e53aa3615bd1c2fc219ea6f2aadb50dfd8` |
| `docs/reward_audit/ebeta2/config_draft_rc30.yaml` | `runs/reward_audit_ebeta2/report/config_draft_rc30.yaml` | `c6245282892586e630d5311cad636c7f0ac335da745af255f0abd922f3620634` |

> **注释补丁（2026-10-01，P4 收尾；仅注释，数值不变）**：`config_draft_rc{1,3,10,30}.yaml`
> 入库副本追加 max_step 裁定注释；YAML 数值段与原始 `runs/` 副本**逐位一致**（`diff <(grep -v '^#')` 校验）。
> 补丁后 tracked sha256：rc1 `136befea…` / rc3 `14b07cec…` / rc10 `eed27deb…` / rc30 `a2c73323…`
> （上表 sha256 列为补丁前入库副本 = 原始 `runs/` 副本，保持溯源）。

## 裁定记录（max_step 终局值 + horizon；Orchestrator，2026-10-01；P4 收尾）

- **max_step 终局值采用 E-β″ 点估计**（rc=1 **−46** / rc=3 −48 / rc=10 −54 / rc=30 −71；
  `config_draft_rc*.yaml` 数值即该裁定值，本次仅补注释锚）。依据：profile 一致性——
  rc=1 审计半区 mean **−15.050**、Δ **−5.050**（边界；方向 B mean −4.750、Δ +5.250）；
  若沿用 P2 −23 则 rc=1 审计半区 mean ≈ **+8.05**、Δ +18（= 超时正收益）。
- **标注**：max_step 每类 **n=8/向、不判通过**（§7.1 ④ 结构性不足）、**首臂后复核**、
  **不扩采**（~5-6 h 超预算）。
- ~~**horizon 显式接受**：训练截断 600 策略步 vs 审计/评测 1000 物理步；max_step 在训练侧更稀有；
  终局值不变。~~ **（已被 Gate4 修正替代，2026-10-01）**
- **Gate4 修正（替代上条）**：审计 `--max-steps 1000` 物理步 = 100 s = **200 策略步**；训练
  截断**对齐 200 策略步**（=100 s；旧 600 = 300 s = 3×，−46 被 dense（+0.168/步、不衰减）
  稀释 → 超时正收益 ≈ +54）。**终局值不变**；实现/复核（首臂后复核 + 非 max_step 两向
  |Δ|≤5）见 `docs/v6_program_prereg.md` §7.1/§7.4。
- 其余三类用 E-β″ 定稿值（rc=1 +29/−22/−14；rc=3 +27/−23/−15；rc=10 +20/−28/−18；
  rc=30 +0/−41/−26；两向非 max_step 类 |Δ|≤5）。
- P4 臂配置组装见 `config/arms/`（8 臂；逐臂单变量/预期见其 README；dry-run 单测
  `tests/test_p4_arm_configs.py`）。

## 溯源

- 工具落库 commit：`abffdf6`（本目录产物的生成代码 `tools/reward_audit*.py` + `tests/test_reward_audit.py`；
  相对生成时 HEAD `fb4f8c1` 的差异仅限审计工具/测试，不影响产物口径）。

- 基座 ckpt：`runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt`（= final 同权重；sha256 `sha256:723db1c27d9da33ed8da17375af718cb421058e000a54555a2e5c667f5a56fa5`）。
- 代码：HEAD `fb4f8c15a4e3963d0da5563a345c9f9662c1570f`（分析）/ `fb4f8c15a4e3963d0da5563a345c9f9662c1570f`（采集；含 P4 前置-B max_step 接线 fb4f8c1）；`--code-mode current`。
- 池：主 `/workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_val.json`（val−eval500） + 补充池 `/workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_train_5k.json`（seeded 切片 order seed `20261001`；`--drop-exclude-ids`）；排除 `/workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_eval500.json`。
- 采集：2500 episode（分类计数 {'arrive_dest': 1127, 'out_of_road': 1211, 'collision': 146, 'max_step': 16}）；6 workers；max_steps=1000（与评测同口径）；tracker=lqr / eval_reference=plan。
- 划分：分层随机 split seed `20261001`；A/B 互换两向各一次（B 向 `reward_audit_swap.json`）。
- 判定：rc=1/3/10/30 剖面——**非 max_step 类两向全档 |Δ|≤5**（最大 ≈2.2）；max_step 每类 n<50（结构性不足，不判通过，见报告附录 C.4）。

## 已知偏差

- **max_step n < 50**：E-β″ 超时率 ~0.7%（1000 步口径），无法在预算内达 n≥50/半；按 §7.1 兜底只报 n 与区间、该类不判通过（参考点估计见附录 C.4）。
- 采集器 obs 帧错位（post-step vs 训练 pre-step）沿用 P2 口径，影响 ≪ 容差（Gate2 发现 ②）。
- 数值 id 交集：主池/补充池在 (id, seed) pair 级交集 0；补充池另做数值 id 级剔除（场景身份=(id,seed)，项目隔离口径）。
