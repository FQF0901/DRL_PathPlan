# P0 评测兼容设计（v0.1）

> 依据：`docs/deepseekv4p1_argue.md` §0.9.5 锁定版 + ChatGPT `cd4b559` 的 6 条硬约束。
> 目的：让 P0 的每个数字**可比较、可复算、口径明确**；先约定边界，再实现与跑批。
> 状态：v0.1（实现 lane 依此执行；任何偏离须回写本文档）。

---

## 1. 版本层与模块划分

**两条模型栈**：

| 栈 | 代码 | 用于 |
|---|---|---|
| **v8 栈（当前 HEAD）** | obs v6 + v8 网络 | Arm P / pri512 / IDM / oracle / laneplan / D1 / D2 / E1 / E2 / F |
| **v7p2 栈（`c37acbb` archive；`v7-lock` 已验证不可用）** | obs v4 + v7p2 网络（type_embed 5、无 lane/ttc） | **仅 s11** |

- v7p2 栈落地方式：`git archive c37acbb | tar -x -C /tmp/opencode/s11_v7_c37acbb/`（`git worktree` 被权限禁用）；**不得**用 v8 代码加载 v7 ckpt（shape mismatch → 静默随机初始化）。
- **更正（fix-11 实测，2026-10-09）**：s11 训练（10-05 06:01–06:19）早于 obs v5 提交（`0291f3a`，10-05 07:25）→ 实为 **obs v4 + v7p2**（type_embed 5、无 lane/ttc）；`v7-lock-20261006` archive 加载其 ckpt 会 `missing=4 / shape_mismatch=1`（拒绝），**唯一可用 archive = `c37acbb`**（0/0/0，154/154；证据与 runner 见 `.slim/deepwork/s1_audit/`）。
- s11 资产：`runs/BTC20261005-0601_v7p2_s11_arm1/`（`ckpt_u150.pt`/`final.pt` + `config.snapshot.yaml` + `model.snapshot.yaml` + manifest）。

**统一语义层（尽量复用同一实现/同一数据；做不到的显式标注）**：

| 项 | 处理 |
|---|---|
| spec 文件 | 统一用 `env/specs/scenarios_eval500.json`（同一份；v7 栈复制或软链使用） |
| 环境 / MetaDrive | 同一 venv、同一 metadrive 版本（v7 栈直接调用同一解释器） |
| tracker 参数 | 显式传同一组参数（`--tracker lqr` + 同一 tracker config；v7 侧默认若不同则显式覆盖） |
| termination / KPI 聚合 | **目标：统一到当前实现**；若 v7 归档无法直接复用，则 s11 数字标记为"跨版本参考"（见 §3） |

**禁止**：跨 evaluator 的 2–3pt 差异解释为网络差异；用"部分加载"的模型产出正式数字。

---

## 2. 评测安全前置（P0-A 规格）

1. **加载 fail-fast（默认开）**：`missing / unexpected / shape_mismatch` 三计数**任一非零 → 打印明细并退出（非零码）**；仅 `--allow-partial-load`（显式）可放行，且放行时强制在日志与 `metrics.json` 记录三计数与 missing 列表。
2. **版本戳**：正式评测落盘：git commit（或 archive 标识）、argv、config 快照（已有）、**ckpt sha256**、evaluator 标识。
3. **回归测试**：新增单测（完整加载通过；缺键/形状不符 → fail-fast；`--allow-partial-load` → 放行且记录）；并用真实产物做一次 CPU 侧"干净加载"冒烟（pri512 ckpt + 其配置）。
4. IDM（A 格）：`tools/baseline_eval.py` 增加 `--episodes-out`（逐 episode JSON 导出；默认关闭）。

---

## 3. 指标可比性矩阵（C1 落地）

| 指标 / 比较 | s11（v7 栈） | v8 栈 | 允许解释 |
|---|---|---|---|
| success / off-road / collision（eval500，同 (id,seed)） | ✓ | ✓ | **同版本内严格**；跨版本（s11 vs v8）**仅方向参考**（若 termination/KPI 未能统一） |
| 口径差：plan vs repeat_action | ✓（同栈内） | ✓（Arm P / pri512） | 各版本内严格比较；口径差判定主证据 |
| B/C/D1/D2/E/F（无版本混用） | — | v8 栈 | 与 E/F 同栈 → 严格可比 |
| 失败归因（T_plan/T_track/T_cross/T_term） | ✓（v7 forensics） | ✓（v8 forensics） | **版本内为主**；跨版本只比较"失败类型构成"（计划块/执行块/recovery），不做百分点归因 |

---

## 4. 审计矩阵（P0-B）与记录字段

| 行 | 参考构建 | 载体 | 状态 |
|---|---|---|---|
| A | IDM 自身 | `baseline_eval`（+逐 episode 导出） | 现有 |
| B | oracle：专家实测未来 3s | forensics `--mode oracle` | 现有（补记录） |
| C | laneplan：车道中心线 3s | forensics `--mode laneplan` | 现有（补记录） |
| D1 | **未来 expert 动作链** → arc_step → 6 点 plan | forensics 新模式 | **新**（privileged ceiling） |
| D2 | **当前 expert 动作** repeat 6 步 → arc_step | forensics 新模式 | **新**（与 F 直接可比） |
| E1 / E2 | 模型 plan + LQR / exact | forensics `lqr`/`exact`（头条数字可再用 eval_runner 复核） | 现有 |
| F | 模型 repeat_action + LQR | forensics 新 `--reference repeat_action`（口径同 `eval_runner.build_eval_references`） | **新** |

**统一约定**：eval500 全量；`--spec env/specs/scenarios_eval500.json`；tracker=lqr（E2=exact）；每行输出 per-episode（id/seed/success/off-road/collision/rc）+ 汇总 + **加载/版本摘要**。

**逐策略步记录字段（分类用，冻结）**：
`step`、`plan (6×2)`、执行位姿、`T_plan / T_track / T_cross / T_term`、`plan_footprint_valid`、`exec_footprint_valid`、`min_signed_margin`（或退化三元组）、`tracking_residual`、`road_class`、`map_id`、`policy_action vs plan_first_action`、`router_topk`、`policy_std`（baseline/expert 行后四项留空）。

---

## 5. footprint / boundary 检查规格

1. `arc_step` 展开 plan → 节点间按 **0.5–1.0 m** 插值；
2. 每个采样位姿生成车辆矩形（车长/宽取 env 参数），采样**四角 + 四边中点 + 中心**（9 点）；
3. 用**环境 off-road 终止同款的几何查询**判定 drivable（实现时须定位该查询并在代码注释记录所用 API）——保证 margin 语义与官方 termination 一致；
4. 记录 `min_signed_margin`；引擎给不出可靠 signed distance 时，退化为 `inside_ratio / first_invalid_pose / invalid_footprint_point_count`（**不伪造**连续距离）。

**时间戳定义**：
- `T_plan` = 首次计划 footprint 不可行；
- `T_track` = 首次实际轨迹明显偏离"仍可行的计划"（tracker residual 超显式阈值）；
- `T_cross` = 首次实际 footprint 越界；
- `T_term` = 环境正式终止。

**分类规则（层级 + 多因素）**：`T_plan < T_cross` 且计划越界 → plan-originated；计划始终可行但先 tracking 偏离后越界 → tracker-originated；进入危险状态后仍存在可行 recovery 而持续失败 → recovery failure；几何判定与 termination 不一致 → anomaly。**同时保留所有 contributing factors**，不只输出单一标签。

---

## 6. D1 / D2 规格

- **首点**：不做 mu 覆盖（expert 链首点 = 其下一步动作，天然等价）。
- **镜像项**：单位 / 归一化 / 裁剪 / 0.5 s 间隔 / SE(2) 累积 / LQR 参数与模型 plan **逐项一致**；转换复用 `net/model.py::arc_step` 与采集侧 `_window_actions` 同款 pose-delta 口径——**不重写近似转换**。
- **D1**：对同 (spec,seed) 运行规则专家取得实测轨迹 → 按 0.5 s 窗口生成 6 步 (ds,dθ)（open-loop；**标记 privileged/oracle ceiling**）。
- **D2**：当前 expert 动作 = 专家策略在**现状态**下应执行的 0.5 s (ds,dθ)（shadow 前向 5 物理步或等价口径），repeat 6 步。

---

## 7. 执行顺序与产物

1. **P0-A 落地**：fail-fast + 版本戳 + 测试（+ IDM 逐 episode 导出）。
2. **s11 兼容 runner（已完成）**：`c37acbb` archive + 接线 + CPU 加载烟测（0/0/0、154/154）→ GPU 空窗跑两口径（`.slim/deepwork/s1_audit/run_s11.sh`）。
3. **审计工具（已完成）**：forensics 增补 `d1`/`d2`/`--reference` + 记录扩展（footprint 有效性、四时间戳、层级分类）+ footprint 检查器（复用 env `on_lane` 同款查询；引擎只给布尔 → 退化三元组，不伪造 signed distance）+ 单测（27 passed；smoke16 全模式跑通）。注：全量 eval500 每行预计 ~1–1.5h（footprint 射线开销），矩阵按行顺序跑；每行需与 ckpt 匹配的 `config/model.yaml`（矩阵脚本按行切换）。
4. **GPU 空出后**（排摸收尾）：矩阵跑批（每行附加载摘要）→ **《P0 审计报告》**（逐 episode paired 数据 + 分类 + 推荐 P1 分支）。
5. 纪律：P0 完成前冻结 WM/MoE/encoder/PPO scope；每行强制加载校验；产物按 §2 落版本戳。

### 7.1 审计勘误与记录（2026-10-10）

- **laneplan 参考构造 bug（已修）**：旧 `LanePlanController` 把中心线折算成 6×2 (ds,dθ) 交 tracker，**丢弃自车相对中心线的横向/航向偏差** → 参考退化为"沿当前航向的切线"（开环无回正）→ LQR 横向自激振荡无阻尼增长直至出界（eval500 前 20 ids：0/20 succ、off 20/20、rc 0.196；id=9 于 79 步出界）。修复：tracker 输入改为中心线**自车系 (N,3) 位姿**（与 oracle 同口径）；`plan/mu/reference` 记录字段不变。修复后同 20 ids：**6/20 succ、0/20 off、rc 0.815**（对照 oracle 9/20、d2 8/20）；id=9：831 步到达、rc 0.98。→ **全量重跑该一行**（旧 bug 版产物留档 `p0_laneplan_bug.json`）。
- **d1 崩溃 = 真实结果（非 bug，保留原样）**：链重建（`_window_actions`/`relocate_cursor`/组装）逐位复核正确、cursor 逻辑与 oracle 同源；根因是"开环 (6,2) 专家链"不随自车误差回正（d2 用的是**现状态**专家反馈动作，天然含回正信号）→ 这是"(ds,dθ) 表示 + 开锚"的脆弱性证据，进入审计结论。
- **anomaly 语义确认（非 bug）**：env `_is_out_of_road` 含黄/白实线与行道 flag（`out_of_road_done` + `on_continuous_line_done`），而 footprint 判定只查 lane 面 → "anomaly"是两套判据口径差的必然产物；报告以"判定口径差异"解释该 bucket，建议后续单独记录这些 flag（未实施）。
- **eval 侧 `model_config` 漏传 `spatial`（已修，2026-10-10）**：评测任务构造只透传 hidden_dim/moe/policy/value/world_model/plan_anchor，遗漏 `spatial` → stg3 臂（layers=3）被按默认 2 层建模型（`unexpected=10`），**fail-fast 拦截 rc=2（未产生错数字；fail-fast 之前的旧行为会静默错评）**。修复：抽出 `eval_runner.task_model_config()` 补 `spatial` 透传 + 守卫单测（`tests/test_eval_model_config_forwarding.py`）。其余臂 `spatial` 为默认 2 层、加载均 0/0/0，不受影响；**stg3 四个筛选评测已排队补跑**。
