# Round-35 总报告（导航职责审计 + 动态 hooks + 数据/字段盘点）

> 依据：`docs/deepseekv4p1_argue.md` §5.49（GPT `07f8c4b` WP1/WP2 + 我方 F3）。纪律：只读、零训练、零既有文件改动；默认关 bit-exact。
> 产物：`docs/navigation_scope_audit.md`（F1 交付件）、`runs/round35/{f1_nav_audit,f2_hooks,f3_data_census}.*`；脚本 `.slim/deepwork/p1_v2/round35/f*.py`。

## 1. F1 导航职责审计（5 arms × 18 ep 闭环 + 干预；bit-exact 18/18）

**字段清单（要点）**：`nav`=道路级（env/obs/nav.py → net/model.py:814-829）；`route_world`=道路级（→ mem.py:465-560）；`current_ref_lanes/next_ref_lanes`=**仅被规则消费**（lane_semantic.py:355-405/486-493）；**`target_lane`=规则生成**（lane_semantic.py:355-405，**不查障碍**）；`target_lane_token`=网络强消费（semantic.py:95-102→156/190；model.py:687-691/919-922）；`ld_route_associated`、`lane semantic`、`route command` 逐项 file:line 见 `docs/navigation_scope_audit.md`。

**三干预响应**（vs baseline；首差异决策中位=0，即刻生效）：

| 干预 | μ max\|Δ\| / plan max\|Δ\| | 车道序列差异率 | 结果 |
|---|---|---|---|
| ① swap_target（route 固定、target 换左邻） | 1.35m / 1.94m | 0.71 | **merge 3 例失败→成功**（b1:7559、b1:7608、base:7559）；toll 10/10 仍失败（但变道意图 0→1.0、lane_lat −0.8m=未完成变道） |
| ② route_rot（target 固定、route/命令旋转） | 1.33m / 2.36m | 0.78 | 成功损失 1（base:7608 off-road）⇒ 道路级 route 被强消费 |
| ③ no_target_token / no_target_identity | 1.15 / 1.18m（μ） | 0.76 / 0.73 | 仍自行变道（率同 baseline）；toll plan 进阻塞车道 0.52→0.17 但**无一转成功** ⇒ 自主选择存在雏形、不可靠 |

**失败三分拆（17 例，覆盖率 17/17 = 1.00 ≥ 0.90）**：

| 类 | n | 明细 |
|---|---|---|
| **`external_target_wrong`（外部规则选错）** | **10** | **tollgate 全部**：规则 target 指向被占/障碍车道且同路有空闲替代（4 例 target==F） |
| `policy_path_wrong_target_ok`（target 正确、路径选错） | 7 | **merge 全部**：target 为合法 route 车道，失败在路径横向几何压线/越界 |
| `policy_knows_obstacle_no_reroute` | 0（被优先级吸收；3 例同时满足） | b1:7784、base:7784、base:7790 |

**外部 target-lane 规则正确率（失败集）**：episode 级 **0.412**（10/17 错）；**tollgate 0/10 = 0.0**；merge 7/7 = 1.0 ⇒ **收费站失败集上"外部决定"本身是缺陷源**。

**自证**：默认关 baseline vs 录制 npz **逐键 bit-exact 18/18**（max\|Δ\|=0.0）；swap_target 复跑 3/3 exact（50 键逐位）。

## 2. F2 动态 hooks（默认关；P1–P6 全 PASS）

- **输入桩 40 键**（nav/lane_semantic raw+归一化/LD slots+role/offset/group/route/OD slots+id·presence/static[16:21]/6 帧 mem + `{source,shape,mask,units,frame,normalization,valid_range,schema_version}` 元数据）；**网络消费桩 351 键**（11 探针×phase×12 统计，含 **effective rank**〔全仓首例〕/有效 slot 比/act_norm/sat_frac）；**rollout 桩 17 键**（ref 6 步/30 点/世界系/弧长/速度、plan_pose/ego_pose_delta/exec_action、曲率/进度、候选分数接口=router/expert）；**LQR reference 11 键**（D3 M1 缺口）。
- 自证：默认关逐位、开共有键逐位、固定 batch 前向 27 输出 **max\|Δ\|=0**（54 次捕获）、RNG 零消耗、ckpt/config/参数 sha256 不变。D3 M1–M5 ✅、M6 ◐（warp 前后 anchor 未落）。
- Caveat：**候选路径分数接口就绪但 K1 无候选头**（`num_anchors=0`）——正是 WP3 要新增。

## 3. F3 场景普查 + 字段可用性

- **计数**：expert5k_v8 **455/441/906/907**（toll/merge/路口/普通，**≥目标 100/100/100/200 ✓**）；3d3 235/236/471/471；indep 24/24/48/49；calib50 5/4/9/9。
- **真实缺口**：①带 v7 字段的独立场景（54/55/109/110）②policy 失败 episode（21/16/28/28）。
- **字段矩阵**：车道几何/route/dynamic slots `in_data`；static slots / candidate 几何 / calib 侧车道几何 `offline_extract`；车道连接真值/calib dynamic slots `env_rerun`；v6 target-lane 重算**未验证**。
- **数据计划**：**DP-1（离线提取 ~2002 ep 候选几何+slots，6 workers ~0.5–1h）+ DP-4（static slots，toll 优先）先行**；DP-3(b) 失败集定向扩采（500 ep ≈10–20 min/臂）后置；覆盖门**不必因总量阻塞**。

## 4. 综合结论（R35 审计答案）

1. **外部 target-lane 规则是缺陷源**（toll 失败集 0/10 正确；规则不查障碍）⇒ GPT"外部不得决策车道/闸口"原则获**直接证据支持**；toll 失败 = "外部错误 target + Policy 跟随"（换 target 即产生变道意图、未完成变道）。
2. **merge 失败 = 横向路径几何**（target 7/7 正确；路径压线/越界）⇒ 需**车道锚定**横向生成（E3 `lane_center` 可行）。
3. **token 与 route 均被强消费**（干预即刻生效：μ 1.35/1.33、plan 1.94/2.36）。
4. **无 token 时模型仍会自行变道但不可靠**（toll 进阻塞车道 0.52→0.17 但 0 成功）。
5. **数据与 hooks 就绪**（expert5k 满足四类总量；DP-1/DP-4 离线提取；F2 hooks 可用）。

## 5. 建议（待裁）

- **诊断臂（cheap，建议先行）**：**障碍感知 target-lane 规则**（默认关、仅诊断）——量化"正确 target 下 Policy 的天花板"（toll 10/10 外部错 ⇒ 可能大幅改善 toll）；**不作生产路径**（GPT 原则不变），纯为归因/期望值评估。
- **WP3 设计输入**：候选须含**收费站空闲闸口车道**（从可见车道生成，不依赖外部规则）；横向路径**车道锚定**（E3）；merge 先保"本车道中心线续接"。
- **数据**：DP-1+DP-4 先行（离线提取）；失败集定向扩采后置。
- **下一步**：WP3 最小原型（候选车道/路径生成 + selector）——待 CG 确认后开工。
