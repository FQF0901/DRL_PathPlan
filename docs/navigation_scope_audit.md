# Round-35 lane F1：导航与 target-lane 责任审计（只读 + 运行时干预）

- 生成：2026-10-14T20:49:35+0800；spec=`runs/p1_v2/round25/specs_calibration50.json`；协议：`{'ld_mode': 'semantic', 'rollout_semantic': 'static_role_only', 'moe_off': True, 'kappa_compose': True, 'max_steps': 1000}`
- 失败集：tollgate 10 + merge 7 = 17（d1 口径）；对照：['base:7608']
- 干预：`swap_target`（①）/ `route_rot`（②，α=-30.0°）/ `no_target_token` / `no_target_identity`（③）；每 arm 同协议闭环复跑

## 0. 结论摘要（自动汇总）

- **自证**：baseline 默认关 bit-exact 18/18（bitexact=True）；干预复现（swap_target 复跑）3/3 逐位（reproducible=True）。
- **① swap_target**（route 固定、target 换左邻车道）：首个决策步即改变行为（首差异决策中位=0）；μ max|Δ| 中位=1.3482 m、plan max|Δ| 中位=1.9366 m、车道序列差异率中位=0.7147；merge 成功增益=['b1:7559', 'b1:7608', 'base:7559']，无成功损失 ⇒ target lane 是强消费信号；但 tollgate 10 例无一转成功（车辆出现左移/变道意图但未完成变道）⇒ 仅换 target 不足以修复。
- **② route_rot**（target 固定、道路级 route/命令旋转 -30°+命令 right）：首差异决策中位=0；μ max|Δ| 中位=1.3344、plan max|Δ| 中位=2.3647、车道序列差异率中位=0.7841；成功损失=['base:7608'] ⇒ 道路级 route/命令通道被网络强消费，非旁观量。
- **③ 移除 target-lane token**：μ max|Δ| 中位=1.1518、plan max|Δ| 中位=1.8589、车道序列差异率中位=0.755；成功增益=['base:7559']，成功损失=-（identity：增益=-、损失=['base:7608']）。tollgate 上 plan 进入阻塞车道的比例显著下降（token 是把 plan 拉向 target 的信号之一），但无任何 tollgate 例转成功 ⇒ **不能证明'无 token 时具备正确的自主车道选择能力'**；策略仍会自行变道（变道率与 baseline 同量级），属'有行为、不可靠'。
- **失败三分拆**（17 例）：`{'external_target_wrong': 10, 'policy_path_wrong_target_ok': 7}`；覆盖率=17/17=1.0（≥0.90 ✓）。tollgate 10/10 归 'external_target_wrong'（规则 target 指向被占/障碍车道，同路有空闲替代）；merge 7/7 归 'policy_path_wrong_target_ok'（target 为合法 route 车道，失败在路径横向几何）。
- **第三类 'Policy 知障碍未改道'**：按优先级（B1>B3>B2）无 episode 单独归入，但 3 例同时满足 B3 条件（接近窗静态障碍 present=1 且失败窗内无变道响应，已被 B1 吸收：b1:7784, base:7784, base:7790）。若把 B3 置于 B1 之前，计数将变为 B1=7 / B3=3。
- **merge 的耦合证据**：① 交换 target 后 3 例失败转成功（b1:7559, b1:7608, base:7559）⇒ merge 失败与 target lane 选择耦合（规则用 lane_id 平局规则选中 1 号车道；直接缺陷是路径横向几何压线，但 target 选择的任意性是触发条件）。
- **外部 target-lane 规则正确率（失败集）**：episode 级 0.4118（10/17 规则错）；按类：tollgate 两类 0.0、merge+unattributed 1.0；step 级（失败窗）0.8789（被 merge 长窗稀释）。⇒ 在 tollgate 失败集上'外部决定'本身是缺陷源。

## 1. 输入清单（逐字段）

| 字段 | 来源 | 层级 | 规则生成 | 模拟器真值 | 部署可得 | 网络消费（file:line） |
|---|---|---|---|---|---|---|
| nav（obs 通道，1×11） | MetaDrive navigation：get_checkpoints()（世界系 → 自车系）+ navi_arrow_dir 命令 + route_completion；env/obs/nav.py::NavChannel.build:49-79、navigation_command:28-39 | 道路级（checkpoint=路段端点；命令=下一分叉转向；rc=全程进度） | 半规则：命令由 navi_arrow_dir 两段航向差 10° 阈值规则化（base_vehicle.py:254-277 同口径）；checkpoint/rc 为上游导航原值 | 是（simulator navigation 状态，非几何估计） | 可（env 提供；obs 默认通道） | net/model.py::_context_tokens:814-829 → embed_context:760-812 → nav_token；plan_step:832-890 融合输入；_head_tokens:892-940（T=39 中的 nav key） |
| nav 段（others[0:11]，规范输入） | env/obs/others.py 把 NavChannel 同口径写入 others 前 11 维；env/obs/schema.py::OTHERS_LAYOUT:116-138 | 道路级 | 同上（半规则） | 是 | 可（others 是规范上下文输入） | net/mem.py::mem_from_obs:288-（others_hist）→ MemEncoder.encode:688-706 → others_ctx；plan_step 融合输入 |
| route_world（64×2）+ route_world_mask | env/obs/world.py::route_world_polyline:49-74（nav.checkpoints + road_network.graph 路段终点，road 中心横向偏移）；RouteWorldChannel.build:106-120 | 道路级（世界系折线，覆盖整条路线） | 是（由导航路线确定性构造；几何来自地图真值） | 是（地图/路线几何） | 可（obs 默认通道，不进 6 帧历史） | net/model.py::world_state_from_obs:226-242 → _rollout:1113-1300 逐步 rebuild_nav_from_world:245-270 → net/mem.py::nav_features_from_world:465-560 |
| current_ref_lanes（env nav 属性） | MetaDrive node_network_navigation.py:120（road_network.graph[start][end]）、update_navigation_info:159-187 | 道路级 → 当前路段车道集合（多车道时多条） | 否（上游导航按 route 更新） | 是 | 可（env 对象；不入 obs 张量，仅被规则消费） | 间接：env/obs/lane_semantic.py::select_target_lane:355-405（target lane）、slot_route_associated:486-493（ld_route_associated）；env/obs/ld.py::candidate_lanes:143-179（LD 采样优先级） |
| next_ref_lanes（env nav 属性） | 同上 node_network_navigation.py:125,184-187 | 道路级 → 下一路段车道集合 | 否 | 是 | 可（同上，仅被规则消费） | 间接：select_target_lane 的 continuation/next 分支（lane_semantic.py:334-352,384-401）；classify_slot_roles 的 continuation（425-464）；LD candidate_lanes |
| target_lane（车道对象，1 条） | env/obs/lane_semantic.py::select_target_lane:355-405（唯一纯函数） | 车道级（唯一车道） | 是（route_current → route_continuation → route_next → ego_fallback；契约 docs/r21_target_lane_selection.md） | 否（由 route 车道 + ego 车道规则选出；不查障碍/可行性） | 可（纯函数，输入均 env 可给） | 间接（对象本身不入网络）：LaneSemanticChannel.build:581-637（12 维特征）、env/obs/ld.py::_build_supplements:315-357（逐槽 role/group）→ net/semantic.py::SemanticWiring.forward:132-191 |
| target_lane_token（1×H） | net/semantic.py::LaneSemanticEncoder.forward:95-102（lane_semantic → 归一化 → 2 层 MLP，mask/valid=0 ⇒ 严格 0） | 车道级（token） | 否（网络学得表示；输入=规则生成的 lane_semantic） | 否 | 可（网络内部张量） | net/semantic.py::SemanticWiring.forward:156,190（readout 第 1 组）；net/model.py::build_geometry_context:687-691（target_eff=β_token·token）；_head_tokens:919-922（target_proj 折叠进 ld_pool 槽）；encode:996-1018 与 _rollout_step_geometry:547-583（rollout 逐步注入） |
| ld_route_associated（16 槽伴随数组） | env/obs/lane_semantic.py::slot_route_associated:486-493（槽车道 ∈ current_ref ∪ next_ref，按对象身份）；env/obs/ld.py::_build_supplements:315-357 | 车道级（逐 LD 槽） | 是（集合成员判定） | 是（route 车道集合来自导航真值） | 可（obs 伴随数组，随 ld 进 6 帧历史 <key>_hist） | net/model.py::_semantic_inputs_from_obs:621-658 → net/semantic.py::SemanticWiring.forward:162,168（route_embed）、178,187（route pool） |
| lane semantic（1×12，相对 target lane） | env/obs/lane_semantic.py::LaneSemanticChannel.build:581-637（lat/heading_error/curvature/margins/valid/confidence/lane_change/distance）；归一化常数 LANE_SEMANTIC_NORMALIZATION:133-146 | 车道级（相对 target lane；自车系 1 槽） | 是（几何规则计算；不含 raw lane_id） | 否（几何测量 + 规则量；margin 是同向 road-surface 非 legal corridor） | 可（obs 默认通道；不进历史） | net/model.py::_semantic_inputs_from_obs:621-658 → LaneSemanticEncoder:73-102 → SemanticWiring（target token + delta）；rollout static_role_only 动态维置零（net/model.py::_rollout_step_geometry:565-578） |
| route command（one-hot，nav[4:10]） | env/obs/nav.py::navigation_command:28-39（navi_arrow_dir 两段航向差；10° 阈值，forward/left/right + 3 保留） | 道路级（下一分叉转向） | 是（上游 navi_arrow_dir → 离散命令的确定性规则） | 是（navi_arrow_dir 为 simulator 导航状态） | 可（nav + others nav 段） | 同 nav：net/model.py::_context_tokens:814-829 + mem others_hist → MemEncoder.encode:688-706；plan_step/_head_tokens 的 nav token |
| （补充）ld_lane_role / ld_group / ld_offset_s_norm | env/obs/lane_semantic.py::classify_slot_roles:425-464、slot_group_ids:467-483、slot_offset_s_normalized:496-504 | 车道级（逐 LD 槽；role 相对 target lane） | 是 | 否（role 由 target lane 规则派生；group/offset 为几何） | 可（obs 伴随数组） | net/semantic.py::SemanticWiring.forward:159-171（role_embed/group_relation_embed/offset_encoder）、174-189（分组池化） |
| （补充）ego.lane（自车当前车道） | MetaDrive vehicle.lane（map 查询真值）；env/obs/ld.py::candidate_lanes:147 | 车道级 | 否 | 是 | 可 | 间接：select_target_lane 的 tie-break/兜底（lane_semantic.py:320-328,403-405）、_lane_change_flags:640-662、LD primary 槽 |

## 2. 自证

- **默认关 bit-exact（baseline arm vs 录制 npz）**：exact 18/18；bitexact=True
- **干预可复现（同 arm 复跑逐位）**：arm=`swap_target`，exact 3/3；reproducible=True

## 3. 干预响应（vs baseline，逐 episode）

| arm | μ max\|Δ\| 中位 | plan max\|Δ\| 中位 | 车道序列差异率 中位 | 首差异决策 中位 | target 变化率 中位 | 终段=used target 中位 | 终段=base target 中位 | 成功增益 | 成功损失 |
|---|---|---|---|---|---|---|---|---|---|
| swap_target | 1.3482 | 1.9366 | 0.7147 | 0 | 1.0 | 0.1024 | 0.3519 | ['b1:7559', 'b1:7608', 'base:7559'] | - |
| route_rot | 1.3344 | 2.3647 | 0.7841 | 0 | 0.0558 | 0.2156 | 0.1964 | - | ['base:7608'] |
| no_target_token | 1.1518 | 1.8589 | 0.755 | 0 | 0.1669 | 0.4356 | 0.3826 | ['base:7559'] | - |
| no_target_identity | 1.1755 | 1.8173 | 0.7338 | 0 | 0.2188 | 0.4692 | 0.4427 | - | ['base:7608'] |

### swap_target

| run | id | n_dec | μ max|Δ| | plan max|Δ| | 车道序列差异率 | 首差异决策 | target 变化率 | 路径终段=used target | 路径终段=base target | 变道率 |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 7784 | 77 | 1.8904 | 3.3779 | 0.753 | 0 | 1.000 | 0.065 | 0.468 | 0.351 |
| b1 | 7789 | 143 | 1.2094 | 1.9051 | 0.762 | 0 | 1.000 | 0.287 | 0.308 | 0.608 |
| b1 | 7790 | 51 | 2.4385 | 3.0826 | 0.627 | 0 | 1.000 | 0.078 | 0.549 | 0.510 |
| b1 | 7795 | 103 | 1.5405 | 2.3821 | 0.670 | 0 | 1.000 | 0.107 | 0.447 | 0.524 |
| b1 | 7797 | 88 | 0.9764 | 1.2666 | 0.818 | 0 | 1.000 | 0.148 | 0.341 | 0.545 |
| base | 7784 | 57 | 1.5506 | 2.0020 | 0.509 | 0 | 1.000 | 0.070 | 0.456 | 0.456 |
| base | 7789 | 165 | 1.2689 | 2.2417 | 0.758 | 0 | 1.000 | 0.255 | 0.351 | 0.576 |
| base | 7790 | 63 | 1.1023 | 1.4211 | 0.508 | 0 | 1.000 | 0.048 | 0.587 | 0.381 |
| base | 7795 | 105 | 1.4089 | 2.2998 | 0.743 | 0 | 0.876 | 0.067 | 0.352 | 0.514 |
| base | 7797 | 102 | 0.7703 | 0.9453 | 0.804 | 0 | 1.000 | 0.098 | 0.520 | 0.471 |
| b1 | 7559 | 73 | 1.3643 | 2.4006 | 0.644 | 0 | 1.000 | 0.069 | 0.397 | 0.630 |
| b1 | 7579 | 178 | 0.9743 | 1.0657 | 0.730 | 0 | 1.000 | 0.416 | 0.230 | 0.663 |
| b1 | 7580 | 116 | 1.3320 | 3.1732 | 0.767 | 0 | 1.000 | 0.578 | 0.043 | 0.310 |
| b1 | 7608 | 99 | 1.4405 | 1.8958 | 0.737 | 0 | 1.000 | 0.111 | 0.333 | 0.616 |
| base | 7559 | 69 | 1.0756 | 1.2443 | 0.623 | 0 | 1.000 | 0.058 | 0.725 | 0.449 |
| base | 7579 | 200 | 1.5658 | 1.5658 | 0.645 | 0 | 1.000 | 0.375 | 0.315 | 0.590 |
| base | 7580 | 76 | 1.4267 | 1.9682 | 0.579 | 0 | 0.987 | 0.276 | 0.092 | 0.461 |
| base | 7608 | 113 | 0.8284 | 1.2539 | 0.699 | 0 | 1.000 | 0.062 | 0.345 | 0.602 |

### route_rot

| run | id | n_dec | μ max|Δ| | plan max|Δ| | 车道序列差异率 | 首差异决策 | target 变化率 | 路径终段=used target | 路径终段=base target | 变道率 |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 7784 | 48 | 1.2560 | 3.1870 | 0.708 | 0 | 0.021 | 0.208 | 0.208 | 0.792 |
| b1 | 7789 | 152 | 1.1707 | 1.1707 | 0.836 | 0 | 0.033 | 0.230 | 0.217 | 0.770 |
| b1 | 7790 | 63 | 1.7778 | 3.4395 | 0.556 | 0 | 0.032 | 0.429 | 0.429 | 0.571 |
| b1 | 7795 | 104 | 1.5106 | 3.1648 | 0.837 | 0 | 0.077 | 0.154 | 0.115 | 0.846 |
| b1 | 7797 | 96 | 0.8524 | 1.2379 | 0.844 | 0 | 0.031 | 0.146 | 0.125 | 0.854 |
| base | 7784 | 63 | 1.5231 | 2.4919 | 0.524 | 0 | 0.191 | 0.698 | 0.667 | 0.318 |
| base | 7789 | 171 | 0.9098 | 2.1781 | 0.807 | 0 | 0.041 | 0.246 | 0.234 | 0.754 |
| base | 7790 | 72 | 1.5029 | 2.5911 | 0.375 | 0 | 0.028 | 0.639 | 0.611 | 0.375 |
| base | 7795 | 113 | 1.6014 | 2.7620 | 0.858 | 0 | 0.133 | 0.177 | 0.133 | 0.832 |
| base | 7797 | 103 | 1.0503 | 1.0535 | 0.961 | 0 | 0.029 | 0.175 | 0.184 | 0.825 |
| b1 | 7559 | 65 | 1.6395 | 3.4369 | 0.892 | 0 | 0.200 | 0.231 | 0.169 | 0.769 |
| b1 | 7579 | 184 | 1.2456 | 1.2456 | 0.707 | 0 | 0.060 | 0.223 | 0.228 | 0.641 |
| b1 | 7580 | 116 | 1.1151 | 3.0229 | 0.448 | 0 | 0.052 | 0.043 | 0.043 | 0.466 |
| b1 | 7608 | 96 | 1.4015 | 2.2376 | 0.927 | 0 | 0.135 | 0.073 | 0.052 | 0.927 |
| base | 7559 | 67 | 1.6837 | 3.2767 | 0.761 | 0 | 0.090 | 0.224 | 0.224 | 0.776 |
| base | 7579 | 200 | 1.2748 | 1.2748 | 0.675 | 0 | 0.040 | 0.265 | 0.260 | 0.610 |
| base | 7580 | 82 | 1.3940 | 2.0578 | 0.476 | 0 | 0.122 | 0.061 | 0.061 | 0.329 |
| base | 7608 | 103 | 0.9259 | 1.3686 | 0.845 | 0 | 0.068 | 0.165 | 0.146 | 0.835 |

### no_target_token

| run | id | n_dec | μ max|Δ| | plan max|Δ| | 车道序列差异率 | 首差异决策 | target 变化率 | 路径终段=used target | 路径终段=base target | 变道率 |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 7784 | 60 | 1.8464 | 3.2780 | 0.817 | 0 | 0.400 | 0.467 | 0.750 | 0.533 |
| b1 | 7789 | 27 | 0.9620 | 1.4234 | 0.741 | 0 | 0.148 | 0.333 | 0.259 | 0.667 |
| b1 | 7790 | 46 | 1.7911 | 2.9692 | 0.826 | 0 | 0.391 | 0.543 | 0.457 | 0.457 |
| b1 | 7795 | 85 | 0.9898 | 2.7507 | 0.835 | 0 | 0.482 | 0.435 | 0.259 | 0.577 |
| b1 | 7797 | 30 | 0.5337 | 1.2345 | 0.867 | 0 | 0.567 | 0.400 | 0.267 | 0.600 |
| base | 7784 | 63 | 1.1936 | 2.1190 | 0.413 | 0 | 0.143 | 0.730 | 0.730 | 0.270 |
| base | 7789 | 25 | 1.8085 | 1.8509 | 0.640 | 0 | 0.160 | 0.400 | 0.320 | 0.600 |
| base | 7790 | 68 | 0.9359 | 1.5729 | 0.353 | 0 | 0.059 | 0.691 | 0.662 | 0.309 |
| base | 7795 | 25 | 0.8094 | 2.8215 | 0.560 | 0 | 0.120 | 0.360 | 0.360 | 0.640 |
| base | 7797 | 105 | 0.5887 | 1.0245 | 0.743 | 0 | 0.029 | 0.610 | 0.619 | 0.391 |
| b1 | 7559 | 73 | 1.6379 | 2.6713 | 0.767 | 0 | 0.096 | 0.397 | 0.411 | 0.726 |
| b1 | 7579 | 25 | 1.3961 | 1.3961 | 0.800 | 0 | 0.320 | 0.280 | 0.160 | 0.720 |
| b1 | 7580 | 116 | 1.1627 | 3.2133 | 0.862 | 0 | 0.233 | 0.276 | 0.405 | 0.560 |
| b1 | 7608 | 39 | 1.1410 | 2.0841 | 0.769 | 0 | 0.385 | 0.436 | 0.282 | 0.564 |
| base | 7559 | 69 | 1.1106 | 1.1106 | 0.493 | 0 | 0.130 | 0.522 | 0.522 | 0.478 |
| base | 7579 | 23 | 1.5171 | 1.5171 | 0.783 | 0 | 0.174 | 0.304 | 0.217 | 0.696 |
| base | 7580 | 82 | 0.7444 | 1.8670 | 0.695 | 0 | 0.061 | 0.451 | 0.415 | 0.427 |
| base | 7608 | 113 | 1.2815 | 1.2815 | 0.735 | 0 | 0.274 | 0.478 | 0.327 | 0.531 |

### no_target_identity

| run | id | n_dec | μ max|Δ| | plan max|Δ| | 车道序列差异率 | 首差异决策 | target 变化率 | 路径终段=used target | 路径终段=base target | 变道率 |
|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 7784 | 53 | 1.6037 | 3.4783 | 0.679 | 0 | 0.321 | 0.415 | 0.623 | 0.585 |
| b1 | 7789 | 25 | 0.5607 | 1.0260 | 0.720 | 0 | 0.200 | 0.280 | 0.240 | 0.720 |
| b1 | 7790 | 65 | 1.2339 | 3.0466 | 0.708 | 0 | 0.431 | 0.538 | 0.354 | 0.462 |
| b1 | 7795 | 78 | 0.9649 | 3.1150 | 0.795 | 0 | 0.410 | 0.397 | 0.321 | 0.615 |
| b1 | 7797 | 37 | 0.5716 | 1.2060 | 0.838 | 0 | 0.540 | 0.460 | 0.216 | 0.540 |
| base | 7784 | 63 | 1.1879 | 2.0150 | 0.365 | 0 | 0.143 | 0.762 | 0.746 | 0.238 |
| base | 7789 | 57 | 1.1631 | 1.6493 | 0.825 | 0 | 0.597 | 0.368 | 0.526 | 0.632 |
| base | 7790 | 58 | 1.6692 | 2.5269 | 0.345 | 0 | 0.017 | 0.776 | 0.759 | 0.224 |
| base | 7795 | 28 | 1.0268 | 2.8757 | 0.643 | 0 | 0.214 | 0.464 | 0.393 | 0.536 |
| base | 7797 | 99 | 0.6080 | 0.9945 | 0.748 | 0 | 0.030 | 0.606 | 0.576 | 0.394 |
| b1 | 7559 | 65 | 1.5409 | 2.8746 | 0.785 | 0 | 0.108 | 0.308 | 0.292 | 0.631 |
| b1 | 7579 | 19 | 0.8694 | 0.8694 | 0.789 | 0 | 0.263 | 0.263 | 0.210 | 0.737 |
| b1 | 7580 | 116 | 2.1596 | 3.6779 | 0.802 | 0 | 0.035 | 0.474 | 0.457 | 0.414 |
| b1 | 7608 | 42 | 1.0991 | 1.5948 | 0.762 | 0 | 0.429 | 0.476 | 0.357 | 0.524 |
| base | 7559 | 69 | 1.2457 | 1.4774 | 0.565 | 0 | 0.174 | 0.565 | 0.638 | 0.435 |
| base | 7579 | 200 | 1.0323 | 1.2477 | 0.665 | 0 | 0.320 | 0.415 | 0.495 | 0.540 |
| base | 7580 | 59 | 1.5584 | 1.9853 | 0.848 | 0 | 0.085 | 0.831 | 0.763 | 0.186 |
| base | 7608 | 112 | 1.4321 | 1.6473 | 0.634 | 0 | 0.223 | 0.482 | 0.429 | 0.518 |

### 3.1 ③ 无 target token 时的车道选择能力

| run | id | baseline 变道率 | no_token 变道率 | no_token 终段=基线target | no_token 终段=自车车道 | no_identity 变道率 | no_identity 终段=基线target |
|---|---|---|---|---|---|---|---|
| b1 | 7784 | 0.472 | 0.472 | 0.717 | 0.528 | 0.585 | 0.623 |
| b1 | 7789 | 0.680 | 0.720 | 0.280 | 0.280 | 0.720 | 0.240 |
| b1 | 7790 | 0.391 | 0.457 | 0.457 | 0.543 | 0.435 | 0.500 |
| b1 | 7795 | 0.577 | 0.615 | 0.282 | 0.385 | 0.615 | 0.321 |
| b1 | 7797 | 0.767 | 0.600 | 0.267 | 0.400 | 0.633 | 0.267 |
| base | 7784 | 0.333 | 0.270 | 0.730 | 0.730 | 0.238 | 0.746 |
| base | 7789 | 0.680 | 0.600 | 0.320 | 0.400 | 0.560 | 0.400 |
| base | 7790 | 0.362 | 0.293 | 0.707 | 0.707 | 0.224 | 0.759 |
| base | 7795 | 0.560 | 0.640 | 0.360 | 0.360 | 0.560 | 0.440 |
| base | 7797 | 0.748 | 0.353 | 0.657 | 0.646 | 0.394 | 0.576 |
| b1 | 7559 | 0.692 | 0.692 | 0.462 | 0.308 | 0.631 | 0.292 |
| b1 | 7579 | 0.632 | 0.789 | 0.210 | 0.210 | 0.737 | 0.210 |
| b1 | 7580 | 0.422 | 0.560 | 0.405 | 0.440 | 0.414 | 0.457 |
| b1 | 7608 | 0.590 | 0.564 | 0.282 | 0.436 | 0.538 | 0.359 |
| base | 7559 | 0.536 | 0.478 | 0.522 | 0.522 | 0.435 | 0.638 |
| base | 7579 | 0.739 | 0.696 | 0.217 | 0.304 | 0.696 | 0.304 |
| base | 7580 | 0.356 | 0.475 | 0.339 | 0.525 | 0.186 | 0.763 |
| base | 7608 | 0.661 | 0.527 | 0.330 | 0.473 | 0.518 | 0.429 |

**tollgate：plan 终段进入阻塞车道 F 的比例**（token 是把 plan 拉向 target 的信号之一）

| run | id | F | baseline | no_token | no_identity |
|---|---|---|---|---|---|
| b1 | 7784 | ['1S0_0_\|2$0_0_\|1'] | 0.5195 | 0.1667 | 0.1321 |
| b1 | 7789 | ['2S0_0_\|3$0_0_\|1'] | 0.0263 | 0.0 | 0.0 |
| b1 | 7790 | ['1S0_0_\|2$0_0_\|1'] | 0.1846 | 0.0 | 0.0441 |
| b1 | 7795 | ['2S0_0_\|3$0_0_\|1'] | 0.037 | 0.0 | 0.0 |
| b1 | 7797 | ['2S0_0_\|3$0_0_\|1'] | 0.0612 | 0.0 | 0.0 |
| base | 7784 | ['1S0_0_\|2$0_0_\|1'] | 0.254 | 0.0 | 0.0317 |
| base | 7789 | ['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1', '2S0_0_\|3$0_0_\|2'] | 0.0625 | 0.0 | 0.0 |
| base | 7790 | ['1S0_0_\|2$0_0_\|1'] | 0.2838 | 0.2206 | 0.069 |
| base | 7795 | ['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1'] | 0.0609 | 0.0 | 0.0 |
| base | 7797 | ['2S0_0_\|3$0_0_\|0'] | 0.0283 | 0.0095 | 0.0303 |

## 4. 失败三分拆（tollgate 10 + merge 7）

**① tollgate 例的变道意图响应**（lane_change flag 置位率 / 自车横向漂移，lane_lat 后半段均值）

| run | id | baseline chg_flag 率 | swap chg_flag 率 | baseline lane_lat(后半) | swap lane_lat(后半) | swap 结局 |
|---|---|---|---|---|---|---|
| b1 | 7784 | 0.0 | 1.0 | -0.006 | -0.765 | out_of_road(失败) |
| b1 | 7789 | 0.0 | 1.0 | 0.162 | -0.78 | out_of_road(失败) |
| b1 | 7790 | 0.0 | 1.0 | 0.003 | -0.849 | out_of_road(失败) |
| b1 | 7795 | 0.0 | 1.0 | 0.21 | -0.754 | out_of_road(失败) |
| b1 | 7797 | 0.0 | 1.0 | 0.141 | -0.805 | out_of_road(失败) |
| base | 7784 | 0.0 | 1.0 | -0.106 | -0.869 | out_of_road(失败) |
| base | 7789 | 0.0 | 1.0 | -0.069 | -0.921 | out_of_road(失败) |
| base | 7790 | 0.0 | 1.0 | -0.079 | -0.875 | out_of_road(失败) |
| base | 7795 | 0.0 | 1.0 | -0.051 | -0.869 | out_of_road(失败) |
| base | 7797 | 0.0 | 1.0 | -0.009 | -0.902 | out_of_road(失败) |

- 计数：`{'external_target_wrong': 10, 'policy_path_wrong_target_ok': 7}`；覆盖率 = 17/17 = **1.0**（目标 ≥0.90）

| run | id | class | F（失败车道） | target∈F | target 被占(有替代) | 无变道(有障碍) | 路径终段∈F | ① swap 结果 | 三分拆 | conf | 证据 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 | 7784 | wrong_gate_selection | ['1S0_0_\|2$0_0_\|1'] | True | True | True/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：True；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1S0_0_\|2$0_0_\|1']，F=['1S0_0_\|2$0_0_\|1']） |
| b1 | 7789 | wrong_gate_selection | ['2S0_0_\|3$0_0_\|1'] | False | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：False；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1C0_1_\|2S0_0_\|1']，F=['2S0_0_\|3$0_0_\|1']） |
| b1 | 7790 | path_hits_obstacle | ['1S0_0_\|2$0_0_\|1'] | True | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：True；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1S0_0_\|2$0_0_\|1', '>>>\|1S0_0_\|1']，F=['1S0_0_\|2$0_0_\|1']） |
| b1 | 7795 | wrong_gate_selection | ['2S0_0_\|3$0_0_\|1'] | False | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：False；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1C0_1_\|2S0_0_\|1']，F=['2S0_0_\|3$0_0_\|1']） |
| b1 | 7797 | wrong_gate_selection | ['2S0_0_\|3$0_0_\|1'] | False | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：False；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1S0_0_\|2S0_0_\|1']，F=['2S0_0_\|3$0_0_\|1']） |
| base | 7784 | wrong_gate_selection | ['1S0_0_\|2$0_0_\|1'] | True | True | True/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：True；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1S0_0_\|2$0_0_\|1']，F=['1S0_0_\|2$0_0_\|1']） |
| base | 7789 | path_hits_obstacle | ['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1', '2S0_0_\|3$0_0_\|2'] | False | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：False；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1C0_1_\|2S0_0_\|1']，F=['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1', '2S0_0_\|3$0_0_\|2']） |
| base | 7790 | wrong_gate_selection | ['1S0_0_\|2$0_0_\|1'] | True | True | True/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：True；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1S0_0_\|2$0_0_\|1']，F=['1S0_0_\|2$0_0_\|1']） |
| base | 7795 | path_hits_obstacle | ['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1'] | False | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：False；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1C0_1_\|2S0_0_\|1']，F=['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1']） |
| base | 7797 | path_hits_obstacle | ['2S0_0_\|3$0_0_\|0'] | False | True | False/True | True | out_of_road(失败) | **external_target_wrong** | high | target∈F(障碍车道)：False；target 自身走廊被占且有同路空闲替代：True（win=5 步，T=['1S0_0_\|2S0_0_\|1']，F=['2S0_0_\|3$0_0_\|0']） |
| b1 | 7559 | path_out_of_drivable | ['2y1_1_\|2y0_0_\|0'] | False | False | False/False | True | arrive_dest(成功, gain) | **policy_path_wrong_target_ok** | high | target∉F 且 target 未被占；policy 路径终段∈F=True 或窗内路径本身无效=True（win=15 步，T=['1S0_0_\|2y0_0_\|1', '>>>\|1S0_0_\|1']，F=['2y1_1_\|2y0_0_\|0']） |
| b1 | 7579 | path_out_of_drivable | ['>>>\|1C0_0_\|0'] | False | False | False/False | True | out_of_road(失败) | **policy_path_wrong_target_ok** | high | target∉F 且 target 未被占；policy 路径终段∈F=True 或窗内路径本身无效=True（win=156 步，T=['1C0_0_\|1C0_1_\|1', '1C0_1_\|2S0_0_\|0', '2S0_0_\|3y0_0_\|0', '>>>\|1C0_0_\|1']，F=['>>>\|1C0_0_\|0']） |
| b1 | 7580 | path_out_of_drivable | ['2y1_2_\|2y0_0_\|0'] | False | False | False/False | True | out_of_road(失败) | **policy_path_wrong_target_ok** | high | target∉F 且 target 未被占；policy 路径终段∈F=True 或窗内路径本身无效=True（win=15 步，T=['1S0_0_\|2y0_0_\|0']，F=['2y1_2_\|2y0_0_\|0']） |
| b1 | 7608 | path_out_of_drivable | ['3y1_1_\|3y0_0_\|0'] | False | False | False/False | True | arrive_dest(成功, gain) | **policy_path_wrong_target_ok** | high | target∉F 且 target 未被占；policy 路径终段∈F=True 或窗内路径本身无效=True（win=18 步，T=['1S0_0_\|2S0_0_\|1', '2S0_0_\|3y0_0_\|1']，F=['3y1_1_\|3y0_0_\|0']） |
| base | 7559 | path_out_of_drivable | ['2y1_1_\|2y0_0_\|0'] | False | False | False/False | True | arrive_dest(成功, gain) | **policy_path_wrong_target_ok** | high | target∉F 且 target 未被占；policy 路径终段∈F=True 或窗内路径本身无效=True（win=19 步，T=['1S0_0_\|2y0_0_\|1', '>>>\|1S0_0_\|1']，F=['2y1_1_\|2y0_0_\|0']） |
| base | 7579 | unattributed | [] | False | False | False/False | False | max_step(失败) | **policy_path_wrong_target_ok** | low | target∉F 且 target 未被占；policy 路径终段∈F=False 或窗内路径本身无效=True（win=134 步，T=['1C0_0_\|1C0_1_\|1', '1C0_1_\|2S0_0_\|0', '>>>\|1C0_0_\|1']，F=[]） |
| base | 7580 | path_out_of_drivable | ['2y1_2_\|2y0_0_\|0'] | False | False | False/False | True | out_of_road(失败) | **policy_path_wrong_target_ok** | high | target∉F 且 target 未被占；policy 路径终段∈F=True 或窗内路径本身无效=True（win=6 步，T=['1S0_0_\|2y0_0_\|0']，F=['2y1_2_\|2y0_0_\|0']） |

## 5. 外部 target-lane 规则正确率（失败集）

- step 级（失败窗）：413 步中 50 步规则 target 错（target==F 或 target 被占且有空闲替代）⇒ **正确率 = 0.8789**
- episode 级（≥50% 窗口步错 ⇒ 规则错）：{'n_episodes_rule_wrong': 10, 'correct_rate': 0.4118} ⇒ **正确率 = 0.4118**
  - `wrong_gate_selection`：6/6 规则错 ⇒ 正确率 0.0
  - `path_hits_obstacle`：4/4 规则错 ⇒ 正确率 0.0
  - `path_out_of_drivable`：0/6 规则错 ⇒ 正确率 1.0
  - `unattributed`：0/1 规则错 ⇒ 正确率 1.0
- 分项：target==F 的 episode=4/17；target 被占（有替代）的 episode=10/17

| run | id | class | F | 窗口步 | 窗口错步 | 窗口错误率 | 全 episode 错误率 | target==F |
|---|---|---|---|---|---|---|---|---|
| b1 | 7784 | wrong_gate_selection | ['1S0_0_\|2$0_0_\|1'] | 5 | 5 | 1.0 | 0.7013 | True |
| b1 | 7789 | wrong_gate_selection | ['2S0_0_\|3$0_0_\|1'] | 5 | 5 | 1.0 | 0.125 | False |
| b1 | 7790 | path_hits_obstacle | ['1S0_0_\|2$0_0_\|1'] | 5 | 5 | 1.0 | 0.4308 | True |
| b1 | 7795 | wrong_gate_selection | ['2S0_0_\|3$0_0_\|1'] | 5 | 5 | 1.0 | 0.2778 | False |
| b1 | 7797 | wrong_gate_selection | ['2S0_0_\|3$0_0_\|1'] | 5 | 5 | 1.0 | 0.1837 | False |
| base | 7784 | wrong_gate_selection | ['1S0_0_\|2$0_0_\|1'] | 5 | 5 | 1.0 | 0.5556 | True |
| base | 7789 | path_hits_obstacle | ['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1', '2S0_0_\|3$0_0_\|2'] | 5 | 5 | 1.0 | 0.1364 | False |
| base | 7790 | wrong_gate_selection | ['1S0_0_\|2$0_0_\|1'] | 5 | 5 | 1.0 | 0.4595 | True |
| base | 7795 | path_hits_obstacle | ['2S0_0_\|3$0_0_\|0', '2S0_0_\|3$0_0_\|1'] | 5 | 5 | 1.0 | 0.313 | False |
| base | 7797 | path_hits_obstacle | ['2S0_0_\|3$0_0_\|0'] | 5 | 5 | 1.0 | 0.1698 | False |
| b1 | 7559 | path_out_of_drivable | ['2y1_1_\|2y0_0_\|0'] | 15 | 0 | 0.0 | 0.0 | False |
| b1 | 7579 | path_out_of_drivable | ['>>>\|1C0_0_\|0'] | 156 | 0 | 0.0 | 0.0 | False |
| b1 | 7580 | path_out_of_drivable | ['2y1_2_\|2y0_0_\|0'] | 15 | 0 | 0.0 | 0.0 | False |
| b1 | 7608 | path_out_of_drivable | ['3y1_1_\|3y0_0_\|0'] | 18 | 0 | 0.0 | 0.0 | False |
| base | 7559 | path_out_of_drivable | ['2y1_1_\|2y0_0_\|0'] | 19 | 0 | 0.0 | 0.0 | False |
| base | 7579 | unattributed | [] | 134 | 0 | 0.0 | 0.0 | False |
| base | 7580 | path_out_of_drivable | ['2y1_2_\|2y0_0_\|0'] | 6 | 0 | 0.0 | 0.0 | False |

## 6. Caveats

- 干预为**运行时 obs 层改写**（新脚本内 patch env.obs.lane_semantic/ld.select_target_lane 与 builder.build 返回值），不改任何既有文件；baseline arm 不触达任何改写路径 ⇒ 逐键 bit-exact 自证。
- ① swap_target 的 target lane 是**反事实车道**：选择函数的 source/confidence 保留 route_current/1.0（route 未变，target 被替换）⇒ lane_semantic 的 change flags/margins/curvature 均按交换后车道重算，LD role/group 同步（同一被 patch 的函数）。相邻车道不存在时退化到 next_ref/current_ref 替代（swap_kind 记录），均无则 no-op（target 不变，已标注）。
- ② route_rot 是**合成反事实路线**（-30° 旋转 + 命令 right），不是地图上真实存在的另一条路线；route_completion 保持原值、ld_route_associated 保持原值（车道级关联不动）⇒ 该 arm 只检验道路级 route/命令通道的敏感性，不可解读为'真实改道'。
- ③ no_target_token 只把 lane_semantic 置 0（mask=0 ⇒ LaneSemanticEncoder 输出严格 0）；no_target_identity 另把 ld_lane_role 中性化为 other。route/route_assoc/LD 几何全部保留；语义 delta 的 role_embed/group_relation 在 no_target_identity 下退化为'无 target 身份'口径。
- A 链（d1）证据用于失败车道 F 的确定；B 链（本工具）baseline arm 提供**规则 target lane 真值**（同一 select_target_lane 函数在真实 nav 状态上求值）与 policy 路径车道序列（plan 6 步几何 → 世界系 → ray_localization 最近车道，env 同款 heading 过滤）。
- 三分拆是启发式、规则写死（见 md 规则段）；阈值/优先级改动会改变计数。
- merge 失败的 F = 失败窗内首个 env 口径无效位姿所在车道（多为压连续线/sidewalk），与 tollgate 的障碍车道语义不同：merge 类外部 target lane 的'选错'判定只看 target==F（不计入 B1）或 target 走廊被占（merge 无静态障碍 ⇒ 不计）。
- target lane '被占' 判定 = 该车道自身走廊（半宽+1 m，不含相邻车道）内 50 m 前方有静态建筑；建筑位于 target lane 的**延长线/下一段同一物理车道**时（tollgate 前一段车道）仍计入 ⇒ b1:7789/7795/7797、base:7797 的 target（gate 前一段车道）因此判为被占；若只算 target lane 自身弧长范围内，这些例的 blocked 会变为 False（计数会变，结论方向不变：target 指向的走廊仍被岗亭占据）。
- ③ 的 no_target_token/no_target_identity 是**分布外消融**（训练时该 token 恒在）：行为整体退化是预期现象，不能把'退化'直接读作'该通道无用'；本审计只用它回答'去掉 token 后是否仍能正确选道'（答案：无证据支持，且部分 merge 例反而成功 ⇒ 存在行为、不可靠）。
- B 链复跑依赖单进程 GPU 确定性；bit-exact 自证在 baseline arm 上逐键比较（无 z_* 键，见 SAVE_DEC_KEYS）。
