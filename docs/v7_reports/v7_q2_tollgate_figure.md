# v7 Q2：tollgate 双面板可视化（spec 34，ckpt s11-u150）

- ckpt: `/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`
- spec: `env/specs/scenarios_eval500.json` id=34 blocks=CS$（CS$：curve;straight;tollgate）
- ckpt 运行: termination=collision steps=962（重放口径与 eval500 一致）
- 选样理由：id 34 是 s11 eval500 的 tollgate 碰撞之一（crash_building，rc=0.928，962 步，为 tollgate 碰撞中步数最长），闸口前有完整 curve+straight 接近段，适合展示“接近前/决策点/闸口”三时刻；id 243 在 s11 为 off_road（未到闸口），故不选。

## PNG 清单

- `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_A_first_sighting_step0821.png`（2100 px 宽）— A_first_sighting step=821
- `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_B_decision_zone_39m_step0869.png`（2100 px 宽）— B_decision_zone_39m step=869
- `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_C_gate_entry_step0929.png`（2100 px 宽）— C_gate_entry step=929
- `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_D_final_crash_step0961.png`（2100 px 宽）— D_final_crash step=961

## 运行时度量

- 岗亭：{"object_id": "ec3ae5cf-7926-4c01-b594-b072f0f16903", "x": 16.595294952392578, "y": -202.424072265625, "heading": -2.211431977554472, "length": 10.0, "width": 3.5, "lane_index": ["2S0_0_", "3$0_0_", "1"], "lane_id": 1, "lane_width": 3.5, "s_booth": 17.7516891759141, "lat_booth": 8.105525921564549e-07}
- 岗亭车道：['2S0_0_', '3$0_0_', '1']（lane_id=1，奇数），lane_width=3.5 m，lat_booth=8.105525921564549e-07 m
- `$` block 车道运行时限速：{"('2S0_0_', '3$0_0_', 0)": 5.6, "('2S0_0_', '3$0_0_', 1)": 5.6, "('2S0_0_', '3$0_0_', 2)": 5.6, "('-3$0_0_', '-2S0_0_', 0)": 5.6, "('-3$0_0_', '-2S0_0_', 1)": 5.6, "('-3$0_0_', '-2S0_0_', 2)": 5.6}
- LD offsets 实际出现（step 0）：[5.0, 10.0, 15.0, 20.0, 30.0]
- 注：任务描述里的 LD offset {20,40,60,80} 是当前 repo v5 口径；s11 冻结协议（pre_v5 git 2f4450e）实际为 {5,10,15,20,30}（`env/obs/ld.py` 默认），本图按实际标注。
- others.static 首次 present=1：step=821，dist_booth_center=54.71 m
- road_class=tollgate 首次=1：step=930，dist_booth_center=16.995 m
- 进入 `$` block：step=929，dist_booth_center=17.257 m
- IDM baseline：termination=arrive_dest steps=458；首次静态扫描（pp_static_gap>0）：step=362 dist=43.819 m；首次变道：step=378 dist=34.653 m lane_id=2

## 文字结论

1. **tollgate 无专用 LD 特征**：LD 16 槽只编码车道中心线采样（dx/dy/heading/curvature/speed_limit/线型）；岗亭是 BaseBuilding，不在 LD；决策点时刻 LD 槽位全部是车道几何。
2. **专用建模 = `others.static`（P1-A，扫描 50 m）+ `road_class`（迟到）**：`others.static` 首次 present=1 在 dist=54.71 m；`road_class=tollgate` 直到进入 `$` block 才=1（dist=17.257 m）——变道决策点在 25–39 m，彼时 road_class 仍为 0（迟到信号）。
3. **当前 top-K LD 不能表述岗亭**：LD 候选只有“自车车道 + 参考车道 + 同路相邻车道”，16 槽远端环会被裁；岗亭在车道正中（不在车道线上），无法由 LD 表达，只能靠 static 段/IDM 式建筑扫描。
4. **典型 tollgate 形态**：岗亭位于 奇数车道 lane_id=1 正中（lat≈8.105525921564549e-07 m）；运行时 `$` 车道限速={"('2S0_0_', '3$0_0_', 0)": 5.6, "('2S0_0_', '3$0_0_', 1)": 5.6, "('2S0_0_', '3$0_0_', 2)": 5.6, "('-3$0_0_', '-2S0_0_', 0)": 5.6, "('-3$0_0_', '-2S0_0_', 1)": 5.6, "('-3$0_0_', '-2S0_0_', 2)": 5.6} m/s（任务描述的“限速 3 m/s”与实测不符，实测 5.6 m/s）；需绕行（变道至自由车道）；IDM 在 dist≈34.653 m 首次变道（P1 triage 对照 ~39.5 m；本测首次静态扫描 pp_static_gap>0 在 43.819 m）。

- 总耗时 12.1 s