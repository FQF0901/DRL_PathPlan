# env/expert

目的：提供两类"专家/参考"能力。

1. **规则基线控制器**（KPI 协议必需件）：`pure_pursuit_idm.PurePursuitIDMPolicy` ——
   横向 pure-pursuit 跟踪导航路线（自车车道锚定 + 后继车道链兜底），纵向 IDM 跟车 + 限速约束；
   输出 `[steer, throttle]`（`BasePolicy` 子类），无随机数 → 确定性，参数全部可配置。
2. **BC 冷启动数据源**：过滤后的 MetaDrive 内置 `IDMPolicy` 轨迹（仅保留"在路内 + 无碰撞 +
   有完整 6 点目标"的片段）。用户指示：专家只用于冷启动，最终性能靠 RL。

## 接口
- 基线：`PurePursuitIDMPolicy(control_object, ...)`，`act() -> [steer, throttle]`；
  可调项如 `steer_gain` / `lookahead_time` / IDM 参数（见文件内参数表）。
- BC 采集：见 `pipeline/stage_a.py`（P2）。

## 评测
```bash
tools/venv-python tools/baseline_eval.py --specs <specs.json> --workers 8 --max-steps 1000 --out runs/baseline_eval/<name>.json
```
KPI 口径与 `config/eval.yaml` 一致（route_completion / 碰撞 / 出界 / a_lon·a_lat 均值与 p95 / speed_ratio，
含 by_difficulty / by_geometry 分组）。

## 已知边界（实测）
- no-event easy 切片（n=23，max-steps=1000）：overall success 0.783，rc 0.896；失败集中在
  `bidirection`（0/3）、`roundabout`（0/1）、`tollgate`（1/3）→ 已派修复（根因诊断中）。
- "相对基线"KPI 在基线自身较弱的类别上需谨慎解释，必要时改用绝对阈值。
- MetaDrive 的 `ExpertPolicy` 预训练权重未纳入关键路径。
