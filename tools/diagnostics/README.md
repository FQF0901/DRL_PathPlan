# tools/diagnostics

一次性诊断 / 复现脚本（只读运行，不改行为代码）。默认参数多为历史 run 路径，使用前先核对或显式覆盖。

| 脚本 | 用途 | 典型调用与产物 |
| --- | --- | --- |
| `forensics_offline.py` | 离线取证：四类焦点几何（curve/roundabout/uturn/tollgate）的样本量 / 预瞄头误差（stage_b ckpt 前向 vs 数据集 traj6 / action） | `tools/venv-python tools/diagnostics/forensics_offline.py --section counts` 或 `--section plan --rows 20000`；参数 `--npz`（默认 `runs/bc_expert_2k_v2/expert_bc.npz`）、`--ckpt`（默认 `runs/train/il_v2_10x10_b_fixed/stage_b/final.pt`）、`--section {counts,plan,all}`、`--batch`(1024)、`--device`（默认 cuda）、`--out`（默认 `runs/forensics/offline.json`） |
| `forensics_closed_loop.py` | 闭环取证：复现冻结评测协议并逐 env-step / 逐策略步记录，区分 plan 不可跟 / tracker 不适配 / 动作错误 | `tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode lqr --ids 0,5,11 --out runs/forensics/closed_lqr.json`；参数 `--mode`（见下）、`--ids`（逗号分隔，缺省=spec 内焦点几何）、`--spec`（默认 `env/specs/scenarios_val_slice50.json`）、`--ckpt`（默认同上一行历史路径）、`--tracker-json`、`--dtheta-gain`、`--arc-radius/--arc-speed`、`--speed-mode {hold,limit}`/`--speed-cap`、`--max-steps`(1000)、`--device`（默认 cuda）、`--out`（必填） |
| `forensics_report.py` | 闭环结果分析：plan-vs-执行偏差、tracker 误差、失败形态与 baseline/exact 对比 | `tools/venv-python tools/diagnostics/forensics_report.py --in runs/forensics/closed_lqr.json --baseline-in runs/forensics/closed_baseline.json --out runs/forensics/closed_lqr_summary.json`；参数 `--in`（必填）、`--baseline-in`、`--exact-in`、`--out`（必填） |

`forensics_closed_loop.py` 的 `--mode`（`tools/diagnostics/forensics_closed_loop.py:602`）：

- `lqr`：完整评测协议（ckpt + LqrTracker）带逐 step 记录；
- `lqr_gain`：同上，用 `--tracker-json` 覆盖 LqrTracker 参数；
- `exact`：ckpt + ExactTracker（plan 精确置位）；
- `baseline`：PurePursuitIDMPolicy 规则专家参照；
- `arc`：合成圆弧参考（`--arc-radius/--arc-speed`），测 LQR 曲率-速度可行域；
- `laneplan`：每 0.5 s 用车道中心线构造 3 s 参考，交给同一个 LQR；
- `oracle`：参考 = 同 spec 上规则专家实测轨迹的未来 3 s，判定执行栈能否跟上一条好路径。

注意事项：

- 三个脚本都只读分析；输出 JSON 由 `--out` 指定（示例路径 `runs/forensics/`，`runs/` 为 gitignore）。
- 默认 `--device cuda`；无 CUDA 时显式 `--device cpu`。
- 默认 ckpt / spec 为历史路径（`forensics_closed_loop.py:37-38`），npz 为历史路径（`forensics_offline.py:228-229`），文件可能已不存在，跑前先核对。
- 约定：新的一次性诊断脚本放本目录并在上表登记；常驻入口放 `tools/` 根。
