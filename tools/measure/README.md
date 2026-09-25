# tools/measure

目的：P0/P1a/P2 实测脚本（只读使用 MetaDrive，不改官方库）。统一 `tools/venv-python` 运行。

| 脚本 | 验证什么 / 典型命令 | 关键结论（实测） |
| --- | --- | --- |
| `smoke_env.py` | headless 冒烟：reset 耗时 / FPS / RSS；`--seed 1000 --steps 200` | reset 0.16 s、**394.7 FPS**、RSS 633 MB（判定阈值 ≥150 FPS） |
| `measure_mapgen.py` | 建图耗时：随机 block_num vs 显式序列；`--seeds 20` | random 3/5/8 块 = 0.21/0.33/0.57 s；显式序列不更快 → R11 退档 |
| `check_determinism.py` | 同 seed 重复/跨进程动作指纹；`--workers 3 --steps 50` | 同进程与 spawn×3 指纹完全一致 → 冻结 val / KPI 可比 |
| `measure_leak.py` | RSS 随 reset 次数斜率；`--resets 1000 --mode False` | `store_map=False` **+11 MB/1000 resets**；`True` +3129 MB |
| `idm_expert_spike.py` | IDMPolicy 能否作 BC 源；`--episodes 5` | 导航/6 点目标可用（67–539 点/ep），但 5 ep 中 3 出界 1 碰撞 |
| `obs_smoke.py` | 观测 shape/mask + 单步耗时分解；`--steps 60` | obs 0.68 ms/step，env.step 1.57 ms → **637.9 FPS** |
| `p1a_integration.py` | spec→build_env→obs→labels→短 rollout；`--limit 5 --steps 50` | 限速写入、obs 形状/标签正常（报告 `runs/p1a_integration.json`） |
| `p2_inventory.py` | 导入各 P2 模块并打印公开 API（无 env 依赖） | 用于对照 p2-contract 核对已落地接口 |
| `api_recon.py` | 只读探测观测/后处理所需 MetaDrive API；`--map SCX` | 防御式打印，供自定义通道与后处理定案 |

- 计时/内存数字出处：`docs/p0-measurements.md` 与 `.slim/deepwork/planner-rl-feasibility.md`。
- 运行纪律：同一时间只跑一个 env-heavy 多进程任务；启动前查 MemAvailable（≥ 任务预算）。
