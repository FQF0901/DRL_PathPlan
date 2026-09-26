# tools/measure

目的：P0/P1a/P2 实测脚本（只读使用 MetaDrive，不改官方库）。统一 `tools/venv-python` 运行。

| 脚本 | 验证什么 / 典型命令 | 关键结论（实测） |
| --- | --- | --- |
| `smoke_env.py` | headless 冒烟：reset 耗时 / FPS / RSS；`--seed 1000 --steps 200` | reset 0.16 s、**394.7 FPS**、RSS 633 MB（判定阈值 ≥150 FPS） |
| `check_determinism.py` | 同 seed 重复/跨进程动作指纹；`--workers 3 --steps 50` | 同进程与 spawn×3 指纹完全一致 → 冻结 val / KPI 可比 |
| `obs_smoke.py` | 观测 shape/mask + 单步耗时分解；`--steps 60` | obs 0.68 ms/step，env.step 1.57 ms → **637.9 FPS** |

- 计时/内存数字出处：`docs/p0-measurements.md` 与 `.slim/deepwork/planner-rl-feasibility.md`。
- 运行纪律：同一时间只跑一个 env-heavy 多进程任务；启动前查 MemAvailable（≥ 任务预算）。
