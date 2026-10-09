# tools/measure

headless 环境 / 观测实测脚本（只读使用 MetaDrive，不改官方库）。统一 `tools/venv-python` 运行。

| 脚本 | 验证什么 | 典型调用与输出 |
| --- | --- | --- |
| `smoke_env.py` | `MetaDriveEnv` 直接冒烟：reset 耗时 / FPS / RSS | `tools/venv-python tools/measure/smoke_env.py --seed 1000 --steps 200`；参数 `--traffic-density`(0.1)、`--map`（数字串=block 数，否则显式序列如 `SCX`）；stdout 打印 `reset_s=… fps=… rss_MB start/after_reset/end` |
| `check_determinism.py` | 同种子确定性：同进程重复建图 + spawn 跨进程动作指纹是否一致 | `tools/venv-python tools/measure/check_determinism.py --workers 3 --steps 50`；参数 `--seed`(1000)、`--map`(3)；打印同进程 `MATCH/MISMATCH` 与跨进程指纹 |
| `obs_smoke.py` | `build_env` + `ObservationBuilder` 全链路：每通道 shape/mask、6 帧历史 valid、限速写入情况、单步耗时分解（obs/physics/traffic/reward）与 fps | `tools/venv-python tools/measure/obs_smoke.py --seed 1000 --density 0.2 --steps 60`；参数 `--blocks`（显式 block 序列；缺省用 generator 的 spec，失败回退兜底 spec）、`--no-events`（清空 cut-in/cut-out）、`--lru`（地图 LRU 容量，0=关、≤32）、`--steer/--throttle` |

注意事项：

- 同一时间只跑一个 env-heavy 任务：MetaDrive 每进程只能有一个 engine（`pipeline/stages.py:31-32`）。
- `smoke_env.py` / `check_determinism.py` 直接用官方 `MetaDriveEnv`（原生口径）；`obs_smoke.py` 走项目 `env.metadrive_env.build_env` + 观测链。三者口径不同，不要互相混比。
- `obs_smoke.py` 的耗时分解是实例级 hook（仅本脚本，不改库代码），且必须在 `env.reset()`（lazy_init）之后才可用。
