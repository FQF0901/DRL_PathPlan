# tools/measure — P0 实测脚本

这些脚本回答 P0 的四个 go/no-go 问题，全部只读地使用 MetaDrive，不修改任何官方库。

| 脚本 | 测量内容 | 判定口径（暂定，P0 结束冻结） |
|---|---|---|
| `smoke_env.py` | headless 冒烟、reset 耗时、FPS、RSS | 单实例 ≥150 FPS（10 交通车） |
| `measure_mapgen.py` | 建图耗时：随机 block_num vs 显式 `BLOCK_SEQUENCE`，按 block 数分档 | ≤2s/张（复杂图 ≤5s） |
| `check_determinism.py` | 跨进程 / 同进程重复建图的种子确定性 | 同种子指纹一致 |
| `measure_leak.py` | RSS 随 reset 次数的增长斜率 | ≤1GB/千次 reset |

用法（安装完成后）：

```bash
.venv/bin/python tools/measure/smoke_env.py --seed 1000 --steps 200
.venv/bin/python tools/measure/measure_mapgen.py --seeds 20
.venv/bin/python tools/measure/check_determinism.py --workers 3
.venv/bin/python tools/measure/measure_leak.py --resets 1000
```
