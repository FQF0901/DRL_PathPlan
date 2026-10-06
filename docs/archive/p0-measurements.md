# P0 实测报告：环境与四测（2026-09-24）

> 结论：**P0 全部通过，R11/R14 风险退档**；发现 1 个需要在 P1 解决的新问题（BC 专家质量）。
> 环境修复链（venv / opencv / GL）见 §1；四测数字与判定见 §2；冻结参数见 §3；新发现见 §4。

---

## 1. 环境修复记录（三个问题，全部根因定位，均未改动系统库）

| # | 现象 | 根因 | 修复（仅 venv 内） |
|---|---|---|---|
| 1 | `python3 -m venv` 失败 | 系统缺 `ensurepip`（无 python3.10-venv） | `python3 -m venv --without-pip --system-site-packages .venv` + 系统 pip `--python .venv/bin/python` |
| 2 | `ImportError: libGL.so.1`（cv2） | `opencv-python` 需要系统 libGL | venv 内换成 `opencv-python-headless` |
| 3 | 引擎初始化 `IndexError`（MetaDrive 日志行 `"Known Pipes: {}".format(*pipes)`） | panda3d 的 `libpandagl.so`/`libp3headlessgl.so` 都依赖 `libGL.so.1`，本机没有 → 0 个 pipe → 触发 MetaDrive 0.4.3 的 buggy 日志 | `apt-get download libgl1 libegl1 libglvnd0 libglx0` → `dpkg -x` 到 `.venv/gl/`，用 `tools/venv-python`（wrapper 设 `LD_LIBRARY_PATH`）→ `n_pipes: 1 (glxGraphicsPipe)` |

附带修正：venv 内被误装的 numpy 2.2.6（opencv-headless 5.x 拉入）会破坏系统 scipy 1.12/torch 2.3（numpy 1.x ABI）→ 已钉死 **numpy==1.26.4 + opencv-python-headless==4.10.0.84**（MetaDrive 本身也要求 numpy<2）。

工具：`tools/setup_gl_libs.sh`（可复现重建 glvnd 解包）、`tools/venv-python`（**所有 MetaDrive 相关运行必须走它**）。

---

## 2. 四测结果与 go/no-go

### ① 冒烟（`smoke_env.py`，map=3 blocks，traffic_density=0.1）

| 指标 | 实测 | 暂定阈值 | 判定 |
|---|---|---|---|
| reset（建图+重置） | **0.16 s** | ≤2 s | ✅ |
| FPS（单实例） | **394.7** | ≥150 | ✅ |
| RSS/进程 | **633 MB** | 8 进程 ≤6 GB | ✅ |

### ② 建图耗时（`measure_mapgen.py`，20 seeds/档）

| 配置 | mean | median | max |
|---|---|---|---|
| random 3 blocks | 0.21 s | 0.22 s | 0.25 s |
| random 5 blocks | 0.33 s | 0.33 s | 0.54 s |
| random 8 blocks | 0.57 s | 0.52 s | 1.02 s |
| explicit `S` | 0.22 s | 0.22 s | 0.23 s |
| explicit `SC` | 0.26 s | 0.26 s | 0.27 s |
| explicit `SCXRO` | 0.48 s | 0.41 s | 1.08 s |

**结论：R11 退档**。显式序列并不比随机生成更快（回溯在这些规模下不是瓶颈），保留显式序列的理由变成**可复现/类别可控**，不是性能。10k 校验 ≈ 10k×0.4s/8 workers ≈ **8 分钟**。社区 362s 的数字应是 60 blocks 级大地图。

### ③ 种子确定性（`check_determinism.py`，50 步动作序列指纹）

- 同进程重复：`bf351f8ee4cecb3f` == `bf351f8ee4cecb3f` ✅
- 跨进程（spawn ×3）：三个指纹完全一致 ✅
- **结论：冻结验证集与 KPI 可比性成立**；本用法下未复现 issue #758。

### ④ 内存泄漏（`measure_leak.py`）

| 模式 | 增长 | 判定 |
|---|---|---|
| `store_map=False`（重建地图） | **+11 MB / 1000 resets** | ✅ 可忽略（R14 退档；worker 重启周期可放宽到 5000） |
| `store_map=True`（保留地图） | +3129 MB / 1000 resets（≈3.1 MB/张） | ❌ 不能全量常驻；也不需要（见 §3） |

### ⑤ IDM 专家数据 spike（`idm_expert_spike.py`，5 episodes）

- 可用性：导航接口正常（`navigation_command ∈ {forward,left,right}`、`route_completion` 可读），
  0.5s×6 点目标充足（每 episode 67–539 个），采集极快（0.4–2.8 s/episode）。
- **质量问题：5 个 episode 中 3 次出界（out_of_road）、1 次碰撞，仅 1 次干净完成**（rc=0.99）。

---

## 3. P0 冻结参数（写入 config）

| 参数 | 冻结值 | 依据 |
|---|---|---|
| env workers | **8（provisional）** | 8×0.65GB ≈ 5.2GB；Gate 2：未计 trainer（torch CUDA context + replay ≈1.5GB）→ F2 复测含 trainer 后转正 |
| `store_map` | **False** | 重建仅 0.2–0.6s/次；泄漏 +11MB/1000 |
| 地图缓存 | **不做**（可选 ≤32 张 LRU） | 重建成本低于缓存收益；大缓存内存代价 3.1MB/张 |
| `worker_restart_resets` | **5000** | 泄漏实测极低 |
| eval wall-clock | 1000 场景 / 8 worker ≈ **3–5 min**；10k 校验 ≈ **12–25 min**（Gate 2 修正：含重采样/复杂序列/6P+8E 调度） | 一次性成本，不阻塞 |
| 地图规模 | 默认 **3–5 blocks**（显式序列） | 覆盖度 vs 建图耗时平衡 |
| KPI 验收口径 | **相对规则基线**（pure-pursuit+IDM 专家，同一冻结 1000 场景） | 自校准，避免拍脑袋阈值；数据集一致性违例数 = 0 |

---

## 4. 新发现与 P1 行动项

**F1（Gate 2 修订 + 用户指示）**：内置 `IDMPolicy` 无 route-level 导航，出界是预期，**过滤只能提纯不能修复**；但用户指示：**它只用于冷启动**，最终性能靠 RL。因此：
- BC 冷启动数据 = IDMPolicy 轨迹**仅保留"在路内 + 无碰撞 + 有完整 6 点目标"的片段**（目标记录未来实际位置，保证因果性）；数据量不足时再考虑自建专家。
- 仍需一个**规则基线控制器**（pure-pursuit + IDM，~150 行，`BasePolicy` 子类）——这是 KPI 相对基线协议的**必需件**（同时可作 BC 备选数据源）；基线仅用 train 集调参后冻结。
- `ExpertPolicy` 预训练权重：可选 30 min 查证，不进关键路径。

**F2（待 P1 复测）**：本轮 FPS/RSS 是"3 blocks + 10% 交通"口径；P1 需在**真实场景配比**（5 blocks、拥挤、cut-in、多车）与 **8 并发 worker** 下复测吞吐与内存，确认 P2 训练预算。

**F3（P1 设计）**：`runs/` 已用于落盘 spike 数据；`tools/venv-python` 成为所有 MetaDrive 运行的唯一入口（文档/脚本统一）。

---

## 5. Gate 2 评审结论（Oracle，2026-09-24）与修复

**总判定**：measurement validity = 部分 sound（smoke 合格、外推不足）；no-cache = sound；expert = needs-change（路线正确）；KPI 协议 = needs-change；**无硬阻塞**。

**Must-fix（已落实）**
1. **KPI 协议收紧**（`config/eval.yaml`）：零基线保护 ε（`≤ max(baseline, 0.01)`）、出界显式界、效率守卫 `speed_ratio ≥ baseline×0.9`（防爬行刷分）、`a_lat_p95` 尾部指标、per-category `n≥30` 才判定（否则仅报告 + Wilson CI）、dataset gate 与 policy KPI 分离。
2. **基线协议冻结**：`pure_pursuit_idm` 仅用 train 集调参后冻结；baseline 与 policy 同 harness/同种子/同 scripted traffic 跑 val（否则不可比）。
3. **确定性矩阵**（P1a 执行，P1b 冻结 val 前设 gate）：多类别 × 完整 episode × 同 env 重复 reset × 异构多进程 × scripted cut-in；P0 的单 seed/50 步只证明"无跨进程污染"。
4. **workers=8 标 provisional**：F2 复测必须含 trainer 进程 RSS、真实 obs/mix、5 min soak。

**P1a 必须补测（Gate 2 指定，含"单项最重要复测"）**
- **自定义 `BaseObservation` 在真实配比（5-block、dense、cut-in）下 8 并发 worker 的吞吐 + RSS（含 trainer）** ← 最重要，同时决定 P2 预算与 workers 是否转正。
- dense/5-block 的 reset 成本；post-build 限速/线型后处理在 `store_map=False` 每次重建后的**重放**验证（易漏）。
- scripted cut-in 的确定性；step-time breakdown（obs/physics/traffic/reward）。
- P2 小课程集（20–50 seeds 反复采样）时启用小 LRU（命中率才有意义；训练全量采样时命中率 <1%，维持 no-cache）。
- 计算修正：10k 校验 ≈12–25 min（非 8 min）；RSS 外推需含 trainer。

**注**：Oracle 建议自建 pure-pursuit+IDM 专家作为 BC 数据与 KPI 基线**双用**；用户指示 IDMPolicy 仅冷启动即可 → 采纳折中：**基线控制器必做**（KPI 协议需要），BC 数据优先用过滤后的 IDMPolicy。
