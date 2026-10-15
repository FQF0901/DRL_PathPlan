# R38-C3：`path_*` 七字段最终分类与 schema 冻结建议（草稿）

- 状态：**已定稿（编排裁定 2026-10-15）**；正文数字全部取自 `runs/round38/c3/*`（不手抄），细节见 `.slim/deepwork/p1_v2/round38/c3_report.md`
- 日期：2026-10-15 ｜ 分支：`Planner_RL` ｜ 范围：GPT R38 §二.4「`path_*` 七字段旁路审计与最终分类」
- 口径：离线 teacher-forcing 受控实验（n = 7133，stratified cap 300，seed 0；ckpt = `runs/round36/j2/t1_keep_res/ckpt_epoch001.pt`，与 A1/A3 同源同集）+ 评测侧只读探针（3 ep × 300 step）
- 约束：**零训练、不改 `net/`/`env/`/`pipeline/`、不 commit**；闭环翻转证据引用 M3/M6/M7/A3/A4，不重跑闭环臂
- 产物：`.slim/deepwork/p1_v2/round38/c3_report.md` / `c3_matrix.py` / `c3_unit_check.py` / `c3_eval_probe.py` / `c3_report.py`；`runs/round38/c3/c3.json` / `c3_tables.md` / `c3_unit_check.json` / `c3_eval_probe.json`

---

## 0. TL;DR

1. **「7 键全缺」的旧读法应修正为「3 键 0 + 4 键回退」**（评测侧实测）：C1 协议闭环里 7 个 `path_*` 键 present 率 = 0/180，但 `lane_semantic` / `nav` 在场率 = 1.0 / 1.0 ⇒ 头实际收到 `left_margin`/`right_margin` = 5.25/5.25 m（`lane_semantic[4]/[5]`）、`nav_progress` p50 0.12（`nav[:,-1]`）、`d2end` p50 22.0 m（`lane_semantic[11]`）；**只有 `static_distance` / `dynamic_distance` / `dynamic_ttc` 静默变 0**（无回退）。这与 M3/M6/M7 聚焦的"三键"一致。
2. **`present` 掩码未被头消费**（`net/path_head.py:1255-1260` 只吃 `long_exog` 值）：在 3/4/5/7 列上「缺键 0」与「静默 0」**逐位相同**（全精度差 0.0）⇒ 现状无法区分"缺失"与"值为 0"；任何"mask"方案必须先把掩码接进头（⇒ 与重训绑定）。
3. **边际影响排序（本器五档矩阵）**：`dynamic_ttc` ≫ `nav_progress`(仅"被写 0"时) ≈ `dynamic_distance` ≈ `d2end` > `margins` ≫ `static_distance`。
   - `dynamic_ttc` 缺失/零值档：Δds0 = +0.123 m、|Δds| 至 0.482 m/步、ΔΣds = −1.816 m/plan、Δ|Σds−真| = +0.757（slow+accel 子集 +1.725）；
   - 其余 6 列 |Δds| ≤ 0.15/步、ΔΣds ≤ 0.54 m/plan（`static_distance` ≤0.001/步）。
4. **分类结论**：2 列（TTC / 动态距离）归"并入标准 observation"；4 列（左右 margin / nav_progress / d2end）归"由 WM 状态 anchors 确定性派生"（其中 margin 的两处确定性缺陷已由 A4 修复）；1 列（`static_distance`）归"仅保留为诊断（不进 Policy）"。
5. **落地建议（禁静默填 0）**：t0 段 8 列**必须由生产者给出**（缺失 ⇒ 报错；eval 侧用同公式真值/回退），k≥1 段由 WM 状态确定性派生（A4 margin 修复 + 进度收缩 + nav 重建），头输入段 `long_exog (8) + long_present (8)` 且掩码必须进头（与重训绑定）。

---

## 1. 消费路径审计（谁注入 / 缺键解析 / 掩码 / eval 实测 / k≥1 现状）

审计对象 = `net/path_head.py:229-238 LONG_EXOG_OBS_KEYS[1:]`；解析 = `long_exog_from_obs`（`:912-995`）；消费 = `StructuredPolicyHead.forward`（`:1255-1260`，`long_features = κ_near ⊕ long_exog`）。

| 字段 | 训练注入 | 缺键解析 | head 消费 present | eval 实测（3 ep） | k≥1（想象步）现状 |
|---|---|---|---|---|---|
| `path_left_margin` | I1 `_long_exog_row`（lane_semantic 同公式） | `lane_semantic[4]` | ✗ | present 1.0，p50 5.253 | 生产重算 `W/2 − lat_k`（符号反，A4 已修） |
| `path_right_margin` | 同上 | `lane_semantic[5]` | ✗ | present 1.0，p50 5.247 | 生产重算 `W/2 + lat_k`（符号反） |
| `path_static_distance` | `scan_static_blocker`（走廊 50 m） | **无 ⇒ 0 / present=0** | ✗ | present 0.0，zero 100% | 进度收缩 `clamp(d_t0 − progress, 0)` |
| `path_dynamic_distance` | OD presence=1 槽矩形最近距 | **无 ⇒ 0 / present=0** | ✗ | present 0.0，zero 100% | hold（A3 kin_b 可选，收益≈0） |
| `path_dynamic_ttc` | OD 槽 `dx/(−vx)` cap 60 | **无 ⇒ 0 / present=0** | ✗ | present 0.0，zero 100% | hold（A3 kin_b 不改善） |
| `path_nav_progress` | 源 npz `route_completion` | `nav[:,-1]` | ✗ | present 1.0，p50 0.120 | hold t0（建议改为 route 重建） |
| `path_distance_to_target_lane_end` | `max(lane.length − s_eff, 0)` | `lane_semantic[11]` | ✗ | present 1.0，p50 22.009 | 进度收缩（= 派生，逐位同生产） |

- 证据：`runs/round38/c3/c3_unit_check.json`（合成 obs 逐列钉死键在场/键缺/键缺+semantic/键缺+nav）+ `c3_eval_probe.json`（`--workers 1` 进程内 monkeypatch `net.model.long_exog_from_obs`，3 ep × 300 step，n = 180 次 head 调用；rc=0，wall 30.4 s）；eval 侧通道在场由 `pipeline/eval_runner.py:945`（`obs_config = task.obs_config or {}` ⇒ `env.obs.builder` 默认全通道）保证。
- 自证：`nav[:,-1]` 与数据集列 6 **逐位相等**（max|Δ| = 0.0）；链复刻与生产 `forward` 逐位相等（plan/ds/κ max|Δ| = 0.0）；nav 派生在 t0 恒等位姿下 MAE 0.0079（p50 0.0027）。

---

## 2. 五档矩阵摘要（7 列 × {缺失 / 零值 / 中性值 / 真值 / WM 派生}）

- 基线 = E6（生产链逐位复刻，与臂 W 逐位同）；Δ 均为配对差（同 n=7133 样本）。
- 「真值」= 行 t+k 自己的真值（oracle 上界）；「WM 派生」= 由 t0 观测 + 想象状态确定性重算（margin：`W/2±lat` fix 口径；static/d2end：进度收缩（≡ 生产）；dyn dist/ttc：kin_b；nav rc：`nav_features_from_world`）。

| 列 | 缺失/零值 Δds0 | 缺失/零值 ΔΣds | 缺失/零值 Δ\|Σds−真\| | 真值档 \|Δds\|/步 | WM 派生档 \|Δds\|/步 | 判读 |
|---|---|---|---|---|---|---|
| `dynamic_ttc` | +0.123 | −1.816 | +0.757 | ≤0.275 | ≤0.131（Δ\|Σds−真\| −0.008） | **主项**；0 = 分布外「立即碰撞」 |
| `nav_progress` | +0.034（零值） | −0.537（零值） | +0.114（零值） | ≤0.041 | ≤0.018（Δ\|Σds−真\| −0.001） | 真·缺失 = nav 回退（≡ 基线）；**禁写 0** |
| `dynamic_distance` | +0.024 | −0.335 | +0.053 | ≤0.044 | ≤0.027（≈hold，A3 同） | 现状恒 0 = 静默 0 |
| `distance_to_target_lane_end` | +0.026 | −0.331 | +0.060 | ≤0.049 | ≡ 基线（生产已派生） | 现状已回退；禁写 0 |
| `right_margin` | +0.008 | +0.110 | −0.016 | ≤0.023 | ≤0.099（Δ末端\|lat\| −0.334） | 纵向影响小 / 横向几何相关 |
| `left_margin` | +0.002 | +0.103 | −0.017 | ≤0.008 | 同上（两列共享 W） | 同上 |
| `static_distance` | +0.000 | −0.004 | +0.002 | ≈0 | ≡ 基线 | 近零；present 2.6% / eval 0 |

关键自证：零值 ≡ 缺失（列 3/4/5/7 逐位 0.0，列 1/2/6 差异来自生产者语义：margin 两列经 `W = left+right` 耦合、缺失令派生停摆；nav 缺失触发回退）；margin 生产重算 ≡ 基线、列 3/7 的 WM 派生 ≡ 基线 ⇒ 生产已实现这两类派生；t0 未改档在 k=0 与基线逐位相同。

---

## 3. 最终分类表（7 行 × 四类）

| 字段 | 分类 | 缺失语义（禁静默 0） | 主要依据 | 影响面（schema / 注入口 / 训练一致性） |
|---|---|---|---|---|
| `path_left_margin` | **由 WM 状态 anchors 确定性派生** | 报错（生产者契约）+ `lane_semantic` 回退 | t0 由回退供给真值；k≥1 由 `W/2±lat_k` 确定性重算；S6 全样本证明生产反号、A3 口径 fix 贴回；A4 已落地修复 | 列数与 `step_geometry` 不变；注入口 = `step_geometry` 几何口径；重训后生效 |
| `path_right_margin` | 同上 | 同上 | 同 left（两列共享 W，须成对） | 同上 |
| `path_static_distance` | **仅保留为诊断（不进 Policy）** | mask + 0（显式主态） | present 2.6%（0 占 97.5%）/ eval 0；五档 ≤0.001/步、ΔΣds ≤0.004 | 生产者保留（I1 列 3），Policy 输入中性/掩蔽；重训或冻结常量 |
| `path_dynamic_distance` | **并入标准 observation** | 中性+mask（主态 0；优先 env 同公式真值） | eval 恒 0（无回退）；缺失/零值 ΔΣds −0.335；kin_b 派生与 hold 等价 | 保留列 + 掩码进头；注入口 = M7 公式（`m7_fill.py`）；必与重训绑定 |
| `path_dynamic_ttc` | **并入标准 observation** | 中性+mask（真值回填优先；0 = 禁止） | 全表最大边际（Δds0 +0.123 / ΔΣds −1.816 / Δ\|Σds−真\| +0.757）；M6 闭环中性填充 success +8pp；0 = 分布外 | 保留列 + 禁写 0；注入口 = M6 中性档 / M7 真值档；必重训 + 闭环复测 |
| `path_nav_progress` | **由 WM 状态 anchors 确定性派生** | 保持 nav 回退（确定性）+ 显式 present | 回退 = 真值（逐位）；写 0 则 ΔΣds −0.537；route 重建派生 t0 恒等 MAE 0.0079 | 列数不变；注入口 = `nav_features_from_world`（生产 rollout 已用同一纯函数）；可与重训解耦 |
| `path_distance_to_target_lane_end` | **由 WM 状态 anchors 确定性派生** | 报错 + `lane_semantic[11]` 回退 | 回退 = 真值；k≥1 生产已做进度收缩（≡ 基线）；真值档差 ≤0.049/步 | 列数不变；注入口 = `step_geometry`（列 3/7 收缩）；回退消失需 fail-fast |

---

## 4. 缺失语义与 schema 冻结建议（"禁止静默填 0"落地）

### 4.1 每字段缺失语义

| 字段 | 缺失语义 | 理由 |
|---|---|---|
| `path_dynamic_ttc` | **中性+mask**（首选 env 同公式真值回填；0 = 禁止） | 训练 present 恒 1；0 = 分布外「立即碰撞」；全表最大动作偏差；M6 中性填充闭环 +8pp |
| `path_dynamic_distance` | **中性+mask**（主态 = 0，显式） | 训练 11.5% 缺失即主态 0；缺失/零值 ΔΣds −0.335 |
| `path_static_distance` | **mask + 0**（显式主态；不进 Policy） | 训练 97.5% 为 0、eval 恒 0；五档 ≤0.001/步 |
| `path_left_margin` / `path_right_margin` | **报错**（t0 必给；eval 由 `lane_semantic` 回退） | 训练 present 100%；两列经 W 耦合，单列缺失令派生停摆 |
| `path_nav_progress` | **保持 nav 回退**（确定性）+ present 位 | 回退 = 真值；写 0 则 ΔΣds −0.537 |
| `path_distance_to_target_lane_end` | **报错 + `lane_semantic[11]` 回退** | 回退 = 真值；写 0 则 ΔΣds −0.331 |

### 4.2 三段式 schema 冻结建议

1. **t0 段（obs → head）**：8 列**必须由生产者给出**（eval 侧 static/dyn/ttc 用 `m7_fill.py` 同公式扫描回填；margin/d2end 用 `lane_semantic`；rc 用 `nav`），并附 `long_present (8)`；任何键缺失 ⇒ **报错**（取消"缺键回退 0"隐式路径）。
2. **k≥1 段（想象步）**：由 WM 状态确定性派生 —— margin = `W/2 ± lat_k`（数据集符号 + foot mask，A4 已落）、static/d2end = 进度收缩、rc = `nav_features_from_world`；dyn dist/ttc 保持 hold（kin_b 收益 ≈0）。
3. **头输入段**：`long_exog (8) + long_present (8)`，掩码**必须进头**（现状 forward 不接收掩码 ⇒ 0 值即有效观测，逐位证据见 §2）；此步改变输入语义 ⇒ **必须与重训绑定**。
4. **诊断列**：`static_distance` 下沉为诊断（不进 Policy）；若采用"保留列但恒中性"的过渡方案，须在训练侧同步该列常量输入。
5. **迁移顺序建议**（供编排选）：(a) 先落"生产者补全 + 禁 0 + fail-fast"（不动头，零训练即可验证 obs/接口一致性，附逐位 off 证据）；(b) 再做"掩码进头 + 静态列下沉"（与下一次重训同批，验收含闭环 Calib50 + M6/M7 对照）。

---

## 5. 闭环与分布交叉证据（引用；本 lane 不重跑闭环）

| 项 | 来源 | 结论 |
|---|---|---|
| 三键（ttc/dyn/static）中性填充 | **M6**（Calib50，C1 同协议） | success 0.34→0.42（+4/−0，p=0.125）、collision 0.14→0.10、timeout −2 ep、rc +0.058、速度 +0.31 m/s；同 t0 Δds0 = +1.13 m/步；A1 档上填充硬安全退化 ⇒ 不晋级 |
| 三键真值回填 | **M7** | success 0.40、off-road 0.12→0.18 ⇒ Hard 先看**不晋级**；公式与训练侧逐位一致；分布差来自状态分布（eval TTC cap-60 0.671 vs 训练 0.365） |
| margin 两缺陷（foot padding / 反号） | **A3**（证据）+ **A4**（落地） | 链上 65.5% 想象步 foot 落 padding（本器复算 0.6552）；恒等位姿下生产 left 更贴 t0 **right**（全 n 复现），fix 贴回 left；修复后 Δ末端\|lat\| −0.334 m、Δ\|Σds−真\| +0.074（横向改善/纵向微退化） |
| nav_progress / d2end | 本 lane（无独立闭环臂） | 两列在 eval 由回退供给、从未"缺失"；离线量级 = 若写 0：ΔΣds −0.537 / −0.331 |

训练/评测分布对照（present 率、p50）见 `.slim/deepwork/p1_v2/round38/c3_report.md` §D：训练行池（n = 265,801）`ttc 1.000/p50 18.05`、`dyn 0.885/p50 18.66`、`static 0.026/p50 21.21`、`margin 1.000/p50 5.27/5.24`、`nav 1.000/p50 0.42`、`d2end 1.000/p50 40.0`；eval 实测（n = 180）`ttc/dyn/static = 0`、`margin 1.0/p50 5.25`、`nav 1.0/p50 0.12`、`d2end 1.0/p50 22.0`。

---

## 6. 编排裁定（2026-10-15；原"待确认项"逐条结案）

1. **`static_distance` → 仅保留为诊断**（删除收益为零且会失去 tollgate 诊断能力）。
2. **`nav_progress` → WM 派生**（与 route 重建同步；不并入标准 obs，避免额外 schema 面；两写法动作差 ≤0.018/步）。
3. **迁移节奏 → 先接口后掩码**：(a) 生产者补全 + 禁 0 + fail-fast（零训练可验，附逐位 off 证据）；(b) 掩码进头 + static 列下沉（**与下一次重训同批**，验收 = 离线矩阵 + Calib50 + M6/M7 对照）。
4. **`c3_eval_probe.py` → 纳入常规工具**（"头入口雷达"：任何 eval 装配变更后必跑）。
5. **docs 索引登记** → 本次提交一并更新（`docs/README.md`）。

## 附：复现

```bash
tools/venv-python .slim/deepwork/p1_v2/round38/c3_unit_check.py          # 缺键解析单元核验（CPU）
tools/venv-python .slim/deepwork/p1_v2/round38/c3_matrix.py --device cuda --cap-per-stratum 300 \
    --out runs/round38/c3                                                # 五档矩阵（≈270 s）
tools/venv-python .slim/deepwork/p1_v2/round38/c3_eval_probe.py          # 评测侧只读探针（≈30 s）
tools/venv-python .slim/deepwork/p1_v2/round38/c3_report.py --run runs/round38/c3 \
    --out .slim/deepwork/p1_v2/round38/c3_report.md                      # 报告再生成
```
