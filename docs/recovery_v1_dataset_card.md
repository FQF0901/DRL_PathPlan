# recovery_v1 数据集卡

> 正式采集完成：2026-10-11（P1-1）。字段契约见 `datasets/recovery_v1/schema.json` 与
> `tools/controls/recovery_quality_gates.py` docstring。

## 采集 provenance

- **学生起点 ckpt**：`runs/BTC20261009-0824_sw_pri512/stage_b/ckpt_epoch020.pt`（P1-0A 定案 = pri512@e020）
- **配置**：`runs/p1_0/configs/p1a_pri512.yaml`；协议 = 两遍确定性重放 + d2 expert 接管（每窗 6 决策步；接管步一致性检查 strict）
- **池**：eval500 + clean500 + dagger r1–r3（5×500）；参数 `--stratify-per-geometry 30 --limit 1500 --no-stop-at-target`，6 分片并行
- 全部 sha256 / args / 分片来源：`datasets/recovery_v1/manifest.json`（`sources` 6 片；`git_commit` 字段）
- 日期 / 操作者：2026-10-11 / orchestrator（collect_recovery lane 实现）

## 规模与数据门（§7.10-A5；`data_gate_report.json`）

| 项 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 有效行 | ≥10,000 | **17,256**（采集 18,643） | PASS |
| 独立 episode（含 ≥1 有效行） | ≥500 | **1,493**（采集 1,500） | PASS |
| 单 episode 最多行 | ≤20 | 18 | PASS |
| 质量门通过率 | ≥80% | **92.56%** | PASS |
| prevention | ≥4,000 | 4,992 | PASS |
| correction | ≥2,500 | 8,886 | PASS |
| late | ≥1,000 | 3,378 | PASS |
| tollgate | ≥300 行且 ≥30 eps | 1,320 行 / 135 eps | PASS |
| merge | 同上 | 1,662 行 / 134 eps | PASS |
| roundabout | 同上 | 1,794 行 / 136 eps | PASS |

## 质量构成（`quality_report.json`）

- 剔除原因：post_takeover_failure 1,153（off-road 1,091 / collision 62）；margin_no_improvement 322；其余门 0。
- 确定性：接管步状态一致性 strict 全通过（0 分歧 / 0 跳过）。
- split：`split_manifest.json`（episode 级 80/20、无泄漏）。

## 已知缺口（`manifest.field_gaps` 摘要）

1. `min_signed_margin` 恒 None（引擎仅布尔 on-lane）→ margin 用 −|lane_lat| 代理；质量门 margin 端点采用窗口级口径，`min_margin_after_takeover` 单列。
2. 线压线类 out_of_road 的 footprint T_cross 恒 None → 回退 env 事件首步（`T_cross_source=env_event`）。
3. 窗口分类为接管时刻实况；prevention 取最晚、correction 取最早、late 取最晚（每档 ≤1）。
4. k≥1 行的 student 字段 = Pass 1 同 step 反事实对照（该时刻 Pass 2 已由专家驱动）。
5. `expert_recovery_traj` = 标签链积分；实际执行轨迹在 `expert_actual_recovery_traj`。
6. 困难路型计数：episode 级用 spec geometry、行级用 road_class（两列都报）。

## 用途与边界

- P1-2 监督目标来源（真实 recovery 行，混采 20–30%；`traj_aux` 仅真实行，禁合成）。
- 本数据集 = 学生失败窗 + 特权专家接管重规划，**不得用于部署声明**；仅用于训练与审计。
- 复算入口：`tools/controls/merge_recovery_shards.py`（分片→合并）、`tools/controls/recovery_quality_gates.py`（质量门）。
