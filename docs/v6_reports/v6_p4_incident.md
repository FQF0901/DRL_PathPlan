# P4 v6 批执行 · 事件记录：arm0 首臂后复核误停（driver tag 前缀 bug）

- 时间：2026-10-02 00:04:55 – 00:20:31（batch elapsed 927.8 s）
- HEAD：`b657fea`（工作区 clean）；driver sha256 `7b9e02013aa4b9e17f63b465770dac3c0464325265d8c98ded1e34e03bbe27af`
  （**未被修改**，mtime 2026-10-01 23:59:49）；臂配置未动。
- batch PID 3142209（setsid 独立 session）；GPU 串行；当前无残留进程、GPU 空闲（452 MiB）。
- 走向：arm0 训练 rc=0（927 s，200/200 updates，收尾断言全过）→ 首臂后复核误判 `needs_probe300`
  → **停批**。batch `completed=[arm0]`，arm1–7 均未启动（无 GPU 消耗）。

## 根因（driver bug，非训练异常）
`first_arm_review()`（`/tmp/opencode/v6_p4_driver.py:650-653`）按**无前缀** tag 读取：
`episodes/termination_counts/{reason}`、`episodes/mean_return_by_reason/{reason}`、
`episodes/mean_steps_by_reason/{reason}`。

但 `pipeline.monitoring` 递归展平嵌套指标时统一加 `train/` 前缀（`pipeline/trainer.py:59`），
实际 CSV tag 为 `train/episodes/termination_counts/…` 等（arm0 `monitor/metrics.csv` 实证）。
同一函数内的 `train/reward/terminal` 与 `readback_assertions` 的 `train/reward/<term>` 均带前缀且正常。
⇒ 复核读到 max_step n=0、arrive n=0 → 误判"训练窗口证据不足"→ 触发 §7.1 的 300 s 探针分支并停批。

影响面：`first_arm_review` 仅在 arm0 执行（arm1–7 不走该分支），故误停只发生在 arm0 之后。

## 正确前缀复算的 arm0 复核读数（同一 metrics.csv；与 driver `weighted()` 同法）
| 读数 | 值 | 阈值 | 判定 |
|---|---|---|---|
| max_step n | **140**（123/200 updates 有 ≥1；末 update 1 条） | ≥ 2 | ✓ |
| R_max（count 加权） | **−48.5104** | ≤ 0 且 ≤ 0.5×R_arrive(=26.23) | ✓ |
| S_max | **200.0** | ≈ 200 | ✓ |
| R_arrive / n | +52.4559 / 3 | — | — |
| collision n / R | 7 / −31.177 | — | — |
| out_of_road n / R | 319 / −23.5917 | — | — |
| terminal mean_last20 | −0.214316 | 显著为负 | ✓ |

⇒ 正确复核结论：**trip=false、needs_probe300=false → arm0 复核通过、无需 300 s 探针**。
（300 s 探针未执行；"n<2" 为读数伪影。）

## 现场保留（未动）
- `runs/BTC20261002-0004_p4_arm0/`：ckpt_u025…u200（8 个候选）+ monitor/metrics（791 KB）等。
  - ckpt_u050 sha256 `9727b537c478aebf7bb29dbc1f5c4697f485093b5c42f4bda0b376e493ff66c7`
  - ckpt_u200 sha256 `4ed0d90e7144ad46373da03bc56841cb3ac3de05d474f598175f1dc690253902`
- `/tmp/opencode/v6_p4_arm0.{log,json}`、`v6_p4_batch.{log,json}`、`v6_p4_status.txt`
- arm0 的 u50 闸 / keep-best / clean500 / eval500 **未执行**（driver 在此前停批）。

## 建议（待指示；driver 与臂配置未改）
最小修复 = `first_arm_review` 三行 tag 加 `train/` 前缀（或保留无前缀 fallback）：
```diff
-        counts_tag = f"episodes/termination_counts/{reason}"
-        ret_tag = f"episodes/mean_return_by_reason/{reason}"
-        steps_tag = f"episodes/mean_steps_by_reason/{reason}"
+        counts_tag = f"train/episodes/termination_counts/{reason}"
+        ret_tag = f"train/episodes/mean_return_by_reason/{reason}"
+        steps_tag = f"train/episodes/mean_steps_by_reason/{reason}"
```
修复后无 GPU 复现验证（回放已有 arm0 产物）：期望 n=140、trip=false、needs_probe300=false。

恢复选项（成本估计）：
- **A（推荐）**：打最小补丁 + 重跑整批 `--batch --budget-h 4.0`。arm0 重训 ~15.5 min（seed=0 同 pins，
  确定性），后续 45–55 min/臂 → 总 ~4–5 h；协议最净。
- B：打补丁 + `--batch --only arm1,…`：arm0 终评（keep-best/clean500/eval500）缺失；
  arm0 是 bundle 底座臂，不推荐。
- C：不改 driver、手工复刻 u50/keep-best 评测：协议漂移风险高，不推荐。

预算：已耗 0.26 h / 4 h。GPU 空闲、无残留进程。
