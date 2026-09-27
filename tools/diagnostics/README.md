# tools/diagnostics

一次性诊断 / 复现脚本（只读运行，不改行为代码）。保留目的是复现历史结论；
结论与用法出处：`docs/forensics-2026-09-26.md`。

| 脚本 | 用途 | 对应文档 |
| --- | --- | --- |
| `forensics_offline.py` | 离线取证：四类几何的样本量 / 预瞄头误差（`--section counts\|plan`） | `docs/forensics-2026-09-26.md` |
| `forensics_closed_loop.py` | 闭环取证：复现冻结评测协议并逐 env-step 记录，区分 plan 不可跟 / tracker 不适配 / 动作错误（`--mode lqr\|lqr_gain\|exact\|baseline\|arc`） | `docs/forensics-2026-09-26.md` |
| `forensics_report.py` | 闭环结果分析：plan-vs-执行偏差 / tracker 误差 / 失败形态（`--in` / `--out`） | `docs/forensics-2026-09-26.md` |

约定：新的一次性诊断脚本放本目录并在上表登记；常驻入口放 `tools/` 根。
