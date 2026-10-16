"""（历史）参数预算乘性常量快照 —— R41-D 正式语义移除后的兼容位。

出处（逐行拷贝的历史定义）
--------------------------
- ``TOTAL_LIMIT = 1.02``：R39 总参数乘性上限系数（``net/param_probe.py`` R39-A2 定义）。
  R40 §五.5 起总参数门改为**绝对上限** ``TOTAL_CAP = 2,000,000``，乘性规则不再参与断言；
  **R41-D 起正式语义整体移除**（``net/param_probe.py`` 不再导出该常量）——本文件仅存数值
  供历史脚本（如 ``.slim/deepwork/p1_v2/round39/a2_run.py``）import 兼容。
- ``ACTIVE_LIMIT = 1.05``：R39 激活上限系数。R41-D 起 production 语义 = **警告阈值**
  （``net.param_probe.ACTIVE_LIMIT_WARN``，报告 ``warnings.active_budget_over``，**非硬门**）。

纪律
----
- 本文件**不被生产代码导入**（``net/``/``pipeline/``/``tools/*.py`` 零引用；见
  ``tests/test_r38_ci_guards.py`` 守卫 ① 与 ``tools/legacy/README.md``）；
- 不参与任何断言/门；**禁止新代码 import**（新代码用 ``net.param_probe`` 的现行常量）。
"""

from __future__ import annotations

__all__ = ["ACTIVE_LIMIT", "TOTAL_LIMIT"]

#: R39 总参数乘性上限系数（R40 废止 → R41-D 正式语义移除；仅历史脚本 import 兼容）。
TOTAL_LIMIT = 1.02
#: R39 激活上限系数（R41-D 起 = 警告阈值；production 常量见 ``net.param_probe.ACTIVE_LIMIT_WARN``）。
ACTIVE_LIMIT = 1.05
