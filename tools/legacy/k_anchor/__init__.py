"""``tools.legacy.k_anchor``：K-anchor 计划头的历史复现快照（R38-P2a 减枝迁移）。

范围（R38 弃用清单 B 组）：``net/anchor.py``（整文件）、
``net/plan_head.py`` 的锚分支（整文件）、``tools/fit_plan_anchors.py``、锚字典
``plan_anchors_k6.json``。

纪律：与 ``tools.legacy`` 相同——**不被生产代码导入**；出处/快照 commit/复现约束见
``README.md`` 与各模块 docstring。旧 ckpt 的 ``plan_head.anchor*`` 键由
``tools/legacy/legacy_key_mapper.py`` 登记忽略（不参与正式 forward）。
"""

__all__: list[str] = []
