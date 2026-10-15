"""``tools.legacy``：已退役路径的历史复现快照（**不被生产代码导入**）。

纪律（R38 弃用清单 §纪律-1 / `docs/repo_handbook.md` 附录 D §5 守卫 #1）：

- 本包只放"从主链删除的历史实现快照"；每个模块头部必须写清出处（``file:line``）+ 快照
  commit，并在 ``tools/legacy/README.md`` 登记用途与运行方式；
- ``net/``、``pipeline/``、``tools/`` 官方 CLI **禁止**导入本包（CI 守卫 #1 的扫描面）；
- 快照**不作为生产依赖**，也不保证与当前 net 接口兼容（详见各模块 docstring 的"需旧 net
  版本"说明）。
"""

__all__: list[str] = []
