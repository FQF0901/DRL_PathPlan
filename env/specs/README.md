# env/specs

目的：存放生成的场景 spec。

- 内容为 JSON，如 `scenarios_train.json`、`scenarios_val.json`；schema 见 `env/scenario/README.md`。
- 由 `tools/gene_env.sh` 生成；spec 内含随机种子与参数哈希，保证可复现。
- 目录内 `*.json` 已加入 `.gitignore`，只提交本说明与 `__init__.py`。
- 校验失败或手工改过的 spec 不得用于训练（validator 会拦截）。
