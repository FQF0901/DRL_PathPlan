# docs/ 索引与卫生规则（CLEAN-1，2026-10-15）

> 本文件 = `docs/` 的**唯一索引**。`docs/` 常驻 = **5 个顶层文件（不允许子目录）**；历史文档一律走 git，不设归档桶。
> 历史文档的**要点压缩 + 逐条取档命令**见 `repo_handbook.md` §附录 D。

## 1. 常驻 5 件（只读这 5 个）

| 文件 | 角色 |
|---|---|
| `repo_handbook.md` | **仓库手册**：网络架构 / 训练流程 / 当前问题（第 1–3 章，代码核验）+ 附录 A（漂移）/ B（纪律）/ C（索引）/ D（归档指针） |
| `deepseekv4p1_argue.md` | **决策台账**（主文档）：**按"当期快照"维护——每开新一轮可整篇重写**；旧轮一律走 git |
| `chatgpt5p6sol_review.md` | GPT 往来原文（每轮拉取后追加入库） |
| `architecture_contract.md` | 架构契约（生成器/辅助边界/调用图/训练-部署同构）；代码按路径引用重，保留 |
| `README.md` | 本文件（索引 + 卫生规则） |

## 2. 卫生规则（强制执行）

1. **当期快照制，不新建 md**：**台账（`deepseekv4p1_argue.md`）按当期快照维护——每开新一轮可整篇重写**；手册（`repo_handbook.md`）按章节就地更新；旧信息一律走 git（`git log --diff-filter=D -- <path>` / `git show <commit>:<path>`）；**禁止**在 `docs/` 新建文件或子目录（也不再设 `archive/`、`v6_reports/`、`v7_reports/`、`cleanup/` 等桶）。
2. **历史一律走 git**（取档模板）：
   ```bash
   git log --diff-filter=D -- <oldpath>     # 找删除提交
   git show <commit>^:<oldpath>             # 取回删除前内容
   git ls-tree -r <commit>^ -- docs/        # 列某时点 docs 全貌
   ```
   代码/流程仍引用的历史文档，其**要点与逐条取档命令**见 `repo_handbook.md` 附录 D。
3. **runs 命名域**：official run = `runs/BTC<YYYYMMDD-HHMM>_<name>/`（官方基线/发布链，沿用既有命名）；dev/诊断/实验 = `runs/dev/roundNN/<arm>/`（本轮起新产物一律 dev 域；`runs/` 根不再新增散落目录）。
4. **doc_stats**（工具将落地）：`tools/venv-python tools/doc_stats.py` —— 统计 `docs/` 文件/行数，并核查"代码 → docs 引用只允许指向上表 5 件；`docs/` 无非 5 件 md/子目录"；每次 docs 或引用改动后跑一次。
5. **数据/spec 新增须登记**：数据集 → `datasets/<name>/`（数据集卡随数据集内放置），并同步 `datasets/README.md`（如未建则随首次登记创建）；spec → `env/specs/`（生成/再生成后更新 `env/specs/README.md` 清单）。**禁止**把数据/spec 放进 `docs/`。

## 3. 当前状态

- `docs/` 实际内容（自检）：`find docs -name '*.md'` = 5；`find docs -mindepth 1 -type d` = 0（无子目录）。
- 本收敛批（CLEAN-1，2026-10-15）：代码引用去文档化（63 文件 +117/−117 行：70 处 `docs/*.md` 引用行 + `deprecation_manifest.yaml`、`reward_audit/`、`archive/`、`v7_reports/specs/` 等非 md 引用；含 2 个测试夹具路径迁 `config/arms/reward_drafts/`）+ 删除全部历史 md/子目录（172 项 docs 移除 = 167 删除 + 5 迁移至 `config/arms/reward_drafts/`，git 记 R）+ `.slim` 过期 md 清理（45 删 / 28 留）；全部历史走 git（`b847e51` 及之前的 docs 提交即全文，要点见 `repo_handbook.md` 附录 D）。pytest 全量仍 **1068 passed**。

## 附录：doc_stats 快照

> 生成: 2026-10-16 00:29（CST）｜生成器: `tools/venv-python tools/doc_stats.py --json .slim/deepwork/doc_stats.json --md`｜口径: refs = 路径/basename 字面量在整仓（排除 .venv/.git/runs*/__pycache__/.pytest_cache 与本快照自身）的去重引用文件数；category = keeper > code-ref(refs>0) > stale(refs==0 且陈旧≥30d) > recent。
> 完整逐文件明细（path/refs/last_write/age/category）: `.slim/deepwork/doc_stats.json`（gitignored）；重跑方法见工具 docstring。

| root | 文件 | keeper | code-ref | recent | stale | 体积 |
| --- | --- | --- | --- | --- | --- | --- |
| `docs/` | 5 | 5 | 0 | 0 | 0 | 231.1 kB |
| `.slim/` | 974 | 1 | 463 | 510 | 0 | 45.9 MB |
| `tools/` | 62 | 0 | 62 | 0 | 0 | 1.0 MB |
| `tests/` | 128 | 0 | 128 | 0 | 0 | 6.2 MB |
| `env/specs/` | 9 | 0 | 9 | 0 | 0 | 14.0 MB |
| `datasets/` | 617 | 0 | 39 | 578 | 0 | 1.47 GB |
| `runs/` | 12539 | 0 | 3187 | 9352 | 0 | 6.56 GB |

stale 候选（refs==0 且陈旧 ≥30 天，共 0 项；**只列不删**，全量 `--stale-list`）Top-20:

（无）
