# R38-C5 Checkpoint 迁移契约（GPT R38 §六-4 / §八 第四件交付物）

> **目的**：把"旧 ckpt 只经 migration layer 加载、不参与正式 forward"从口号落成键面清单 + 映射草案 + 接口设计，
> 供 R38-C 减枝时**先做迁移器、再动实现**（§七-D 的顺序要求）。
> **性质**：设计/审计文档——**本文件不改代码**；`checkpoint_migration.py` / `legacy_key_mapper.py` 只是约定的
> 落点与 API 草案。
> **机器证据**：`.slim/deepwork/p1_v2/round38/{c5_ckpt_keys.json, c5_ckpt_families.json}`（由
> `c5_reach_scan.py --ckpt-keys` / `--ckpt-families` 生成，只读）。
> **配套交付物**：`docs/code_reachability_audit.md`（引用面与判定）、`docs/deprecation_manifest.yaml`（逐项弃用清单，
> 其中 `checkpoint_keys` 列与本文件 §3/§4 同源）、`docs/runtime_entrypoint_matrix.md`（入口 × 装配口径）。

---

## 1. 键面普查方法（可复现）

```bash
# ① 全仓键族普查（522 个 .pt 全量，聚合"键前缀族 → ckpt 数/键数"）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py --ckpt-families
# ② 代表 ckpt 全键面（逐 key 落盘）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py --ckpt-keys
# ③ 与"默认模型键面"对账（本文件 §2 的 missing/unexpected 数字）
tools/venv-python -c "..."   # 见 §2 脚注的一行式命令
```

口径：
- ①=**存在性普查**（哪些 ckpt 带哪些族）；②=**逐 ckpt 键面**；③=**可加载性判定**（`missing` = 模型需要但 ckpt 没有 ⇒
  必须初始化；`unexpected` = ckpt 有但模型没有 ⇒ 必须丢弃或改变装配）。
- 522/522 个 `runs/**/*.pt` 可读（`c5_ckpt_families.json` 全量覆盖）；`runs/**` 的 119 个 >2 MB json 未参与文本扫描，
  但键面由 ckpt 二进制直接读出，不受影响。

## 2. 默认模型键面 vs 代表 ckpt（可加载性实测）

默认模型 = `pipeline.stages.build_model(load_config("config/model.yaml"))`（`plan_anchor.enabled=false`）→ **168 键**；
`model.set_path_head(True)` 后 → **186 键**。

| 代表 ckpt | ckpt 键数 | vs 默认模型(168) | vs 结构化模型(186) | 键面特征 |
|---|---|---|---|---|
| `runs/BTC20261009-0824_sw_pri512/stage_b/final.pt`（当前正式基线） | 168 | **missing 0 / unexpected 0** | missing 18（`path_head.*`）/ unexpected 0 | 干净基线 |
| `runs/round36/j1/t1_keep/ckpt_epoch001.pt`（K1 结构化头） | 186 | missing 0 / **unexpected 18**（`path_head.*`） | **missing 0 / unexpected 0** | 新族；默认装配下无法加载 |
| `runs/BTC20261005-0856_v7struct_v5/stage_b/final.pt`（v7 K-anchor） | 174 | **missing 14 + unexpected 20** | missing 32 / unexpected 20 | 缺 `*_transition`/`policy.trunk.2`，多 `encoders.lane/ttc`+`plan_head.anchor*` |
| `runs/p1_v2/round28/K1/final.pt`（R21 语义接线） | 188 | missing 0 / **unexpected 20** | missing 18 / unexpected 20 | `semantic_wiring.*`(10)+`target_proj.*`(2)+`beta_context`+`beta_token` |
| `runs/round33/canary_v2/b1/ckpt_last.pt`（R32/R33 canary） | 176 | missing 0 / **unexpected 8** | missing 18 / unexpected 8 | `ego_phys.0/2`(4)+`st_gnn.action_ego`(4) |
| `runs/p1_v2/round22/M2b/final.pt`（LD set-attention） | 177 | missing 0 / **unexpected 9** | missing 18 / unexpected 9 | `ld_aggregator.*`(7)+`ld_set_attn.*`(2) |
| `runs/_refs_rlbase/e_beta_prime/final.pt`（更早结构） | 135 | **missing 34 / unexpected 1** | missing 52 / unexpected 1 | `policy.log_std`（**改名键**）；缺 trunk.2/norm_out/query/`*_transition` |
| `runs/p1_v2/gated_residual_pilot/heads_final.pt`（分片头文件） | 12 | missing 168 / unexpected 12 | — | `gate.*`+`residual.*`（**非完整模型**，分片加载专用） |

> 数字来源（可复现，CPU，秒级）：`build_model(load_config("config/model.yaml"))` → 与 `torch.load(ckpt)["model"]`
> 做集合差；`set_path_head(True)` 后重复一次。
> 关键结论：上表 7 个代表 ckpt × 2 种装配（默认 168 / structured 186）**共 14 个组合里只有 2 个干净**
> （旧正式 ckpt×默认、K1 ckpt×structured）——每个历史阶段都留下一族"条件键"（惰性构造的消融/实验模块）。
> 因此 migration layer 必须是**显式白名单 + 显式初始化 + 显式丢弃**三档，而不是"strict=False 一把梭"
> （那正是静默口子：会把"装配不匹配的 20 个键"与"真该丢弃的 4 个 legacy 键"混为一谈）。

## 3. 键族普查（全仓 522 ckpt）

| 键族 | 覆盖 ckpt | 键数 | 现状判定 | 处置动作 |
|---|---|---|---|---|
| `encoders.{ego,ld,nav,od,others,signal,type_embed,id_embed,norm}` | 514 | 16/ckpt | 现行结构 | load |
| `encoders.lane.*` / `encoders.ttc.*` | **15** | 4 | **v8 已删通道**（`net/encoders.py` 注明"lane/ttc 已删除"） | **drop（legacy ignore 集）** |
| `mem_encoder.*` | 514 | 24/ckpt | 现行结构 | load |
| `st_gnn.{spatial,entry,od_head,ld_head,presence,step_embed}` | 514 | 33/ckpt | 现行结构 | load |
| `st_gnn.{ego,ld,od}_transition.*` | 427 | 12 | v8 前结构（旧 ckpt 有、新模型有 ⇒ 新版 ckpt 缺） | **init（缺失侧初始化）或按"旧 ckpt 结构"装配**（见 §4.2） |
| `plan_head.{fusion,moe,norm,ego_next}` | 514 | 51/ckpt | 现行结构 | load |
| `plan_head.{anchors,anchor_ds,anchor_dth,anchor_embed,anchor_head,residual_head,speed_head}` | **13** | 16 | **K-anchor（B 组，长期关闭）** | **drop（ignore 集）**；`checkpoint_migration` 记录"曾被丢弃的族" |
| `plan_head.fusion.2.*` | 514 | 2 | **K1 训练 allowlist 之一**（现行） | load（**勿与 K-anchor 混淆**） |
| `policy.*` | 514 | 19–21/ckpt（trunk 深度 4 或 6） | 现行结构（旧头；structured 模式下仅诊断/默认关路径） | load |
| `policy.log_std` | **8** | 1 | **改名键**（现名 `policy.logstd`） | **remap**：`policy.log_std → policy.logstd` |
| `value.*` | 514 | 13/ckpt | 现行结构 | load |
| `path_head.{lateral,longitudinal,lateral_geom,longitudinal_geom}` | **106** | 18 | **K1 结构化头（新族）** | load（装配含 structured）；旧 ckpt 缺失时 → **init from calibration**（§4.1） |
| `semantic_wiring.*` | 60 | 10 | R21-3A 惰性构造（`set_ld_mode("semantic")` 才建） | **条件键**：装配匹配则 load，否则 drop + 记录 |
| `target_proj.*` / `beta_context` / `beta_token` | 45 | 4 | R21-25A 惰性构造 | 同上 |
| `ld_aggregator.*` / `ld_set_attn.*` | 8 / 3 | 7 / 4 | R20/R22 消融惰性构造 | **条件键**；消融退役后 → drop 白名单 |
| `gate.*` / `residual.*` / `semantic_aux.*` | 8 / 8 / 4 | 6 / 6 / 2 | 分片头/消融实验模块 | drop（非完整模型，或历史消融） |
| `st_gnn.action_ego.*` / `st_gnn.alpha_ego` | 26 / 19 | 4 / 1 | R32 canary 动作条件化（惰性） | **条件键**（canary 归档后 → drop） |
| `ego_phys.*` | 21 | 2 | R32 canary 物理锚 decoder（惰性） | **条件键**（同左） |
| `policy.action_high/action_low`（buffer） | 514 | 2 | 现行结构（动作界） | load（buffer；缺失不致命但须显式） |

> 全部族与计数见 `c5_ckpt_families.json::families`。

## 4. 迁移映射草案（逐族动作）

### 4.1 走进结构化头（旧 ckpt → 结构化装配）

| 源 | 目标 | 动作 | 说明 |
|---|---|---|---|
| 旧 ckpt（168 键，无 `path_head.*`） | `path_head.*` 18 键 | **init（标定）** | 头构造期已按 `runs/round36/k1/calibration.json` 载入 `ds_init`/κ 限幅/几何归一化；migration layer 须**显式登记**"这 18 键来自标定而非训练"，并写入 `metrics.json::meta.ckpt_load.init_from_calibration={file, keys}` |
| 标定文件缺失 | `path_head.*` | **fail-fast** | 头内有常量 fallback（`ds_init=3.5`/`κ_limit=0.1`，`net/path_head.py:30-31`）——migration layer **不得**默认走 fallback，必须要求显式 `--allow-calibration-fallback` |
| `meta.path_head_ablation` / `meta.path_head_mode` | — | **读出后丢弃（不 forward）** | 只作装载守卫输入（`oneshot` ⇒ 拒绝，除非显式诊断开关）；A 组迁移后该键仍被 migration layer 识别 |

### 4.2 丢弃族（legacy ignore 集，建议冻结为常量）

```yaml
# legacy_key_mapper.IGNORE_PREFIXES（草案；顺序无关，前缀匹配）
- encoders.lane.          # v8 删除通道（15 ckpt）
- encoders.ttc.           # 同上
- plan_head.anchors       # K-anchor buffer（13 ckpt）
- plan_head.anchor_ds
- plan_head.anchor_dth
- plan_head.anchor_embed
- plan_head.anchor_head.
- plan_head.residual_head.
- plan_head.speed_head.
- ld_aggregator.          # R20/R22 消融（8 ckpt，条件键）
- ld_set_attn.            # （3 ckpt，条件键）
- gate.                   # 分片头/消融（8 ckpt）
- residual.               # 同上
- semantic_aux.           # R24 消融（4 ckpt）
```

要求：**每次丢弃必须逐族计数并落盘**（`dropped_keys_by_family`），禁止"总计 N 个 unexpected"一笔带过
（这正是 §六-4 要隔离的东西）。

### 4.3 条件族（装配决定是否 load）

| 族 | 装配条件 | 缺装配时 |
|---|---|---|
| `semantic_wiring.*` / `target_proj.*` / `beta_context` / `beta_token` | `model.set_ld_mode("semantic")`（R21-3A/25A） | drop + 记录（**不静默**）；或在迁移报告里给出"该 ckpt 需 semantic 装配"的提示 |
| `st_gnn.action_ego.*` / `st_gnn.alpha_ego` / `ego_phys.*` | `model.set_action_ego(True)` / `set_ego_phys(True)`（R32 canary；`@ .slim/round32/canary_train.py`） | drop + 记录（canary 归档后进 §4.2 常驻 ignore 集） |
| `st_gnn.{ego,ld,od}_transition.*` | v8 前结构装配（现模型构造**不再含**这些键） | 新版 ckpt 缺这 12 键 ⇒ 迁移目标 = 旧 ckpt 加载到"旧结构装配"；若统一到新装配则须**显式 init**（不得静默零初始化） |

### 4.4 改名 / 形状族

| 源键 | 目标键 | 动作 |
|---|---|---|
| `policy.log_std` | `policy.logstd` | rename（值直传，形状一致；8 个 ckpt 受影响） |
| 形状不一致（历史 `hidden_dim` 变化，如 H96→H128） | — | 现状：`_filter_checkpoint_state` 过滤 + 三计数上报（`pipeline/eval_runner.py:756-796`）。迁移层须把"形状不符 ⇒ 丢弃"升级为**显式清单 + 影响面报告**（哪些层被丢弃 = 该 ckpt 不能用于闭环评测） |

## 5. 接口设计草案（不改代码；R38-C 实施时的目标形态）

```
tools/legacy/checkpoint_migration.py          # 入口级迁移 API（唯一允许理解"旧键"的模块）
tools/legacy/legacy_key_mapper.py             # 纯字典/前缀映射（无 torch、无 IO，便于单测）
```

```python
# legacy_key_mapper.py（纯函数，无副作用）
IGNORE_PREFIXES: tuple[str, ...]              # §4.2 集
RENAME_MAP: dict[str, str]                    # {"policy.log_std": "policy.logstd"}
INIT_FROM_CALIBRATION: dict[str, dict]        # {"path_head.": {"source": "runs/round36/k1/calibration.json"}}

def classify_keys(ckpt_keys, model_keys) -> MigrationPlan:
    """→ MigrationPlan(load=[...], rename={...}, init=[...], drop=[...], reject=[...], reasons={...})"""

# checkpoint_migration.py（入口级）
def load_checkpoint(model, path, *, assembly: AssemblySpec, allow_calibration_fallback: bool = False,
                    on_partial: Literal["fail", "record"]) -> LoadReport:
    """① classify → ② 逐族动作 → ③ 写 LoadReport（含 dropped_by_family/init_from_calibration/
    assembly_fingerprint）→ ④ 只把 load+rename+init 的结果 load_state_dict 进模型。"""
```

**硬约束（写入契约）**：
1. **migration layer 只在装载路径被调用**：`net/model.py` 的 `forward`/`rollout` **不得** import 它；
   `net/` 侧不出现"旧键名"字符串（CI 守卫 #1 的检查面）。
2. **默认 fail-fast 保持不变**：`missing/unexpected/shape_mismatch` 非零仍拒载；migration 的"三档动作"
   只把**已登记**的族从 `unexpected` 里移走（登记动作 = 代码里的常量表，而不是参数开关）。
3. **装配指纹**：`LoadReport.assembly_fingerprint = {model_keys 数量, structured?, num_anchors, ld_mode,
   action_ego, ego_phys, ...}` ⇒ 与 `metrics.json::meta.ckpt_load` 同源，供 Clean500/Eval500 的 `protocol_hash` 使用。
4. **可审计**：每条 drop/init/rename 都带 `count + family + 理由`；`--allow-partial-load` 保留为**显式逃生门**，
   其输出必须同时写 `explicit_escape=true`。
5. **不参与 forward**：装载后模型 forward 的输入/输出键集合与"干净加载"路径**逐位一致**（单测：同一 ckpt
   经 migration 与经 `set_path_head(True)` 直载，输出 `max|Δ| = 0`）。

## 6. 门与验收（R38-C 实施时）

1. **键面单测**：`legacy_key_mapper` 的 `classify_keys` 对 8 个代表 ckpt（§2 表）给出期望动作；
   含 `policy.log_std → policy.logstd`、`plan_head.anchor*` 丢弃、`path_head.*` 标定初始化三类断言。
2. **加载回归**：522 个 ckpt 全量跑 `classify_keys`，断言"可装载集合"（含显式 init/drop）≥ 现有可装载集合，
   且**不出现新增的静默丢弃**（任何 drop 必须有 family 记录）。
3. **逐位证据**：对当前主线 ckpt（T1/t2_term2 等）跑 3 口径逐位（默认关 / structured / structured+migration），
   断言 migration 路径与直载路径 `max|Δ| = 0`。
4. **与 L-A 契约断言不冲突**：装载后 `action_generator_names()`、`path_head_mode()` 等运行时断言照常生效
   （migration layer 不得修改这些属性）。
5. **§5.69-① 的 oracle fixture 是另一类冻结物**：C 组 legacy `set_ego_warp` 的 w1/w2 逐位期望**不属于** ckpt 键面，
   须与其同批"冻结为 fixture 后移入 legacy probe"——两类遗留物在 R38-C 同轮处理，但**不要**混进 migration layer
   （migration layer 只管键面，不管 warp 语义）。

## 7. 未决

1. `st_gnn.*_transition.*`（427 ckpt 有 / 新版无）的最终处置：**统一到新装配 + 显式 init** 还是 **保留旧结构装配**？
   （影响 522-427=95 个新版 ckpt 与 427 个旧版 ckpt 的可加载性，须 CG 裁决。）
2. 形状不符族（历史 H96/H128）是否纳入 migration 的显式清单（现状只过滤，不记录族级影响面）。
3. `policy.*`（旧 PolicyHead）在 D 组收编后的键命名空间（`model.policy.*` 是否改名）——决定 migration 的 rename 表规模。
4. `removal_commit` 与 `LoadReport` 的落点约定（是否随 `metrics.json::meta.ckpt_load` 一起进产物）。
