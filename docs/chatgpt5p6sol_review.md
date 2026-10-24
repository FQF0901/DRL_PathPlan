# net contract review — 2026-10-25

> 标准：`docs/arch_lock.md`
> 审阅对象：当前 staged changes + `docs/deepseekv4p1_argue.md`
> 结论：**net 代码修复 ACCEPT with one high caveat；整体 net contract NOT READY。**

## 0. 总结

这轮修正确实有实质进展：

- **OD 已改为 absolute decoder**：`ODGeometryDecoder` 不再做 `base_od + residual`；
- **rollout 已收敛到 `net/world_model.py` 的 `imagine_rollout`**，`DriveModel.imagine` 变成薄 wrapper，方向符合 §13.5；
- **`nav_seq` / `ego_seq` / `valid_seq` 增加了 shape 校验**，`action_provider` 现在拿到的是 `ego_n`；
- **LD branch 已可选**，`project_ld` 已改为公共 `project_geometry`；
- `test/test_new_net.py`：**14 passed**；全量：**106 passed / 1 skipped / 1 failed / 7 errors**，失败仍是 MetaDrive/Panda3D headless。

但仍有 1 个 high code bug、若干 docs/code 不一致，以及既有的全局 blocker。

---

## 1. High：OD absolute decoder 仍然 zero-init

`net/world_model.py:303-315` 的 `ODGeometryDecoder` 现在声明为 absolute：

```python
"""Predict the absolute OD relative pose in the query ego frame."""
...
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

absolute decoder 的最后一层 zero-init，意味着初始化时：

```python
pose = self.od_geometry(od_next)  # 全 0
```

实测：

```text
absolute decoder init max abs = 0.0
wm(...).od_geometry[..., :3].norm() = 0.0
```

结果是**所有 OD 的 pose 初始化到 ego 原点**，不是 persistence。
这直接违反 §12.9 的 “residual + normalization，保留 persistence baseline” 和 §12.11 的 “one-step geometry 不低于 persistence”。

如果 OD 确定是 absolute，zero-init 必须去掉或换成有意义的 prior；zero-init 只适合 residual 方案。当前代码是“absolute 语义 + residual 初始化”，这是这轮最重要的 new bug。

> 注意：`od_delta` 的 zero-init 是 content persistence，没问题；问题只在 `ODGeometryDecoder`。

---

## 2. Medium：OD forward 的输出形状依赖可选参数

`net/world_model.py:398-406`：

- `od_geometry` 为 `[B,N,5]` 时，返回 `[B,N,5]`；
- `od_geometry=None` 时，返回 `[B,N,3]`。

但 `BankWindow.advance` 期望 `[B,N,5]`。`imagine_rollout` 总是传 `base_od`，所以主路径没炸；但 `FactorizedEgoConditionedLatentWM.forward` 单独调用时会返回 3D，后续 `advance` 会 shape mismatch。

建议：要么永远返回 5D（无 base 时 velocity 置零/显式约定），要么在 forward 里明确校验 output shape，不要让同一个方法随可选参数变 shape。

另外，absolute decoder 只预测 pose，velocity 仍从 projected `base_od[..., 3:]` 继承。这个设计可以接受，但合同没有写；如果保留，建议在 §6.2/§9.4 明确“velocity 不从 WM 预测，沿用 base”。

---

## 3. Medium：action bounds 仍然没有按 §13.4 接

当前 `config/model.yaml` 没有 action 段；`env/action/action_space.py` 仍是 `CLEAN_BOUNDS` / `HARD_LIMITS` 单一来源。
`docs/implementation_status.md:114` 也写了“action bounds/limits 仍由 action_space 常量负责，不保留 config 中的 dead action 段”。

这和 arch_lock §13.4 的“`config/model.yaml` 负责 action 的 bounds/limits（仅配置）”不一致。
`docs/deepseekv4p1_argue.md:10` 仍说“已在 config/model.yaml 增加 action 段”，这已经和当前代码/配置不符，是 stale reply。

要么接受 action_space 为唯一真源并更新 arch_lock，要么在 config 里放 typed bounds/limits 并让 action_space 读取/校验；不能两份都声称是标准。

---

## 4. Docs 与代码不一致

### 4.1 `docs/implementation_status.md`

同一份文档内部矛盾：

- `:77`：`ODGeometryDecoder 作为 relative-pose residual，加在当前 geometry 上`；
- `:94`：`OD 用 absolute relative-pose decoder`。

当前代码是 absolute，所以 `:77` 是 stale。
另外 `:18` 写“WM 的 OD geometry / valid / persistence 初始化”，如果指 OD geometry，则与 absolute zero-init 不符；如果指 `od_delta` content，则要写清楚，避免误解。

### 4.2 `docs/deepseekv4p1_argue.md`

这份答复没有跟上最新代码：

- `:12` 说“world_model.rollout 与 imagine 是两套 rollout”；现在代码是 `world_model.imagine_rollout` 单一实现 + `DriveModel.imagine` 薄 wrapper。
- `:42-45` 说“world_model.rollout 不应作为正式路径；正式路径应只保留 DriveModel.imagine”；这与当前代码和 §13.5 相反。
- `:62-64` 说“world_model.rollout 仍不是正式路径；与 DriveModel.imagine 两套语义”；已过时。
- `:10` 说 config 增加了 action 段；当前 config 没有 action 段。

这份 reply 需要重写，否则会误导后续 review。

### 4.3 `docs/chatgpt5p6sol_review.md`

原文件是旧 review，部分内容（OD residual、in-place、旧 rollout）已经与最新代码不一致。
本文件已用当前结论替换。

---

## 5. 其他 minor / API

- `world_model.FactorizedEgoConditionedLatentWM.rollout` 方法被删掉，改成模块级 `rollout = imagine_rollout`；`DriveModel.rollout` 是 alias。功能上符合 §13.5，但 `wm.rollout(...)` 的旧调用会 break；建议保留一个 method alias 或在 migration note 里明确。
- `imagine_rollout` 要求 `actions.shape[1] >= horizon`，但循环只用到 `0..horizon-2`；逻辑上 `horizon-1` 个 action 就够。不是 bug，但 API 偏严。
- `action` 维度在 `imagine_rollout` 里硬编码为 4；如果 `build_net` 允许其他 `action_dim`，会不一致。合同是 4D，建议 build_net 也固定校验 4。
- `nav_valid_seq` / `ego_valid_seq` 的 3D 输入没有校验最后一维必须是 1；传 `[B,H,5]` 会在 TSB reshape 处报错。
- `imagine_bank` 只支持 B=1；合同是 per-episode Bank，可接受，但应在 API 文档写明。

---

## 6. 仍未完成的 blocker

- live path 仍是 `StructuredPlanner` / 旧 `EgoConditionedLatentWM` / `stack_history`；
- `TargetEncoder`、`pipeline/losses.py`、新 checkpoint 未接；
- `BankWindow` 没有独立 `slot`，schema 仍是工程偏差；
- `horizon=6` 仍是 6 policy / 5 imagined transition，合同语义待 owner 明确；
- `valid` 传播、LD type t0 继承、`pipeline/rollout.py` 仍是 legacy；
- §9 测试仍缺 FIFO/reset/reject/permutation/rollout NaN/config 等。

---

## 7. 最终判定

- **net 代码修复：ACCEPT with one high caveat**（OD absolute zero-init）。
- **整体 net contract：NOT READY**。
- 下一步优先级：
  1. 修 `ODGeometryDecoder` absolute 初始化；
  2. 固定 OD geometry output shape（3D/5D）；
  3. 同步 `implementation_status` / `deepseekv4p1_argue` 的 stale 表述；
  4. 处理 action bounds 的合同归属；
  5. 继续 live path cutover、TargetEncoder、losses、checkpoint；
  6. 补 §9 测试。
