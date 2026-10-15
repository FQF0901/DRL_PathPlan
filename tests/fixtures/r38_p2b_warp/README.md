# R38-P2b warp oracle fixture（legacy `set_ego_warp` w1/w2 逐位冻结）

**用途**：R38-P2b 删除生产 `net/` 的 legacy warp 实现（`set_ego_warp("w1"/"w2")` /
`DrivingModel._ego_warp_pose_k`）后，M1/L-D 等价性证据链的**逐位参照物**（§5.69-① 硬前置）。
相关 Gate/单测改读本目录（`manifest.json` 逐键 sha256 可核验），不再调用已删实现。

## 生成（必须在 legacy net 下运行）

```bash
# 生成时点 HEAD = dd0a8c2（R38-P2b 前；net/ 未改、w1/w2 仍在）
tools/venv-python .slim/deepwork/p1_v2/round38/p2b_oracle_freeze.py --out tests/fixtures/r38_p2b_warp

# 事后重生成（net 已删 w1/w2 时）：取旧 net 快照再跑
git worktree add /tmp/legacy_net dd0a8c2
cd /tmp/legacy_net && tools/venv-python .slim/deepwork/p1_v2/round38/p2b_oracle_freeze.py \
    --out /workspace/01_Proj/DRL_PathPlan/tests/fixtures/r38_p2b_warp
```

## 内容

| 文件 | 口径（recipe） | 关键内容 |
|---|---|---|
| `pure_cases.pt` | d1_ego_warp_tests::check_r33b_bit_exact 用例表（float32/float64 × 5 形状 + 3 边界 ×2） | 每例输入 `ego/od/ld/vel` + 期望 `legacy_node_pose_k`/`warp_pose_k`/`assemble_{off,w1,w2}`（生成期已断言 `_ego_warp_pose_k == warp_pose_k` 逐位） |
| `model_poses.pt` | 小型模型（hidden=16, experts=8, wm_steps=6；`torch.manual_seed(0)`）+ `tests.test_net_shapes.make_obs(batch=2, seed=0)` | 三模式（off/w1/w2）st_gnn 实收 `pose_k_by_step` + `traj_xy/traj_theta/plan` + 重建输入（`od_pose_t0/ld_pose_t0/od_vel_rel/dt`）+ state_dict |
| `m1_arms.pt` | `round37/m1_gate_a.py::run_arms`（M0 ckpt + BTC…v8 数据集，seed 0，batch 4） | 五臂全输出（w1/w2 = 25 键，off=25，cheap=7，structured=73） |
| `b_compat_arms.pt` | `round33/b_warp_compat.py::check_forward_parity`（K1e2 semantic ckpt，batch 8） | `out_off/out_w1/out_w2`（27 键/臂） |
| `manifest.json` | — | git rev、逐文件 sha256 / 逐键 sha256 / 数据与 ckpt 身份、recipe 常量 |

## 消费方

- `.slim/deepwork/p1_v2/round37/m1_gate_a.py`（w1/w2 臂 → fixture；call-site 参照更新）
- `.slim/deepwork/p1_v2/round33/b_warp_compat.py`（w1/w2 接线探针 → fixture）
- `.slim/deepwork/p1_v2/round36/d1_ego_warp_tests.py`（模型内部逐位 → fixture + legacy 纯函数快照）
- `tests/test_r38_p2b_warp_legacy_removed.py`（残留守卫 + fixture 完整性）
- 纯函数复算：`tools/legacy/legacy_ego_warp.py`（自包含快照，不依赖 fixture）
