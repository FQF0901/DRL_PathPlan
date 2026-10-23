# Real/Imagine Memory Bank + ST-GNN integration

## Added modules

- `net/memory_bank.py`: entity/frame records, Real/Imagine FIFO banks, copy-on-write rollout snapshot, lazy SE(2) alignment, stable-ID temporal edges, birth/death representation, current-frame spatial edges.
- `net/st_gnn.py`: dependency-free typed temporal/spatial message passing.
- `net/bank_planner.py`: bank-native planner core and dynamic-OD-only world-model head.
- `test/test_memory_bank_stgnn.py`: FIFO/reset, SE(2), stable-ID/birth/death, no-writeback, ST-GNN smoke tests.

## Intended call path

1. `env.obs` returns **one current structured frame only**.
2. Existing OD/LD/Nav/Ego encoders encode that frame.
3. Convert encoded tokens and geometry anchors to `EntityRecord`s and append one `FrameRecord` to `RealMemoryBank`.
4. `BankViewBuilder.build(real_bank.frames)` performs lazy SE(2) compensation and graph construction.
5. `BankPlannerCore(view)` produces plan/value/risk.
6. For rollout, call `real_bank.snapshot()`, append generated `Source.IMAGINED` frames, and rebuild the view each depth.

## Deliberate boundaries

- The bank does not align history to current slots. Stable IDs create temporal graph edges.
- Dead entities remain historical nodes but are excluded by `readout_mask`.
- New entities have `birth=True` and no temporal parent.
- Real bank rejects imagined frames; imagined frames never mutate the real bank.
- Geometry is transformed analytically; ordinary content latent is never rotated.
- `DynamicODWorldModel` predicts only dynamic OD residual and survival. Ego pose is advanced analytically; LD/Nav must be reprojected or reconstructed by route progress.

## Required repository integration work

The supplied source dump does not expose a clean single-frame encoder API. Before deleting the old path, split `StructuredPlanner._encode_frame` into a public `encode_current_frame(obs)` method returning per-entity content latent and geometry anchors. Then replace all `stack_history` construction in collection/evaluation/rollout with bank append calls. Do not maintain both history systems after migration.

The placeholder `advance_ego_pose()` must be replaced by the repository's canonical 4-D action decoder from `net/action_space.py`; it is intentionally marked as a placeholder to avoid silently inventing action semantics.
