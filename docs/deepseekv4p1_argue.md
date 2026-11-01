# Bank-native raw-history training: final plan

Status: raw-only implementation complete; final adversarial review ACCEPT (no P0/P1). No commit/push.
Scope: IL/BC and DAgger training; no DAgger recollection in this pass.

## 1. Decision: raw history only

Canonical training, validation and closed-loop deployment must use the same
history representation:

```text
offline:
  raw NPZ (history_mode=1, real per-frame pose)
  → BankWindow
  → TSB
  → Planner / WM

online:
  env observation
  → LiveBankRunner
  → RealMemoryBank
  → BankWindow
  → TSB
```

`stack_history` query-frame projected shards are no longer accepted by the
loader. They are legacy/archive data only. The loader must reject missing or
`history_mode=0` data with a clear regeneration error.

Why: a projected historical frame is expressed in each row's query ego frame.
The same physical past frame appears differently in row `t` and row `t+1`, while
`LiveBankRunner` keeps raw frames and real pose deltas. A clip/GroupSampler
changes sampling order, but cannot undo this feature-coordinate mismatch.

## 2. Raw data generation

Regenerate IL and validation data with the current collectors. No DAgger yet.

```bash
python -m dataset.collect \
  --specs data/specs/train_2000.json \
  --out data/episodes/train_2000/il_expert_raw \
  --workers 8 --max-steps 600

python -m dataset.collect \
  --specs data/specs/val_500.json \
  --out data/episodes/val_500/expert_raw \
  --workers 8 --max-steps 600
```

Expected canonical paths:

```text
data/episodes/train_2000/il_expert_raw
data/episodes/val_500/expert_raw
```

Old projected shards should be moved under `archive/` or removed after raw
training is verified. They must not remain in the canonical training path.

## 3. Fast BankWindow path

`net/model.py` provides a raw-only tensor builder:

- flatten `[B,T,...]` to `[B*T,...]`;
- one encoder call for all six frames;
- vectorized duplicate-ID sanitization;
- stable-ID reindexing into the current frame's slot order;
- direct `BankWindow` tensor assembly;
- real per-frame `ego_pose`;
- malformed/legacy batches fall back to the record builder.

The fast path must match `BankWindowBuilder` field-for-field (FP tolerance) over
real shipped raw rows and TSB/action outputs.

## 4. Sequence batches and FIFO

`pipeline/scene_batches.py` groups raw rows by episode, preserves frame order,
shuffles episode/clip sequences, and packs `scenes_per_batch` sequences.

`run_scene_epoch`:

- folds a short final TBPTT tail into the previous chunk;
- performs one optimizer step per chunk;
- weights chunk loss by plan-weight sum;
- uses tensor FIFO only for raw history;
- rebuilds the six-frame window when an entity re-enters, a row step is not
  consecutive, or the FIFO is otherwise unsafe;
- never carries a post-optimizer FIFO into the next chunk;
- masks finished scenes and never crosses episode boundaries.

Default schedule:

```yaml
sequence_batch: true
scenes_per_batch: 256
clip_length: 100
tbptt_chunk: 8
```

The default step count is lower than row-wise training; convergence must be
verified before treating it as final.

## 5. TensorBoard

Loss tags are grouped:

```text
train_loss/total
train_loss/nll
train_loss/point
train_loss/heading
train_loss/velocity
train_loss/curvature
train_loss/accel

eval_loss/total
...
```

Closed-loop KPIs are separate:

```text
closed_loop/success_rate
closed_loop/collision_rate
closed_loop/off_road_rate
closed_loop/route_completion_mean
closed_loop/fallback_rate
closed_loop/infeasible_count_mean
```

No weight histograms, model graph, or images are added.

## 6. Closed-loop selection

Fixed val500 only; never test500.

```yaml
closed_loop: true
closed_loop_specs: data/specs/val_500.json
closed_loop_split_manifest: data/splits/val_500.json
closed_loop_limit: 100
closed_loop_final_limit: 500
closed_loop_every: 5
closed_loop_patience: 3
closed_loop_max_steps: 600
```

Every 5 epochs run a fixed closed-loop subset. Final best checkpoint runs the
full 500. Patience counts evaluations, so 3 evaluations means about 15 epochs.

Score is lexicographic success-first:

```python
(
    success_rate,
    -collision_rate,
    -off_road_rate,
    route_completion_mean,
    -fallback_rate,
    -infeasible_count_mean / max_steps,
)
```

No scalar weighted sum and no projected diagnostic opt-in. Projected data is
rejected before training.

## 7. Verification

Targeted tests:

- raw loader accepts `history_mode=1` and rejects projected/missing markers;
- fast vs record BankWindow parity on real raw rows;
- TSB/action parity;
- raw FIFO re-entry and non-consecutive-step rebuild;
- `run_scene_epoch` vs `run_epoch` raw metrics/MAE;
- tail fold and plan-weighted chunk loss;
- lexicographic closed-loop score;
- closed-loop empty/non-finite and environment-failure guards;
- timestamp/config/TB tags.

Headless MetaDrive/Panda3D failures are environmental and must not be counted as
logic regressions.

## 8. Next steps

1. Generate raw IL/val shards.
2. Run raw sequence BC, inspect TB and closed-loop outputs.
3. Select best checkpoint on lexicographic closed-loop ranking.
4. Only then begin a new DAgger round with that checkpoint as student.
5. Keep official conda/Python/site-packages read-only; do not commit/push until
   adversarial review is complete.

## 9. Selection failure policy

`closed_loop: true` is a required selection stage for this workflow. If the
closed-loop evaluator fails or produces no selectable `best_closed_loop.pt`,
`train_bc` writes metrics/checkpoints and exits non-zero. Downstream WM/PPO
must not silently consume a stale or missing selected checkpoint.

`od_age` is currently a zero-valued schema slot in both raw and live paths;
it is not a train/serve skew, but it is a dead feature and must be removed or
documented in the next schema cleanup.

## 10. GPU memory note

The initial 1024-scene / 16-frame-chunk sequence settings exceeded the 12 GB
RTX 4070 budget (CUDA OOM while building the initial encoder batch).  The
checked-in default is now 256 scenes and an 8-frame TBPTT chunk; this measured
about 5.4 GB peak in a one-batch smoke.  Increase only after measuring peak
memory for the target GPU.
