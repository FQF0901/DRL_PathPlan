# PNP implementation status (P0/P1/P2)

Last updated: 2026-10-21 (Asia/Shanghai)

## Environment

- Project-local `.venv` with `--system-site-packages`; system Python/conda untouched.
- `metadrive-simulator==0.4.3`, `numpy==1.26.4`, `opencv-python-headless`.
- glvnd libs unpacked into `.venv/gl` to satisfy `libGL.so.1`; use `tools/venv-python`.
- Headless smoke: reset ~0.17 s, ~892 FPS, RSS ~634 MB.
- GPU RTX 4070 12 GB; CPU 20; RAM 15 GiB (swap 63 GiB).

## Current canonical implementation

The working tree contains one version. The previous 3D/legacy implementation is available only from git history/tag `pre-consolidation`, not as duplicate live modules.

- `env/obs.py`: structured schema v2 (data version), road-level Nav from `current_road/next_road/final_road/checkpoints`; no lane endpoint/later_middle in Student input; stable IDs/taxonomy/anchor; same-ID 6-frame temporal alignment.
  - OD 16 x 18, LD 16 x 5 x 16, Nav 108, Ego 12.
- `net/action_space.py`: canonical PSTL execution layer for `(dx,dy,dpsi,v_next)`; geometric Hermite path + cubic arc-length time law; exact endpoint/terminal speed; feasibility diagnostics and v_next interval; no silent action rewrite.
- `net/model.py`: 1,053,142-param `StructuredPlanner`; learned `q_lane/q_od`; `encode_entities()` for per-entity WM targets; plan input is exactly six latents; lane_role/TTC/gap are auxiliary/supervision, not plan shortcuts.
- `dataset/collect.py`: expert collection with measured 4D endpoint labels, PSTL diagnostics, expert audit, full-row outcomes; scene-sharded.
- `dataset/collect_dagger.py`: student roll-in with measured expert 0.5 s replay + lift/residual, intervention/source/privileged audit, spawn workers.
- `dataset/splits.py`: scene-seed split; no episode/seed leakage.
- `pipeline/train_bc.py`: BC/DAgger training, scene-seed split, truncated-Normal NLL, physical plan losses, aux/value/risk/cost heads, query-selection supervision.
- `pipeline/train_ppo.py`: PPO contract canary (bounded TruncatedNormal, stored sampled-vs-executed separation, per-episode GAE, truncation bootstrap, reward/cost separation).
- `pipeline/eval.py`: clean exact/LQR closed-loop evaluator (`--controller exact|lqr`), fallback/v1 projection diagnostics.
- `pipeline/distributions.py`, `pipeline/eval_metrics.py`, `pipeline/rollout.py`: bounded distribution, paired/bootstrap metrics, canonical plan helpers.
- Tests: `100 passed, 1 skipped` (`tools/venv-python -m pytest -q`).

## Stage ledger (owner policy: proceed, do not use LQR/exact A/B as hard blockers)

| stage | status | evidence / caveat |
|---|---|---|
| Stage 0 | partial | schema/split/expert collector exist; Expert audit and action-bucket quotas still missing. |
| A1 exact | **provisional pass for progression** | r3b Val500 67.6% vs Expert 73.0%; success A/B CI crosses 0; LQR 0.5 s is a low-threshold canary only. |
| A2 history | **provisional** | inference 1/2/4/6 = 16/38/52/68%; training ablation and per-checkpoint LQR shadow still open. |
| B DAgger | **provisional / data debt** | terminal failed segment is now saved and `terminal_failed` is consumed; old r3b/r4/r5/r6 data predates this fix and is not gate-clean. |
| C PPO canary | **not clean; wiring only** | `runs/ppo_v8_canary` 20 episodes / 399 transitions; per-transition logp diff 0; batch 1.1–2.6e-5; strict init; cost critic detached from trunk. Actual post-update trust region is still bad: update 1 approx_kl .189 / clip .639, update 5 .015 / .212; no performance claim. |
| D1 WM | **one-step smoke passes baseline** | `runs/wm_d1_final`: 167,952 params; val 0.616 vs persistence 1.051 (od .162/.185, ld .401/.648, nav .007/.021, ego .046/.198). Static-target/ego-dynamics semantics still open. |
| D2 WM | **probe implemented** | held-out 1/2/3/6 autoregressive with recorded actions; depth 6 WM od .522 vs persistence .673, ld .990 vs 1.182, ego .133 vs .617; not a trained 1/2/3/6 curriculum gate. |
| E/F | **not started** | blocked on D1 fidelity and the 3 s assembly; do not claim. |
| Test500 | **not run** | frozen manifest exists; only run after the final gate. |

**Owner decisions recorded:** LQR 0.5 s is a spot-check/canary, not a performance blocker; exact A/B need not be a comprehensive win; exact is close enough to Expert to proceed to RL/WM. Implementation proceeds with these debts recorded, not with false gate claims.

## Current evidence

**P0 status (partial):** P0-1/P0-3 are fixed and covered by tests; P0-2 now has shared `pipeline/execution.py` bookkeeping and measured `realized_action` in eval/PPO; unit parity tests pass, but the LQR canary/realized distribution is still open; P0-4 loads `done/shield/feasible/...`, weights failed expert only at the terminal segment and separates shield from replay, but multi-solution/privileged semantics remain open. The four P0s are not all "closed".

### Canonical data layout

`datasets/` now has exactly four logical datasets (the old versioned directories were moved to `/tmp/opencode/datasets_archive_20261022/`, not deleted):

- `datasets/train_bc_2k/` — 137,787 rows, 2,000 scenes (4 shards)
- `datasets/dagger_r1/` — 122,872 rows, 2,000 scenes
- `datasets/dagger_r2/` — 133,559 rows, 2,000 scenes
- `datasets/dagger_r3_hard500/` — 36,458 rows, 500 scenes

The canonical shards are hard-linked to the old payloads; the training loader accepts a directory of `*.npz` shards. Expert training success is **70.6%**; the expert itself is not a 100% oracle.

### Expert ceiling on Val500

I collected the expert on the 500 frozen validation seeds:

- expert success **73.0%**, collision **14.4%**, off-road **6.6%**, timeout **6.0%**, stuck 0%.

This is an upper bound for imitation under the current Expert. The clean model cannot be expected to reach 100% on this validation set without a stronger expert or RL that exceeds it. That is a root-cause finding, not a tuning excuse.

### Val500 exact

| model | success | collision | off-road | stuck | RC |
|---|---:|---:|---:|---:|---:|
| Legacy ad-hoc | 65.0% | 18.2% | 12.0% | 0.0% | 0.854 |
| Clean r3 (`runs/val500_clean_r3_p0fixed`) | 66.2% | 12.2% | 8.6% | 0.4% | 0.859 |
| Clean r3b (`runs/val500_clean_r3b_p0fixed`) | **67.6%** | 15.6% | **7.4%** | 4.6% | 0.857 |

Paired r3b − legacy: success **+2.6 pp** (95% CI −1.2..+5.8), collision −2.6 pp (CI −6.0..+0.6), off-road **−4.6 pp** (CI −7.0..−2.2). Success and collision are not formally significant. Legacy metrics have **no stuck field**, so the r3b stuck=4.6% is not a paired regression and must not be cited as one. **This is not a gate pass and r3b is not promoted.**

r4 (DAgger from r3) was **62% / 16% / 12% / 10% on 50 val**; r5 (DAgger from r3b) was **66% / 8% / 14% / 12% on 50 val**, but on Val500 only **64.8% / 13.8% / 11.8%** (stuck 5.0); r6 with student-failure upweighting was only 60/14/12/14 on 50 val. None is promoted.

### History context ablation (inference, same r3b checkpoint/seed)

Masking older history frames through `hist_valid` on the same 50 val seeds:

| frames | success | collision | off-road | RC |
|---:|---:|---:|---:|---:|
| 1 | 16% | 20% | 70% | 0.536 |
| 2 | 38% | 26% | 38% | 0.656 |
| 4 | 52% | 34% | 10% | 0.784 |
| 6 | **68%** | 16% | 10% | 0.857 |

This is a same-checkpoint inference ablation (not a retrained 1/2/4/6 model); it is strong evidence that history is causal, but the contract-required training ablation is still open.

### D1/D2 latent WM smoke

- `net/world_model.py`: ego-action-conditioned per-entity latent WM; frozen planner/encoder target; 167,952 params (hidden 80, one layer) in the smoke.
- `pipeline/train_wm.py`: mixed Expert/DAgger sampling, executed-action condition, fixed per-element persistence; `runs/wm_d1_final` val .616 vs persistence 1.051 (per-modality better). D1 one-step smoke passes; static LD/Nav/ego target semantics remain open.
- `pipeline/probe_wm.py`: held-out 6-depth autoregressive probe; `runs/wm_d2_probe_final2` depth 6 WM od .522 vs persistence .673, ld .990 vs 1.182, ego .133 vs .617. This is still not a trained 1/2/3/6 curriculum gate.
- Oracle review R1–R7 was incorporated where possible; R7 (static-target/ego-dynamics semantics), action-conditioning tests, and full D2 curriculum remain open. Stage E/F not started.


### PPO P3 wiring canary (r3b)

`runs/ppo_v7_canary`: 20 episodes / 5 updates / 398 transitions, max 100 steps; per-transition logprob diff 0, ratio-at-old 1.0; TF32 disabled; cost enters policy; strict checkpoint init. Batch diagnostic is 1.1–2.6e-5, with a separate 1e-4 diagnostic tolerance; update uses per-transition logp. LQR shadow 50 = 40/2/14/44. No performance or P3 gate claim.

### LQR canary (50 val; contract canary, not final 3 s)

| model | success | collision | off-road | timeout | RC |
|---|---:|---:|---:|---:|---:|
| Legacy ad-hoc | 44% | 8% | 26% | — | 0.776 |
| Clean r3 | 34% | 2% | 12% | 52% | 0.740 |
| Clean r3b | 48% | 6% | 10% | 36% | 0.797 |

r3b tracking: xy 0.024 m, heading 0.009 rad, speed 0.247 m/s; steer saturation 0.0034, throttle saturation 0.0. On the same 50 seeds, exact r3b succeeds on 34 and LQR fails on 11 of those (**32.4% Exact-success→LQR-failure**); this is a contract canary metric, not a final 3 s score. The 0.5 s canary is still a gate and is **not closed**; the 3 s / 6-point interface is not implemented. The tiny-negative-speed LQR crash found during this run was a numerical bug, now clamped/fallback; it was not a planner failure.

### PPO contract canary

`runs/ppo_v2_canary/contract.json` is still 4 transitions, 25% mask, and a pre-P0 `bc_v2` checkpoint. It is wiring evidence only; not P3 and not a performance claim.

### Test500

`config/splits/val500.json` and `test500.json` now carry seeds/SHA; `scenarios_test500.json` has `split="test"`; `pipeline/eval.py --split-manifest ... --split test` verifies it. Test500 has **not** been run; no conclusions.

### Cleanup

- `datasets/`: four logical dirs, 2.1 GiB; old versioned dirs are archived outside the project.
- `runs/`: 69 MiB of current lineage/evidence; old run dirs are archived outside the project. `runs/logs` and `runs/monitor` are retained.
- Tests: **100 passed, 1 skipped**.

## Design decisions

These are implementation decisions that `docs/arch_lock.md` leaves open: candidates, measured evidence, choice, constants, and rollback conditions. They are not review rebuttals; debate belongs in `docs/deepseekv4p1_argue.md`.

### 2026-10-20 — 4D action execution formulation

- **Contract:** §6.1–6.5, §7.2, §22.2–5.
- **Candidates:**
  - A: speed-scaled time-Hermite. Endpoint and endpoint velocity exact, but speed is not monotone, geometry/speed conflicts can bulge/loop, and there is no explicit `a_lon`/jerk/curvature gate.
  - B: 2D quintic time polynomial. C2-capable, but path and speed are implicit; zero endpoint acceleration raises peak acceleration; PPO bounds are high-order inequalities.
  - C: **PSTL** — geometric cubic-Hermite path in a shape parameter `u`, plus a cubic arc-length time law `s(tau)` with `s(0)=0`, `s(T)=L`, `sdot(0)=v0`, `sdot(T)=v1`.
- **Chosen:** C. It executes `v_next` exactly, keeps the 4D action and one Plan Head, yields five 0.1 s position/heading/speed points, makes the endpoint/speed conflict explicit (`r_v=v1-v_cons`, `j_req=6r_v/T²`), and supports a monotone-speed check.
- **Interpretation note:** the contract sentence “tangent magnitude from speed boundary” is applied to the **time law**, not the geometric path. The geometric path uses a chord-scaled tangent (`m=rho*||p1||`, `rho=1.0`). This is deliberate and recorded; if the contract owner rejects it, A is the fallback but must be labeled non-monotone.
- **Evidence (expert BC 2k, 34,847 rows, `bc_2k_c0`):**
  - geometry-only terminal-speed error: mean |e| 0.133 m/s, p95 0.386, p99 1.409;
  - PSTL hard gate (`a_lon in [-10, 3.5]`, `|j| <= 20`) passes 97.3%; failures: jerk 1.6%, a1 0.6%, a0 0.26%, vmin 0.2%;
  - feasible `v_next` interval width p5/p50/p95 = 0.84/1.16/1.27 m/s; width <0.1 m/s = 0.6%;
  - expert `v_next` outside hard interval = 2.3%;
  - `v_lo-v0` median -0.78 m/s, `v_hi-v0` median +0.43 m/s.
- **Provisional constants:** `rho=1.0`; static action box dx [0,7.5], dy [-1.5,1.5], dpsi [-0.8,0.8], v [0,15]; `HARD_LIMITS`: a_lon [-10,3.5], |a_lat|<=8, |kappa|<=0.345, |kappa_rate|<=1.0, |jerk|<=20/20; `COMFORT_LIMITS`: a_lon [-4,2], |a_lat|<=3, |kappa|<=0.20, |kappa_rate|<=0.05, |jerk|<=5/5. These pass ~97.2% of 2k expert rows under PSTL. All values remain provisional until LQR/closed-loop calibration is logged.
- **Rollback conditions:** expert hard-gate pass <95%; feasible v interval p90 width <0.1 m/s; LQR speed/lateral tracking regresses on paired scenes; round-trip ADE >0.1 m after rho calibration; PPO log-prob/ratio consistency fails.
- **Next:** implement `net/action_space.py` and the Exact/LQR speed-profile path; then BC NLL/aux and PPO probability contract. Do not connect the 6-step same-context preview as a 3 s WM/plan claim.

### 2026-10-20 — Data split and bounds

- Data split must be by `scene_seed`, not `episode_id`; the historical BC+DAgger mix had 800 duplicate seeds and a 28.5% contaminated internal val.
- Historical expert 2k baseline: 70.2% success / 15.6% collision / 5.8% off-road / 8.3% timeout. The 80%/5% on 20 val is a small-sample slice and is not an expert ceiling.
- Action bounds above are calibrated from BC 2k quantiles and DAgger outliers; they are not yet LQR-trackability calibrated.

## Root causes found and fixed

1. **`q_lane` was near-uniform**: current lane role0 received 0.011 attention while a lat=-7 m lane got 0.60; the car drifted off-road even on straight scenes. Added query-selection auxiliary supervision (current lane / urgent OD attention targets). This is auxiliary only; current lane/TTC are not plan inputs.
2. **BC-to-closed-loop distribution shift**: on the current re-collected 2k data, BC 2k was 8% success / 70% off-road on 50 val; DAgger r1b/r2/r3 were 44%/50%/58% success. The historical “50% then 64%” came from the old data/evaluator and is not the current r3 number. This remains the dominant P2 issue.
3. **DAgger lift bias**: old kinematic bicycle labels had +0.53 m dx / +1.12 m/s v_next bias and censored braking; the canonical collector stores measured expert replay plus lift/residual/source validity.
4. **LQR canary gap**: clean r3 34% / 52% timeout and r3b 48% / 36% timeout on the 50-val canary vs exact 66.2/67.6%. The evaluator only executes the first 0.5 s (5 points), so this is a canary gate result, not the final 3 s controller-realized score. Report tracking/trackability/saturation and fixed-seed Exact-vs-LQR failures now; the full 3 s/6-point interface is the focus after the ego-conditioned latent WM/imagined rollout. Do not replace the canary with a NaN/crash-only check.

## Current state

- No training/collection process is active; 20-minute monitor is running (`runs/monitor/tasks_20m.log`).
- Data: four canonical dirs only: `datasets/train_bc_2k`, `datasets/dagger_r1`, `datasets/dagger_r2`, `datasets/dagger_r3_hard500`.
- Best clean checkpoints: `runs/bc_2k_dagger_r3b` (exact 67.6/15.6/7.4, stuck 4.6; 0.5 s LQR 48/6/10/36) and `runs/bc_2k_dagger_r3` (exact 66.2/12.2/8.6, stuck 0.4; LQR 34/2/12/52). Neither is promoted: `best.pt` is still selected by open-loop validation loss (r3b epoch 4/15), not by a closed-loop gate. Safety-prefer r3; success-prefer r3b, but the paired differences are not gate-significant.
- Legacy reference checkpoint (ad-hoc, not a formal A/B): `runs/p1_weight_hi` with (`runs/val500_legacy_p0fixed`, 65.0/18.2/12.0/0.854). The checkpoint uses the old schema/model and requires code at tag `pre-consolidation`; the canonical tree cannot load it, and the metrics artifact has no config/commit provenance.
- 10k collection was stopped in its unsharded form: the parent process aggregated all episodes and exhausted RAM/swap. 2k is sharded and stable; final 10k needs shard-stream training or a disk-backed dataset.
- Val/Test split is now machine-readable: `config/splits/val500.json` and `config/splits/test500.json`; `scenarios_test500.json` has `split="test"`; `dataset/freeze_splits.py` regenerates them; `pipeline/eval.py --split-manifest ... --split test` verifies seeds/SHA. Test500 has **not** been run.
- Stage numbering: P1 single-frame BC; **P2 = real history + DAgger**; P3 short PPO canary; P4 WM. D1/D2 WM were started out of the original order under the owner's progression policy; this is recorded as a deviation, not a gate pass. Imagine Bank/Ego Dynamics/3 s LQR remain unstarted.

## Reproduce (current machine)

```bash
# Environment already created; use the glvnd wrapper.
tools/venv-python -m env.scenario.cli --n-train 10000 --n-val 1000 --slice 200 --workers 4 --no-validate

# Expert/DAgger shards are collected with --out <canonical>/shard_XXXX.npz;
# the loader accepts a directory of *.npz shards.
tools/venv-python -m dataset.collect --specs env/specs/scenarios_train.json \
  --out datasets/train_bc_2k --offset 0 --limit 500 --workers 8 --max-steps 600
tools/venv-python -m dataset.collect_dagger \
  --specs env/specs/scenarios_train.json --out datasets/dagger_r2 \
  --offset 0 --limit 500 --ckpt runs/bc_2k_dagger_r1b/best.pt \
  --workers 8 --max-steps 600 --expert-ratio 0.3 --shield-ttc 2.0 --round 2 --device cuda

# Train/evaluate only through the frozen manifest.
tools/venv-python -m pipeline.train_bc \
  --data datasets/dagger_r2 --init runs/bc_2k_dagger_r1b/best.pt \
  --out runs/bc_2k_dagger_r2 --epochs 15 --batch-size 256 --lr 1e-4 \
  --w-infeasible 0.2 --w-shield 3 --device cuda
tools/venv-python -m pipeline.eval --specs env/specs/scenarios_val500.json \
  --split-manifest config/splits/val500.json --split val --strict-checkpoint \
  --ckpt runs/bc_2k_dagger_r3b/best.pt --out runs/val500_clean_r3b_p0fixed \
  --limit 500 --max-steps 600 --controller exact --device cuda
tools/venv-python -m pipeline.eval --specs env/specs/scenarios_val500.json \
  --split-manifest config/splits/val500.json --split val --strict-checkpoint \
  --ckpt runs/bc_2k_dagger_r3b/best.pt --out runs/lqr50_clean_r3b_canary \
  --limit 50 --max-steps 600 --controller lqr --device cuda

# PPO contract canary.
tools/venv-python -m pipeline.train_ppo \
  --specs env/specs/scenarios_train_slice200.json --ckpt runs/bc_2k_dagger_r3/best.pt \
  --out runs/ppo_v2_canary --updates 1 --episodes 1 --max-steps 20 --device cpu
```

## Known gaps

- Val500 exact is **not a formal pass**: r3b success +2.6 pp vs legacy (95% CI −1.2..+5.8), collision −2.6 (CI −6.0..+0.6), off-road −4.6 (CI −7.0..−2.2). Legacy has no stuck field; r3b stuck 4.6% is not a paired regression. r3 is safer, r3b has higher success; do not promote either on this table.
- **Expert ceiling is 73.0% success on Val500** (14.4 collision, 6.6 off-road, 6.0 timeout); these are episode-level counts. The clean r3b is 67.6%; 100% is not attainable on this validation set with the current expert/imitation pipeline. Need a stronger expert or RL/curriculum that exceeds it, not just more epochs.
- 0.5 s LQR canary is a current contract gate and is **not passing/closed**: r3b 48% success / 36% timeout; r3 34% / 52%. It is not the final 3 s controller-realized score. Report tracking/trackability/saturation and Exact→LQR failure ratio; the full 3 s/6-point interface follows after WM/imagined rollout.
- 10k/500/500 + Test500 final scale and shard-stream loader are not implemented; no Test500 conclusions.
- Dataset cleanup is done: four canonical dirs, 2.1 GiB; old dirs archived outside the project. Existing `runs/*best.pt` is not a closed-loop-selected final checkpoint.
- P0-4 multi-solution/privileged semantics are still open; DAgger failure-prioritized collection beyond the current r4/r5/r6 experiments is needed. The history ablation is inference-only; a retrained 1/2/4/6 ablation, Expert-GT 3 s LQR controller smoke, PPO reward/cost hacking audit, and full 3 s plan/WM path remain open.
- r6 with `w_student_failure=2` was 60% on 50 val; do not promote it.
- terminalfix2 BC was 12% on 50 val; terminalfix2 DAgger r1 was 48% / 12% / 34% on 50 val and **47.8% / 19.0% / 26.4%** on Val500 (stuck 1.6%). It does not close the B gate; old data debt remains visible.
- Closed-loop selector on 50 val: selected r3b; r3 eligible (58/12/10), r3b eligible (68/16/10), terminalfix r1 ineligible (off-road 34%). Selection is exact-only, no LQR branch yet, and 50-scene selection is not final. See `runs/selected_checkpoint50/selection.json`.
- D1/D2 WM smoke exists, but Stage E/F (joint Plan+WM, 3 s LQR, imagined RL) are not started.
- Checkpoint selection is still open-loop; `best.pt` is not a final model.
- Stage C is wiring only: per-transition logp passes, but batched diagnostic logp is 1.1–2.6e-5 (>1e-5); cost has no limit/dual; mask/postprocess ~31%; LQR shadow is only 50 val and low-threshold.
- D1/D2 are not full gates: D1 one-step smoke now passes persistence (0.616 vs 1.051) but static-target/ego-dynamics semantics are incomplete; D2 is an exploratory held-out autoregressive probe, not a trained 1/2/3/6 curriculum; Stage E/F are not started.
- Old BC/DAgger r3b/r4/r5/r6 data predates `terminal_failed`; `terminalfix2` BC data now has terminal rows and partial terminal labels, but the DAgger terminalfix run needs a fresh pass after the `_backfill` change. Old data is not clean B-gate data.
- Stage D1/D2 code is a smoke; Imagine Bank/Ego Dynamics, joint Plan+WM, and the 3 s LQR/P5 formal path are not started.
- Expert audit fields are stored, but the multi-solution downweighting policy is still conservative.
- Comfort violation rate is 100% in current evals under the strict comfort bounds; it is a diagnostic, not yet a training gate.
