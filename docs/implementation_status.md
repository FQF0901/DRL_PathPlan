# PNP implementation status (P0/P1)

Last updated: 2026-10-20 (Asia/Shanghai)

## Environment

- Project-local `.venv` with `--system-site-packages`; system Python/conda untouched.
- `metadrive-simulator==0.4.3`, `numpy==1.26.4`, `opencv-python-headless`.
- glvnd libs unpacked into `.venv/gl` to satisfy `libGL.so.1`; use `tools/venv-python`.
- Headless smoke: reset ~0.17 s, ~892 FPS, RSS ~634 MB.
- GPU RTX 4070 12 GB; CPU 20; RAM 15 GiB (swap 63 GiB).

## Implemented

- `env/metadrive_env.py`, `env/tracking.py`, `env/expert/pure_pursuit_idm.py`, `env/scenario/` copied from the previous branch as environment/control references (not the old net/schema).
- New `env/obs.py`: fixed-shape OD/LD/Nav/Ego, 6-frame history, ego-motion alignment.
  - OD: 16 x 11 (dx,dy,cos,sin,vx_rel,vy_rel,L,W,type,static,presence)
  - LD: 16 x 5 x 9 (dx,dy,heading_rel,curvature,speed_limit,left_type,right_type,lane_role,lat_offset)
  - Nav 11, Ego 8, history 6.
- `dataset/collect.py`: expert roll-in, 4D action labels `(dx,dy,dpsi,v_next)`.
- `dataset/collect_dagger.py`: student roll-in, expert 4D label via kinematic bicycle; recovery states.
- `net/model.py`: 921k deployable params, H=128, 4 heads, temporal attention + 2-layer spatial transformer, role0 lane readout, nearest-OD and TTC readouts, value/risk heads (heads exist; value/risk loss not yet trained).
- `pipeline/train_bc.py`: BC/fine-tune, episode-level split, normalized action MSE.
- `pipeline/rollout.py`: 4D action -> cubic Hermite segment; 6-step composition.
- `pipeline/eval_exact.py`: exact kinematic closed loop.
- `pipeline/eval_lqr.py`: P1 single-0.5 s-segment LQR canary.
- `pipeline/train_ppo.py`: short PPO canary (sampled normalized 4D action, exact tracker, GAE, clipped PPO). Smoke: 1 update / 1 episode / 100 steps at `runs/ppo_canary_smoke`; not a performance claim.
- `config/`, `net/param_probe.py`, `tools/monitor_tasks.sh`.
- New schema-v2 path (2026-10-20): `env/obs_v2.py` with road-level Nav, stable IDs/taxonomy, fixed 6-frame same-ID history; `net/action_space.py` PSTL 4D→speed-profile execution; `net/clean_model.py` 1,053,142-param clean planner with `q_lane`/`q_od`, aux heads and `encode_entities()`; `dataset/collect_v2.py` measured expert labels + audit; `pipeline/train_bc_v2.py`; `pipeline/eval_clean.py`; `pipeline/distributions.py`; `pipeline/eval_metrics.py`; `dataset/splits.py`.
- Test suite: 89 passed, 1 skipped (`tools/venv-python -m pytest -q`). Legacy path remains intact.

## Current results (10 validation scenes, exact tracking, 300 env steps)

| model | success | collision | off-road | route completion |
|---|---:|---:|---:|---:|
| BC 200 scenes (`runs/bc_200_v2`) | 0.00 | 0.10 | 0.50 | 0.54 |
| BC + role0/nearest OD (`runs/bc_odnear`) | 0.40 | 0.40 | 0.10 | 0.69 |
| BC + DAgger100 (`runs/bc_dagger_100`) | 0.50 | 0.40 | 0.10 | 0.70 |
| BC + TTC readout (`runs/bc_ttc`) | 0.40 | 0.30 | 0.10 | 0.75 |
| Stage1: c0 + DAgger200 (`runs/p1_s1`) | 0.55 | 0.20 | 0.25 | 0.79 |
| Stage1 LQR canary (`runs/p1_s1_lqr`, 10 scenes) | 0.60 | 0.00 | 0.20 | 0.84 |
| mix c0+c500+DAgger (`runs/p1_mix1`) | 0.35 | 0.10 | 0.55 | 0.72 |
| stage2 c500+c1000 (`runs/p1_s2`) | 0.30 | 0.20 | 0.50 | 0.53 |
| replay c0+DAgger200+c500DAgger (`runs/p1_replay_c500`) | 0.55 | 0.35 | 0.05 | 0.77 |
| replay c500 LQR canary (`runs/p1_replay_c500_lqr`, 10) | 0.60 | 0.20 | 0.10 | 0.81 |
| safety-weighted replay (`runs/p1_weight`, 50 val) | 0.64 | 0.26 | 0.08 | 0.82 |
| safety-weighted LQR (`runs/p1_weight_lqr20`, 20 val) | 0.55 | 0.20 | 0.05 | 0.79 |
| stronger safety weights 8/4 (`runs/p1_weight_hi`, 50 val) | 0.70 | 0.16 | 0.10 | 0.86 |
| stronger safety weights 8/4 LQR (`runs/p1_weight_hi_lqr`, 20 val) | 0.55 | 0.10 | 0.15 | 0.79 |
| stronger safety weights 8/4 LQR (`runs/p1_weight_hi_lqr50`, 50 val) | 0.56 | 0.12 | 0.18 | 0.79 |
| 8/4 LQR + curvature feedforward (`runs/p1_weight_hi_lqr_ff50`, 50 val) | 0.54 | 0.06 | 0.22 | 0.81 |
| safety weights 12/6 (`runs/p1_weight_hi2`, 50 val) | 0.54 | 0.06 | 0.18 | 0.81 |
| Expert PurePursuitIDM (`runs/expert_val20.json`, 20 val) | 0.80 | 0.05 | 0.05 | 0.95 |

Stage1 exact by geometry: straight 0.69 success / 0.15 coll / 0.15 off; curve 0.29 success / 0.29 coll / 0.43 off. LQR canary is a controller-interface canary only, not a final performance claim.

### Data audit (2026-10-20)

- Full 2k BC collection: 137,831 rows / 2,000 episodes / 2,000 unique seeds. Expert success **70.2%**, collision **15.6%**, off-road **5.8%**, timeout/no-flag **8.3%**. The older “80% / 5% on 20 val” number is a small-sample slice (95% half-width ~±18 pp), not an expert ceiling.
- Current DAgger labels are open-loop kinematic-bicycle lifts. On 800 matched initial states, lift minus measured BC label is **+0.53 m dx** and **+1.12 m/s v_next** on average; `a_long` is censored at -3.0, and there are no full-stop/stop-to-go rows. DAgger must be re-collected with measured expert 0.5 s replay, residual, source and validity fields.
- Current training mix has 800 duplicate scene seeds across BC and DAgger shards; the episode-level 10% internal val split can leak 28.5% of val rows by seed. The external 50-scene val seeds are disjoint, but the final split must be by `scene_seed` (Val500/Test500).
- Current PPO sample-vs-executed clamp rate is **61.8% any-dim** (probe on `runs/p1_weight_hi`); log-prob is for the pre-clamp sample while reward/next state use the clamped action. This is a wiring probe, not a valid PPO result.
- Calibrated action quantiles (BC 2k): dx p99.9 6.94, dy p0.1/p99.9 -0.79/+0.57, dpsi -0.28/+0.18, v_next max 13.90. Provisional new bounds: dx [0,7.5–8.0], dy [-1.5,1.5], dpsi [-0.8,0.8], v_next [0,15]. Hard feasibility candidate: a_long [-10,3.5], |a_lat|≤8, |κ|≤0.345, |κ_rate|≤0.20, |jerk|≤20/5. All remain provisional until P0/LQR calibration is logged; see `docs/decision_log.md`.
- Current code path is still the legacy schema/3D execution; schema v2 and the 4D speed-profile path are being implemented. Do not claim P1/P2 pass from the numbers above.

### Clean schema B/C progress (2026-10-20)

- 2k v2 expert collection: 4 x 500 shards, 137,764 rows, 2,000 seeds; expert success 70.6%, collision/off-road/other per episode close to legacy; PSTL clean-gate pass 95.6–96.1%.
- Clean 2k BC with learned `q_lane`/`q_od`: the first model had a real query bug — `q_lane` attention was near-uniform (current lane 0.011; a lat=-7 m lane got 0.60), so even straight driving drifted off-road. Added query-selection auxiliary supervision (current lane role0; urgent OD) and retrained. This is a change to auxiliary supervision only; current lane/TTC are still not plan inputs.
- Same 50-val exact protocol: clean 2k BC after query supervision = **8% success / 14% collision / 80% off-road / 0.46 RC** (still far below legacy 70/16/10).
- DAgger v2 round 1 (2k scenes, 50% measured expert replay): clean model = **50% success / 22% collision / 18% off-road / 0.755 RC**. BC+DAgger mix was worse (42/26/30), so DAgger-only is the current clean best.
- DAgger v2 round 2 (2k scenes, 30% replay) trained from the round-1 model: same 50-val exact = **64% success / 12% collision / 12% off-road / 0.831 RC**. This is `runs/bc_v2_2k_q_dagger_r2`.
- Val500 exact (same first 500 val seeds, paired by scene):
  - legacy `runs/val500_legacy`: **64.0% success / 19.2% collision / 12.2% off-road / 0.853 RC**;
  - clean round-2 `runs/val500_clean`: **58.8% / 16.8% / 18.2% / 0.819**;
  - paired bootstrap diff (clean - legacy): success **-5.2 pp** (95% CI -9.2..-1.2), collision **-2.4 pp** (-5.8..+1.2), off-road **+6.0 pp** (+2.8..+9.4).
- **A/B gate not passed**: clean has better collision but worse success and off-road. Do not promote clean over legacy. This is not a 50-scene noise artifact.
- Hard/curve round 3 DAgger (500 hard train scenes, 30% replay) trained from round-2: 50-val exact = **64% / 8% collision / 18% off-road / 0.834 RC**; collision improved, off-road/success did not. Val500 exact: **59.6% success / 15.8% collision / 16.6% off-road / 0.827 RC**.
- Val500 paired diff (clean round-3 - legacy): success **-4.4 pp** (95% CI -8.6..-0.2), collision **-3.4 pp** (-7.2..+0.2), off-road **+4.4 pp** (+1.2..+7.8). The A/B gate still fails on success/off-road; collision is not worse. Clean is **not promoted**.
- LQR controller-realized canary on clean round-2 (50 val): **26% success / 4% collision / 18% off-road / 52% timeout**, versus legacy LQR ~55/10/15. The exact→LQR gap is unresolved; do not use exact numbers as final controller-realized claims. The likely cause is speed/profile tracking + timeout on curves, not a clean planner conclusion.
- 50-scene numbers are smoke/regression evidence only; 500-scene paired numbers are the gate.
- PPO v2 contract canary is implemented (`pipeline/train_ppo_v2.py`): bounded TruncatedNormal, sampled-vs-executed separation, per-episode GAE with truncation bootstrap, reward/cost separation; tests 89 passed / 1 skipped. A 1-update CPU canary (`runs/ppo_v2_canary`) produced boundary_rate 0.0, logprob recompute diff 0.0, ratio-at-old 1.0, mask/postprocess 25% (one infeasible transition); it is a contract canary, not a performance or P3 gate pass.
- The 10k collection was stopped in its unsharded form because the parent process aggregated all episodes and exhausted RAM/swap. 2k is sharded and stable; final 10k needs shard-stream training or a disk-backed dataset. This is a hardware/data-path issue, not a schema result.

A naive fine-tune on c500+c1000+c1500 degraded closed-loop performance (30% success) despite low val loss. Root cause: no replay of the c0/DAgger domain and no DAgger coverage on the new maps. Adding c500 DAgger with explicit c0/DAgger replay restores 55% exact / 60% LQR. Safety weighting (braking and near-lead rows) improves exact success from 64% (3x/2x) to 70% (8x/4x) on 50 val; LQR is 56% success / 12% collision / 18% off-road on 50 val; curvature feedforward reduces collision to 6% but raises curve off-road to 39%, so the Exact→LQR gap is primarily curve tracking/off-road, not straight-line or speed control. Pushing to 12x/6x cuts collision to 6% but collapses success to 54% and raises off-road to 18%, and LQR success to 30%: over-braking is the failure mode. 8x/4x is the current P1 trade-off. The expert is 80% success / 5% collision on 20 val, so collision avoidance and the Exact→LQR gap remain the main gaps.

## Root causes found and fixed

1. **LD had no lateral error signal**: current-lane points were sampled at ego projection, so `dy=0`; added explicit `lat_offset` feature.
2. **4D action geometry dropped `dy`**: old `(ds,dpsi)` arc discarded lateral offset; replaced with cubic Hermite through `(dx,dy,dpsi)`.
3. **Plan head averaged all LD tokens**: current-lane signal was diluted; added role0 readout + lat0.
4. **Plan head averaged all OD tokens**: nearest vehicle diluted; added nearest-OD and min-TTC readouts.
5. **LQR canary used the unvalidated 3 s preview**: Stage A1 should use only the current 0.5 s segment; fixed.

## Running / current state

- `tools/monitor_tasks.sh`: 20 min task/resource monitor, `runs/monitor/tasks_20m.log`; independent of session.
- Data: `datasets/bc_2k_c0..c1500` (4 x 500 scenes) complete; `datasets/dagger_*` complete but are lift labels and have duplicate scene seeds with BC shards. They are legacy diagnostics, not final training data.
- Current P1 baseline: `runs/p1_weight_hi` (70% exact success / 16% collision / 10% off-road on 50 val; LQR 55/10/15 on 20 val; LQR+FF 54/6/22 on 50 val).
- Safety-first alternative `runs/p1_weight_hi2` (54% exact / 6% collision / 18% off-road; LQR 30% success) is not promoted.
- Expert baseline: 80% success / 5% collision / 5% off-road on 20 val; full 2k collection is 70.2% / 15.6% / 5.8% / 8.3% timeout.
- PPO canary: `runs/ppo_canary_smoke` and `runs/ppo_canary_2u` prove wiring only; 61.8% any-dim sample/execute mismatch; not a PPO result.
- Current state: no training/collection process is active. Clean/legacy Val500 exact are complete; clean A/B is **not promoted**. 10k/500/500 + Test500 remains the final data scale; the unsharded 10k collector was stopped because it aggregated all episodes in one parent process and exhausted RAM/swap. Shard-stream training is required before 10k.
- Next root fixes before more training: LQR speed/preview calibration and exact→LQR failure attribution; off-road failure buckets on Val500; failure-prioritized DAgger on actual train-scene failures (not only geometry buckets); 10k shard-stream loader; Test500 only after a P1/P2 gate passes.
- 200-scene clean smoke (`runs/bc_v2_200`, 30 epochs) completed: val physical MAE dx 0.067 m, dpsi 0.010 rad, dvy 0.18 m/s; closed-loop 10 val exact = 0/10 success, 90% off-road with the current fallback projection. This is a small-data smoke, not an A/B result and not promoted.
- 2k clean runs: `runs/bc_v2_2k_q2` (BC query-supervised), `runs/bc_v2_2k_q_dagger` (round-1 DAgger, current clean best), `runs/bc_v2_2k_q_mix` (worse mix), plus 50-val metrics in `runs/*_eval50`.
- Stage numbering: P1 single-frame BC; **P2 = real history + DAgger**; P3 short PPO canary; P4 WM. Do not start P4 before the P1/P2 gates pass.

## Reproduce (current machine)

```bash
# environment already created; use wrapper for glvnd
tools/venv-python -m env.scenario.cli --n-train 10000 --n-val 1000 --slice 200 --workers 4 --no-validate
tools/venv-python -m dataset.collect --specs env/specs/scenarios_train_slice200.json --limit 200 --workers 4 --max-steps 600 --out datasets/bc_200
tools/venv-python -m pipeline.train_bc --data datasets/bc_200_v2 datasets/dagger_100 --out runs/bc_dagger_100 --epochs 20 --batch-size 256 --lr 5e-4 --device cuda
tools/venv-python -m pipeline.eval_exact --specs env/specs/scenarios_val_slice50.json --ckpt runs/bc_dagger_100/best.pt --out runs/eval --limit 50 --max-steps 600 --device cuda
tools/venv-python -m pipeline.eval_lqr --specs env/specs/scenarios_val_slice50.json --ckpt runs/bc_dagger_100/best.pt --out runs/eval_lqr --limit 20 --max-steps 600 --device cuda
tools/venv-python -m pipeline.train_ppo --ckpt runs/p1_weight/best.pt --specs env/specs/scenarios_train_slice200.json --out runs/ppo_canary --updates 2 --episodes 2 --max-steps 200 --device cuda
```

## Known gaps

- Value/risk heads are not yet trained.
- WM, PPO, imagined RL are not implemented yet.
- Traffic-light/toll/cone scenarios exist in specs but the observation builder's static-object path needs a dedicated check.
- The expert label in DAgger is a kinematic-bicycle lift, not a replayed expert 0.5 s rollout; audit it against a small replayed sample.
- Test set is not yet separated from validation in the current run; use `scenarios_val_slice50.json` only for smoke until the Val/Test split is frozen.
- No replay buffer in `train_bc`; fine-tuning on a new map chunk can forget the c0/DAgger domain. Use explicit replay or DAgger per domain.
- Exact tracker currently uses the geometric segment; `v_next` is used as a label but not yet enforced as a separate speed profile in the tracker/LQR.
