# PNP implementation status (P0/P1/P2)

Last updated: 2026-10-20 (Asia/Shanghai)

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
- Tests: `82 passed, 1 skipped` (`tools/venv-python -m pytest -q`).

## Current evidence

**P0 rerun status:** Nav history units, eval/PPO decision-state parity, exact steering, LD stable-ID/OD acceleration units, and feasibility/label_source training weights were fixed and committed `a6b2628` (plus the `w_infeasible`/safety-weight refinement). The exact numbers below are from re-collected 2k data and the fixed evaluator. LQR remains controller-realized diagnostic only.

### 2k expert data (canonical re-collection)

- `datasets/bc_2k_s0..s3`: 4 x 500 scenes, 137,764 rows, 2,000 seeds.
- Expert success 70.6%; PSTL hard-gate pass 95.6–96.1%.
- `expert_bc.npz` is the canonical filename; old `bc_v2_*` data is historical and not used.

### 50-val exact after P0 fixes

| model | success | collision | off-road | RC |
|---|---:|---:|---:|---:|
| BC 2k (`runs/bc_2k`) | 8% | 22% | 70% | 0.42 |
| DAgger r1 (`runs/bc_2k_dagger_r1b`) | 44% | 24% | 32% | 0.733 |
| DAgger r2 (`runs/bc_2k_dagger_r2`) | 50% | 32% | 12% | 0.774 |
| DAgger r3 hard500 + safety weights (`runs/bc_2k_dagger_r3`) | 58% | 12% | 16% | 0.844 |

### Val500 exact (frozen first 500 val seeds; paired by scene)

| model | success | collision | off-road | RC |
|---|---:|---:|---:|---:|
| Legacy fixed evaluator (`runs/val500_legacy_p0fixed`) | 65.0% | 18.2% | 12.0% | 0.854 |
| Clean r3 (`runs/val500_clean_r3_p0fixed`) | 66.2% | 12.2% | 8.6% | 0.859 |

Paired bootstrap (clean r3 − legacy):

- success **+1.2 pp** (95% CI −2.2..+4.6)
- collision **−6.0 pp** (CI −9.4..−2.6)
- off-road **−3.4 pp** (CI −6.0..−0.8)
- stuck +0.4 pp (CI 0.0..+1.0)

**Exact A/B passes the success/no-worse and safety checks.** Clean is not yet promoted as the full P1 controller-realized target because the LQR canary still fails the gap gate.

### LQR controller-realized canary (50 val)

| model | success | collision | off-road | timeout | RC |
|---|---:|---:|---:|---:|---:|
| Legacy fixed LQR (`runs/lqr50_legacy_p0fixed`) | 44% | 8% | 26% | — | 0.776 |
| Clean r3 LQR (`runs/lqr50_clean_r3_p0fixed`) | 34% | 2% | 12% | 52% | 0.740 |

Clean LQR has lower collision/off-road but **10 pp lower success and 52% timeout**; tracking xy 0.016 m, heading 0.008 rad, speed 0.210 m/s. The exact→LQR gap is still a blocker; do not present exact tracking as the final controller-realized result.

### PPO contract canary

`pipeline/train_ppo.py` 1-update CPU canary (`runs/ppo_v2_canary`):

- boundary_rate 0.0; logprob recompute diff 0.0; ratio-at-old 1.0;
- mask/postprocess 25% (one infeasible transition);
- contract only, not a performance or P3 gate claim.

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
2. **BC-to-closed-loop distribution shift**: 2k BC clean was 8% success / 80% off-road on 50 val; measured-expert DAgger raised it to 50% then 64%. This is the dominant P2 issue.
3. **DAgger lift bias**: old kinematic bicycle labels had +0.53 m dx / +1.12 m/s v_next bias and censored braking; the canonical collector stores measured expert replay plus lift/residual/source validity.
4. **Exact→LQR gap**: clean LQR 26% vs exact 64% on 50 val; not yet root-caused to controller speed/preview vs planner feasibility.

## Current state

- No training/collection process is active; 20-minute monitor is running (`runs/monitor/tasks_20m.log`).
- Data: `datasets/bc_2k_s0..s3` (2k expert), `datasets/dagger_2k_r1_s0..s3`, `datasets/dagger_2k_r2_s0..s3`, `datasets/dagger_2k_r3_hard500`.
- Best clean checkpoint: `runs/bc_2k_dagger_r3` (Val500 66.2/12.2/8.6/0.859).
- Legacy reference checkpoint for exact A/B: `runs/p1_weight_hi` with fixed evaluator (`runs/val500_legacy_p0fixed`, 65.0/18.2/12.0/0.854). The checkpoint uses the old schema/model and requires code at tag `pre-consolidation`; the canonical tree cannot load it.
- 10k collection was stopped in its unsharded form: the parent process aggregated all episodes and exhausted RAM/swap. 2k is sharded and stable; final 10k needs shard-stream training or a disk-backed dataset.
- Val500/Test500 specs are frozen in `docs/eval_split_manifest.json` (`env/specs/scenarios_val500.json` / `scenarios_test500.json`). Test500 has not been run; it remains untouched until the LQR/P1 gate passes.
- Stage numbering: P1 single-frame BC; **P2 = real history + DAgger**; P3 short PPO canary; P4 WM. Do not start P4 before the P1/P2 gates pass.

## Reproduce (current machine)

```bash
# Environment already created; use the glvnd wrapper.
tools/venv-python -m env.scenario.cli --n-train 10000 --n-val 1000 --slice 200 --workers 4 --no-validate

# 2k expert data, sharded (avoid parent-process aggregation).
tools/venv-python -m dataset.collect --specs env/specs/scenarios_train.json \
  --out datasets/bc_2k_s0 --offset 0 --limit 500 --workers 8 --max-steps 600
# repeat offsets 500/1000/1500, then concatenate via --data for training.

# DAgger round 2 (student roll-in + measured expert replay).
tools/venv-python -m dataset.collect_dagger \
  --specs env/specs/scenarios_train.json --out datasets/dagger_2k_r2_s0 \
  --offset 0 --limit 500 --ckpt runs/bc_2k_dagger_r1b/best.pt \
  --workers 8 --max-steps 600 --expert-ratio 0.3 --shield-ttc 2.0 --round 2 --device cuda

# Clean BC/DAgger training and closed-loop evaluation.
tools/venv-python -m pipeline.train_bc \
  --data datasets/dagger_2k_r2_s0 datasets/dagger_2k_r2_s1 \
         datasets/dagger_2k_r2_s2 datasets/dagger_2k_r2_s3 \
  --init runs/bc_2k_dagger_r1b/best.pt --out runs/bc_2k_dagger_r2 \
  --epochs 15 --batch-size 256 --lr 1e-4 --w-infeasible 0.2 --device cuda
tools/venv-python -m pipeline.eval --specs env/specs/scenarios_val.json \
  --ckpt runs/bc_2k_dagger_r3/best.pt --out runs/val500_clean_r3_p0fixed \
  --limit 500 --max-steps 600 --controller exact --device cuda
tools/venv-python -m pipeline.eval --specs env/specs/scenarios_val_slice50.json \
  --ckpt runs/bc_2k_dagger_r3/best.pt --out runs/lqr50_clean_r3_p0fixed \
  --limit 50 --max-steps 600 --controller lqr --device cuda

# PPO contract canary.
tools/venv-python -m pipeline.train_ppo \
  --specs env/specs/scenarios_train_slice200.json --ckpt runs/bc_2k_dagger_r3/best.pt \
  --out runs/ppo_v2_canary --updates 1 --episodes 1 --max-steps 20 --device cpu
```

## Known gaps

- Clean A/B fails on success/off-road; do not promote it over the legacy checkpoint.
- LQR clean canary is far below exact and legacy LQR; controller/speed/preview calibration is unresolved.
- 10k/500/500 + Test500 final scale and shard-stream loader are not implemented; no Test500 conclusions.
- Failure-prioritized DAgger on actual train-scene failures is still needed; round 3 used a hard-geometry bucket.
- P4 WM / Imagine Bank / Ego Dynamics and P5 full 3 s LQR/joint PPO are not started.
- Expert audit fields are stored, but the multi-solution downweighting policy is still conservative.
- Comfort violation rate is 100% in current evals under the strict comfort bounds; it is a diagnostic, not yet a training gate.
