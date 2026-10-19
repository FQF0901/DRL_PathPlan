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
- `config/`, `test/` (7 tests pass), `net/param_probe.py`, `tools/monitor_8h.sh`.

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
| Expert PurePursuitIDM (`runs/expert_val20.json`, 20 val) | 0.80 | 0.05 | 0.05 | 0.95 |

Stage1 exact by geometry: straight 0.69 success / 0.15 coll / 0.15 off; curve 0.29 success / 0.29 coll / 0.43 off. LQR canary is a controller-interface canary only, not a final performance claim.

A naive fine-tune on c500+c1000+c1500 degraded closed-loop performance (30% success) despite low val loss. Root cause: no replay of the c0/DAgger domain and no DAgger coverage on the new maps. Adding c500 DAgger with explicit c0/DAgger replay restores 55% exact / 60% LQR. Safety weighting (braking and near-lead rows, 3x/2x) further improves to 64% exact / 55% LQR on 50/20 val scenes. The expert is 80% success / 5% collision on the same 20 val scenes, so collision avoidance remains the main P1 gap.

## Root causes found and fixed

1. **LD had no lateral error signal**: current-lane points were sampled at ego projection, so `dy=0`; added explicit `lat_offset` feature.
2. **4D action geometry dropped `dy`**: old `(ds,dpsi)` arc discarded lateral offset; replaced with cubic Hermite through `(dx,dy,dpsi)`.
3. **Plan head averaged all LD tokens**: current-lane signal was diluted; added role0 readout + lat0.
4. **Plan head averaged all OD tokens**: nearest vehicle diluted; added nearest-OD and min-TTC readouts.
5. **LQR canary used the unvalidated 3 s preview**: Stage A1 should use only the current 0.5 s segment; fixed.

## Running

- `datasets/bc_2k`: 2,000 train scenes, 6 workers; log `runs/logs/collect_2k.log`.
- `tools/monitor_8h.sh`: 24 x 20 min resource/task log `runs/monitor/monitor_8h.log`.
- Current P1 baseline: `runs/p1_weight` (64% exact success / 26% collision / 8% off-road on 50 val; 55% LQR success / 20% collision on 20 val).
- DAgger c1000 finished/soon; next train with c0+DAgger replay and safety weighting and compare against `runs/p1_weight`.
- Expert baseline on 20 val: 80% success / 5% collision / 5% off-road / 0.95 rc.
- Next: DAgger on c1000/c1500 if replay training holds; then freeze P1 baseline and run the 50-scene Val slice. 10k/500/500 remains the final scale.

## Known gaps

- Value/risk heads are not yet trained.
- WM, PPO, imagined RL are not implemented yet.
- Traffic-light/toll/cone scenarios exist in specs but the observation builder's static-object path needs a dedicated check.
- The expert label in DAgger is a kinematic-bicycle lift, not a replayed expert 0.5 s rollout; audit it against a small replayed sample.
- Test set is not yet separated from validation in the current run; use `scenarios_val_slice50.json` only for smoke until the Val/Test split is frozen.
- No replay buffer in `train_bc`; fine-tuning on a new map chunk can forget the c0/DAgger domain. Use explicit replay or DAgger per domain.
- Exact tracker currently uses the geometric segment; `v_next` is used as a label but not yet enforced as a separate speed profile in the tracker/LQR.
