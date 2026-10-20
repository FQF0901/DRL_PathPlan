# Decision log

This file records implementation choices that `docs/arch_lock.md` leaves open. It is evidence-oriented: candidates, measured data, choice, and rollback conditions. It is not a review/rebuttal log; debate belongs in `docs/deepseekv4p1_argue.md`.

## 2026-10-20 — 4D action execution formulation

- **Contract:** §6.1–6.5, §7.2, §22.2–5.
- **Candidates:**
  - A: speed-scaled time-Hermite. Endpoint and endpoint velocity exact, but speed is not monotone, geometry/speed conflicts can bulge/loop, and there is no explicit `a_lon`/jerk/curvature gate.
  - B: 2D quintic time polynomial. C2-capable, but path and speed are implicit; zero endpoint acceleration raises peak acceleration; PPO bounds are high-order inequalities.
  - C: **PSTL** — geometric cubic-Hermite path in a shape parameter `u`, plus a cubic arc-length time law `s(tau)` with `s(0)=0`, `s(T)=L`, `sdot(0)=v0`, `sdot(T)=v1`.
- **Chosen:** C. It executes `v_next` exactly, keeps the 4D action and one Plan Head, yields five 0.1 s position/heading/speed points, makes the endpoint/speed conflict explicit (`r_v=v1-v_cons`, `j_req=6r_v/T²`), and supports a monotone-speed check.
- **Interpretation note:** the contract sentence “tangent magnitude from speed boundary” is applied to the **time law**, not the geometric path. The geometric path uses a chord-scaled tangent (`m=rho*||p1||`, `rho=1.0`). This is deliberate and recorded; if the contract owner rejects it, A is the fallback but must be labeled non-monotone.
- **Evidence (expert BC 2k, 34,847 rows, `bc_2k_c0`):**
  - current geometry-only terminal-speed error: mean |e| 0.133 m/s, p95 0.386, p99 1.409;
  - PSTL hard gate (`a_lon in [-10, 3.5]`, `|j| <= 20`) passes 97.3%; failures: jerk 1.6%, a1 0.6%, a0 0.26%, vmin 0.2%;
  - feasible `v_next` interval width p5/p50/p95 = 0.84/1.16/1.27 m/s; width <0.1 m/s = 0.6%;
  - expert `v_next` outside hard interval = 2.3%;
  - `v_lo-v0` median -0.78 m/s, `v_hi-v0` median +0.43 m/s.
- **Provisional constants:** `rho=1.0`; static action box dx [0,7.5], dy [-1.5,1.5], dpsi [-0.8,0.8], v [0,15] (outer); hard gate `CLEAN_LIMITS`: a_lon [-10,3.5], |a_lat|<=8, |kappa|<=0.345, |kappa_rate|<=1.0, |jerk|<=20/20; comfort `CLEAN_COMFORT_LIMITS`: a_lon [-4,2], |a_lat|<=3, |kappa|<=0.20, |kappa_rate|<=0.05, |jerk|<=5/5. These pass ~97.2% of 2k expert rows under PSTL. All values remain provisional until LQR/closed-loop calibration is logged.
- **Rollback conditions:** expert hard-gate pass <95%; feasible v interval p90 width <0.1 m/s; LQR speed/lateral tracking regresses on paired scenes; round-trip ADE >0.1 m after rho calibration; PPO log-prob/ratio consistency fails.
- **Next:** implement `net/action_space.py` + `pipeline/segment_ref.py` + Exact/LQR speed profile; then BC NLL/aux and PPO probability contract. Do not connect the 6-step same-context preview as a 3 s WM/plan claim.

## 2026-10-20 — Data split and bounds

- Data split must be by `scene_seed`, not `episode_id`; current BC+DAgger mix has 800 duplicate seeds and a 28.5% contaminated internal val.
- Current expert 2k baseline: 70.2% success / 15.6% collision / 5.8% off-road / 8.3% timeout. The 80%/5% on 20 val is a small-sample slice and is not an expert ceiling.
- Action bounds above are calibrated from BC 2k quantiles and DAgger outliers; they are not yet LQR-trackability calibrated.
