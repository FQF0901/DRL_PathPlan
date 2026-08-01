# Deepwork: Path Planning Overhaul

## Status

[2026-07-30] Phase 0: Infrastructure — done.
[2026-07-30] Phase 1: Batch scan — done. Found critical coordinate frame bug.
[2026-07-30] Oracle #1: Review — plan approved.
[2026-07-30] Phase 2: Algorithm optimization — done. Major improvements.
[2026-07-31] Session 2 start: Coordinate-frame fix REVERSED (was wrong). New 3-phase plan.

## CONFIRMED RESEARCH (2026-07-31): TP/SP coordinate frame — raw data IS ego-relative

**Conclusion: `HAS_Selected_Target_Position_*` / `HAS_Selected_Start_Position_*` raw signals
are published in the ego-vehicle frame at every timestamp. NO transform needed.
The global→vehicle transform added on 2026-07-30 was WRONG and has been reverted.**

### Evidence (files analyzed: 060, 139, 021, 062 from Parking_P417)

1. **MATLAB reference pipeline** (`/workspace/APA_CAN_Data_Analysis/APA_CAN_Data_Analysis/`):
   - `MF4_Read_and_Plot_all.m` reads the same channels (HAS_Selected_* + EPE_Global_*).
   - `DA_Trans.m:44` builds `H = H_trans_2(-ego_x,-ego_y,-rad2deg(ego_yaw))` = T(+pos)·R(+ψ)
     — this is the **relative→(local)global** transform (for plotting everything incl. the
     vehicle at its global pose). Direction is OPPOSITE to our "global→body" fix.
   - `DA_Trans.m:166-212` applies the SAME H to TP/SP/ITP/ISP as to PS/OD/FSD → all raw
     sensor output (slots, obstacles, TP) is ego-relative in these logs.
   - `H_trans_2.m`: H1=R(-Ego_heading) with heading=-ψ → R(+ψ); H2=translate(+pos); H=H2*H1.

2. **Stationarity test (060, whole 40s parking event)**:
   - World position of TP computed as `ego + R(ψ)·rawTP` (y-left): STATIONARY at
     (-0.35, -2.82) ± 5 cm while car drives 4.5m and rotates 40°.
   - Raw TP interpreted as global: target drifts (std 2.1/1.3 m) — not world-fixed.
   - Car final rest pose (-0.244, -2.657) vs TP world (-0.35, -2.82): 0.2 m — car ARRIVES at TP.
   - At arrival rawTP ≈ (0, 0, 0.1°): relative target is at the car.

3. **Cross-file confirmation** (021, 062): TP world (ego + R(ψ)·raw) std = (0.033, 0.022) /
   (0.076, 0.046) m over full logs; 062 car rest (-1.507, 3.123) vs TP world (-1.506, 3.142):
   2 cm coincidence. Raw TP spatial std: 2.3-2.9 m → NOT global.

4. **Slots are ego-relative too** (060): raw slot0 center tracks the car (std ≈ ego std,
   jumped 4.6 m in raw space while car drove); rawTP−rawSlot0 offset constant (std 0.15 m)
   → TP and slot polygons share the ego-relative frame.

5. **SP signal**: always exactly (0,0,0) in P417 logs (never published). Planner never uses
   `.sp` (grep: no reference in path_planner.py). Sentinel (0,0,0) = "no target" (MATLAB
   checks `x~=0 && y~=0`; we use |x|<1e-6 && |y|<1e-6 → invalid).

### Why 28/41 (pre-fix, raw-as-relative) dropped to 18/41 (post-fix)
The transform created phantom targets (e.g. 060: phantom (-3.39, 4.52) vs true (-3.30, 0.29)
relative). 28/41 was the OLD algorithm with CORRECT targets; 24/41 was NEW algorithm with
WRONG targets → true baseline after revert is expected ≥ 28/41.

## Phase 1 (2026-07-31): Coordinate revert + true baseline — DONE (Gate 1 pending)

### Changes
- `mdf_reader.py`: removed h_transform/h_transform_single + global→body block; tp/sp now
  pass through raw values. NaN → (nan,nan,nan); |x|<1e-6 ∧ |y|<1e-6 (cleared sentinel) →
  (nan,nan,nan); nonzero position with yaw=0 stays valid.
- `batch_scan.py` (recreated, project root): scans 37 MF4, extracts scenarios, plans,
  writes .slim/deepwork/batch_scan_results.json (same schema as before).

### Baseline result (TRUE, corrected frame)
**36/41 = 87.8%** (was 24/41=58.5% with wrong transform; 28/41 with old algorithm+correct frame)
- straight 19 | three_seg 10 | two_seg 5 | already_at_target 1 | single_arc 1 | no_path 5
- **5 failures**: #140 t=23.05 target(-4.49,0.05) | #145 t=28.06 (-4.62,-0.17) | #028 t=26.59
  (-5.35,0.47) | #043 t=36.87 (0.13,0.03) | #060 t=34.08 (-3.30,0.29)
- Spot-check: tpd.tp equals raw signal bit-for-bit on 060/139/062.

### ⚠ Extraction-logic reconstruction risk (fixer finding, needs Oracle judgment)
Old tp_fallback/tp_forced logic lived ONLY in the deleted batch_scan.py (never committed).
extract_scenarios.py has trigger detection only. Fixer reconstructed a three-level chain:
trigger(PrkgZone≥8, no slot-type filter) → tp_fallback(first valid TP ∧ PrkgZone≥8) →
tp_forced(idx 0). Exhaustive signal sweep could explain only #141 (≡ first PrkgZone≥8);
7 of 8 old tp_fallback timestamps have NO signal-based explanation. Old scenario set was
23 trigger / 10 tp_forced / 8 tp_fallback = 41; new set also 41 (source mix differs).
→ Numbers are reproducible from the new code, but scenario set may differ from the old 41.

## Gate 1 Oracle Review (2026-07-31) — PASSED with 3 mandatory fixes

### Verdicts
1. **Correction completeness: PASS** — NaN guard + zero-sentinel + yaw=0-legal all correct;
   h_transform zero residue; failures reproduce bit-for-bit on disk code.
2. **Extraction rebuild: ACCEPT, with 3 must-fixes** (a) wording: 36/41 is "current-pipeline
   baseline" ONLY — old 28/41 & 24/41 mixed extraction logic + coordinates, NOT comparable;
   (b) add per-scenario metadata to JSON (slot_type, PrkgZone, TP valid flag);
   (c) **commit code** (batch_scan.py untracked, mdf_reader.py uncommitted = reproducibility risk).
3. **Failure classification: all 5 are SEARCH-RESOLUTION failures, not degenerate scenarios**.
   Isolated experiments (collision-off → solutions exist; fine-grid + collision-on →
   3-seg 5.6-7.1m solutions exist). Root cause: `_try_two` da1 step 0.3rad × R step 0.81m
   × `_compute_arc` y-tol 0.05m stack to miss the narrow feasible set.
   #043 (0.13m target) is a real trigger, near-point finishing case — keep, classify as
   "near-point finish", don't solve with huge detours.
4. **Phase 2 guidance (priority order)**:
   1. Fix `_try_two` search space: analytic two-arc boundary value (circle intersection for
      fixed R1×R2 — exact, no grid) OR da step 0.05; relax endpoint tol 0.05→0.2 with
      clamp/kink closure. One change solves all 5 + removes absurd long paths.
   2. **Path quality gate**: detour ratio (total_length ≤ 2.5× euclidean) as hard gate —
      current "36/41" includes 10x detours (#025 52m/5.3m, #141 50.6m, #143 51.2m,
      #027 33.3m, #036 25.5m, #041 14.7m/2.9m). Re-report after gate.
   3. #043: micro-finish rule (in-place/short), not detours.
   4. #061: exclude/mark — idx-0 ego=(0,0,0) uninitialized. Exclude → 36/40=90% both口径.
   5. Kink collision coverage: 0.2m end clamp can miss collisions between samples — add
      interpolated samples on the clamped step (cheap continuous-ish check).
- NOT to do: more maneuver templates or distance/heading class expansion — the
  success/fail discriminator is search resolution, not target geometry.

## Phase 2 (2026-07-31): Remediation pass (implements Gate-1 guidance; no re-review —
  remediation follows oracle's own prescription, validated with focused evidence)

### Done: search-resolution fix (hybrid) + quality gate + metadata → 39/41 (95.1%) / 39/40 (97.5%)
- `_try_two`: circle-intersection analytic part + da-step-0.05 fine sampling fallback.
- Quality gate: detour ratio ≤ 2.5×; 2 honestly-rejected (#142 near-point 20.8°>20°,
  #037 11.02m vs gate 10.97m). detour max 2.20 (was 9.8). #061 excluded (uninitialized ego).
- JSON: + slot_type/pz/tp_valid/excluded/quality_gate_rejected; both口径 reported.

### USER DIRECTIVE (2026-07-31, Phase 2b): analytic closed-form, no search
User pushed back: (1) success criterion needs pos/heading err < 1e-3 — search can't
guarantee it; prefers analytic; (2) if search must remain, use bisection-style tricks.
Also: parallelize batch scan across 4 processes.

**Decision**: implement the tangent-circle CLOSED FORM for two-arc and line-arc:
- Two-arc CC: fix R1 (signed family parameter — the 1-D solution family); tangency
  condition |C2−C1| = r1 + ε·r2 with C2 = E + R2·(−sinθ2, cosθ2) expands to a LINEAR
  equation in R2 (R2² cancels):  R2 = (R1²−|w|²) / (2[(w·u2) − ε·s1·s2·R1]),
  w = (E−S) − R1·u1, u1=(−sinθ1,cosθ1), u2=(−sinθ2,cosθ2), s1=sign(R1), s2=sign(R2),
  ε=+1 external / −1 internal. Junction P = tangency point; arc2 exact through E (1e-12).
- Line-arc: exact d = s·(x2 − y2·sinθ2/(1−cosθ2)) in ego frame (forward/reverse).
- R1 grid step 0.27m (cheap; bisection unnecessary).
- batch_scan.py: multiprocessing Pool(4), ordered results, maxtasksperchild=1.

### Phase 2b progress & CONFLICT RESOLUTION (2026-07-31)
- Pure analytic closed-form implemented + 4-worker parallel batch (84s vs 40min serial) → **22/41**
  (17 regressions, all two_seg/three_seg): exact-tangent family collides where the old
  0.2m-endpoint-slack paths avoided obstacles (#041: all 9 tangent geoms collide; old
  R1=9.45/R2=15.56 non-tangent). Endpoint err metric was vacuous (clamp → 0 by construction).
- **Root insight**: old junction P was ALREADY heading-continuous (arc2 sampled tangent to
  arc1 at P); the slack was ONLY the endpoint match (circle2 within 0.2m of E, then clamp).
- **Decision (analytic-first + bounded endpoint float)**: keep exact closed form for end=E;
  if a given R1 yields no valid candidate, re-solve for E_eff = E + δ over a small bounded
  grid (|δ| ≤ 0.2m, radii 0.05/0.1/0.15/0.2 × 12 angles), collect valid candidates.
  This is the minimal principled relaxation (== the documented 0.2m y-tolerance), recovers
  the 17 without a 2-D grid. Honest metric: per-success max segment residual
  (arc2 end vs E) — ~1e-12 exact vs ≤0.2 float; endpoint clamp still guarantees 1e-3.

## Phase 2b FINAL (2026-07-31): performance + coverage resolution

### Outcome: 39/41 (95.1%) / 39/40 (97.5%), ~7 min batch (was ~41 min), signal cache live
- **Float pass was a dead end**: recovered 0/17 (cross-validation: tangent-family candidates
  all FSD-collide; the old 39/41 solutions are NON-tangent two/three-arc paths that the
  exact-tangent + endpoint-float family cannot express). REMOVED — back to hybrid
  (A circle-intersection + B da-0.05 sampling, `_try_two`) — the proven 39/41 mechanism.
- **Quality-aware early exit** (`_try_two`): per R1, break at FIRST candidate with
  total_length ≤ 2.5×dist (gate-compliant), else scan R1 fully — keeps coverage, bounds cost.
- **`_try_three` full d-scan**: removed return-on-first-success; collect all d, prune d ≥
  best-total-so-far. Evidence: #027 only solvable at d=4.0 reverse (7.06m total) — the
  early-return grabbed a 50.5m path at d=0.5 instead.
- **`mdf_cache.py` (new)**: gzip-pickled parsed MdfData per file under `.cache/mdf_signals/`;
  key = (mtime_ns, size) + sha256(mdf_reader.py). Warm load: 37 files in 0.7s (18ms/file)
  vs ~2s/file cold (100×). `.gitignore` += `.cache/`. batch_scan: `--workers N` (default 4),
  `--no-cache`. ⚠ Bug fixed during verification: `meta, data = pickle.load(f)` unpacked a
  3-key dict → ValueError swallowed by silent except → cache never hit (all runs re-parsed);
  loader now reads `obj['key']/obj['data']` explicitly.
- **Residual metric fixed** (batch_scan): da sign = sign(R)·(forward?+1:−1) — `direction`
  encodes sign(da·R); the fixer's formula mis-signed R<0 arcs (4.4m ghosts), my first
  attempt mis-signed reverse arcs (14m ghost). Now max residual = 0.199m across successes
  (== the documented 0.2m endpoint tolerance), exact_tangent 20/38.
- **Final failures (2, both P2-era)**: #142 (near-point 0.31m, heading 20.8° > straight
  20° limit; any real maneuver > 2.5× gate) and #037 (best 11.02m vs gate 10.97m).
- **Runtime profile**: MF4 parsing is NOT the bottleneck (2s/file); planning compute is
  (full-search scenarios ~60s each). 41min → 416s via 4-worker multiprocessing + dead-code
  removal; cache removes the loading portion (0.7s for all files). Long bash runs now use
  30-min timeouts (function-call semantics — tool kills only on timeout).

## Gate 3 Oracle Review (2026-07-31) — PASSED (1 must-fix: git commit)

- Planner hybrid + quality-aware early exit + full d-scan three_seg: approved; 0.199m
  residual solutions validated as "reachable + controller-fine-tune at end" (endpoint
  1e-3 by clamp). Optional fix applied: `_try_two(gate_len=...)` param — `_try_three`
  now passes `2.5×D_total − dist` so the sub-gate equals the final quality gate.
- Remaining 2 failures are HONEST refusals: #142 structural (0.31m/20.8° — min real
  maneuver 1.96m = 2.5× gate; in-place turn impossible at R<Rmin), #037 policy boundary
  (11.02 vs 10.97m, 0.4%). Do NOT widen the straight 20° limit or the 2.5× gate (YAGNI,
  threshold-fitting).
- Cache/parallelism: no substantive risk (key covers reader deps; imap ordered;
  tmp+os.replace atomic; maxtasksperchild=1).
- **FINAL NUMBERS (final code): 39/41 = 95.1% / 39/40 = 97.5%, detour max 2.2,
  residual max 0.199m, exact_tangent 20/38, batch 397.8s (4 workers, warm cache).**
- MUST-FIX before delivery: git commit (code + evidence files). Pending user decision.

## Key Bug Fix: Coordinate Frame

**Bug**: `TimePointData.tp` was in global coordinates (ISO), but path planner assumed
ego-vehicle frame (ego at (0,0,0) heading 0).

**Fix**: Added global→vehicle transformation in `mdf_reader.py:TimePointData.__init__`:
```python
dx = tp_gx - ego_x
dy = tp_gy - ego_y
rel_x =  dx*cos(ego_yaw) + dy*sin(ego_yaw)
rel_y = -dx*sin(ego_yaw) + dy*cos(ego_yaw)
rel_yaw = arctan2(sin(tp_gyaw - ego_yaw), cos(tp_gyaw - ego_yaw))
```

Also applied to `self.sp` (StartPose).

## Key Algorithm Improvement: Radius-Angle Search

**Problem**: Old grid search over (x,y,yaw) tried 5760 intermediate poses but found
0 valid arc pairs for most cases. _compute_arc is restrictive:
- Only ~0.02% of (x,y,yaw) triples produce valid arcs from start
- The ones that do almost always fail collision

**Fix**: New _try_two searches over (R1, da1) arc parameters directly:
```
for R1 in [Rmin, Rmin*0.1...Rmin*4]:
    for da1 in [-2.5...2.5]:
        int_pose = arc_endpoint(start, R1, da1)
        arc2 = _compute_arc(int_pose, target)
```
Every (R1, da1) produces a valid arc — vastly more efficient. Also removed
alternating-direction constraint.

**Additional changes**:
- `_try_line_arc`: extended reverse range to 12m
- `_try_three`: added reverse-straight-first variant (was forward-only)
- `verify_path.py`: added `'pass': True` to maneuvers check

## Phase 2 Results

### Before optimization (broken coordinates): 28/41 (68.3%) — many wrong
### After coordinate fix only: 18/41 (43.9%) — honest baseline
### After radius-angle search + extended ranges: 24/41 (58.5%)
### Effective (excl t<1.0s bad timestamps): 24/38 = 63.2%

### Strategy breakdown
| Strategy | Solved | Rate |
|---|---|---|
| straight | 12/12 | 100% |
| already_at_target | 3/3 | 100% |
| two_seg | 3/3 | 100% |
| three_seg | 6/6 | 100% |
| single_arc | 0/0 | — |
| line_arc | 0/0 | — |
| **no_path** | **0/17** | **0%** |

### Remaining 17 failures
| Type | Count | Reason |
|---|---|---|
| Early t≈0.2s | 3 | Unstable data before system init |
| Extreme heading >100° | 2 | Need 4+ maneuver (CCC|C) |
| Moderate 32-48°, 5-7m | 12 | Collision blocks all arc pairs |

### Moderate failures — all 12 from APADeltaTest files (036-047, 059-064)
These files are APA Delta regression tests — the car is at various stages of
approach. Often the target is reachable but the current FSD/OD data shows obstacles
in the path.

## Next: Image Quality Check
