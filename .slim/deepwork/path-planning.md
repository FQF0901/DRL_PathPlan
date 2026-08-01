# Geometric Path Planning — Deepwork Progress

## Goal
From ego pose to TargetPose: collision-free geometric path planning for APA horizontal parking, covering all 5 extracted scenarios.

## Deliverables

| File | Lines | Purpose |
|---|---|---|
| `mdf_reader.py` | 469 | Signal reading layer (MdfData, TimePointData) |
| `viz_env.py` | 298 | Visualization + `draw_path()` for trajectory overlay |
| `collision.py` | 185 | AABB + SAT collision detection |
| `extract_scenarios.py` | 245 | Phase 1: scenario extraction CLI |
| `path_planner.py` | 303 | Phase 2: geometric path planning engine |
| `.slim/deepwork/path-planning.md` | — | Progress tracking |

## Signal Mapping

| Concept | Channel | Notes |
|---|---|---|
| Selected SlotType | `SA_Ego_PS_SlotType_NU` | 2=horizontal |
| Parking Zone | `Psi_PrkgInfo.PrkgZone_u8` | 0–11, >=8 near target |
| Trigger | `HAS_Event_Request_Trajectory_Trigger` | Rising edge |
| Parking Slot | `Rte_Irv_..._HI_ZF_Fusion_Parking_Slot_Set_IRV` | Structured dtype |
| OD | `SIFOR1_Valid_Fusion_Object_Set` | 64 max |
| FSD | `SIFOR1_Valid_Free_Space_Boundary_Set` | 64 max |
| EgoPose | `EPE_Global_Estimated_X/Y/Yaw` | Global (but visualized in ego frame) |
| TargetPose | `HAS_Selected_Target_Position_X/Y/Yaw` | In ego frame at trigger moment |

## Coverage Results (post-fix: 4/5)

| Scenario | File | TP (ego frame) | Result | Collision | Seg≥0.2m | \|R\|≥Rmin | End Err |
|---|---|---|---|---|---|---|---|
| 139 R11A03 | `...139.mf4` | (-0.66, 0.06, -1°) | **1把 reverse 0.66m** | PASS | 0.66m ✅ | inf (straight) ✅ | 0.06m ✅ |
| 021 R12A02 | `...021.mf4` | (0.03, 0.00, -4°) | 0把 (at target) | N/A | N/A | N/A | N/A |
| 060 R12A00 | `...060.mf4` | (-3.30, 0.29, -40°) | **NO-PATH** | — | — | — | — |
| 062 R12A00 | `...062.mf4` | (-0.61, -0.14, 12°) | **1把 reverse 0.62m** | PASS | 0.62m ✅ | inf (straight) ✅ | 0.01m ✅ |
| 063 R12A00 | `...063.mf4` | (0.58, -0.01, 9°) | **1把 forward 0.58m** | PASS | 0.58m ✅ | inf (straight) ✅ | 0.06m ✅ |

## Bug History

| Bug | Found by | Fix | Impact |
|---|---|---|---|
| `_compute_arc` y-consistency sign (P0) | Oracle review #2 | Unified signed formula `center=(x-Rsinθ, y+Rcosθ)` | Right-turn arcs were all rejected |
| `_try_single_arc` dead loop (P1) | Oracle review #2 | Single call, no iteration | Useless 150x loop |
| `_try_two` early return (P1) | Oracle review #2 | Collect all candidates | Missed better multi-segment paths |
| `_sample_line` ignores direction (P0) | Oracle review #3 | Added `reverse` parameter | Reverse-path sampled forward (wrong region for collision check); fixed scenario 139 → NO-PATH→PASS |

## Parking Planning Experience Summary

### 1. All data is in ego-vehicle frame
At the trigger moment, FSD, OD, PS, and TP are all expressed relative to the vehicle's current pose. The ego is at (0,0,0) with heading 0. This eliminates coordinate transforms for planning — the path planning problem is a simple "go from origin to a relative target."

### 2. Most scenarios are final adjustments, not full maneuvers
With PrkgZone>=8, the vehicle is 0.03–0.66m from the target (except 060 at 3.3m). These are trajectory REPLANNING triggers during an already-executing maneuver, not START-OF-PARKING triggers. Straight-line or gentle-arc paths work in most cases.

### 3. Circular-arc-only connecting is geometrically restrictive
A circular arc between two arbitrary poses (x1,y1,θ1) and (x2,y2,θ2) has a unique center that must satisfy both the x-equation (`R = (x2-x1)/(sinθ2-sinθ1)`) AND the y-equation (`y1+Rcosθ1 == y2+Rcosθ2`). These rarely agree for real pose pairs. For most maneuvers, NO single arc connects the start and end poses — the algorithm must fall back to multi-segment strategies.

### 4. Two-segment paths work when segments are independent
The key insight: the vehicle can STOP between forward and reverse movements and CHANGE the steering angle. This means the two arcs DON'T need tangent continuity — each arc has its own center. A grid search over intermediate poses (x,y,yaw) finds feasible transitions.

### 5. The most effective strategy is "fallback"
The planner tries strategies in order of preference:
1. **Already-at-target** (dist<0.1m) → zero-length success
2. **Straight line** (heading change<20°) → forward or reverse
3. **Single arc** → rare but optimal when it exists
4. **Two-segment grid search** → most common for real parking
5. **Three-segment** → forward straight + two-segment

### 6. Remaining gap: scenario 060
The 3.3m, -40° heading-change case has no feasible circular-arc path (single or double) with radius ≥ Rmin=5.4m. This is a GENUINE geometric constraint — the physical min-turn radius prevents connecting these specific pose pairs with only 1-2 arcs. A full Reeds-Shepp implementation (3-5 segments, including cusp maneuvers) would be needed.

### 7. What a production system would need
- **Reeds-Shepp curves** (6 primitives, 48 words) for complete pose-to-pose coverage
- **Collision-aware pruning** (already implemented with AABB+SAT)
- **Continuous curvature** (clothoids for smoother steering)
- **Hybrid A\*** for obstacle-aware global planning
- **State machine** to distinguish START triggers from REPLAN triggers
