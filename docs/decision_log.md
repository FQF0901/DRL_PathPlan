# Decision log

## 2026-10-21 — LQR priority

- **Decision:** 0.5 s LQR is a low-threshold canary/interface check, not the final controller-realized score; full 3 s / 6-point LQR is deferred until after the ego-conditioned latent WM and imagined rollout.
- **Why:** the current evaluator executes only the first 0.5 s; using success/timeout as a performance gate confuses interface and planner quality.
- **Owner:** project contract owner clarification; arch_lock §13/§22.4 remains the source for the final 3 s gate.

## 2026-10-22 — Progression policy

- **Decision:** proceed with Stage B/C/D1/D2 while recording open debt; exact A/B need not be a comprehensive win, and LQR 0.5 s is not a blocker. Exact 67.6% is close to the Expert 73.0% ceiling (episode-level: collision 14.4%, off-road 6.6%, timeout 6.0%).
- **Consequences:** the stage ledger marks A1/A2/B as provisional; no claim that failed-expert/DAgger semantics or checkpoint selection are closed. Test500 remains read-only until the final gate.
- **Data debt:** old r3b/r4/r5/r6 DAgger data predates terminal-failed semantics; terminalfix2 is the replacement lineage.

## 2026-10-22 — WM metric correction

- **Decision:** D1/D2 metrics use per-element latent MSE; `masked_mse` denominator includes the latent dimension. Prior 128×-inflated D1 numbers are invalid.
- **Result:** D1 one-step smoke 0.616 vs persistence 1.051; D2 held-out depth 6 WM od 0.522 vs persistence 0.673. Static LD/Nav/Ego target semantics remain open; not a Stage E gate.

## 2026-10-22 — Checkpoint selection

- **Decision:** open-loop `best.pt` is a training diagnostic, not the selected final model. `pipeline/select_checkpoint.py` uses hard collision/off-road thresholds -> success -> RC, refuses Test500, and refuses to promote when no candidate is eligible. The 0.5 s LQR branch is still open.
