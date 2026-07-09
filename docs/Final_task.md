# Final_task.md — WK9 NEU Defect CNN, remaining work

Content is complete (all 20 questions, Parts 0/A/B/C, present and executed with no errors).

## Done

- **Item 2 (Q15 code/prose mismatch):** Fixed. Rewrote the Q15 discussion cell in the notebook to state the actual code values (`N_TRIALS=4`, `epochs=4`, not the stale 12/8), and added a direct answer to the assignment's question using the real printed numbers: Optuna `val_acc=0.633` vs grid's `0.636` — essentially tied, grid marginally ahead, as expected at this trial/epoch budget. No code cell touched, no re-run needed — this was a markdown-only cell, so nothing else in the notebook is affected.
- **Item 3 (README totals):** Fixed. Recomputed Part A/B/C epoch total from actual code values: Part A (15) + Part B (30) + Q12 grid (60) + Q14 (10) + Q15 (16) = **131**, not 246. Rewrote the runtime line, the Contents section's stale "12 trials" mention, and the speed-tip paragraph, which now correctly points at Q12's grid search (~46% of the total) as the dominant cost instead of Q15 (now the smallest, at ~12%). The new ~76min/~105min figures are labeled as **estimates** (proportional scaling from this notebook's own claimed per-CNN-epoch rate) — not a measurement I made, since I don't have the dataset to actually execute training. Replace with your real local timing once you run it.

## Remaining — you said you'll run locally

- **Item 1 (Q14 execution order):** No code change was needed or made. File order already has `EPOCHS_C = 5` before the StepLR training cell; a clean **Restart kernel → Run All** enforces that order and resolves the mismatch by itself. Expect the Q14 numbers to change from the currently-displayed `0.719`/`0.661` (which were actually 15-epoch runs) to new 5-epoch numbers, and the plot to show 5 points instead of 15 — that's correct behavior, not a regression.

## Verification checklist for your local run

1. Restart kernel, Run All top to bottom.
2. Confirm execution_count is strictly increasing across all cells (no cell has a lower count than one above it).
3. Note the actual Q14 val_acc numbers and actual total wall-clock time.
4. If the measured Part A/B/C time differs meaningfully from the ~76min estimate now in README.md, update that figure with your real number.
5. `best_model.pt` does not need regenerating — these fixes don't touch Part B (`model_b9`) weights.
