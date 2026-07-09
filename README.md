# WK9 — NEU Steel Defect CNN Classifier + Hardening

Fusemachines AI Fellowship 2026 · Week 9 Neural Network Assignment

**Notebook:** `W9_NEU_Defect_CNN_Assignment.ipynb`
**Status:** Built, syntax-validated, all code paths smoke-tested on reduced-scale runs. The notebook is complete as a deliverable; the remaining optional step is a full-scale rerun if you want benchmark numbers instead of the practical CPU-budget settings used here.
**Due:** 9 Jul 2026, 20:45 NPT

## Contents

- **Part 0** (Qs 1–5): NN foundations on flattened NEU images — 2-layer `nn.Module`, ReLU vs Sigmoid, CE vs MSE justification, SGD/Momentum/Adam + GPU memory estimate, BatchNorm1d vs Dropout ablation.
- **Part A** (Qs 1–6): CNN defect classifier — `ImageFolder` pipeline, dataset-computed normalization, Conv→ReLU→MaxPool ×2 architecture, 15-epoch training loop, curve analysis, per-class F1 + misclassified crazing/patches inspection.
- **Part B** (Qs 7–11): Hardening — train-only augmentation, BatchNorm2d, Dropout(0.4), cumulative comparison plot, written reflection.
- **Part C** (Qs 12–15): Tuning — 2×2 grid search, results table, StepLR scheduler, Optuna Bayesian search (4 trials).

## Setup

Python 3.12.10 is the current local environment version used for this WK9 setup.

```bash
pip install torch torchvision optuna scikit-learn matplotlib numpy pillow --break-system-packages --no-cache-dir
```

Download the NEU-DET dataset from [Kaggle: NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database?authuser=0), then unzip `archive.zip` into `data/NEU-DET/` before running — expects `data/NEU-DET/{train,validation}/images/<class>/`.

This notebook auto-detects CUDA only via `torch.cuda.is_available()`, so Intel Arc will not be used automatically by this setup. If you want GPU acceleration on Intel Arc, you will need a compatible accelerator backend; otherwise, run on CPU and no code changes are required.

## Execution note

**CPU-only full-scale run time: ~105 minutes (estimated)** (Part 0: ~29min for 140 total epochs across Qs 2/4/5 — measured; Part A/B/C CNN training: ~76min estimated for 131 total epochs). Breakdown of the 131: Q12's 2×2 grid search (60 epochs, ~46% — now the single largest contributor), Part B's two hardening reruns (30 epochs), Part A's baseline (15 epochs), Q14's StepLR comparison (10 epochs), and Q15's Optuna search (16 epoch-equivalents: 4 trials × 4 epochs each — the smallest contributor). The ~76min figure is a proportional estimate scaled from this notebook's own per-CNN-epoch rate, not a fresh measurement — replace it with your actual local run time. The notebook was finalized with reduced-budget tuning settings in Q14/Q15 so it stays practical on CPU; if you want the original full-budget benchmark, rerun those cells on a GPU (e.g. Colab). Every cell's logic has been independently smoke-tested at reduced epoch counts/data subsets to confirm correctness before delivery; the only expected differences at full scale are the actual accuracy numbers and total wall time, not runtime errors.

If speed is needed and no GPU is available, the fastest lever is now Q12's grid search — reduce `EPOCHS_C` (currently 15) for the 2×2 sweep, or shrink the grid itself — since it's the single largest contributor at ~46% of the Part A/B/C epoch budget. Q15's Optuna search (`N_TRIALS=4`, `epochs=4` per trial) is already at a minimal, CPU-practical budget and is no longer a meaningful lever.

## Repo structure

```
fuseAiF_wk9_neu_defect_cnn/
├── data/NEU-DET/                          # dataset (gitignored)
├── W9_NEU_Defect_CNN_Assignment.ipynb     # main deliverable
├── plots/                                 # populated on execution
├── assignment/                            # best_model.pt saved on execution
├── docs/W9_TaskPlan.md                    # planning doc
├── misc/                                  # scratch, not graded
├── README.md
├── LICENSE
└── .gitignore
```
