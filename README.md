# WK9 — NEU Steel Defect CNN Classifier + Hardening

Fusemachines AI Fellowship 2026 · Week 9 Neural Network Assignment

**Notebook:** `W9_NEU_Defect_CNN_Assignment.ipynb`
**Status:** Built, syntax-validated, all code paths smoke-tested on reduced-scale runs. The notebook is complete as a deliverable; the remaining optional step is a full-scale rerun if you want benchmark numbers instead of the practical CPU-budget settings used here.
**Due:** 9 Jul 2026, 20:45 NPT

## Contents

- **Part 0** (Qs 1–5): NN foundations on flattened NEU images — 2-layer `nn.Module`, ReLU vs Sigmoid, CE vs MSE justification, SGD/Momentum/Adam + GPU memory estimate, BatchNorm1d vs Dropout ablation.
- **Part A** (Qs 1–6): CNN defect classifier — `ImageFolder` pipeline, dataset-computed normalization, Conv→ReLU→MaxPool ×2 architecture, 15-epoch training loop, curve analysis, per-class F1 + misclassified crazing/patches inspection.
- **Part B** (Qs 7–11): Hardening — train-only augmentation, BatchNorm2d, Dropout(0.4), cumulative comparison plot, written reflection.
- **Part C** (Qs 12–15): Tuning — 2×2 grid search, results table, StepLR scheduler, Optuna Bayesian search (12 trials).

## Setup

Python 3.12.10 is the current local environment version used for this WK9 setup.

```bash
pip install torch torchvision optuna scikit-learn matplotlib numpy pillow --break-system-packages --no-cache-dir
```

Download the NEU-DET dataset from [Kaggle: NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database?authuser=0), then unzip `archive.zip` into `data/NEU-DET/` before running — expects `data/NEU-DET/{train,validation}/images/<class>/`.

This notebook auto-detects CUDA only via `torch.cuda.is_available()`, so Intel Arc will not be used automatically by this setup. If you want GPU acceleration on Intel Arc, you will need a compatible accelerator backend; otherwise, run on CPU and no code changes are required.

## Execution note

**CPU-only full-scale run time: ~172 minutes** (Part 0: ~29min for 140 total epochs across Qs 2/4/5; Part A/B/C CNN training: ~143min for 246 total epochs, dominated by the 96-epoch-equivalent Optuna search in Q15). The notebook was finalized with reduced-budget tuning settings in Q14/Q15 so it stays practical on CPU; if you want the original full-budget benchmark, rerun those cells on a GPU (e.g. Colab). Every cell's logic has been independently smoke-tested at reduced epoch counts/data subsets to confirm correctness before delivery; the only expected differences at full scale are the actual accuracy numbers and total wall time, not runtime errors.

If speed is needed and no GPU is available, the fastest lever is cutting Q15's Optuna `N_TRIALS` (currently 12) and/or its per-trial `epochs` (currently 8) — that cell alone accounts for roughly 56% of total projected runtime.

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
