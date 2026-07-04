# WK9 Task Plan — NEU Steel Defect CNN Classifier + Hardening

**Repo:** `fuseAiF_wk9_neu_defect_cnn`
**Assignment:** Neural Network Assignments — AI Fellowship 2026 (Rakshya Lama Moktan, posted 1 Jul)
**Due:** 9 Jul 2026, 20:45 NPT · 100 pts
**Status at plan time:** environment verified, dataset verified, no code written yet.

---

## 1. Verified ground truth (do not re-derive, just use)

### 1.1 Environment
- Sandbox has **no CUDA** (`torch.cuda.is_available() == False`). Confirmed by direct check, not assumed.
- Installed: `torch 2.12.1+cu130`, `torchvision 0.27.1+cu130`, `optuna 4.9.0`, `scikit-learn 1.8.0`, `matplotlib 3.10.8`, `numpy 2.4.4`, `pillow 12.1.1`.
- `pip install` must use `--break-system-packages --no-cache-dir` (disk is tight — cache purge was needed once already; do not let pip cache balloon again).
- Network whitelist blocks `download.pytorch.org` — only default PyPI resolves. CPU wheel came down as the `+cu130` build anyway (fine, since `torch.cuda.is_available()` gates all device logic — code must never hardcode `.cuda()`).
- **Implication for notebook:** every `device = torch.device(...)` call must be `"cuda" if torch.cuda.is_available() else "cpu"`. 15-epoch CNN training on CPU for 1440 images at 200×200 will be slow (rough order: minutes, not seconds, per epoch depending on architecture) — architecture should stay small enough that Part A can still hit the 70-min-live-checkpoint 75% val-acc bar in reasonable wall time. Keep the CNN shallow (2 conv blocks, as the spec's minimum literally asks for) rather than over-building.

### 1.2 Dataset (extracted from `archive.zip`, verified via bash — see session log)
- Path: `data/NEU-DET/{train,validation}/{images,annotations}/<class>/`
- 6 classes: `crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`
- **Pre-split, not flat:** train = 240 img/class (1440 total), validation = 60 img/class (360 total). No overlap in filenames (checked: train IDs 1–240, val IDs 241–300 per class, zero intersection).
- All images: **200×200 px**, PIL reads them as **RGB mode** even though the XML annotations report `<depth>1</depth>` (i.e. source is grayscale but saved as 3-identical-channel JPG). Confirmed on samples from all 6 classes.
- Annotations are Pascal-VOC-style XML with bounding boxes — **not needed for this assignment** (classification only, not detection), but present if a stretch task wants them later. Ignore for now.
- Total dataset = 1800 images, matches the assignment's stated "1,800 grayscale images... 300 images per class."

### 1.3 Assignment-document cross-references (from `Required_Materials` + `Course_Plan.docx`)
- **Come-prepared question** (Required_Materials): chain rule multiplies *local gradients* layer-by-layer; long chains of small (<1) derivatives multiply toward zero → vanishing gradients. This is background knowledge for Part 0 Q2/Q5 reasoning, not a separate deliverable — but my written explanations in those cells should reflect it correctly.
- **Course_Plan §Memory requirements** gives the exact formula I must use for Part 0 Q4's GPU memory estimate:
  `P × (4B weights + 4B grads + 4B momentum-vectors + activation buffers)`
  — SGD adds 0 extra vector storage, SGD+momentum adds 1× model size, Adam adds 2× (m and v) → Adam ≈ 6–10× raw model size once activations are counted. I will compute actual parameter count P from the built network and plug into this, not hand-wave a number.
- **Course_Plan Part B framing**: "clean vs rotated/relit defect images" — augmentation exists to simulate exactly this. My Part B write-up frames it that way, not generically.
- **Course_Plan mid-week discussion prompt** ("what goes wrong if you augment the validation set too?") directly maps to assignment Q7's "why not augment val/test" — answer must cover: val/test exists to estimate real deployment performance; augmenting it makes the metric measure robustness-to-augmentation rather than true generalization, and makes epoch-to-epoch val numbers non-comparable since the val set itself is randomly changing.
- **Live checkpoint constraint** (70 min into class): CNN must hit >75% val accuracy *before* Part B is taught. This means Part A (Qs 1–6) is effectively a pre-class deliverable, not something to build live — reinforces building the full notebook now rather than doing it during the session.
- **Crazing vs patches confusion** (assignment Q6) — this is a well-known real confusion pair for NEU-DET surface defects (both present as diffuse, low-contrast textural anomalies vs. the sharper geometric signatures of scratches/rolled-in-scale). My Q6 answer will ground the "why" in actual visual inspection of misclassified samples, not just assert this.

---

## 2. Execution plan by part

### Part 0 — NN Foundations (Qs 1–5)
Use **synthetic/flattened NEU data** (flatten 200×200×3 → 120000-dim input) for the from-scratch 2-layer NN, since Part 0 is architecture-agnostic pedagogy, not the CNN itself. This matches Course_Plan's "Recap: Foundations Bridge" segment which explicitly shows "an underperforming flattened-image Linear network" before CNNs are introduced — so using flattened NEU images (not a toy dataset like MNIST) is the intended continuity.

| Q | Deliverable | Method |
|---|---|---|
| 1 | 2-layer NN, explicit `nn.Module`, no `nn.Sequential` | `__init__`/`forward` defined manually |
| 2 | ReLU vs Sigmoid, 20 epochs | Same net, swap activation, plot both loss curves, note vanishing-gradient-adjacent slower convergence for Sigmoid |
| 3 | CrossEntropy vs MSE justification | Written explanation: CE + softmax gives well-behaved gradients for multi-class one-hot targets and models a categorical distribution; MSE treats classes as ordinal/continuous and saturates badly with softmax outputs |
| 4 | SGD vs SGD+momentum(0.9) vs Adam, 1 plot, GPU memory estimate | 3 training runs, overlaid loss plot, memory formula computed from actual param count |
| 5 | BatchNorm1d vs Dropout(0.3) ablation | Two more runs, report validation loss effect + mechanism explanation |

### Part A — Defect Classifier (Qs 1–6)
Build the actual `ImageFolder`-based pipeline on the real train/validation split.

- **Q1**: `ImageFolder` on `data/NEU-DET/train/images` (root has 6 class subfolders — this *is* the ImageFolder-correct structure already, no need to merge/reshuffle). Report per-class counts (240/class) and image dims (200×200) — already know these numbers from verification, but the notebook cell must derive them from the loaded dataset object, not hardcode them (grading likely expects live inspection).
- **Q2**: `transforms.Normalize` — compute actual per-channel mean/std from the training set rather than guessing ImageNet defaults, since this is a domain-specific grayscale-as-RGB dataset, not natural images. Justify normalization mechanically (keeps activations in the range where ReLU/optimizers behave well, prevents scale mismatch across channels).
- **Q3**: CNN — Conv2d→ReLU→MaxPool2d ×2 →Flatten→Linear(6). Keep filter counts modest (e.g. 16→32) given CPU-only training constraint noted in §1.1.
- **Q4**: Full manual loop, 15 epochs, on **train** split; validate against **validation** split (not a random subsplit of train — the pre-existing folder split is deliberately the one to use, this is what the assignment dataset structure is *for*).
- **Q5**: Plot train/val accuracy + loss curves on same figure; identify overfit onset epoch from the actual curves.
- **Q6**: Per-class F1 via `sklearn.metrics.classification_report`; pull a small grid of misclassified crazing/patches images for visual side-by-side.

### Part B — Hardening (Qs 7–11)
- **Q7**: Augmentation (`RandomHorizontalFlip`, `RandomRotation(15)`, `RandomCrop(180)`) applied only via a separate `train_transform` vs `eval_transform` — two separate `ImageFolder` instantiations pointing at the same folders with different transform pipelines, so val is never augmented. Written "why not augment val" answer per §1.3.
- **Q8**: `BatchNorm2d` after each Conv2d, retrain, compare loss curves to Part A baseline, mechanism explanation (reduces internal covariate shift, stabilizes input distribution to each layer, allows higher effective LR).
- **Q9**: `Dropout(0.4)` before final Linear. Explain train-mode (stochastic zeroing, ~1/(1-p) inverted scaling) vs eval-mode (dropout disabled, full network used) behavior explicitly, since the question asks for that distinction directly.
- **Q10**: 3-line validation-accuracy plot: baseline / +aug / +aug+BN+dropout, single figure.
- **Q11**: Written reflection, single technique choice + justification.

### Part C — Hyperparameter Tuning (Qs 12–15)
- **Q12**: Grid over `lr ∈ {0.001, 0.01}` × `batch_size ∈ {16, 32}` = 4 runs on the Part B best config, record val accuracy each.
- **Q13**: Markdown table of the 4 results + written comparison of which hyperparameter moved the needle more.
- **Q14**: `StepLR(step_size=5, gamma=0.5)` added to the best run from Q12, compare final test accuracy with/without.
- **Q15 (Extended)**: Optuna Bayesian search, `lr` log-uniform(1e-4, 1e-1) × `batch_size` int(8, 64), compare best trial vs grid search result from Q12.

---

## 3. Repo structure (target, mirrors WK7/WK8 convention from fellowship history)

```
fuseAiF_wk9_neu_defect_cnn/
├── data/NEU-DET/                          # extracted dataset (gitignored — too large for repo)
├── W9_NEU_Defect_CNN_Assignment.ipynb     # main deliverable, all 15 Qs
├── W9_TaskPlan.md                         # this file → moves to docs/ pre-commit
├── plots/                                 # exported figures referenced in notebook
├── assignment/                            # any saved model artifacts (e.g. best_model.pt)
├── scripts/                               # standalone utility scripts if any emerge
├── docs/                                  # this taskplan + any resource guide
├── misc/                                  # dev/discussion scratch, not graded
├── README.md
├── LICENSE
└── .gitignore                             # must exclude data/ (raw dataset) and __pycache__/
```

## 4. Pre-commit gate (per fellowship 12-Factor standard)

Before final commit: run the standard sequence — secret-scan diff, `.env` check (n/a, no secrets in this repo), confirm `data/` is gitignored (raw dataset, ~28MB zipped, should not go into git history), confirm no stray `print()` (use minimal logging or notebook cell output only, acceptable in notebook context), `git status` clean except staged.

## 5. Open decisions deferred to build time (not blocking plan approval)

- Exact CNN channel widths (16→32 vs 32→64) — will tune down if CPU epoch time is prohibitive during Part A build.
- Whether Part 0's flattened-NEU-image approach needs a dataset subsample (120000-dim input × 2-layer FC is large) — will check parameter count and drop to a resized/smaller flatten input if Part 0 training is impractically slow, since Part 0 is about the *mechanics* (activation/loss/optimizer behavior) not about achieving good accuracy on defects.
