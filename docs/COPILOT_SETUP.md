# VS Code / Copilot — Environment Setup

Target: run `W9_NEU_Defect_CNN_Assignment.ipynb` locally.

## 1. Python
Python 3.12.x (tested on 3.12.3). Verify: `python3 --version`

## 2. Virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

## 3. Install dependencies
```bash
pip install torch torchvision optuna scikit-learn matplotlib numpy pillow nbformat ipykernel
```
No `--break-system-packages` needed inside a venv (that flag was only required in the sandboxed container where this was built, which had no venv).

Pinned versions used when this notebook was authored/validated:
```
torch==2.12.1
torchvision==0.27.1
scikit-learn==1.8.0
optuna==4.9.0
matplotlib==3.10.8
numpy==2.4.4
pillow==12.1.1
```

## 4. GPU (optional, strongly recommended)
If you have an NVIDIA GPU, install the CUDA-matched wheel from https://pytorch.org/get-started/locally/ instead of the default `pip install torch`. The notebook auto-detects via `torch.cuda.is_available()` — no code changes needed either way.

Full CPU run ≈ 172 min. GPU cuts this to low single-digit minutes.

## 5. Jupyter kernel in VS Code
```bash
python -m ipykernel install --user --name=neu-defect-cnn
```
Then in VS Code: open the `.ipynb` → top-right kernel picker → select `neu-defect-cnn`.

## 6. Dataset
Extract `archive.zip` so the structure is:
```
data/NEU-DET/train/images/<class>/*.jpg
data/NEU-DET/validation/images/<class>/*.jpg
```
6 classes: `crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches`. 240 train / 60 val images per class, 200×200 RGB.

If using the provided `fuseAiF_wk9_neu_defect_cnn.zip`, `data/` is already extracted inside it — just unzip the whole repo and the notebook's relative paths (`data/NEU-DET/...`) resolve as-is from the repo root.

## 7. Run
Open the notebook in VS Code, select the `neu-defect-cnn` kernel, Run All. To speed up a CPU run, reduce Q15's `N_TRIALS` (12→fewer) and/or its per-trial `epochs` (8→fewer) — that cell is ~56% of total runtime.
