# Adversarial Attacks and Robustness: CIFAR-10 ResNet-18

Deep Learning (BCS714A) Activity-Based Learning project, Group G20.
PyTorch implementation of **FGSM**, **PGD** and **PGD adversarial training**, compared with a standard baseline.

| Item | Setting (as in the report) |
|---|---|
| Data | CIFAR-10, 50,000 train / 10,000 test, random-crop + flip |
| Model | ResNet-18 for 32x32 inputs (3x3 stem, no max-pool), 11.17M parameters |
| Threat model | L-infinity, eps = 8/255, step alpha = 2/255 |
| Training attack | PGD-10 with random start (min-max objective, Madry et al.) |
| Evaluation | clean, FGSM, PGD-20 (random start) on the test set |

## Files

| File | Purpose |
|---|---|
| `Adversarial_Robustness_Colab.ipynb` | **Main deliverable**: run on Google Colab (T4 GPU). Cells mirror report Figures 2-5. |
| `advrob.py` | The whole implementation: data, ResNet-18, FGSM, PGD, training, evaluation, plots |
| `train.py` | CLI to train one model (`standard`, `pgd_at`, `fast_at`) |
| `evaluate.py` | CLI to evaluate trained models and write the table and figures |
| `prepare_data.py` | Optional fast CIFAR-10 download from a mirror (used if the official host is slow) |
| `make_notebook.py` | Rebuilds the notebook from `advrob.py` (so they never differ) |
| `run_cpu_demo.ps1` | Reduced-scale run for a laptop CPU |

## 1. Full experiment on Google Colab (recommended for the submission)

1. Upload `Adversarial_Robustness_Colab.ipynb` to <https://colab.research.google.com>.
2. `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.
3. Wait: about 10 min for the standard model and 60-90 min for PGD adversarial training.
   (To see it work first, set `QUICK = True` in the configuration cell: about 2 min.)
4. The notebook prints the results table, shows all figures, and downloads `results.zip`.
5. Copy **your own numbers** into Section 5.1 of the report and the figures into Figures 2-6. Paste the notebook's share link into Section 8 of the report.

The numbers in the current report (94 / 84 / 48 %) are literature reference values and the report itself says to replace them.
Your run will land close to them (typically: standard ~93-95% clean, ~0% PGD-20; PGD-AT ~78-85% clean, ~42-50% PGD-20) but not identical. Report what you measure.

## 2. Running locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt

# full-scale (needs an NVIDIA GPU; on CPU use run_cpu_demo.ps1 instead)
.\.venv\Scripts\python train.py --mode standard --epochs 20 --amp
.\.venv\Scripts\python train.py --mode pgd_at   --epochs 25 --amp
.\.venv\Scripts\python evaluate.py              # writes results\ (table, json, figures)
```

`--mode fast_at` trains with single-step FGSM + random start (about 10x cheaper than PGD-AT, slightly less robust). It is the "faster AT" item in the report's future scope.

On a CPU-only laptop a full PGD-AT epoch takes about 3 hours, so `run_cpu_demo.ps1` trains on 4,000 images with 3-step PGD. It is only a demonstration of the effect.

## 3. How it maps to the report

* Section 3.1 (FGSM / PGD maths) -> `fgsm_attack`, `pgd_attack` in `advrob.py`
* Section 3.2 (min-max training) -> `train_epoch(mode="pgd_at")`: inner loop `pgd_attack`, outer loop the SGD step
* Section 4.1 (projection, eval-mode attacks, strong evaluation) -> `pgd_attack` clamps to the eps-ball and to [0,1] after every step, switches the model to `eval()` while attacking, and `evaluate` uses PGD-20 with a random start on fresh test images
* Section 5 (results table) -> `evaluate.py` / last cells of the notebook
* Section 6 (limitations) -> the epsilon-sweep plot shows robustness is specific to the eps = 8/255 threat model

## 4. Implementation notes (useful for the viva)

* Mean/std normalisation is inside the model, so eps is measured in real pixel units and images stay in [0, 1].
* Attack gradients use `torch.autograd.grad` (no weight gradients are computed or accumulated).
* Training: SGD (momentum 0.9, Nesterov, weight decay 5e-4), one-cycle learning-rate schedule, mixed precision on GPU.
* PGD-AT costs about (K+1) forward/backward passes per batch, which is where the 8-11x training-time overhead comes from.
* Only the adversarial loss is optimised in PGD-AT (as in Madry et al.), which is why clean accuracy drops.
