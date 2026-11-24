"""Generate Adversarial_Robustness_Colab.ipynb from advrob.py (single source of truth).

    python make_notebook.py
"""
import re

import nbformat as nbf

src = open("advrob.py", encoding="utf-8").read()
parts = re.split(r"^# ===== SECTION: (.*) =====\n", src, flags=re.M)
sections = list(zip(parts[1::2], parts[2::2]))          # [(title, code), ...]

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""# Adversarial Attacks and Robustness: CIFAR-10 ResNet-18
**Deep Learning (BCS714A) - Activity-Based Learning** &nbsp;|&nbsp; FGSM, PGD and PGD adversarial training

**Before running:** `Runtime -> Change runtime type -> T4 GPU`, then `Runtime -> Run all`.

| Step | What happens |
|---|---|
| Cells 1-2 | imports, CIFAR-10 loaders (random-crop + flip) |
| Cells 3-4 | ResNet-18 (CIFAR variant) and the FGSM / PGD attacks |
| Cells 5-6 | standard, PGD adversarial (min-max) and fast adversarial training |
| Cells 7-8 | robust evaluation (clean / FGSM / PGD-20) and plots |
| Rest | run both experiments, report table, figures, download results |

Threat model: L-infinity, eps = 8/255, alpha = 2/255, PGD-10 for training, PGD-20 for testing.""")

for i, (title, body) in enumerate(sections, 1):
    md(f"## Cell {i} - {title}")
    code(body)

md("## Configuration\nSet `QUICK = True` for a 2-minute smoke test. Leave `False` for the full experiment (T4 GPU: roughly 10 min "
   "standard + 60-90 min PGD adversarial training).")
code('''
QUICK = False

device = get_device()
AMP = device.type == "cuda"            # mixed precision on GPU
set_seed(0)
if QUICK:
    TRAIN_SUBSET, TEST_SUBSET, EPOCHS_STD, EPOCHS_AT, AT_STEPS, PGD_EVAL_STEPS = 1000, 500, 1, 1, 2, 3
    MONITOR = dict(eval_batches=1, monitor_steps=2)   # cheap per-epoch monitoring
else:
    MONITOR = {}
    TRAIN_SUBSET = TEST_SUBSET = None  # full CIFAR-10: 50,000 train / 10,000 test
    EPOCHS_STD, EPOCHS_AT, AT_STEPS, PGD_EVAL_STEPS = 20, 25, 10, 20
print(f"device={device}  amp={AMP}  epochs: standard={EPOCHS_STD}, PGD-AT={EPOCHS_AT}  (PGD-{AT_STEPS} in training)")
train_loader, test_loader = get_loaders("./data", 128, TRAIN_SUBSET, TEST_SUBSET)
print("train batches:", len(train_loader), " test batches:", len(test_loader))
''')

md("## Experiment 1 - standard training (baseline)")
code('std_model, std_hist = fit("standard", EPOCHS_STD, train_loader, test_loader, device, amp=AMP, **MONITOR)')

md("## Experiment 2 - PGD adversarial training (Madry et al.)\n"
   "Each batch: the inner loop finds the worst-case perturbation with PGD, the outer loop updates the weights on it.\n"
   "For a faster (but weaker) alternative use `fit(\"fast_at\", 15, ...)` (FGSM with random start, about 10x cheaper).")
code('adv_model, adv_hist = fit("pgd_at", EPOCHS_AT, train_loader, test_loader, device, pgd_steps=AT_STEPS, amp=AMP, **MONITOR)')

md("## Robust evaluation: clean / FGSM / PGD-20 on the test set")
code('''
results = {
    "Standard training": evaluate(std_model, test_loader, device, pgd_steps=PGD_EVAL_STEPS, amp=AMP),
    "PGD adversarial training": evaluate(adv_model, test_loader, device, pgd_steps=PGD_EVAL_STEPS, amp=AMP),
}
t_std = sum(r["seconds"] for r in std_hist) / len(std_hist)
t_adv = sum(r["seconds"] for r in adv_hist) / len(adv_hist)
print("| Model | Clean | FGSM | PGD-%d | Train time / epoch |" % PGD_EVAL_STEPS)
print("|---|---|---|---|---|")
for (name, r), t in zip(results.items(), (t_std, t_adv)):
    print(f"| {name} | {r['clean']:.1f}% | {r['fgsm']:.1f}% | {r['pgd']:.1f}% | {t:.0f} s |")
s, a = results["Standard training"], results["PGD adversarial training"]
print(f"\\nClean-accuracy drop: {s['clean']-a['clean']:.1f} points | PGD gain: {a['pgd']-s['pgd']:.1f} points | "
      f"training overhead: {t_adv/t_std:.1f}x per epoch")
plot_accuracy_bars(results)
''')

md("## Figures: adversarial examples, robustness vs epsilon, training curves")
code('''
show_adversarial_examples(std_model, test_loader, device, title="Standard model")
show_adversarial_examples(adv_model, test_loader, device, title="PGD-adversarially trained model")
sweeps = {"Standard training": eps_sweep(std_model, test_loader, device, steps=PGD_EVAL_STEPS, amp=AMP),
          "PGD adversarial training": eps_sweep(adv_model, test_loader, device, steps=PGD_EVAL_STEPS, amp=AMP)}
plot_eps_sweep(sweeps)
plot_history({"Standard training": std_hist, "PGD adversarial training": adv_hist})
''')

md("## Save and download results")
code('''
import shutil
os.makedirs("results", exist_ok=True)
save_checkpoint(std_model, "results/standard.pt", std_hist)
save_checkpoint(adv_model, "results/pgd_at.pt", adv_hist)
json.dump({"results": results, "eps_sweep": {k: {str(e): v for e, v in s.items()} for k, s in sweeps.items()},
           "sec_per_epoch": {"standard": t_std, "pgd_at": t_adv}}, open("results/results.json", "w"), indent=2)
shutil.make_archive("results", "zip", "results")
try:
    from google.colab import files
    files.download("results.zip")
except ImportError:
    print("saved results.zip")
''')

md("""## Notes for the report
* Replace the reference numbers in Section 5.1 of the report with the table printed above.
* Expected pattern (exact values vary with epochs and seed): standard model ~93-95% clean and ~0% under PGD-20; PGD-AT model ~78-85% clean and ~42-50% under PGD-20.
* FGSM accuracy > PGD-20 accuracy for the adversarially trained model, which is the gap that shows why weak attacks overestimate robustness.
* Robustness is specific to this threat model (L-infinity, eps = 8/255): see the epsilon-sweep plot.""")

nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU",
                  "colab": {"name": "Adversarial_Robustness_Colab.ipynb", "provenance": []},
                  "kernelspec": {"display_name": "Python 3", "name": "python3"}}
nbf.write(nb, "Adversarial_Robustness_Colab.ipynb")
print("wrote Adversarial_Robustness_Colab.ipynb with", len(cells), "cells")
