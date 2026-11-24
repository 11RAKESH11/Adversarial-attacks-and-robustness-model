"""Evaluate trained models under attack and produce the numbers/figures for the report.

    python evaluate.py                       # evaluates runs/standard and runs/pgd_at (full 10,000 test images)
    python evaluate.py --n-test 1000         # quicker, on a random 1,000-image test subset

Outputs (in --out, default 'results'): results.json, results_table.md, accuracy_bars.png,
adversarial_examples_<model>.png, eps_sweep.png, training_curves.png
"""
import argparse
import json
import os

import advrob as A

NAMES = {"standard": "Standard training", "pgd_at": "PGD adversarial training",
         "fast_at": "Fast (FGSM-RS) adversarial training"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--n-test", type=int, default=None, help="evaluate on N random test images (default: all 10,000)")
    p.add_argument("--pgd-steps", type=int, default=20)
    p.add_argument("--sweep-batches", type=int, default=4, help="test batches (x256) used for the eps sweep")
    p.add_argument("--sweep-steps", type=int, default=20, help="PGD steps used in the eps sweep")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    A.set_seed(a.seed)
    device = A.get_device()
    _, test_loader = A.get_loaders(a.data_dir, test_subset=a.n_test, seed=a.seed)

    results, sweeps, hists, metas, models = {}, {}, {}, {}, {}
    for key, name in NAMES.items():
        path = os.path.join(a.runs, key, "model.pt")
        if not os.path.exists(path):
            continue
        model, hist, meta = A.load_model(path, device)
        print(f"== {name}  ({path})")
        r = A.evaluate(model, test_loader, device, pgd_steps=a.pgd_steps)
        print(f"   clean {r['clean']:.2f}%   FGSM {r['fgsm']:.2f}%   PGD-{a.pgd_steps} {r['pgd']:.2f}%")
        results[name], hists[name], metas[name], models[name] = r, hist, meta, model
        sweeps[name] = A.eps_sweep(model, test_loader, device, steps=a.sweep_steps, max_batches=a.sweep_batches)
        A.show_adversarial_examples(model, test_loader, device, path=os.path.join(a.out, f"adversarial_examples_{key}.png"),
                                    title=name)
    if not results:
        raise SystemExit(f"No trained models found in '{a.runs}'. Run train.py first.")

    A.plot_accuracy_bars(results, os.path.join(a.out, "accuracy_bars.png"))
    A.plot_eps_sweep(sweeps, os.path.join(a.out, "eps_sweep.png"))
    A.plot_history({n: h for n, h in hists.items() if h}, os.path.join(a.out, "training_curves.png"))

    # ---- markdown table that mirrors Section 5.1 of the report ----
    train_min = {n: m.get("total_train_seconds", 0) / 60 for n, m in metas.items()}
    ep = {n: max(len(hists[n]), 1) for n in hists}
    lines = ["| Model | Clean | FGSM | PGD-%d | Train time / epoch |" % a.pgd_steps, "|---|---|---|---|---|"]
    for n, r in results.items():
        lines.append(f"| {n} | {r['clean']:.1f}% | {r['fgsm']:.1f}% | {r['pgd']:.1f}% | {train_min[n]*60/ep[n]:.0f} s |")
    base, adv = results.get(NAMES["standard"]), results.get(NAMES["pgd_at"])
    if base and adv:
        lines += ["", f"Clean-accuracy drop: {base['clean']-adv['clean']:.1f} points; "
                      f"PGD-{a.pgd_steps} gain: {adv['pgd']-base['pgd']:.1f} points; "
                      f"training-time overhead: {(train_min[NAMES['pgd_at']]/ep[NAMES['pgd_at']])/(train_min[NAMES['standard']]/ep[NAMES['standard']]):.1f}x per epoch."]
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.out, "results_table.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump({"n_test": a.n_test or 10000, "pgd_steps": a.pgd_steps, "results": results,
                   "eps_sweep": {n: {str(k): v for k, v in s.items()} for n, s in sweeps.items()},
                   "train_minutes": train_min, "epochs": ep}, f, indent=2)
    print(f"\nFigures and tables written to '{a.out}/'")


if __name__ == "__main__":
    main()
