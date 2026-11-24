"""Train one model.  Examples:

    python train.py --mode standard --epochs 30            # baseline
    python train.py --mode pgd_at   --epochs 30            # PGD-10 adversarial training (Madry)
    python train.py --mode fast_at  --epochs 15            # cheap FGSM-RS adversarial training

    # small CPU-friendly run (subset of the data):
    python train.py --mode pgd_at --epochs 3 --train-subset 5000 --pgd-steps 5
"""
import argparse
import json
import os

import torch

import advrob as A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standard", "pgd_at", "fast_at"], required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--pgd-steps", type=int, default=10, help="PGD steps used during adversarial training")
    p.add_argument("--train-subset", type=int, default=None, help="use only N training images")
    p.add_argument("--test-subset", type=int, default=None, help="use only N test images")
    p.add_argument("--eval-batches", type=int, default=2, help="test batches (x256) for the per-epoch monitor eval")
    p.add_argument("--monitor-steps", type=int, default=10, help="PGD steps for the per-epoch monitor eval")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    A.set_seed(a.seed)
    device = A.get_device()
    amp = a.amp and device.type == "cuda"
    train_loader, test_loader = A.get_loaders(a.data_dir, a.batch_size, a.train_subset,
                                              a.test_subset, a.workers, a.seed)
    model, history = A.fit(a.mode, a.epochs, train_loader, test_loader, device, lr=a.lr,
                           pgd_steps=a.pgd_steps, amp=amp, eval_batches=a.eval_batches,
                           monitor_steps=a.monitor_steps)

    path = os.path.join(a.out, a.mode, "model.pt")
    meta = vars(a) | {"total_train_seconds": sum(r["seconds"] for r in history)}
    A.save_checkpoint(model, path, history, meta)
    with open(os.path.join(a.out, a.mode, "history.json"), "w") as f:
        json.dump({"meta": meta, "history": history}, f, indent=2)
    print(f"saved {path}   total training time {meta['total_train_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
