"""
advrob.py  -  Adversarial Attacks and Robustness (CIFAR-10, ResNet-18)
=======================================================================
Deep Learning (BCS714A) Activity-Based Learning project.

Implements, from scratch in PyTorch:
  * ResNet-18 adapted to 32x32 inputs (3x3 first conv, no max-pool, ~11.17M params)
  * FGSM   (Goodfellow et al., 2015)  - one-step gradient-sign attack
  * PGD    (Madry et al., 2018)       - iterative attack, random start, L-inf projection
  * Standard training, PGD adversarial training (min-max) and "fast" FGSM-RS adversarial training
  * Robust evaluation: clean / FGSM / PGD-20 accuracy, epsilon sweep, plots

Threat model: L-infinity ball, eps = 8/255, step alpha = 2/255.
All images live in [0, 1]; mean/std normalisation is done INSIDE the model so
that eps is measured in raw pixel space.

The file is split into "# ===== SECTION: ... =====" blocks; make_notebook.py turns
each block into one cell of the Google Colab notebook.
"""

# ===== SECTION: 1. Imports, configuration and helpers =====
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

EPS = 8 / 255          # L-inf perturbation budget
ALPHA = 2 / 255        # PGD step size
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===== SECTION: 2. CIFAR-10 data loaders (normalise inside the model, augment here) =====
class CIFAR10Arrays(torch.utils.data.Dataset):
    """CIFAR-10 from uint8 arrays (used with data/cifar10.npz made by prepare_data.py)."""

    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.x[i])), int(self.y[i])


def get_loaders(data_dir="./data", batch_size=128, train_subset=None,
                test_subset=None, workers=None, seed=0):
    """Returns (train_loader, test_loader). Pixels are in [0,1]; no mean/std here."""
    if workers is None:
        workers = 0 if sys.platform == "win32" else 2
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    test_tf = T.ToTensor()
    npz = os.path.join(data_dir, "cifar10.npz")
    if os.path.exists(npz):                      # fast local copy (see prepare_data.py)
        d = np.load(npz)
        train_set = CIFAR10Arrays(d["xtr"], d["ytr"], train_tf)
        test_set = CIFAR10Arrays(d["xte"], d["yte"], test_tf)
    else:                                        # standard torchvision download (fast on Colab)
        train_set = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(seed)
    if train_subset:
        idx = torch.randperm(len(train_set), generator=g)[:train_subset].tolist()
        train_set = Subset(train_set, idx)
    if test_subset:
        idx = torch.randperm(len(test_set), generator=g)[:test_subset].tolist()
        test_set = Subset(test_set, idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
                              pin_memory=pin, drop_last=True, persistent_workers=workers > 0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=workers,
                             pin_memory=pin, persistent_workers=workers > 0)
    return train_loader, test_loader


# ===== SECTION: 3. ResNet-18 for 32x32 images (CIFAR variant) =====
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 with a 3x3 stem and no max-pool (~11.17M parameters)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        for s in [stride] + [1] * (blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):                      # x in [0, 1]
        x = (x - self.mean) / self.std         # normalisation lives inside the model
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ===== SECTION: 4. FGSM and PGD attacks (torch.autograd.grad, L-inf threat model) =====
def _input_grad(model, x, y, amp=False):
    """Gradient of the CE loss w.r.t. the input. Uses reduction='sum' so fp16 grads do not underflow."""
    x = x.clone().detach().requires_grad_(True)
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
        logits = model(x)
    loss = F.cross_entropy(logits.float(), y, reduction="sum")
    (grad,) = torch.autograd.grad(loss, x)
    return torch.nan_to_num(grad)


def fgsm_attack(model, x, y, eps=EPS, amp=False):
    """FGSM:  x_adv = clip( x + eps * sign( grad_x L ) )   (single step)."""
    was_training = model.training
    model.eval()                                 # keep BatchNorm statistics fixed while attacking
    grad = _input_grad(model, x, y, amp)
    model.train(was_training)
    return (x + eps * grad.sign()).clamp(0, 1).detach()


def pgd_attack(model, x, y, eps=EPS, alpha=ALPHA, steps=10, random_start=True, amp=False):
    """PGD:  x^0 = x + U(-eps, eps);  x^{t+1} = Proj_{x+S}( x^t + alpha * sign( grad_x L ) )."""
    was_training = model.training
    model.eval()
    delta = (torch.rand_like(x) * 2 - 1) * eps if random_start else torch.zeros_like(x)
    delta = ((x + delta).clamp(0, 1) - x).detach()
    for _ in range(steps):
        grad = _input_grad(model, x + delta, y, amp)
        delta = (delta + alpha * grad.sign()).clamp(-eps, eps)   # project onto the eps-ball
        delta = ((x + delta).clamp(0, 1) - x).detach()           # keep a valid image in [0, 1]
    model.train(was_training)
    return (x + delta).detach()


# ===== SECTION: 5. Standard / PGD adversarial / fast adversarial training =====
def train_epoch(model, loader, optimizer, scheduler, scaler, device, mode="standard",
                eps=EPS, alpha=ALPHA, pgd_steps=10, amp=False):
    """mode: 'standard' | 'pgd_at' (Madry min-max) | 'fast_at' (FGSM + random start, Wong et al. 2020)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if mode == "pgd_at":        # inner maximisation: worst-case delta inside the eps-ball
            x = pgd_attack(model, x, y, eps, alpha, pgd_steps, True, amp)
        elif mode == "fast_at":
            x = pgd_attack(model, x, y, eps, 1.25 * eps, 1, True, amp)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)    # outer minimisation: update weights
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, 100.0 * correct / seen


def _log(msg):
    print(msg, flush=True)


def fit(mode, epochs, train_loader, test_loader, device, lr=0.1, weight_decay=5e-4,
        pgd_steps=10, amp=False, eval_batches=2, monitor_steps=10, log=_log):
    """Trains a fresh ResNet-18 and returns (model, history). One history row per epoch.
    After every epoch clean and PGD-`monitor_steps` accuracy are measured on `eval_batches` test batches."""
    model = ResNet18().to(device).to(memory_format=torch.channels_last)   # ~25% faster convs on CPU/GPU
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader), pct_start=0.25)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    history = []
    log(f"[{mode}] ResNet-18 params={count_params(model)/1e6:.2f}M  device={device}  epochs={epochs}  "
        f"batches/epoch={len(train_loader)}")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, device,
                                      mode, pgd_steps=pgd_steps, amp=amp)
        seconds = time.time() - t0
        quick = evaluate(model, test_loader, device, attacks=("clean", "pgd"), pgd_steps=monitor_steps,
                         amp=amp, max_batches=eval_batches)
        row = dict(epoch=epoch, loss=loss, train_acc=train_acc, test_clean=quick["clean"],
                   test_pgd=quick["pgd"], seconds=seconds)
        history.append(row)
        log(f"[{mode}] epoch {epoch:>3}/{epochs}  loss {loss:.3f}  train-acc {train_acc:5.1f}%  "
            f"test clean {quick['clean']:5.1f}%  PGD-{monitor_steps} {quick['pgd']:5.1f}%  ({seconds:.0f}s)")
    return model, history


# ===== SECTION: 6. Robust evaluation (clean, FGSM, PGD-20) =====
def evaluate(model, loader, device, eps=EPS, alpha=ALPHA, pgd_steps=20, amp=False,
             attacks=("clean", "fgsm", "pgd"), max_batches=None):
    """Returns accuracy (%) per requested attack. PGD uses a random start, adversarial examples
    are generated fresh from the test images and never reused from training."""
    model.eval()
    correct = {a: 0 for a in attacks}
    seen = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        for a in attacks:
            if a == "clean":
                xa = x
            elif a == "fgsm":
                xa = fgsm_attack(model, x, y, eps, amp)
            elif a == "pgd":
                xa = pgd_attack(model, x, y, eps, alpha, pgd_steps, True, amp)
            else:
                raise ValueError(a)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                correct[a] += (model(xa).argmax(1) == y).sum().item()
        seen += y.size(0)
    return {a: 100.0 * c / seen for a, c in correct.items()}


def eps_sweep(model, loader, device, eps_list=(0, 1, 2, 4, 8, 12, 16), steps=20, amp=False, max_batches=4):
    """PGD accuracy as the budget grows (eps in units of 1/255). Step size = eps/4."""
    out = {}
    for e in eps_list:
        eps = e / 255
        if e == 0:
            out[e] = evaluate(model, loader, device, attacks=("clean",), amp=amp, max_batches=max_batches)["clean"]
        else:
            out[e] = evaluate(model, loader, device, eps=eps, alpha=eps / 4, pgd_steps=steps,
                              attacks=("pgd",), amp=amp, max_batches=max_batches)["pgd"]
    return out


def save_checkpoint(model, path, history=None, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": history or [], "meta": meta or {}}, path)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ResNet18().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("history", []), ckpt.get("meta", {})


# ===== SECTION: 7. Plots (accuracy bars, adversarial examples, epsilon sweep) =====
def plot_accuracy_bars(results, path=None):
    """results = {"Standard training": {"clean":..,"fgsm":..,"pgd":..}, "PGD adversarial training": {...}}"""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Clean", "FGSM", "PGD-20"]
    keys = ["clean", "fgsm", "pgd"]
    colors = ["#14213d", "#0f9d8a", "#d64545", "#888888"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.8 / len(results)
    for i, (name, r) in enumerate(results.items()):
        vals = [r[k] for k in keys]
        bars = ax.bar(np.arange(3) + i * w - 0.4 + w / 2, vals, w, label=name, color=colors[i % 4])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("CIFAR-10 ResNet-18: accuracy under attack (eps = 8/255)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160); plt.close(fig)
    else:
        plt.show()


def show_adversarial_examples(model, loader, device, n=6, eps=EPS, path=None, title="Standard model"):
    """Rows: original / PGD-20 adversarial / perturbation (amplified). Titles show predictions."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n].to(device), y[:n].to(device)
    xa = pgd_attack(model, x, y, eps, ALPHA, 20)
    with torch.no_grad():
        p0, p1 = model(x).argmax(1), model(xa).argmax(1)
    fig, axes = plt.subplots(3, n, figsize=(2.1 * n, 6.6))
    for j in range(n):
        pert = ((xa[j] - x[j]) / (2 * eps) + 0.5).clamp(0, 1)
        for r, (img, ttl, col) in enumerate([
                (x[j], f"{CLASSES[p0[j]]}", "green" if p0[j] == y[j] else "red"),
                (xa[j], f"{CLASSES[p1[j]]}", "green" if p1[j] == y[j] else "red"),
                (pert, "perturbation", "black")]):
            axes[r, j].imshow(img.permute(1, 2, 0).cpu().numpy())
            axes[r, j].set_title(ttl, fontsize=9, color=col); axes[r, j].axis("off")
    axes[0, 0].text(-0.15, 0.5, "original", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.15, 0.5, "PGD-20", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle(f"{title}: true labels = {', '.join(CLASSES[k] for k in y.tolist())}", fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160); plt.close(fig)
    else:
        plt.show()


def plot_eps_sweep(sweeps, path=None):
    """sweeps = {"Standard training": {eps: acc, ...}, ...}"""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, s in sweeps.items():
        ax.plot(list(s.keys()), list(s.values()), marker="o", label=name)
    ax.axvline(8, ls="--", c="gray", lw=1)
    ax.text(8.2, 95, "training eps", fontsize=8, color="gray")
    ax.set_xlabel("PGD-20 budget eps (x 1/255)"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Robustness vs attack strength"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160); plt.close(fig)
    else:
        plt.show()


def plot_history(histories, path=None):
    """histories = {"Standard training": [rows], ...}: clean accuracy and PGD accuracy per epoch."""
    import matplotlib
    if path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["test_clean"] for r in h], marker="o", label=name)
        axes[1].plot(ep, [r["test_pgd"] for r in h], marker="o", label=name)
    axes[0].set_title("Clean test accuracy"); axes[1].set_title("PGD test accuracy (monitor subset)")
    for a in axes:
        a.set_xlabel("epoch"); a.set_ylabel("%"); a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=160); plt.close(fig)
    else:
        plt.show()
