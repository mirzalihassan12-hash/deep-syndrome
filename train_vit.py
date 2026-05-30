# -*- coding: utf-8 -*-
"""
Train ViT-S/16 for Down Syndrome Detection
==========================================
Improvements over baseline:
  - Linear LR warmup (5 epochs) — critical for ViT stability
  - Smaller learning rate (5e-5) — ViT needs lower LR than CNNs
  - Early stopping (patience=8) — ViT converges slower
  - AMP mixed-precision on GPU
  - Label smoothing (0.05)
  - Saves best model by Val AUC

Usage
-----
  python train_vit.py
  python train_vit.py --data-dir "data/Down Syndrome Dataset"
  python train_vit.py --data-dir "..." --epochs 30 --batch-size 16
"""

import os
import sys
import random
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "ViT-S/16"
PTH_NAME   = "ViT_S16_best.pth"


def parse_args():
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    p.add_argument("--data-dir",    type=str, default=None)
    p.add_argument("--output-dir",  type=str, default=str(PROJECT_ROOT))
    p.add_argument("--epochs",      type=int, default=25)          # ViT needs more epochs
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=5e-5)      # ViT: lower LR
    p.add_argument("--warmup-epochs", type=int, default=5)         # LR warmup
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--workers",     type=int, default=0)
    p.add_argument("--no-gradcam",  action="store_true")
    return p.parse_args()


def build_model(num_classes=2):
    import timm
    return timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)


def warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup then CosineAnnealingLR — essential for ViT."""
    import torch
    from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

    warmup = LambdaLR(optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def main():
    args = parse_args()

    from training.dataset import prepare_dataset, build_dataloaders, build_train_transform
    import torch
    import numpy as np

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Dataset path ──────────────────────────────────────────────────────────
    if args.data_dir:
        raw_dir = args.data_dir
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)
            raw_dir = filedialog.askdirectory(title=f"Select dataset for {MODEL_NAME}")
            root.destroy()
        except Exception:
            raw_dir = None
        if not raw_dir:
            print("[ERROR] No dataset selected.")
            print(f"        Run: python train_vit.py --data-dir 'data/Down Syndrome Dataset'")
            sys.exit(1)

    output_dir  = Path(args.output_dir)
    results_dir = output_dir / "results" / "vit"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_dataset_split"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Model         : {MODEL_NAME}")
    print(f"[INFO] Device        : {DEVICE}")
    print(f"[INFO] LR warmup     : {args.warmup_epochs} epochs")
    if torch.cuda.is_available():
        print(f"       GPU          : {torch.cuda.get_device_name(0)}")

    # ── Prepare data ──────────────────────────────────────────────────────────
    prepare_dataset(raw_dir, work_dir, args.seed)

    import albumentations as A
    # ViT: minimal spatial distortion — ViT relies on patch positions
    extra = [A.GaussianBlur(blur_limit=(3, 3), p=0.2)]
    train_tf = build_train_transform(extra_aug=extra)
    train_loader, val_loader, test_loader = build_dataloaders(
        work_dir, args.batch_size, args.workers, train_tf
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model     = build_model().to(DEVICE)
    save_path = output_dir / PTH_NAME

    # ── Warmup scheduler factory ──────────────────────────────────────────────
    warmup_epochs = args.warmup_epochs
    total_epochs  = args.epochs

    def scheduler_fn(optimizer):
        return warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs)

    # ── Train ─────────────────────────────────────────────────────────────────
    from training.engine import train_model, evaluate_test
    history = train_model(
        model         = model,
        model_name    = MODEL_NAME,
        save_path     = save_path,
        train_loader  = train_loader,
        val_loader    = val_loader,
        device        = DEVICE,
        epochs        = args.epochs,
        lr            = args.lr,
        label_smoothing = 0.05,
        scheduler_fn  = scheduler_fn,
        early_stopping_patience = 8,   # ViT is slower to converge
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    import torch as _torch
    model.load_state_dict(_torch.load(save_path, map_location=DEVICE))
    scores = [evaluate_test(model, MODEL_NAME, test_loader, DEVICE, results_dir)]

    # ── Plots ─────────────────────────────────────────────────────────────────
    from training.plots import plot_training_curves, plot_roc, run_gradcam, save_metrics_csv
    plot_training_curves(history, MODEL_NAME, results_dir)
    plot_roc(scores, results_dir)
    save_metrics_csv(scores, results_dir)

    if not args.no_gradcam:
        run_gradcam(model, MODEL_NAME, test_loader, DEVICE, results_dir)

    print(f"\n[DONE] {MODEL_NAME} complete!")
    print(f"       .pth  -> {save_path}")
    print(f"       plots -> {results_dir}")
    print(f"\n  Start server: python app.py")


if __name__ == "__main__":
    main()
