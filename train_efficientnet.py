# -*- coding: utf-8 -*-
"""
Train EfficientNet-B3 for Down Syndrome Detection
==================================================
Improvements over baseline:
  - Stronger augmentation (CLAHE + GaussianBlur)
  - Early stopping (patience=7)
  - AMP mixed-precision on GPU
  - CosineAnnealingLR scheduler
  - Saves best model by Val AUC

Usage
-----
  python train_efficientnet.py
  python train_efficientnet.py --data-dir "data/Down Syndrome Dataset"
  python train_efficientnet.py --data-dir "..." --epochs 25 --batch-size 16
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

MODEL_NAME = "EfficientNet-B3"
PTH_NAME   = "EfficientNet_B3_best.pth"


def parse_args():
    p = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    p.add_argument("--data-dir",   type=str, default=None)
    p.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--workers",    type=int, default=0)
    p.add_argument("--no-gradcam", action="store_true")
    return p.parse_args()


def build_model(num_classes=2):
    import timm
    return timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)


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
            print(f"        Run: python train_efficientnet.py --data-dir 'data/Down Syndrome Dataset'")
            sys.exit(1)

    output_dir  = Path(args.output_dir)
    results_dir = output_dir / "results" / "efficientnet"
    results_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_dataset_split"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Model  : {MODEL_NAME}")
    print(f"[INFO] Device : {DEVICE}")
    if torch.cuda.is_available():
        print(f"       GPU    : {torch.cuda.get_device_name(0)}")

    # ── Prepare data ──────────────────────────────────────────────────────────
    prepare_dataset(raw_dir, work_dir, args.seed)

    import albumentations as A
    # EfficientNet: stronger aug — CLAHE for contrast + blur for robustness
    extra = [
        A.CLAHE(clip_limit=3.0, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    ]
    train_tf = build_train_transform(extra_aug=extra)
    train_loader, val_loader, test_loader = build_dataloaders(
        work_dir, args.batch_size, args.workers, train_tf
    )

    # ── Build model ───────────────────────────────────────────────────────────
    model     = build_model().to(DEVICE)
    save_path = output_dir / PTH_NAME

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
        label_smoothing = 0.0,
        early_stopping_patience = 7,
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
