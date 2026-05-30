# -*- coding: utf-8 -*-
"""
Train ResNet-50 for Down Syndrome Detection
============================================
Improvements over baseline:
  - Label smoothing (0.1) — reduces overconfidence
  - Early stopping (patience=7)
  - AMP mixed-precision on GPU
  - CosineAnnealingLR scheduler
  - Saves best model by Val AUC

Usage
-----
  python train_resnet50.py
  python train_resnet50.py --data-dir "data/Down Syndrome Dataset"
  python train_resnet50.py --data-dir "..." --epochs 30 --batch-size 16
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

MODEL_NAME = "ResNet-50"
PTH_NAME   = "ResNet50_best.pth"


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
    import torch.nn as nn
    import torchvision.models as models
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, num_classes),
    )
    return m


def main():
    args = parse_args()

    # ── Dataset path ──────────────────────────────────────────────────────────
    from training.dataset import prepare_dataset, build_dataloaders, build_train_transform
    import torch
    import numpy as np

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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
            print(f"        Run: python train_resnet50.py --data-dir 'data/Down Syndrome Dataset'")
            sys.exit(1)

    output_dir  = Path(args.output_dir)
    results_dir = output_dir / "results" / "resnet50"
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
    # ResNet-50: extra GridDistortion for shape robustness
    extra = [A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3)]
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
        label_smoothing = 0.1,       # reduces overconfidence
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
