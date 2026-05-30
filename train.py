# -*- coding: utf-8 -*-
"""
DeepSyndrome - Train ALL 3 Models
==================================
Runs ResNet-50, EfficientNet-B3, and ViT-S/16 sequentially.
Each model's weights are saved to the project root.

To train a SINGLE model instead:
  python train_resnet50.py    --data-dir "data/Down Syndrome Dataset"
  python train_efficientnet.py --data-dir "data/Down Syndrome Dataset"
  python train_vit.py         --data-dir "data/Down Syndrome Dataset"

Usage
-----
  python train.py
  python train.py --data-dir "data/Down Syndrome Dataset"
  python train.py --data-dir "..." --epochs 20 --batch-size 16
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description="Train all 3 DeepSyndrome models")
    p.add_argument("--data-dir",   type=str, default=None)
    p.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=0)
    p.add_argument("--no-gradcam", action="store_true")
    return p.parse_args()


def pick_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)
        folder = filedialog.askdirectory(title="Select dataset root folder")
        root.destroy()
        return folder or None
    except Exception:
        return None


def main():
    args = parse_args()

    # ── Resolve dataset path ──────────────────────────────────────────────────
    if args.data_dir:
        raw_dir = args.data_dir
    else:
        print("[INFO] No --data-dir provided. Opening folder picker...")
        raw_dir = pick_folder()
        if not raw_dir:
            print("[ERROR] No folder selected.")
            print("        Run: python train.py --data-dir 'data/Down Syndrome Dataset'")
            sys.exit(1)

    print(f"\n[INFO] Dataset  : {raw_dir}")
    print(f"[INFO] Output   : {args.output_dir}")
    print(f"[INFO] Epochs   : {args.epochs}")
    print(f"\n  Will train: ResNet-50 -> EfficientNet-B3 -> ViT-S/16\n")

    # ── Build shared CLI args to forward ─────────────────────────────────────
    base = [
        "--data-dir",   raw_dir,
        "--output-dir", args.output_dir,
        "--batch-size", str(args.batch_size),
        "--workers",    str(args.workers),
    ]
    if args.no_gradcam:
        base.append("--no-gradcam")

    # ── Import and run each script's main() ───────────────────────────────────
    import train_resnet50
    import train_efficientnet
    import train_vit

    for module, extra_epochs in [
        (train_resnet50,    []),
        (train_efficientnet,[]),
        (train_vit,         ["--epochs", str(max(args.epochs, 25))]),  # ViT needs more
    ]:
        sys.argv = [module.__file__] + base + extra_epochs
        module.main()

    print(f"\n{'='*60}")
    print("  ALL MODELS TRAINED")
    print(f"{'='*60}")
    print(f"  ResNet50_best.pth        -> {args.output_dir}")
    print(f"  EfficientNet_B3_best.pth -> {args.output_dir}")
    print(f"  ViT_S16_best.pth         -> {args.output_dir}")
    print(f"\n  Start server: python app.py")
    print(f"  Open UI    : http://localhost:7860")


if __name__ == "__main__":
    main()
