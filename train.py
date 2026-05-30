# -*- coding: utf-8 -*-
"""
DeepSyndrome - Local Training Script
=====================================
Trains ResNet-50, EfficientNet-B3, ViT-S/16 on a Down Syndrome dataset.
Saves .pth files to the project root so the API server can load them.

Usage
-----
  # Option 1: pick dataset folder via dialog
  python train.py

  # Option 2: pass path directly
  python train.py --data-dir "C:/path/to/your/dataset"

  # Option 3: download from Kaggle then train
  python train.py --kaggle

  # Custom epochs / batch size
  python train.py --data-dir "..." --epochs 30 --batch-size 16
"""

import os
import sys
import shutil
import random
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DeepSyndrome Training Script")
    p.add_argument("--data-dir",    type=str,  default=None,
                   help="Root folder of your dataset (subfolders named control/down_syndrome etc.)")
    p.add_argument("--output-dir",  type=str,  default=str(Path(__file__).parent),
                   help="Where to save .pth files (default: project root)")
    p.add_argument("--epochs",      type=int,  default=20)
    p.add_argument("--batch-size",  type=int,  default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--seed",        type=int,  default=42)
    p.add_argument("--kaggle",      action="store_true",
                   help="Download dataset from Kaggle (needs kagglehub installed + API key)")
    p.add_argument("--no-gradcam",  action="store_true",
                   help="Skip Grad-CAM visualization")
    p.add_argument("--workers",     type=int,  default=0,
                   help="DataLoader num_workers (0 = main process, safe on Windows)")
    return p.parse_args()


# ── Dataset folder picker ─────────────────────────────────────────────────────
def pick_folder_dialog(title="Select your dataset folder"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder if folder else None
    except Exception:
        return None


# ── Kaggle download ───────────────────────────────────────────────────────────
def download_from_kaggle():
    try:
        import kagglehub
        print("[INFO] Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("mamunhasan2cs/down-syndrome-dataset")
        print(f"[OK]   Dataset downloaded to: {path}")
        return path
    except ImportError:
        print("[ERROR] kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Kaggle download failed: {e}")
        print("        Make sure your Kaggle API key is set up.")
        sys.exit(1)


# ── Dataset exploration & split ───────────────────────────────────────────────
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DS_KEYWORDS  = ["down", "syndrome", "ds", "positive", "affected"]
CTRL_KEYWORDS= ["control", "normal", "healthy", "negative", "non"]


def find_class_folders(base):
    ds_folders, ctrl_folders = [], []
    for root, _, files in os.walk(base):
        imgs = [f for f in files if Path(f).suffix.lower() in IMG_EXTS]
        if not imgs:
            continue
        name = os.path.basename(root).lower()
        if any(k in name for k in DS_KEYWORDS):
            ds_folders.append((root, imgs))
        elif any(k in name for k in CTRL_KEYWORDS):
            ctrl_folders.append((root, imgs))
    return ds_folders, ctrl_folders


def split_and_copy(folders, class_name, dest_root, seed):
    all_imgs = [os.path.join(f, img) for f, imgs in folders for img in imgs]
    random.seed(seed)
    random.shuffle(all_imgs)
    n    = len(all_imgs)
    n_tr = int(n * 0.70)
    n_vl = int(n * 0.15)
    splits = {
        "train": all_imgs[:n_tr],
        "val":   all_imgs[n_tr : n_tr + n_vl],
        "test":  all_imgs[n_tr + n_vl :],
    }
    for split, imgs in splits.items():
        dest = Path(dest_root) / split / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for src in imgs:
            shutil.copy(src, dest)
        print(f"   [{split:5s}] {class_name}: {len(imgs)} images")
    return splits


def prepare_dataset(raw_dir, work_dir, seed):
    print(f"\n[STEP] Exploring dataset: {raw_dir}")
    ds_folders, ctrl_folders = find_class_folders(raw_dir)
    print(f"       Found DS folders: {len(ds_folders)} | Control folders: {len(ctrl_folders)}")

    if not ds_folders:
        print("[ERROR] No Down Syndrome folders found.")
        print("        Folder names must contain one of:", DS_KEYWORDS)
        sys.exit(1)
    if not ctrl_folders:
        print("[ERROR] No Control folders found.")
        print("        Folder names must contain one of:", CTRL_KEYWORDS)
        sys.exit(1)

    print("\n[STEP] Splitting 70 / 15 / 15 ...")
    split_and_copy(ds_folders,   "down_syndrome", work_dir, seed)
    split_and_copy(ctrl_folders, "control",       work_dir, seed)
    print("[OK]   Split done!\n")
    return work_dir


# ── Imports (heavy libs after arg parse so --help is fast) ───────────────────
def import_libs():
    global torch, nn, optim, models, timm, A, ToTensorV2
    global Dataset, DataLoader, Image, np
    global plt, sns, pd
    global accuracy_score, precision_score, recall_score, f1_score
    global roc_auc_score, confusion_matrix, roc_curve

    print("[INFO] Importing libraries...")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.models as models
    import timm
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix, roc_curve,
    )
    print("[OK]   Libraries ready.")
    return (torch, nn, optim, models, timm, A, ToTensorV2,
            Dataset, DataLoader, Image, np, plt, sns, pd,
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, confusion_matrix, roc_curve)


# ── Dataset class ─────────────────────────────────────────────────────────────
class DownSyndromeDataset:
    def __init__(self, root_dir, split, transform=None):
        from PIL import Image as PILImage
        import numpy as _np
        self._Image   = PILImage
        self._np      = _np
        self.transform = transform
        self.paths, self.labels = [], []
        for cls, lbl in [("control", 0), ("down_syndrome", 1)]:
            d = Path(root_dir) / split / cls
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.suffix.lower() in IMG_EXTS:
                    self.paths.append(str(f))
                    self.labels.append(lbl)
        print(f"   [{split:5s}] total={len(self.paths)}  "
              f"DS={self.labels.count(1)}  ctrl={self.labels.count(0)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self._np.array(
            self._Image.open(self.paths[idx]).convert("RGB")
        )
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


# ── Model builders ────────────────────────────────────────────────────────────
def build_resnet50(num_classes=2):
    import torchvision.models as _models
    import torch.nn as _nn
    m = _models.resnet50(weights=_models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = _nn.Sequential(_nn.Dropout(0.3), _nn.Linear(m.fc.in_features, num_classes))
    return m


def build_efficientnet(num_classes=2):
    import timm as _timm
    return _timm.create_model("efficientnet_b3", pretrained=True, num_classes=num_classes)


def build_vit(num_classes=2):
    import timm as _timm
    return _timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=num_classes)


MODEL_REGISTRY = {
    "ResNet50":        (build_resnet50,     "ResNet50_best.pth"),
    "EfficientNet_B3": (build_efficientnet, "EfficientNet_B3_best.pth"),
    "ViT_S16":         (build_vit,          "ViT_S16_best.pth"),
}


# ── Training / validation ─────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    import torch as _torch
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    import torch as _torch
    import numpy as _np
    from sklearn.metrics import roc_auc_score as _auc
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with _torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
            all_probs.extend(_torch.softmax(out, 1)[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    auc = _auc(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, auc


def train_model(name, build_fn, save_path, train_loader, val_loader,
                device, epochs, lr):
    import torch as _torch
    import torch.nn as _nn
    import torch.optim as _optim

    print(f"\n{'='*58}")
    print(f"  Training: {name}")
    print(f"{'='*58}")

    model     = build_fn().to(device)
    criterion = _nn.CrossEntropyLoss()
    optimizer = _optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = _optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_auc  = 0.0
    history   = {k: [] for k in ["tr_loss", "vl_loss", "tr_acc", "vl_acc", "vl_auc"]}

    for ep in range(1, epochs + 1):
        tr_l, tr_a          = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_l, vl_a, vl_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        for key, val in zip(history, [tr_l, vl_l, tr_a, vl_a, vl_auc]):
            history[key].append(val)

        saved = ""
        if vl_auc > best_auc:
            best_auc = vl_auc
            _torch.save(model.state_dict(), save_path)
            saved = "  [SAVED]"

        print(f"  Ep {ep:02d}/{epochs} | "
              f"Tr L:{tr_l:.4f} A:{tr_a:.3f} | "
              f"Val L:{vl_l:.4f} A:{vl_a:.3f} AUC:{vl_auc:.3f}{saved}")

    print(f"\n[OK]   {name} done!  Best Val AUC: {best_auc:.4f}")
    print(f"       Weights saved -> {save_path}")
    return model, history


# ── Test evaluation ───────────────────────────────────────────────────────────
def evaluate_test(model, name, test_loader, device, results_dir):
    import torch as _torch
    import numpy as _np
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score,
                                  confusion_matrix, roc_curve)
    import matplotlib.pyplot as plt
    import seaborn as sns

    model.eval()
    preds_all, probs_all, lbl_all = [], [], []
    with _torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(device))
            probs_all.extend(_torch.softmax(out, 1)[:, 1].cpu().numpy())
            preds_all.extend(out.argmax(1).cpu().numpy())
            lbl_all.extend(labels.numpy())

    acc  = accuracy_score(lbl_all,  preds_all)
    prec = precision_score(lbl_all, preds_all, zero_division=0)
    rec  = recall_score(lbl_all,    preds_all, zero_division=0)
    f1   = f1_score(lbl_all,        preds_all, zero_division=0)
    auc  = roc_auc_score(lbl_all,   probs_all) if len(set(lbl_all)) > 1 else 0.0

    print(f"\n  {name}")
    print(f"  Accuracy  : {acc:.4f}  {'OK' if acc  >= 0.85 else 'LOW'} (target >=0.85)")
    print(f"  F1-Score  : {f1:.4f}   {'OK' if f1   >= 0.83 else 'LOW'} (target >=0.83)")
    print(f"  AUC-ROC   : {auc:.4f}  {'OK' if auc  >= 0.88 else 'LOW'} (target >=0.88)")

    # Confusion matrix
    cm = confusion_matrix(lbl_all, preds_all)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Control", "Down Syn"],
                yticklabels=["Control", "Down Syn"])
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    safe = name.replace("/", "_").replace(" ", "_")
    plt.savefig(results_dir / f"{safe}_confusion.png", dpi=150)
    plt.close()

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                preds=preds_all, probs=probs_all, labels=lbl_all)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def run_gradcam(model, name, test_loader, device, results_dir, n=6):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("[SKIP] pytorch-grad-cam not installed. Skipping Grad-CAM.")
        return

    import numpy as _np
    import matplotlib.pyplot as plt

    def get_layer(m, model_name):
        if "ResNet"       in model_name: return [m.layer4[-1]]
        if "EfficientNet" in model_name: return [m.conv_head]
        return [m.blocks[-1].norm1]

    model.eval()
    cam = GradCAM(model=model, target_layers=get_layer(model, name))
    imgs_b, lbls_b = next(iter(test_loader))
    imgs_b = imgs_b[:n]; lbls_b = lbls_b[:n]

    MEAN_A = _np.array([0.485, 0.456, 0.406])
    STD_A  = _np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for i in range(n):
        inp    = imgs_b[i].unsqueeze(0).to(device)
        gc     = cam(input_tensor=inp, targets=[ClassifierOutputTarget(1)])[0]
        img_np = (imgs_b[i].permute(1, 2, 0).numpy() * STD_A + MEAN_A).clip(0, 1).astype(_np.float32)
        cam_img = show_cam_on_image(img_np, gc, use_rgb=True)
        lbl = "Down Syn" if lbls_b[i].item() == 1 else "Control"
        axes[0, i].imshow(img_np);  axes[0, i].set_title(f"Original\n{lbl}", fontsize=8); axes[0, i].axis("off")
        axes[1, i].imshow(cam_img); axes[1, i].set_title("Grad-CAM",          fontsize=8); axes[1, i].axis("off")

    plt.suptitle(f"Grad-CAM - {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = name.replace("/", "_").replace(" ", "_")
    plt.savefig(results_dir / f"{safe}_gradcam.png", dpi=150)
    plt.close()
    print(f"[OK]   Grad-CAM saved.")


# ── Training curves ───────────────────────────────────────────────────────────
def plot_training_curves(all_history, results_dir):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, hist in all_history.items():
        ep = range(1, len(hist["tr_loss"]) + 1)
        axes[0].plot(ep, hist["tr_loss"], label=f"{name} Train")
        axes[0].plot(ep, hist["vl_loss"], label=f"{name} Val", linestyle="--")
        axes[1].plot(ep, hist["tr_acc"],  label=f"{name} Train")
        axes[1].plot(ep, hist["vl_acc"],  label=f"{name} Val",  linestyle="--")
        axes[2].plot(ep, hist["vl_auc"],  label=name)

    for ax, title in zip(axes, ["Loss", "Accuracy", "Val AUC"]):
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle("Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "training_curves.png", dpi=150)
    plt.close()
    print(f"[OK]   Training curves saved.")


# ── ROC curves ────────────────────────────────────────────────────────────────
def plot_roc(test_scores, results_dir):
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    for name, res in test_scores.items():
        fpr, tpr, _ = roc_curve(res["labels"], res["probs"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(results_dir / "roc_curves.png", dpi=150)
    plt.close()
    print("[OK]   ROC curves saved.")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── 1. Get raw dataset path ───────────────────────────────────────────────
    raw_data_dir = None
    if args.kaggle:
        raw_data_dir = download_from_kaggle()
    elif args.data_dir:
        raw_data_dir = args.data_dir
    else:
        print("[INFO] No --data-dir provided. Opening folder picker...")
        raw_data_dir = pick_folder_dialog("Select your dataset root folder")
        if not raw_data_dir:
            print("[ERROR] No folder selected. Exiting.")
            print("        Run:  python train.py --data-dir 'C:/path/to/dataset'")
            sys.exit(1)

    if not os.path.isdir(raw_data_dir):
        print(f"[ERROR] Not a valid directory: {raw_data_dir}")
        sys.exit(1)

    print(f"\n[INFO] Dataset root : {raw_data_dir}")
    print(f"[INFO] Output dir   : {args.output_dir}")
    print(f"[INFO] Epochs       : {args.epochs}")
    print(f"[INFO] Batch size   : {args.batch_size}")
    print(f"[INFO] Learning rate: {args.lr}")

    # ── 2. Import heavy libs ──────────────────────────────────────────────────
    import_libs()
    import torch
    import torch.nn as nn
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader
    import numpy as np
    import pandas as pd

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"       GPU: {torch.cuda.get_device_name(0)}")

    # ── 3. Split dataset ──────────────────────────────────────────────────────
    work_dir    = Path(args.output_dir) / "_dataset_split"
    output_dir  = Path(args.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    prepare_dataset(raw_data_dir, work_dir, args.seed)

    # ── 4. Build dataloaders ──────────────────────────────────────────────────
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    IMG  = 224

    try:
        train_tf = A.Compose([
            A.Resize(IMG, IMG),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])
    except TypeError:
        # newer albumentations renamed params
        train_tf = A.Compose([
            A.Resize(IMG, IMG),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
            A.CoarseDropout(num_holes_x=8, hole_height_max=16, hole_width_max=16, p=0.3),
            A.Normalize(MEAN, STD),
            ToTensorV2(),
        ])

    val_tf = A.Compose([A.Resize(IMG, IMG), A.Normalize(MEAN, STD), ToTensorV2()])

    print("\n[STEP] Building datasets...")
    train_ds = DownSyndromeDataset(work_dir, "train", train_tf)
    val_ds   = DownSyndromeDataset(work_dir, "val",   val_tf)
    test_ds  = DownSyndromeDataset(work_dir, "test",  val_tf)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=DEVICE.type == "cuda")
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=DEVICE.type == "cuda")
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=DEVICE.type == "cuda")
    print("[OK]   Dataloaders ready.")

    # ── 5. Train all 3 models ─────────────────────────────────────────────────
    all_models  = {}
    all_history = {}

    for name, (build_fn, pth_name) in MODEL_REGISTRY.items():
        save_path = output_dir / pth_name
        model, history = train_model(
            name, build_fn, save_path,
            train_loader, val_loader,
            DEVICE, args.epochs, args.lr,
        )
        all_models[name]  = (model, save_path)
        all_history[name] = history

    # ── 6. Test evaluation ────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  Test Set Evaluation")
    print(f"{'='*58}")

    test_scores = {}
    for name, (model, save_path) in all_models.items():
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        test_scores[name] = evaluate_test(model, name, test_loader, DEVICE, results_dir)

    # ── 7. Summary table ──────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "Accuracy":  {k: v["acc"]  for k, v in test_scores.items()},
        "Precision": {k: v["prec"] for k, v in test_scores.items()},
        "Recall":    {k: v["rec"]  for k, v in test_scores.items()},
        "F1-Score":  {k: v["f1"]   for k, v in test_scores.items()},
        "AUC-ROC":   {k: v["auc"]  for k, v in test_scores.items()},
    }).round(4)

    print(f"\n  Model Comparison:")
    print(summary.to_string())
    summary.to_csv(results_dir / "metrics.csv")

    best_name = summary["AUC-ROC"].idxmax()
    print(f"\n  Best model: {best_name}  (AUC={summary.loc[best_name, 'AUC-ROC']:.4f})")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_training_curves(all_history, results_dir)
    plot_roc(test_scores, results_dir)

    # ── 9. Grad-CAM on best model ─────────────────────────────────────────────
    if not args.no_gradcam:
        best_model, best_path = all_models[best_name]
        best_model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        run_gradcam(best_model, best_name, test_loader, DEVICE, results_dir)

    # ── 10. Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print("  Training Complete!")
    print(f"{'='*58}")
    print(f"  .pth files saved to: {output_dir}")
    for _, pth_name in MODEL_REGISTRY.values():
        p = output_dir / pth_name
        status = "READY" if p.exists() else "MISSING"
        print(f"    [{status}] {pth_name}")
    print(f"\n  Plots + metrics saved to: {results_dir}")
    print(f"\n  Now run the API server:")
    print(f"    python app.py")
    print(f"  Then open: http://localhost:7860")


if __name__ == "__main__":
    main()
