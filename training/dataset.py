# training/dataset.py
# Dataset preparation, transforms, and dataloaders — shared by all training scripts.

import os
import sys
import shutil
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DS_KEYWORDS   = ["down", "syndrome", "ds", "positive", "affected"]
CTRL_KEYWORDS = ["control", "normal", "healthy", "negative", "non"]

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224


# ── Folder discovery ──────────────────────────────────────────────────────────
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


# ── 70 / 15 / 15 split ───────────────────────────────────────────────────────
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
    print(f"       DS folders   : {len(ds_folders)}")
    print(f"       Ctrl folders : {len(ctrl_folders)}")

    if not ds_folders:
        print("[ERROR] No Down Syndrome folders found.")
        print("        Folder names must contain:", DS_KEYWORDS)
        sys.exit(1)
    if not ctrl_folders:
        print("[ERROR] No Control folders found.")
        print("        Folder names must contain:", CTRL_KEYWORDS)
        sys.exit(1)

    print("\n[STEP] Splitting 70 / 15 / 15 ...")
    split_and_copy(ds_folders,   "down_syndrome", work_dir, seed)
    split_and_copy(ctrl_folders, "control",       work_dir, seed)
    print("[OK]   Split complete.\n")


# ── Dataset class ─────────────────────────────────────────────────────────────
class DownSyndromeDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
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
        print(f"   [{split:5s}] total={len(self.paths):4d}  "
              f"DS={self.labels.count(1):4d}  ctrl={self.labels.count(0):4d}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, self.labels[idx]


# ── Transforms ────────────────────────────────────────────────────────────────
def _coarse_dropout(**kwargs):
    """Handle albumentations API changes across versions."""
    try:
        return A.CoarseDropout(max_holes=8, max_height=16, max_width=16, **kwargs)
    except TypeError:
        return A.CoarseDropout(num_holes_x=8, hole_height_max=16, hole_width_max=16, **kwargs)


def build_train_transform(extra_aug=None):
    """
    extra_aug: optional list of additional A.* transforms inserted after
               basic spatial augmentation (before normalize).
    """
    aug = [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.4),
        _coarse_dropout(p=0.3),
    ]
    if extra_aug:
        aug.extend(extra_aug)
    aug += [A.Normalize(MEAN, STD), ToTensorV2()]
    return A.Compose(aug)


def build_val_transform():
    return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(MEAN, STD), ToTensorV2()])


# ── Dataloaders ───────────────────────────────────────────────────────────────
def build_dataloaders(work_dir, batch_size, workers, train_transform=None):
    if train_transform is None:
        train_transform = build_train_transform()
    val_tf = build_val_transform()

    print("\n[STEP] Building datasets...")
    train_ds = DownSyndromeDataset(work_dir, "train", train_transform)
    val_ds   = DownSyndromeDataset(work_dir, "val",   val_tf)
    test_ds  = DownSyndromeDataset(work_dir, "test",  val_tf)

    pin = True  # pin_memory works on both CPU and GPU
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    print("[OK]   Dataloaders ready.")
    return train_loader, val_loader, test_loader
