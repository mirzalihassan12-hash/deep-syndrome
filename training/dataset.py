# training/dataset.py
# Dataset preparation, transforms, and dataloaders — shared by all training scripts.

import os
import sys
import random
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from face_crop import crop_to_face

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


# ── Near-duplicate grouping (prevents train/val/test leakage) ────────────────
def _ahash(path, hash_size=8):
    """8x8 average-hash — cheap perceptual fingerprint, no numpy needed."""
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for p in pixels:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return bits


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _group_near_duplicates(paths, threshold=0):
    """
    Cluster near-identical images (e.g. re-saved/re-compressed copies) so they
    always land together in one split. threshold=0 (exact aHash match) is
    intentionally strict: a looser threshold starts clustering images that
    merely share similar lighting/composition rather than being duplicates.
    """
    hashes = [_ahash(p) for p in paths]
    n = len(paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _hamming(hashes[i], hashes[j]) <= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(paths[i])
    return list(groups.values())


# ── 70 / 15 / 15 split ───────────────────────────────────────────────────────
def split_and_copy(folders, class_name, dest_root, seed):
    all_imgs = [os.path.join(f, img) for f, imgs in folders for img in imgs]

    # Group near-duplicates first so a duplicate cluster never straddles splits.
    groups = _group_near_duplicates(all_imgs)
    random.seed(seed)
    random.shuffle(groups)

    n    = len(all_imgs)
    n_tr = int(n * 0.70)
    n_vl = int(n * 0.15)

    splits = {"train": [], "val": [], "test": []}
    for group in groups:
        if len(splits["train"]) < n_tr:
            splits["train"].extend(group)
        elif len(splits["val"]) < n_vl:
            splits["val"].extend(group)
        else:
            splits["test"].extend(group)

    multi = [g for g in groups if len(g) > 1]
    if multi:
        print(f"   [dedup] {class_name}: {len(multi)} near-duplicate group(s) "
              f"kept together (sizes: {sorted((len(g) for g in multi), reverse=True)})")

    for split, imgs in splits.items():
        dest = Path(dest_root) / split / class_name
        dest.mkdir(parents=True, exist_ok=True)
        n_face = 0
        for src in imgs:
            img = Image.open(src).convert("RGB")
            cropped, found = crop_to_face(img)
            n_face += int(found)
            cropped.save(dest / Path(src).name)
        print(f"   [{split:5s}] {class_name}: {len(imgs)} images "
              f"(face detected & cropped: {n_face}/{len(imgs)})")
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
    """
    Handle albumentations API changes across versions. Newer versions (2.x)
    silently warn and ignore unknown kwargs instead of raising TypeError, so
    a try/except can't detect a mismatch here — inspect the actual signature.
    """
    params = inspect.signature(A.CoarseDropout.__init__).parameters
    if "num_holes_range" in params:
        return A.CoarseDropout(
            num_holes_range=(1, 8), hole_height_range=(0.03, 0.07),
            hole_width_range=(0.03, 0.07), **kwargs,
        )
    if "max_holes" in params:
        return A.CoarseDropout(max_holes=8, max_height=16, max_width=16, **kwargs)
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
        # Sharpness/noise-invariance: dataset audit found DS images are ~2x
        # sharper/noisier (Laplacian variance) than control images on average —
        # a source/compression artifact, not a real facial feature. Without
        # this, models learn "high-frequency noise = Down Syndrome" (confirmed
        # by pure random noise being classified as DS at 100% confidence).
        # Randomizing sharpness/noise on every image breaks that correlation.
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.GaussNoise(std_range=(0.05, 0.35), p=1.0),
            A.Sharpen(alpha=(0.2, 0.6), lightness=(0.7, 1.3), p=1.0),
            A.ImageCompression(quality_range=(30, 90), p=1.0),
        ], p=0.7),
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
