# training/engine.py
# Training loop, validation, test evaluation, and early stopping.

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)


# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    """Stop training if val AUC doesn't improve for `patience` epochs."""

    def __init__(self, patience=7, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = -1.0
        self.counter    = 0
        self.should_stop = False

    def step(self, val_auc):
        if val_auc > self.best + self.min_delta:
            self.best    = val_auc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ── Training epoch ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Single training epoch. Uses AMP if scaler is provided."""
    model.train()
    total_loss = correct = total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ── Validation ────────────────────────────────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)

            total_loss += loss.item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            total      += labels.size(0)
            all_probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / total, correct / total, auc


# ── Full training run ─────────────────────────────────────────────────────────
def train_model(model, model_name, save_path,
                train_loader, val_loader,
                device, epochs, lr,
                label_smoothing=0.0,
                scheduler_fn=None,
                early_stopping_patience=7):
    """
    Train a model, save best checkpoint by Val AUC.

    Args:
        scheduler_fn: callable(optimizer) -> scheduler, or None for CosineAnnealingLR
        label_smoothing: 0.0 = standard CE, >0 = label smoothing CE
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimizer)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # AMP scaler — only active on CUDA
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    early_stop = EarlyStopping(patience=early_stopping_patience)
    best_auc   = 0.0
    history    = {k: [] for k in ["tr_loss", "vl_loss", "tr_acc", "vl_acc", "vl_auc"]}

    print(f"\n{'='*60}")
    print(f"  Training : {model_name}")
    print(f"  Epochs   : {epochs}  |  LR: {lr}  |  AMP: {scaler is not None}")
    print(f"  Device   : {device}  |  Label smoothing: {label_smoothing}")
    print(f"{'='*60}")

    for ep in range(1, epochs + 1):
        tr_l, tr_a          = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_l, vl_a, vl_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        for key, val in zip(history, [tr_l, vl_l, tr_a, vl_a, vl_auc]):
            history[key].append(val)

        saved = ""
        if vl_auc > best_auc:
            best_auc = vl_auc
            torch.save(model.state_dict(), save_path)
            saved = "  [SAVED]"

        stopped = early_stop.step(vl_auc)
        print(f"  Ep {ep:02d}/{epochs} | "
              f"Tr L:{tr_l:.4f} A:{tr_a:.3f} | "
              f"Val L:{vl_l:.4f} A:{vl_a:.3f} AUC:{vl_auc:.3f}{saved}")

        if stopped:
            print(f"  [EARLY STOP] No improvement for {early_stopping_patience} epochs.")
            break

    print(f"\n[OK]   Best Val AUC : {best_auc:.4f}")
    print(f"       Weights     -> {save_path}")
    return history


# ── Test evaluation ───────────────────────────────────────────────────────────
def evaluate_test(model, model_name, test_loader, device, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    model.eval()
    preds_all, probs_all, lbl_all = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            out = model(imgs.to(device))
            probs_all.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
            preds_all.extend(out.argmax(1).cpu().numpy())
            lbl_all.extend(labels.numpy())

    acc  = accuracy_score(lbl_all,  preds_all)
    prec = precision_score(lbl_all, preds_all, zero_division=0)
    rec  = recall_score(lbl_all,    preds_all, zero_division=0)
    f1   = f1_score(lbl_all,        preds_all, zero_division=0)
    auc  = roc_auc_score(lbl_all,   probs_all) if len(set(lbl_all)) > 1 else 0.0

    print(f"\n  {model_name} — Test Results")
    print(f"  {'─'*38}")
    print(f"  Accuracy  : {acc:.4f}  {'OK' if acc  >= 0.85 else 'LOW'} (target >= 0.85)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}   {'OK' if f1   >= 0.83 else 'LOW'} (target >= 0.83)")
    print(f"  AUC-ROC   : {auc:.4f}  {'OK' if auc  >= 0.88 else 'LOW'} (target >= 0.88)")

    # Confusion matrix
    cm   = confusion_matrix(lbl_all, preds_all)
    safe = model_name.replace("/", "_").replace(" ", "_").replace("-", "_")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Control", "Down Syn"],
                yticklabels=["Control", "Down Syn"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(results_dir / f"{safe}_confusion.png", dpi=150)
    plt.close()

    return dict(model=model_name, acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                preds=preds_all, probs=probs_all, labels=lbl_all)
