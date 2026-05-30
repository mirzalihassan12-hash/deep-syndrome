# training/plots.py
# Visualization utilities: training curves, ROC, Grad-CAM, metrics CSV.

import csv
from pathlib import Path

import numpy as np


def save_metrics_csv(scores, results_dir):
    """Save per-model test metrics to a CSV file."""
    path = Path(results_dir) / "metrics.csv"
    fields = ["model", "acc", "prec", "rec", "f1", "auc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scores:
            w.writerow({k: round(s[k], 4) if isinstance(s[k], float) else s[k]
                        for k in fields})
    print(f"[OK]   Metrics CSV -> {path}")


def plot_training_curves(history, model_name, results_dir):
    """Plot loss / accuracy / AUC curves for a single model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep  = range(1, len(history["tr_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep, history["tr_loss"], label="Train")
    axes[0].plot(ep, history["vl_loss"], label="Val", linestyle="--")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["tr_acc"], label="Train")
    axes[1].plot(ep, history["vl_acc"], label="Val",  linestyle="--")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(ep, history["vl_auc"], color="darkorange")
    axes[2].set_title("Val AUC"); axes[2].grid(alpha=0.3)

    plt.suptitle(f"Training Curves — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = model_name.replace("/", "_").replace(" ", "_").replace("-", "_")
    out  = Path(results_dir) / f"{safe}_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK]   Training curves -> {out}")


def plot_roc(scores, results_dir):
    """Plot ROC curve(s). scores = list of evaluate_test() dicts."""
    from sklearn.metrics import roc_curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    for s in scores:
        fpr, tpr, _ = roc_curve(s["labels"], s["probs"])
        plt.plot(fpr, tpr, lw=2, label=f"{s['model']} (AUC={s['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
    plt.legend(); plt.tight_layout()
    out = Path(results_dir) / "roc_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK]   ROC curve -> {out}")


def run_gradcam(model, model_name, test_loader, device, results_dir, n=6):
    """Grad-CAM visualization for the first n test images."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        print("[SKIP] pytorch-grad-cam not installed. pip install grad-cam")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def get_layer(m, name):
        if "ResNet"        in name: return [m.layer4[-1]]
        if "EfficientNet"  in name: return [m.conv_head]
        return [m.blocks[-1].norm1]  # ViT

    model.eval()
    cam      = GradCAM(model=model, target_layers=get_layer(model, model_name))
    imgs_b, lbls_b = next(iter(test_loader))
    imgs_b   = imgs_b[:n]; lbls_b = lbls_b[:n]

    MEAN_A = np.array([0.485, 0.456, 0.406])
    STD_A  = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    for i in range(n):
        inp    = imgs_b[i].unsqueeze(0).to(device)
        gc     = cam(input_tensor=inp, targets=[ClassifierOutputTarget(1)])[0]
        img_np = (imgs_b[i].permute(1, 2, 0).numpy() * STD_A + MEAN_A).clip(0, 1).astype(np.float32)
        cam_img = show_cam_on_image(img_np, gc, use_rgb=True)
        lbl    = "Down Syn" if lbls_b[i].item() == 1 else "Control"
        axes[0, i].imshow(img_np);  axes[0, i].set_title(f"Original\n{lbl}", fontsize=8); axes[0, i].axis("off")
        axes[1, i].imshow(cam_img); axes[1, i].set_title("Grad-CAM", fontsize=8);          axes[1, i].axis("off")

    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe = model_name.replace("/", "_").replace(" ", "_").replace("-", "_")
    out  = Path(results_dir) / f"{safe}_gradcam.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[OK]   Grad-CAM -> {out}")
