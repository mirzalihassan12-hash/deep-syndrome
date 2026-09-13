# inference.py
# Shared model loading + prediction logic, used identically by the FastAPI
# backend (main.py) and the Hugging Face Gradio app (app_hf.py). Deliberately
# has no FastAPI dependency, so the Gradio deployment doesn't need to install
# fastapi/uvicorn at all.

import io
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from PIL import Image

from face_crop import crop_to_face

IMG_SIZE    = 224
NUM_CLASSES = 2
DEVICE      = torch.device("cpu")
MEAN        = np.array([0.485, 0.456, 0.406])
STD         = np.array([0.229, 0.224, 0.225])

MODEL_PATHS = {
    "ResNet-50":       "ResNet50_best.pth",
    "EfficientNet-B3": "EfficientNet_B3_best.pth",
    "ViT-S/16":        "ViT_S16_best.pth",
}


# ── Model Architecture Definitions ───────────────────────────
def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m


def build_efficientnet():
    return timm.create_model("efficientnet_b3", pretrained=False, num_classes=NUM_CLASSES)


def build_vit():
    return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=NUM_CLASSES)


BUILDERS = {
    "ResNet-50":       build_resnet50,
    "EfficientNet-B3": build_efficientnet,
    "ViT-S/16":        build_vit,
}


# ── Load All Models at Startup ────────────────────────────────
loaded_models = {}


def load_all_models():
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: {path} not found")
            continue
        try:
            model = BUILDERS[name]()
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            loaded_models[name] = model
            print(f"[OK] {name} loaded!")
        except Exception as e:
            print(f"[FAIL] {name} load fail: {e}")

    print(f"\n[READY] {len(loaded_models)}/3 models ready!")
    if loaded_models:
        print(f"   Active: {list(loaded_models.keys())}")


load_all_models()


# ── Image Preprocessing ───────────────────────────────────────
def preprocess(image_bytes: bytes) -> tuple[torch.Tensor, bool]:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img, face_found = crop_to_face(img)
    img    = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - MEAN) / STD
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, face_found


# ── Per-model predictions ──────────────────────────────────────
def predict_individual(tensor: torch.Tensor) -> dict:
    """Each loaded model's own prediction, independent of the ensemble average."""
    individual = {}
    with torch.no_grad():
        for name, model in loaded_models.items():
            out   = model(tensor.to(DEVICE))
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            idx   = int(np.argmax(probs))
            individual[name] = {
                "prediction": "Down Syndrome" if idx == 1 else "Control",
                "confidence": round(float(probs[idx]) * 100, 2),
            }
    return individual


# ── Ensemble Prediction ───────────────────────────────────────
def ensemble_predict(tensor: torch.Tensor):
    """Average all loaded models' probabilities. Returns (avg_probs, pred_idx) or (None, None)."""
    all_probs = []
    with torch.no_grad():
        for model in loaded_models.values():
            out   = model(tensor.to(DEVICE))
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    if not all_probs:
        return None, None

    avg_probs = np.mean(all_probs, axis=0)
    pred_idx  = int(np.argmax(avg_probs))
    return avg_probs, pred_idx
