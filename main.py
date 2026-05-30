"""
DeepSyndrome FastAPI Backend
Ensemble of ResNet-50 + EfficientNet-B3 + ViT-S/16
Deploy on: HuggingFace Spaces (FREE)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from PIL import Image
import numpy as np
import io
import os

app = FastAPI(title="DeepSyndrome Ensemble API", version="2.0.0")

# ── CORS ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Config ────────────────────────────────────────────────────
IMG_SIZE    = 224
NUM_CLASSES = 2
DEVICE      = torch.device("cpu")
MEAN        = np.array([0.485, 0.456, 0.406])
STD         = np.array([0.229, 0.224, 0.225])

# .pth file paths — HuggingFace pe in files ko upload karna hai
MODEL_PATHS = {
    "ResNet-50":       "ResNet50_best.pth",
    "EfficientNet-B3": "EfficientNet_B3_best.pth",
    "ViT-S/16":        "ViT_S16_best.pth",
}


# ── Model Architecture Definitions ───────────────────────────
def build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, NUM_CLASSES)
    )
    return m

def build_efficientnet():
    return timm.create_model(
        'efficientnet_b3', pretrained=False, num_classes=NUM_CLASSES
    )

def build_vit():
    return timm.create_model(
        'vit_small_patch16_224', pretrained=False, num_classes=NUM_CLASSES
    )

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
def preprocess(image_bytes: bytes) -> torch.Tensor:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img    = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - MEAN) / STD
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


# ── Ensemble Prediction ───────────────────────────────────────
def ensemble_predict(tensor: torch.Tensor):
    """
    Teeno models ki probabilities average karo.
    Agar koi model nahi mila to jo available hain unka use karo.
    """
    all_probs = []

    with torch.no_grad():
        for name, model in loaded_models.items():
            out   = model(tensor.to(DEVICE))
            probs = torch.softmax(out, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    if not all_probs:
        return None, None

    # Average of all models
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx  = int(np.argmax(avg_probs))

    return avg_probs, pred_idx


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.get("/api")
def api_info():
    return {
        "message":        "DeepSyndrome Ensemble API",
        "models_loaded":  len(loaded_models),
        "active_models":  list(loaded_models.keys()),
        "endpoint":       "POST /predict"
    }


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_ready":  len(loaded_models),
        "models":        list(loaded_models.keys())
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Image upload → Ensemble prediction

    Returns:
        - prediction     : "Down Syndrome" ya "Control"
        - confidence     : percentage
        - probabilities  : dono classes ki %
        - models_used    : kaunse models ne predict kiya
        - individual     : har model ki alag prediction
    """
    # Validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Sirf image files allowed hain")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image 10MB se badi hai")

    # Demo mode — koi model nahi mila
    if not loaded_models:
        import random
        conf  = round(random.uniform(60, 95), 2)
        label = random.choice(["Down Syndrome", "Control"])
        return {
            "prediction":    label,
            "confidence":    conf,
            "probabilities": {
                "control":       round(100 - conf, 2) if label == "Down Syndrome" else conf,
                "down_syndrome": conf if label == "Down Syndrome" else round(100 - conf, 2),
            },
            "models_used":   [],
            "individual":    {},
            "demo_mode":     True,
            "message":       "Koi .pth file nahi mili — demo results"
        }

    # Real prediction
    try:
        tensor = preprocess(contents)
        avg_probs, pred_idx = ensemble_predict(tensor)

        if avg_probs is None:
            raise HTTPException(500, "Prediction fail")

        pred_label = "Down Syndrome" if pred_idx == 1 else "Control"
        confidence = round(float(avg_probs[pred_idx]) * 100, 2)

        # Har model ki individual prediction
        individual = {}
        with torch.no_grad():
            for name, model in loaded_models.items():
                out   = model(tensor.to(DEVICE))
                probs = torch.softmax(out, dim=1)[0].cpu().numpy()
                idx   = int(np.argmax(probs))
                individual[name] = {
                    "prediction": "Down Syndrome" if idx == 1 else "Control",
                    "confidence": round(float(probs[idx]) * 100, 2)
                }

        return {
            "prediction":    pred_label,
            "confidence":    confidence,
            "probabilities": {
                "control":       round(float(avg_probs[0]) * 100, 2),
                "down_syndrome": round(float(avg_probs[1]) * 100, 2),
            },
            "models_used":   list(loaded_models.keys()),
            "individual":    individual,
            "demo_mode":     False
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
