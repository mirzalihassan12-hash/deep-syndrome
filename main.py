"""
DeepSyndrome FastAPI Backend
Ensemble of ResNet-50 + EfficientNet-B3 + ViT-S/16
Deploy on: any Docker host (e.g. HuggingFace Spaces Docker SDK, if available)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

from inference import loaded_models, preprocess, ensemble_predict, predict_individual

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


# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse("static/index.html")


# ── PWA files — served from root scope so the service worker can control "/" ──
@app.get("/manifest.json")
def manifest():
    return FileResponse("static/manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    return Response(
        content=open("static/sw.js").read(),
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/"},
    )


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
        tensor, face_found = preprocess(contents)
        avg_probs, pred_idx = ensemble_predict(tensor)

        if avg_probs is None:
            raise HTTPException(500, "Prediction fail")

        pred_label = "Down Syndrome" if pred_idx == 1 else "Control"
        confidence = round(float(avg_probs[pred_idx]) * 100, 2)
        individual = predict_individual(tensor)

        return {
            "prediction":    pred_label,
            "confidence":    confidence,
            "probabilities": {
                "control":       round(float(avg_probs[0]) * 100, 2),
                "down_syndrome": round(float(avg_probs[1]) * 100, 2),
            },
            "models_used":   list(loaded_models.keys()),
            "individual":    individual,
            "demo_mode":     False,
            "face_detected": face_found,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")
