# app_hf.py
# Gradio front-end for Hugging Face Spaces. Hugging Face's free tier no
# longer offers Docker Spaces (Docker is now paid) - Gradio (or Static) is
# what's free. This uses the exact same inference.py logic as the FastAPI
# backend (main.py), so results are identical either way; only the web
# layer differs.

import io

import gradio as gr
import spaces
from PIL import Image

from inference import loaded_models, preprocess, ensemble_predict, predict_individual

DISCLAIMER = (
    "⚠️ **Research/educational screening aid only — not a medical diagnosis.** "
    "Always consult a qualified clinician."
)


@spaces.GPU
def run_prediction(image: Image.Image):
    if image is None:
        return "Please upload a facial image.", None

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    if not loaded_models:
        return "⚠️ Demo Mode — no model weights found on this Space.", None

    tensor, face_found = preprocess(image_bytes)
    avg_probs, pred_idx = ensemble_predict(tensor)
    if avg_probs is None:
        return "Prediction failed — no models loaded.", None

    pred_label = "Down Syndrome" if pred_idx == 1 else "Control"
    confidence = float(avg_probs[pred_idx]) * 100
    individual = predict_individual(tensor)

    lines = [
        f"## {'⚠️' if pred_idx == 1 else '✅'} {pred_label} — {confidence:.1f}% confidence",
        f"Face detected in image: {'Yes' if face_found else 'No (used full image as fallback)'}",
        "",
        "### Individual model votes",
    ]
    for name, info in individual.items():
        lines.append(f"- **{name}**: {info['prediction']} ({info['confidence']:.1f}%)")
    lines.append("")
    lines.append(DISCLAIMER)

    probs_out = {
        "Control": float(avg_probs[0]),
        "Down Syndrome": float(avg_probs[1]),
    }
    return "\n".join(lines), probs_out


demo = gr.Interface(
    fn=run_prediction,
    inputs=gr.Image(type="pil", label="Upload a facial image"),
    outputs=[
        gr.Markdown(label="Result"),
        gr.Label(label="Class probabilities"),
    ],
    title="🧬 DeepSyndrome — Ensemble Down Syndrome Screening",
    description=(
        "Ensemble of ResNet-50, EfficientNet-B3, and ViT-S/16. "
        "Upload a facial image to get a screening prediction. " + DISCLAIMER
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
