---
title: Deepsyndrome Api
emoji: 🧬
colorFrom: indigo
colorTo: cyan
sdk: docker
pinned: false
---

# DeepSyndrome — AI Diagnostic System

Ensemble of **ResNet-50 + EfficientNet-B3 + ViT-S/16** for Down Syndrome detection from facial imagery.

---

## Project Structure

```
deep-syndrome/
├── app.py                   # Entry point — starts the server
├── main.py                  # FastAPI app + model loading + API routes
├── train.py                 # Local model training script  ← train here
├── static/
│   └── index.html           # Web UI (served at http://localhost:7860)
├── requirements.txt         # Server dependencies
├── requirements_train.txt   # Training dependencies (includes ML extras)
├── ResNet50_best.pth        # Trained ResNet-50 weights
├── EfficientNet_B3_best.pth # Trained EfficientNet-B3 weights
├── ViT_S16_best.pth         # Trained ViT-S/16 weights
└── results/                 # Created after training — plots & metrics
```

---

## Quick Start — Run the Server

> Pre-trained `.pth` files are already included. Just install and run.

**Step 1 — Install server dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Start the server**
```bash
python app.py
```

**Step 3 — Open the web UI**
```
http://localhost:7860
```

Upload any facial image and get predictions from all 3 models instantly.

---

## Train Your Own Models

Use this when you want to retrain on a new or updated dataset.
After training finishes, the `.pth` files in the project root are automatically replaced.

### Step 1 — Install training dependencies
```bash
pip install -r requirements_train.txt
```

### Step 2 — Prepare your dataset

Your dataset folder should have subfolders named with these keywords:

| Class | Accepted folder name keywords |
|---|---|
| Down Syndrome | `down`, `syndrome`, `ds`, `positive`, `affected` |
| Control (no DS) | `control`, `normal`, `healthy`, `negative`, `non` |

**Example valid structures:**
```
my_dataset/
├── down_syndrome/     ← images here
└── control/           ← images here
```
```
my_dataset/
├── DS_images/
├── DS_more/
└── normal_faces/
```

The script auto-explores all subfolders recursively and splits 70% train / 15% val / 15% test.

### Step 3 — Run training

**Option A — Folder picker dialog (easiest)**
```bash
python train.py
```
A folder selection window opens. Pick your dataset root folder.

**Option B — Pass path directly**
```bash
python train.py --data-dir "C:/path/to/your/dataset"
```

**Option C — Download from Kaggle**
```bash
pip install kagglehub
python train.py --kaggle
```
Requires a Kaggle account and API key (`~/.kaggle/kaggle.json`).

### Step 4 — Custom training options

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | (dialog) | Path to dataset root folder |
| `--output-dir` | project root | Where to save `.pth` files |
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--no-gradcam` | off | Skip Grad-CAM visualization |
| `--workers` | `0` | DataLoader workers (keep 0 on Windows) |
| `--kaggle` | off | Auto-download from Kaggle |

**Examples:**
```bash
# Fast test run
python train.py --data-dir "C:/data/ds_dataset" --epochs 5 --batch-size 16

# Full training run
python train.py --data-dir "C:/data/ds_dataset" --epochs 30
```

### Step 5 — Restart the server to load new weights
```bash
python app.py
```

Training outputs saved to `results/`:
- `training_curves.png` — loss / accuracy / AUC per epoch
- `roc_curves.png` — ROC curve comparison
- `*_confusion.png` — confusion matrix per model
- `*_gradcam.png` — Grad-CAM heatmaps (best model)
- `metrics.csv` — accuracy, F1, AUC-ROC table

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Model load status |
| `GET` | `/api` | API info JSON |
| `POST` | `/predict` | Run prediction — send `multipart/form-data` with `file` field |
| `GET` | `/docs` | Swagger UI (interactive API docs) |

**Example `curl` request:**
```bash
curl -X POST http://localhost:7860/predict \
  -F "file=@face.jpg"
```

**Example response:**
```json
{
  "prediction": "Down Syndrome",
  "confidence": 87.43,
  "probabilities": {
    "control": 12.57,
    "down_syndrome": 87.43
  },
  "models_used": ["ResNet-50", "EfficientNet-B3", "ViT-S/16"],
  "individual": {
    "ResNet-50":       { "prediction": "Down Syndrome", "confidence": 91.2 },
    "EfficientNet-B3": { "prediction": "Down Syndrome", "confidence": 85.6 },
    "ViT-S/16":        { "prediction": "Control",       "confidence": 78.1 }
  },
  "demo_mode": false
}
```

---

## Notes

- Images are processed in-memory and never saved to disk.
- If no `.pth` files are found, the server runs in **Demo Mode** (random predictions).
- On Windows, keep `--workers 0` during training to avoid multiprocessing issues.
- CPU inference is supported. GPU (CUDA) is auto-detected and used if available.
