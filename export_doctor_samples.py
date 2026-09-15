#!/usr/bin/env python3
"""
Offline utility: pulls doctor-confirmed (image, label) pairs from the
`confirmed_samples` MongoDB collection (populated by the Next.js Doctor Mode
feature) and writes them to data/doctor_confirmed/{down_syndrome,control}/*.jpg
so training/dataset.py's existing find_class_folders()/prepare_dataset() picks
them up automatically -- zero changes to the training pipeline.

Usage: python export_doctor_samples.py [--uri ...] [--db ...] [--out ...]
Requires: pip install pymongo
"""
import argparse
import base64
import csv
import os
import sys
from pathlib import Path

from pymongo import MongoClient

LABEL_TO_FOLDER = {"Down Syndrome": "down_syndrome", "Control": "control"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uri", default=os.environ.get("MONGODB_URI"))
    p.add_argument("--db", default=os.environ.get("MONGODB_DB", "deep_syndrome"))
    p.add_argument("--out", default="data/doctor_confirmed")
    args = p.parse_args()
    if not args.uri:
        sys.exit("Error: pass --uri or set MONGODB_URI.")

    collection = MongoClient(args.uri)[args.db]["confirmed_samples"]
    out_root = Path(args.out)
    for folder in LABEL_TO_FOLDER.values():
        (out_root / folder).mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    exported = skipped = 0
    for doc in collection.find({}):
        folder = LABEL_TO_FOLDER.get(doc.get("doctor_label"))
        b64 = doc.get("image_base64")
        if not folder or not b64:
            skipped += 1
            continue
        try:
            raw = base64.b64decode(b64)
        except Exception:
            skipped += 1
            continue

        content_type = doc.get("content_type", "")
        ext = ".png" if "png" in content_type else ".webp" if "webp" in content_type else ".jpg"
        filename = f"doctor_{doc['_id']}{ext}"
        (out_root / folder / filename).write_bytes(raw)

        metadata_rows.append({
            "filename": filename,
            "class": folder,
            "doctor_email": doc.get("doctor_email", ""),
            "model_prediction": doc.get("model_prediction", ""),
            "model_confidence": doc.get("model_confidence", ""),
            "face_detected": doc.get("face_detected", ""),
            "created_at": doc.get("created_at", ""),
        })
        exported += 1

    if metadata_rows:
        with open(out_root / "metadata.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metadata_rows)

    print(f"Exported {exported} image(s) to {out_root.resolve()} (skipped {skipped})")
    print("Folder names match DS_KEYWORDS/CTRL_KEYWORDS in training/dataset.py.")
    print(f"Metadata written to {out_root / 'metadata.csv'}")


if __name__ == "__main__":
    main()
