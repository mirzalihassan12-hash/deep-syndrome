# face_crop.py
# Shared face detection/cropping — used identically by both the training
# pipeline and the live inference server, so train and inference see the
# same kind of input. Deliberately lightweight (OpenCV Haar cascade, no
# extra heavy dependency) since it needs to run on both.
#
# Why this exists: dataset audit + live testing showed the model performs
# near-perfectly on images from this dataset's own distribution, but poorly
# on any other real-world photo — a classic domain-shift symptom. Cropping
# tightly to the detected face removes background/lighting/framing
# differences between sources, forcing the model to actually look at facial
# structure instead of incidental image context.

import cv2
import numpy as np
from PIL import Image

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MARGIN = 0.30  # padding around the detected face box, as a fraction of its size


def crop_to_face(image: Image.Image, margin: float = MARGIN) -> tuple[Image.Image, bool]:
    """
    Detect the largest face in `image` and crop to it with margin.
    Returns (cropped_image, face_found). Falls back to the original image
    (uncropped) if no face is detected, rather than failing — a photo with
    unusual lighting/angle shouldn't just be dropped.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return image, False

    # Largest detected face by area, in case of multiple/false detections.
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    mx, my = int(w * margin), int(h * margin)
    H, W = rgb.shape[:2]
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)

    return image.crop((x0, y0, x1, y1)), True
