"""
Test cases for face_crop.crop_to_face().

Covers: successful face detection + cropping with margin, and the fallback
behaviour (return the original image unchanged) when no face is found —
this fallback matters because a photo with unusual lighting/angle should
never just be dropped from the pipeline.
"""
from pathlib import Path

from PIL import Image

from face_crop import crop_to_face

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_crop_to_face_detects_real_face():
    img = Image.open(FIXTURES / "sample_face.jpg")
    cropped, found = crop_to_face(img)

    assert found is True
    # A real crop should be smaller than (or equal to, if the face fills the
    # frame) the original image, never larger.
    assert cropped.size[0] <= img.size[0]
    assert cropped.size[1] <= img.size[1]
    # The crop must still be a valid, non-empty image.
    assert cropped.size[0] > 0 and cropped.size[1] > 0


def test_crop_to_face_falls_back_when_no_face_present():
    # A flat solid-colour image has no face for the Haar cascade to detect.
    blank = Image.new("RGB", (300, 300), color=(120, 120, 120))
    result, found = crop_to_face(blank)

    assert found is False
    # Fallback must return the original image untouched, not crash or crop blindly.
    assert result.size == blank.size


def test_crop_to_face_margin_is_configurable():
    img = Image.open(FIXTURES / "sample_face.jpg")
    tight, found_tight = crop_to_face(img, margin=0.0)
    wide, found_wide = crop_to_face(img, margin=0.6)

    assert found_tight and found_wide
    # A larger margin should never produce a smaller (or equal on both axes)
    # crop than a tighter margin on the same image.
    wide_area = wide.size[0] * wide.size[1]
    tight_area = tight.size[0] * tight.size[1]
    assert wide_area >= tight_area
