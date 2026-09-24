"""
Test cases for inference.py — image preprocessing and the ensemble averaging
logic. Model checkpoints (.pth files) are intentionally excluded from the
code submission (100MB+ each), so these tests mock the loaded models rather
than requiring the real weights: the goal here is verifying the *math* of
preprocessing and ensembling, which is independent of any specific trained
model.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import inference
from inference import IMG_SIZE, preprocess, predict_individual, ensemble_predict

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class _FakeModel:
    """A stand-in for a loaded nn.Module: returns fixed logits regardless of input."""
    def __init__(self, logits):
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, tensor):
        return self._logits

    def to(self, device):
        return self


def _sample_image_bytes():
    with open(FIXTURES / "sample_face.jpg", "rb") as f:
        return f.read()


def test_preprocess_returns_correctly_shaped_tensor():
    tensor, face_found = preprocess(_sample_image_bytes())

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all(), "normalized tensor must not contain NaN/Inf"
    assert isinstance(face_found, bool)


def test_preprocess_handles_rgba_input(tmp_path):
    # Some uploads are PNGs with an alpha channel; preprocess() must not crash on them.
    rgba_path = tmp_path / "rgba.png"
    Image.new("RGBA", (100, 100), color=(10, 20, 30, 128)).save(rgba_path)

    tensor, _ = preprocess(rgba_path.read_bytes())
    assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_ensemble_predict_averages_softmax_across_models(monkeypatch):
    # Two fake models with different, known logits.
    fake_models = {
        "ModelA": _FakeModel([2.0, 0.0]),   # confidently class 0
        "ModelB": _FakeModel([0.0, 2.0]),   # confidently class 1
    }
    monkeypatch.setattr(inference, "loaded_models", fake_models)

    dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))
    avg_probs, pred_idx = ensemble_predict(dummy_input)

    expected_a = torch.softmax(torch.tensor([2.0, 0.0]), dim=0).numpy()
    expected_b = torch.softmax(torch.tensor([0.0, 2.0]), dim=0).numpy()
    expected_avg = (expected_a + expected_b) / 2

    assert avg_probs is not None
    np.testing.assert_allclose(avg_probs, expected_avg, rtol=1e-5)
    # Both models cancel out symmetrically, so it's a coin flip — argmax must
    # still pick a valid class index (0 or 1), not crash.
    assert pred_idx in (0, 1)


def test_ensemble_predict_returns_none_when_no_models_loaded(monkeypatch):
    monkeypatch.setattr(inference, "loaded_models", {})
    dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))

    avg_probs, pred_idx = ensemble_predict(dummy_input)
    assert avg_probs is None
    assert pred_idx is None


def test_predict_individual_reports_each_models_own_verdict(monkeypatch):
    fake_models = {
        "ResNet-50": _FakeModel([3.0, 0.0]),       # strongly Control
        "ViT-S/16":  _FakeModel([0.0, 3.0]),       # strongly Down Syndrome
    }
    monkeypatch.setattr(inference, "loaded_models", fake_models)

    dummy_input = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))
    result = predict_individual(dummy_input)

    assert set(result.keys()) == {"ResNet-50", "ViT-S/16"}
    assert result["ResNet-50"]["prediction"] == "Control"
    assert result["ViT-S/16"]["prediction"] == "Down Syndrome"
    for entry in result.values():
        assert 0.0 <= entry["confidence"] <= 100.0
