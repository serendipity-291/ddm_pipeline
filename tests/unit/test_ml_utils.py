import numpy as np
import pytest
import torch

from ml_utils import extract_features, predict_window


class DummyScaler:
    def transform(self, x):
        return x


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # 2-class deterministic logits.
        return torch.tensor([[1.0, 0.0]], dtype=torch.float32, device=x.device)


@pytest.mark.unit
def test_extract_features_shape():
    window = np.random.randn(2048)
    feats = extract_features(window)
    assert feats.shape == (1, 22)


@pytest.mark.unit
def test_predict_window_contract():
    window = np.random.randn(2048)
    out = predict_window(
        window=window,
        model=DummyModel(),
        scaler=DummyScaler(),
        class_mapping={0: "B", 1: "Normal"},
    )
    assert "predicted_class" in out
    assert "confidence" in out
    assert "probabilities" in out
    assert set(out["probabilities"].keys()) == {"B", "Normal"}
