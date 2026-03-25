import numpy as np
import pytest
import torch

from ml_utils import FaultMLP


@pytest.mark.model_smoke
def test_faultmlp_forward_shape():
    model = FaultMLP(input_dim=22, num_classes=2)
    batch = torch.from_numpy(np.random.randn(4, 22).astype("float32"))
    logits = model(batch)
    assert logits.shape == (4, 2)
