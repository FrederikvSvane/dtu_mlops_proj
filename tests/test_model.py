import torch
from mlops_proj.model import Classifier


def test_model():
    model = Classifier()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
