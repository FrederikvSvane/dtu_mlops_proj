import torch
from mlops_proj.model import Classifier


def test_model_output_shape():
    model = Classifier()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_model_batch_handling():
    """Model should handle various batch sizes."""
    model = Classifier()
    for batch_size in [1, 8, 32]:
        x = torch.randn(batch_size, 1, 28, 28)
        y = model(x)
        assert y.shape == (batch_size, 10)


def test_model_eval_mode_deterministic():
    """Model should be deterministic in eval mode (dropout disabled)."""
    model = Classifier()
    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert torch.allclose(y1, y2)


def test_model_parameters():
    """Model should have a reasonable number of parameters."""
    model = Classifier()
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 1000, "Model has too few parameters"
    assert num_params < 10_000_000, "Model has too many parameters"
