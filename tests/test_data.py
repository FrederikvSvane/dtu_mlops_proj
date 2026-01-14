import torch
from mlops_proj.data import corrupt_mnist, normalize


def test_normalize():
    """Normalize should produce mean ~0 and std ~1."""
    x = torch.randn(100, 1, 28, 28) * 50 + 100  # arbitrary mean/std
    normalized = normalize(x)
    assert abs(normalized.mean().item()) < 0.01, "Mean should be close to 0"
    assert abs(normalized.std().item() - 1.0) < 0.01, "Std should be close to 1"


def test_data():
    train_set, test_set = corrupt_mnist()
    assert len(train_set) == 30000
    assert len(test_set) == 5000

    # train_set is a Dataset, which is essentially a list-like of tuples
    # so we index once to get the first tuple of (image, label)
    # then we index again to get (image)
    for dataset in [train_set, test_set]:
        for image, label in dataset:
            assert image.shape == (1, 28, 28)
            assert label in range(10)

    train_targets = torch.unique(train_set.tensors[1])
    test_targets = torch.unique(test_set.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()  # .all() checks that it holds for all elements
    assert (test_targets == torch.arange(0, 10)).all()


if __name__ == "__main__":
    test_data()
