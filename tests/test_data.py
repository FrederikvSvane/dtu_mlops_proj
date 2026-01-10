import torch
from mlops_proj.data import corrupt_mnist

# from tests import _PATH_DATA


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
