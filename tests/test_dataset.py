import torch
from mlops_proj.dataset import CorruptMnistDataset, show_image_and_target


def test_dataset_train_loading():
    """Train dataset should load correctly."""
    dataset = CorruptMnistDataset(train=True)
    assert len(dataset) == 30000


def test_dataset_test_loading():
    """Test dataset should load correctly."""
    dataset = CorruptMnistDataset(train=False)
    assert len(dataset) == 5000


def test_dataset_getitem_shape():
    """Dataset should return correct shapes."""
    dataset = CorruptMnistDataset(train=True)
    img, target = dataset[0]
    assert img.shape == (1, 28, 28)
    assert isinstance(target, torch.Tensor)


def test_dataset_all_labels_present():
    """Dataset should contain all 10 digit classes."""
    dataset = CorruptMnistDataset(train=True)
    unique_labels = torch.unique(dataset.target)
    assert len(unique_labels) == 10


def test_show_image_and_target_runs():
    """show_image_and_target should run without error."""
    images = torch.randn(9, 1, 28, 28)
    targets = torch.randint(0, 10, (9,))
    show_image_and_target(images, targets, show=False)
