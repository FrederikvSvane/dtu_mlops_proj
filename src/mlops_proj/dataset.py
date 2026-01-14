from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import torch
import typer
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torchvision.transforms.v2 as transforms


class CorruptMnistDataset(Dataset):
    """Corrupt MNIST dataset for PyTorch.

    Args:
        data_folder: Path to the data folder.
        train: Whether to load training or test data.
        img_transform: Potential image transformation to apply.
        target_transform: Target transformation to apply.
    """

    name: str = "Corrupt MNIST"

    def __init__(
        self,
        data_folder: str = "data",
        train: bool = True,
        img_transform: transforms.Transform | None = None,
        target_transform: transforms.Transform | None = None,
    ) -> None:
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.load_data()

    def load_data(self) -> None:
        """Load images and targets from disk."""
        processed_folder = f"{self.data_folder}/processed/corruptmnist"
        if self.train:
            self.images = torch.load(f"{processed_folder}/train_images.pt")
            self.target = torch.load(f"{processed_folder}/train_labels.pt")
        else:
            self.images = torch.load(f"{processed_folder}/test_images.pt")
            self.target = torch.load(f"{processed_folder}/test_labels.pt")

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor."""
        img, target = self.images[idx], self.target[idx]
        if self.img_transform:
            img = self.img_transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return self.images.shape[0]


def show_image_and_target(images: Tensor, targets: Tensor, show: bool = True) -> None:
    """Display a grid of images with their labels."""
    n = int(len(images) ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap="gray")
            ax.set_title(f"Label: {targets[i].item()}")
        ax.axis("off")
    if show:
        plt.show()


def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics."""
    train_dataset = CorruptMnistDataset(data_folder=datadir, train=True)
    test_dataset = CorruptMnistDataset(data_folder=datadir, train=False)
    print(f"Train dataset: {train_dataset.name}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")
    print("\n")
    print(f"Test dataset: {test_dataset.name}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")

    show_image_and_target(train_dataset.images[:25], train_dataset.target[:25], show=False)
    plt.savefig("corrupt_mnist_images.png")
    plt.close()

    train_label_distribution = torch.bincount(train_dataset.target)
    test_label_distribution = torch.bincount(test_dataset.target)

    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()


if __name__ == "__main__":
    typer.run(dataset_statistics)
