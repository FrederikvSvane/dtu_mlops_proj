from pathlib import Path

import torch
import typer
from torch.utils.data import TensorDataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    return (images - images.mean()) / images.std()


def preprocess_data(data_path: Path, output_folder: Path) -> None:
    """
    Load the datasets from the pt files in the corruptmnist folder

    :return: (train_dataset, test_dataset)
    :rtype: tuple[TensorDataset, TensorDataset]
    """
    train_images_paths: list[Path] = sorted(data_path.glob("train_images_*.pt"))
    train_labels_paths: list[Path] = sorted(data_path.glob("train_target_*.pt"))
    test_images_path: Path = Path(f"{data_path}/test_images.pt")
    test_labels_path: Path = Path(f"{data_path}/test_target.pt")

    train_images_loaded: list[torch.Tensor] = [torch.load(p) for p in train_images_paths]
    train_labels_loaded: list[torch.Tensor] = [torch.load(p) for p in train_labels_paths]

    train_images: torch.Tensor = torch.cat(train_images_loaded, dim=0)
    test_images: torch.Tensor = torch.load(test_images_path)
    train_labels: torch.Tensor = torch.cat(train_labels_loaded, dim=0).long()
    test_labels: torch.Tensor = torch.load(test_labels_path).long()
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, f"{output_folder}/train_images.pt")
    torch.save(train_labels, f"{output_folder}/train_labels.pt")
    torch.save(test_images, f"{output_folder}/test_images.pt")
    torch.save(test_labels, f"{output_folder}/test_labels.pt")


def corrupt_mnist() -> tuple[TensorDataset, TensorDataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/corruptmnist/train_images.pt")
    train_labels = torch.load("data/processed/corruptmnist/train_labels.pt")
    test_images = torch.load("data/processed/corruptmnist/test_images.pt")
    test_labels = torch.load("data/processed/corruptmnist/test_labels.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_labels)
    test_set = torch.utils.data.TensorDataset(test_images, test_labels)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
