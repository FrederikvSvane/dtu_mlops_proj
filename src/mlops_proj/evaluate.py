import torch
from torch.utils.data import DataLoader

from mlops_proj.data import corrupt_mnist
from mlops_proj.model import Classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(checkpoint: str) -> None:
    """Evaluate a trained model (provided a path to a model.pt file)"""
    model = Classifier().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    _, test_set = corrupt_mnist()
    test_loader = DataLoader(test_set, batch_size=32)

    correct, total = 0, 0

    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    print(f"Test accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    evaluate("models/model.pt")
