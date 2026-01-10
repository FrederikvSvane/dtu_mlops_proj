import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from mlops_proj.data import corrupt_mnist
from mlops_proj.model import Classifier


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png"):
    """Visualize features of dataset in 2d space using t-SNE"""
    model: Classifier = Classifier()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    model.fc1 = torch.nn.Identity()  # type: ignore

    _, test_set = corrupt_mnist()
    # print(test_set.tensors[0].shape)
    # print(test_set.tensors[1].shape)

    test_loader = DataLoader(test_set, batch_size=32)
    embeddings_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    with torch.inference_mode():
        for images, labels in test_loader:
            embeds = model(images)
            embeddings_list.append(embeds)
            labels_list.append(labels)

        embeddings: torch.Tensor = torch.cat(embeddings_list)
        labels: torch.Tensor = torch.cat(labels_list)

        # print(embeddings.shape)
        # print(labels.shape)

        tsne = TSNE(n_components=2)

        embeddings_numpy = embeddings.numpy()
        labels_numpy = labels.numpy()
        embeddings_2d = tsne.fit_transform(embeddings_numpy)

        plt.figure(figsize=(10, 10))

        for digit in range(10):
            mask = labels_numpy == digit

            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=str(digit))

        plt.legend()
        plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    visualize("models/model.pt")
