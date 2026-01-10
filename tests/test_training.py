import torch
import torch.nn.functional as F
from mlops_proj.data import corrupt_mnist
from mlops_proj.model import Classifier


def test_model_can_learn_single_batch_flawlessly():
    model = Classifier()
    model.train()

    train_set, _ = corrupt_mnist()
    image, label = train_set[0]
    image = image.unsqueeze(1)  # add "fake" batch dim
    label = torch.tensor([label])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inital_loss = None

    for i in range(100):
        optimizer.zero_grad()
        out = model(image)
        loss = F.cross_entropy(out, label)
        loss.backward()
        optimizer.step()

        if inital_loss is None:
            inital_loss = loss.item()

    final_loss = loss.item()  # type: ignore

    assert final_loss < 0.01, "Model failed to learn a batch"
