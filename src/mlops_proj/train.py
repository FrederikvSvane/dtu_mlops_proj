import os

import matplotlib.pyplot as plt
import torch.nn
from dotenv import load_dotenv
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
from torch import nn

# from torch.profiler import ProfilerActivity, tensorboard_trace_handler # NOTE for profiling
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from mlops_proj.data import corrupt_mnist
from mlops_proj.model import Classifier

load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")

# DEVICE = torch.device("cpu") #NOTE for profiling
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# prof = torch.profiler.profile(
#     activities=[ProfilerActivity.CPU],
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=1),  # just for a better overview of the loop
#     profile_memory=True,
#     on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
# ) #NOTE for profiling


def train(batch_size: int = 32, epochs: int = 10, lr: float = 0.001):
    # print(DEVICE) -> mps
    # prof.start() #NOTE for profiling
    print(f"Learning rate set to {lr}, epochs = {epochs}")

    # statistics = {"train_loss": [], "train_accuracy": []}
    run = wandb.init(project="mlops corrupt mnist", config={"lr": lr, "batch size": batch_size, "epochs": epochs})

    train_set, _ = corrupt_mnist()
    print(f"Train: {len(train_set)} samples")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = Classifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        model.train()

        preds, targets = [], []
        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = loss_fn(logits, labels)
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # prof.step()  # just for a better overview of the loop.
            # statistics["train_loss"].append(loss.item())
            # statistics["train_accuracy"].append(accuracy)
            pbar.set_postfix(loss=loss.item(), acc=accuracy)

            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            preds.append(logits.detach().cpu())
            targets.append(labels.detach().cpu())

            if i % 100 == 0:
                # add a plot of the input images (rescale from normalized to 0-255 for display)
                imgs_display = images[:5].detach().cpu()
                imgs_display = (imgs_display - imgs_display.min()) / (imgs_display.max() - imgs_display.min())  # scale to 0-1
                wandb.log({"images": [wandb.Image(img.squeeze().numpy()) for img in imgs_display]})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.cpu())})  # type: ignore
                # add a custom matplotlib plot of the ROC curves

        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot,
                preds[:, class_id],
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )

        wandb.log({"roc": wandb.Image(plt)})
        plt.close()  # close the plot to avoid memory leaks and overlapping figures
    # prof.stop() #NOTE for profiling
    print("Training complete!")
    torch.save(model.state_dict(), "models/model.pt")
    print("Model saved to models/model.pt")

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))  # type: ignore
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")  # type: ignore
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")  # type: ignore
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")  # type: ignore

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

    """ NOTE for logging with matplotlib instead of WandB
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    print("Loss- and accuracy graphs saved to reports/figures/training_statistics.png")
    """


if __name__ == "__main__":
    train()
