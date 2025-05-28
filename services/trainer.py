import torch
from tqdm import tqdm
from typing import Any
from utils.utils import save_metrics
from services.tester import test_model
from torchvision.transforms import functional as TF


def augment_image(x):
    if torch.rand(1).item() > 0.5:
        x = TF.adjust_gamma(x, gamma=torch.empty(1).uniform_(0.3, 2.0).item())
    if torch.rand(1).item() > 0.5:
        x = TF.gaussian_blur(x, kernel_size=[3, 3])
    if torch.rand(1).item() > 0.5:
        x = x + torch.randn_like(x) * 0.05
    return torch.clamp(x, 0, 1)


def train_model(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    epochs: int,
    patience: int,
    output_dir: str,
    device: torch.device,
    optimizer: Any,
):
    if patience > epochs:
        patience = epochs - 1
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1
    current_patience = 0
    train_losses, acc_values = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        desc = f"Epoch {epoch+1}/{epochs}"
        for data, target in tqdm(train_loader, desc=desc):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            data_aug = torch.stack([augment_image(img) for img in data])
            outputs = model(data_aug)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_acc = test_model(model, val_loader, device)

        train_losses.append(loss.item())
        acc_values.append(val_acc)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.10f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            model_path = f"{output_dir}/model_best.pth"
            print(
                f"New best model with accuracy: {val_acc:.4f}, saving model... {model_path}"
            )
            best_acc = val_acc
            current_patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            current_patience += 1
            print(f"Patience {current_patience}/{patience}")
            if current_patience >= patience:
                print(f"Early stopping triggered, best accuracy{best_acc}")
                break

    save_metrics(
        {
            "train_loss": train_losses,
            "val_accuracy": acc_values,
        },
        f"{output_dir}/metrics.csv",
    )
