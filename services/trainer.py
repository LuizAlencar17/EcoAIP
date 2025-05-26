import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any
from utils.utils import save_metrics
from services.tester import test_model
from torchvision.transforms import functional as TF


def multi_objective_loss(pred, target, x_dip, x_orig, criterion):
    loss_task = criterion(pred, target)
    loss_rec = F.l1_loss(x_dip, x_orig)
    perceptual_loss = F.mse_loss(F.avg_pool2d(x_dip, 4), F.avg_pool2d(x_orig, 4))
    return loss_task + 0.1 * loss_rec + 0.05 * perceptual_loss


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
    loss_computation: str,
):

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
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
            # data_aug = data
            if loss_computation == "augment":
                outputs, x_dip = model(data_aug)
                loss = multi_objective_loss(outputs, target, x_dip, data_aug, criterion)
            else:
                outputs = model(data_aug)
                loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_acc = test_model(model, val_loader, device, loss_computation)

        train_losses.append(loss.item())
        acc_values.append(val_acc)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.10f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            print(f"New best model with accuracy: {val_acc:.4f}, saving model...")
            best_acc = val_acc
            current_patience = 0
            torch.save(model.state_dict(), f"{output_dir}/model_best.pth")
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
