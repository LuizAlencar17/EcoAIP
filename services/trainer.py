import torch
from tqdm import tqdm
from typing import Any
from utils.utils import save_metrics
from services.tester import test_model


def train_model(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    epochs: int,
    patience: int,
    output_dir: str,
    device: torch.device,
    optimizer: Any,
    **kwargs: Any,
):
    if patience > epochs:
        patience = epochs - 1

    is_eco_aip = "EcoAIP" in str(type(model))
    if is_eco_aip:
        optimizer = torch.optim.AdamW(
            model.param_groups(lr_backbone=1e-4, lr_enhancer=3e-5)
        )

    criterion = torch.nn.CrossEntropyLoss()
    best_acc = -1
    current_patience = 0
    train_losses, train_accs, acc_values = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for data, target in pbar_train:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar_train.set_postfix(loss=running_loss / (pbar_train.n + 1))

        avg_train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        _, val_acc = test_model(model, val_loader, device)

        train_losses.append(loss.item())
        acc_values.append(val_acc)
        train_accs.append(train_acc)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.10f}, Train Acc={train_acc:.10f}, Val Acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            model_path = f"{output_dir}/best_model.pth"
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
                print(f"Early stopping triggered, best accuracy: {best_acc}")
                break

    save_metrics(
        {
            "train_loss": train_losses,
            "val_accuracy": acc_values,
            "train_accuracy": train_accs,
        },
        f"{output_dir}/metrics.csv",
    )
