# services/yolo_trainer.py

import os
import torch
import pandas as pd
import torch.nn as nn
import math

from tqdm import tqdm
from services.yolo_tester import test_yolo_model
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss
from torch.optim.lr_scheduler import CosineAnnealingLR


class YoloLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.loss_fn = v8DetectionLoss(model.model)

    def forward(self, preds, batch):
        loss, loss_items = self.loss_fn(preds, batch)
        return loss.sum(), loss_items


def train_yolo_model(
    model,
    train_loader,
    val_loader,
    epochs,
    patience,
    output_dir,
    device,
    optimizer,
    num_classes,
    apply_augment,
    **kwargs,
):
    """
    Loop de treinamento modificado com warm-up manual.
    """
    model.model.args = DEFAULT_CFG
    criterion = YoloLoss(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --- PARÂMETROS DE WARM-UP E SCHEDULER ---
    # Parâmetros do warm-up (padrões do YOLO)
    warmup_epochs = 3
    warmup_momentum = 0.8
    warmup_bias_lr = 0.1
    # Pega o LR final do otimizador (que veio da config, ex: 0.01)
    final_lr = optimizer.param_groups[0]["lr"]

    # O scheduler principal só começará após o warm-up
    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=final_lr * 0.01
    )

    best_metrics = -1
    patience_counter = 0

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.pth")
    metrics_path = os.path.join(output_dir, "metrics.csv")
    training_history = []

    # --- LÓGICA DE WARM-UP ---
    num_warmup_iterations = max(round(warmup_epochs * len(train_loader)), 100)
    it_num = 0  # Contador de iterações global

    for epoch in range(epochs):
        model.model.train()
        total_train_loss = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, targets in pbar_train:
            it_num += 1
            images = images.to(device)

            # --- AJUSTE DO LEARNING RATE DURANTE O WARM-UP ---
            if it_num <= num_warmup_iterations:
                xi = [0, num_warmup_iterations]
                # Interpolação linear para o learning rate e momentum
                for j, pg in enumerate(optimizer.param_groups):
                    pg["lr"] = final_lr * (
                        (
                            (1 - it_num / num_warmup_iterations)
                            * (1.0 - warmup_bias_lr if j == 2 else 0.0)
                        )
                        + it_num / num_warmup_iterations
                    )
                    if "momentum" in pg:
                        pg["momentum"] = warmup_momentum + (
                            optimizer.defaults["momentum"] - warmup_momentum
                        ) * (it_num / num_warmup_iterations)

            optimizer.zero_grad()

            batch = {
                "img": images,
                "batch_idx": torch.cat(
                    [torch.full((len(t),), i) for i, t in enumerate(targets)]
                ).to(device),
                "cls": torch.cat([t[:, 0] for t in targets]).to(device),
                "bboxes": torch.cat([t[:, 1:] for t in targets]).to(device),
            }

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                predictions = model(images)
                loss, loss_items = criterion(predictions, batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar_train.set_postfix(loss=total_train_loss / (pbar_train.n + 1))

        # --- PASSO DO SCHEDULER APÓS O WARM-UP ---
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # ATENÇÃO: test_yolo_model ainda calcula avg_iou, não mAP!

        avg_val_loss, val_metrics = test_yolo_model(
            model=model, test_loader=val_loader, device=device
        )

        # Extrai e imprime as métricas principais
        map_50_95 = val_metrics["map"].item()
        map_50 = val_metrics["map_50"].item()
        map_75 = val_metrics["map_75"].item()

        print(
            f"End of Epoch {epoch+1}: "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"mAP@.50:.95: {map_50_95:.4f} | "
            f"mAP@.50: {map_50:.4f}"
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_map_50": map_50,
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_history.append(epoch_metrics)

        # --- Lógica de Early Stopping ---
        if map_50 > best_metrics:
            best_metrics = map_50
            torch.save(model.state_dict(), model_path)
            print(f"--> New best validation mAP: {map_50:.4f}. Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("--- Patience reached. Stopping training. ---")
                break

    metrics_df = pd.DataFrame(training_history)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nTraining metrics saved to {metrics_path}")
