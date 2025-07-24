import os
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from typing import Any
from ultralytics.utils.ops import non_max_suppression
from torchmetrics.detection import (
    MeanAveragePrecision,
)  # <--- NOVO: Importa a métrica de mAP


def test_yolo_model(
    model: Any,
    test_loader: Any,
    device: torch.device,
    criterion: Any = None,
    output_dir: str = "",
):
    model.model.eval()

    # --- NOVO: Instancia o objeto da métrica de mAP ---
    # Ele irá acumular os resultados de todos os lotes
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.to(device)

    total_loss = 0.0
    all_results_to_save = []
    image_counter = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Validation / Test")
        for images, targets in pbar:
            images = images.to(device)
            h, w = images.shape[2], images.shape[3]

            # Formata os alvos (ground truth) para o formato do torchmetrics
            gts = []
            for target_per_image in targets:
                # Desnormaliza as caixas delimitadoras se necessário
                gt_boxes_xywh = target_per_image[:, 1:]
                gt_boxes_xyxy = torchvision.ops.box_convert(
                    gt_boxes_xywh, in_fmt="cxcywh", out_fmt="xyxy"
                )
                gt_boxes_xyxy[:, [0, 2]] *= w
                gt_boxes_xyxy[:, [1, 3]] *= h

                gts.append(
                    {
                        "boxes": gt_boxes_xyxy.to(device),
                        "labels": target_per_image[:, 0].to(torch.int32).to(device),
                    }
                )

            # Realiza as predições
            predictions_raw = model(images)
            final_predictions = non_max_suppression(prediction=predictions_raw)

            # Formata as predições para o formato do torchmetrics
            preds = []
            for pred_per_image in final_predictions:
                preds.append(
                    {
                        "boxes": pred_per_image[:, :4],  # Formato xyxy
                        "scores": pred_per_image[:, 4],  # Confiança
                        "labels": pred_per_image[:, 5].to(torch.int32),  # Classe
                    }
                )

            # --- NOVO: Atualiza a métrica com as predições e os alvos do lote atual ---
            metric.update(preds, gts)

            # O resto do código para calcular a loss e salvar predições permanece o mesmo
            # (se você ainda precisar dele)
            if criterion is not None:
                batch = {
                    "img": images,
                    "batch_idx": torch.cat(
                        [torch.full((len(t),), i) for i, t in enumerate(targets)]
                    ).to(device),
                    "cls": torch.cat([t[:, 0] for t in targets]).to(device),
                    "bboxes": torch.cat([t[:, 1:] for t in targets]).to(device),
                }
                loss, _ = criterion(predictions_raw, batch)
                total_loss += loss.item()

            if output_dir:
                for i, pred in enumerate(final_predictions):
                    if pred.shape[0] > 0:
                        pred_boxes_xywhn = torchvision.ops.box_convert(
                            pred[:, :4], in_fmt="xyxy", out_fmt="cxcywh"
                        )
                        pred_boxes_xywhn /= torch.tensor([w, h, w, h], device=device)
                        for box_idx in range(pred.shape[0]):
                            all_results_to_save.append(
                                {
                                    "image_index": image_counter + i,
                                    "class_id": int(pred[box_idx, 5]),
                                    "confidence": float(pred[box_idx, 4]),
                                    "x_center": float(pred_boxes_xywhn[box_idx, 0]),
                                    "y_center": float(pred_boxes_xywhn[box_idx, 1]),
                                    "width": float(pred_boxes_xywhn[box_idx, 2]),
                                    "height": float(pred_boxes_xywhn[box_idx, 3]),
                                }
                            )
            image_counter += len(images)

    # --- NOVO: Calcula o mAP final com base em todos os lotes processados ---
    print("\nCalculating mAP metrics...")
    final_metrics = metric.compute()

    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

    if output_dir and all_results_to_save:
        results_df = pd.DataFrame(all_results_to_save)
        csv_path = os.path.join(output_dir, "eval_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Test predictions saved to: '{csv_path}'")

    # --- NOVO: Retorna a loss e o dicionário de métricas do mAP ---
    return avg_loss, final_metrics
