import os
import torch
import torchvision
import pandas as pd
from tqdm import tqdm
from typing import Any, Tuple
from ultralytics.utils.ops import non_max_suppression
from torchvision.ops import box_iou


def test_yolo_model(
    model: Any,
    test_loader: Any,
    device: torch.device,
    output_dir: str = "",  ### NOVO: Parâmetro para controlar o salvamento do CSV
) -> Tuple[int, float]:
    """
    Testa um modelo YOLO, calcula a média do IoU para as detecções correspondentes
    e, opcionalmente, salva todas as predições em um arquivo CSV.
    """

    model.eval()

    total_iou = 0.0
    matched_detections_count = 0
    all_results_to_save = []
    image_counter = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Calculando IoU Médio")
        for images, targets in pbar:

            ### CORREÇÃO: Garante que 'targets' seja um tensor único ###
            if isinstance(targets, list):
                # Se o DataLoader retornou uma lista, converte para um tensor único.
                if not targets or all(t.numel() == 0 for t in targets):
                    targets = torch.tensor([])  # Lida com lotes vazios
                else:
                    # Adiciona o índice do lote (batch index) a cada tensor de alvo
                    for i, t in enumerate(targets):
                        # Cria uma coluna de índices e a concatena
                        batch_idx = torch.full(
                            (t.shape[0], 1), i, device=t.device, dtype=t.dtype
                        )
                        targets[i] = torch.cat([batch_idx, t], dim=1)
                    targets = torch.cat(targets, 0)

            # A partir daqui, 'targets' é garantidamente um tensor
            if targets.numel() == 0:
                if output_dir:
                    image_counter += len(images)
                continue

            # Move o tensor para o dispositivo correto APÓS a concatenação
            targets = targets.to(device)
            images = images.to(device)
            h, w = images.shape[2], images.shape[3]

            predictions_raw = model(images)
            preds = non_max_suppression(
                prediction=(
                    predictions_raw[0]
                    if isinstance(predictions_raw, tuple)
                    else predictions_raw
                )
            )

            # Itera sobre cada imagem no lote para calcular o IoU
            for i, pred_per_image in enumerate(preds):
                gt_indices = targets[:, 0] == i
                gt_boxes_norm = targets[gt_indices][:, 2:]
                gt_labels = targets[gt_indices][:, 1]

                if gt_boxes_norm.numel() == 0:
                    continue

                gt_boxes_xyxy = torchvision.ops.box_convert(
                    gt_boxes_norm, in_fmt="cxcywh", out_fmt="xyxy"
                ).to(device)
                gt_boxes_xyxy[:, [0, 2]] *= w
                gt_boxes_xyxy[:, [1, 3]] *= h

                if pred_per_image.numel() > 0:
                    iou_matrix = box_iou(pred_per_image[:, :4], gt_boxes_xyxy)
                    for pred_idx in range(len(pred_per_image)):
                        label_matches = pred_per_image[pred_idx, 5] == gt_labels
                        if not label_matches.any():
                            continue
                        iou_scores_for_pred = iou_matrix[pred_idx][label_matches]
                        if iou_scores_for_pred.numel() > 0:
                            max_iou, _ = torch.max(iou_scores_for_pred, dim=0)
                            if max_iou.item() > 0.1:
                                total_iou += max_iou.item()
                                matched_detections_count += 1

            ### NOVO: Bloco para coletar os dados para o CSV ###
            if output_dir:
                for i, pred_per_image in enumerate(preds):
                    if pred_per_image.shape[0] > 0:
                        # Converte de xyxy (pixels) para cxcywh (normalizado) para salvar
                        pred_boxes_xywhn = torchvision.ops.box_convert(
                            pred_per_image[:, :4], in_fmt="xyxy", out_fmt="cxcywh"
                        )
                        pred_boxes_xywhn /= torch.tensor([w, h, w, h], device=device)
                        for box_idx in range(pred_per_image.shape[0]):
                            all_results_to_save.append(
                                {
                                    "image_index": image_counter + i,
                                    "class_id": int(pred_per_image[box_idx, 5]),
                                    "confidence": float(pred_per_image[box_idx, 4]),
                                    "x_center": float(pred_boxes_xywhn[box_idx, 0]),
                                    "y_center": float(pred_boxes_xywhn[box_idx, 1]),
                                    "width": float(pred_boxes_xywhn[box_idx, 2]),
                                    "height": float(pred_boxes_xywhn[box_idx, 3]),
                                }
                            )
                image_counter += len(images)

    # Calcula a média final do IoU
    mean_iou = (
        total_iou / matched_detections_count if matched_detections_count > 0 else 0.0
    )

    ### NOVO: Bloco final para salvar o arquivo CSV ###
    if output_dir and all_results_to_save:
        results_df = pd.DataFrame(all_results_to_save)
        csv_path = os.path.join(output_dir, "iou_eval_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nPredições do teste salvas em: '{csv_path}'")

    return mean_iou
