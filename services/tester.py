import torch
from tqdm import tqdm
from typing import Any

from utils.utils import get_device, save_metrics

device = get_device()


def test_model(
    model: Any = None,
    test_loader: Any = None,
    device: torch.device = None,
    output_dir: str = "",
):
    model.eval()
    correct = 0
    total = 0
    predictions = []  # List to store predictions and labels

    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Save predictions and labels
            for i in range(len(target)):
                predictions.append(
                    {
                        "image_index": idx * len(target) + i,
                        "pred": predicted[i].item(),
                        "true": target[i].item(),
                    }
                )
    if output_dir:
        save_metrics(predictions, f"{output_dir}/eval_results.csv")
    accuracy = correct / total
    return accuracy
