import torch
import importlib
import pandas as pd
from pathlib import Path
from torchvision.utils import save_image, make_grid


def save_metrics(data: dict, output_dir: str) -> None:
    pd.DataFrame(data).to_csv(output_dir, index=False)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_name):
    config = importlib.import_module(config_name)
    print("\nConfiguration Parameters:")
    for attr in dir(config):
        if not attr.startswith("__"):  # Ignore special attributes
            print(f"{attr}: {getattr(config, attr)}")
    return config


@torch.no_grad()
def save_side_by_side(
    save_dir: str, x_orig: torch.Tensor, x_enh: torch.Tensor, idx: int
):
    """
    Save original and enhanced images side by side in a specified directory.
    """
    save_max_per_batch = 4
    _save_idx = 0
    x_orig = x_orig.detach().clamp(0, 1).cpu()
    x_enh = x_enh.detach().clamp(0, 1).cpu()

    B = x_orig.size(0)
    n = min(B, save_max_per_batch)
    for i in range(n):
        grid = make_grid([x_orig[i], x_enh[i]], nrow=2, padding=2)  # [orig | enhanced]
        fname = f"pair{idx}_{_save_idx:06d}.png"
        save_image(grid, f"{save_dir}/{fname}")
        _save_idx += 1
