import torch
import importlib
import pandas as pd


def save_metrics(data: dict, output_dir: str) -> None:
    pd.DataFrame(data).to_csv(output_dir, index=False)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_name):
    config = importlib.import_module(config_name)
    print("\n\nConfiguration Parameters:")
    for attr in dir(config):
        if not attr.startswith("__"):  # Ignore special attributes
            print(f"{attr}: {getattr(config, attr)}")
    return config
