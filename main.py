import os
import torch
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from utils.utils import get_device, load_config
from models.loader import get_model
from data.dataset import AnimalDataset
from services.trainer import train_model
from services.tester import test_model
from utils.seed import set_seed


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--config", type=str, required=True, help="Name of the config file (without .py)"
)
parser.add_argument(
    "--mode", type=str, required=False, default="train", help="Train or test?"
)

# Dynamically load the arguments
args = parser.parse_args()
print("\nArguments:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

# Dynamically load the configuration
config = load_config(args.config)
device = get_device()

set_seed(config.SEED)
path_output = f"{config.OUTPUT_DIR}/{config.TASK}/{config.MODEL}_{config.TRAIN_SIZE}"
os.makedirs(path_output, exist_ok=True)

transform = transforms.Compose(
    [transforms.Resize(config.IMAGE_SIZE), transforms.ToTensor()]
)
train_dataset = AnimalDataset(config.DATA_TRAIN_CSV_PATH, transform, config.TRAIN_SIZE)
val_dataset = AnimalDataset(config.DATA_VAL_CSV_PATH, transform, 400)
test_dataset = AnimalDataset(config.DATA_TEST_CSV_PATH, transform, 400)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model, optimizer = get_model(config, device, config.NUM_CLASS)
if args.mode == "train":
    print("Tranning model...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        patience=config.PATIENCE,
        output_dir=path_output,
        optimizer=optimizer,
        apply_augment=config.AUGMENT_IMAGE,
        device=device,
    )

if args.mode == "test":
    print("Testing model...")
    weights_path = f"{path_output}/model_best.pth"
    print(f"Loading weights in {weights_path}")
    model.load_state_dict(torch.load(weights_path), strict=False)
    accuracy = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=path_output,
    )
    print(f"Accuracy: {accuracy:.2f}")
