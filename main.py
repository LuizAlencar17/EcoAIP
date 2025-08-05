import torch
import argparse

from pathlib import Path
from torch.utils.data import DataLoader
from data.classification import ClassificationDataset
from data.detectation import DetectionDataset
from utils.utils import get_device, load_config
from models.loader import get_model
from services.yolo_trainer import train_yolo_model
from services.yolo_tester import test_yolo_model
from services.trainer import train_model
from services.tester import test_model
from utils.seed import set_seed


def get_serengeti_model_path(config):
    serengeti_path = "/data/luiz/dataset/EcoAIP/serengeti"
    output_path = (
        Path(serengeti_path) / config.TASK / f"{config.MODEL}_{config.TRAIN_SIZE}"
    )
    return output_path


def detection_collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    images = torch.stack(images, dim=0)  # this works since image sizes are fixed
    return images, targets


def setup_dataloaders(config, task_type):
    """Encapsulates the creation of datasets and dataloaders to avoid repetition."""
    # Common DataLoader parameters
    loader_params = {
        "batch_size": config.BATCH_SIZE,
        "num_workers": 4,
        "pin_memory": True,
    }
    if task_type in [
        "animal-classifier",
        "species-classifier",
        "species-classifier-cropped",
    ]:
        DatasetClass = ClassificationDataset
    else:
        DatasetClass = DetectionDataset
        loader_params["collate_fn"] = detection_collate_fn

    train_dataset = DatasetClass(
        csv_file=config.DATA_TRAIN_CSV_PATH,
        img_size=config.IMAGE_SIZE,
        n=config.TRAIN_SIZE,
        seed=config.SEED,
        is_train=True,
        bbox_is_normalized=config.BBOX_IS_NORMALIZED,
    )
    val_dataset = DatasetClass(
        csv_file=config.DATA_VAL_CSV_PATH,
        img_size=config.IMAGE_SIZE,
        n=config.VAL_SIZE,
        seed=config.SEED,
        bbox_is_normalized=config.BBOX_IS_NORMALIZED,
    )
    test_dataset = DatasetClass(
        csv_file=config.DATA_TEST_CSV_PATH,
        img_size=config.IMAGE_SIZE,
        n=config.TEST_SIZE,
        seed=config.SEED,
        bbox_is_normalized=config.BBOX_IS_NORMALIZED,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

    return train_loader, val_loader, test_loader


def main(args):
    """Main function that runs the train or test workflow."""

    # Load configuration and set device
    print("\nArguments:", args)
    config = load_config(args.config)
    device = get_device()
    set_seed(config.SEED)

    # Create the output directory robustly with pathlib
    output_path = (
        Path(config.OUTPUT_DIR) / config.TASK / f"{config.MODEL}_{config.TRAIN_SIZE}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = output_path / "best_model.pth"

    # Set up the DataLoaders
    train_loader, val_loader, test_loader = setup_dataloaders(config, config.TASK)

    # Load the model and optimizer
    model, optimizer = get_model(config, device, config.NUM_CLASS)

    if args.mode == "train":
        print("\nTraining model...")
        if "serengeti" not in config.OUTPUT_DIR:
            print("\nLoading Serengeti weights...")
            weights_path = get_serengeti_model_path(config) / "best_model.pth"
            model.load_state_dict(torch.load(weights_path), strict=False)

        trainer_fn = train_yolo_model if "yolo" in config.MODEL.lower() else train_model
        trainer_fn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS,
            patience=config.PATIENCE,
            output_dir=str(output_path),
            optimizer=optimizer,
            device=device,
            num_classes=config.NUM_CLASS,
        )

    elif args.mode == "test":
        print("\nTesting model...")
        print(f"Loading weights from {weights_path}")

        model.load_state_dict(torch.load(weights_path), strict=False)
        tester_fn = test_yolo_model if "yolo" in config.MODEL.lower() else test_model
        metrics = tester_fn(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=str(output_path),
        )
        print(f"Test result: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train or test a computer vision model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the config file (e.g., 'configs.yolo_config').",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Execution mode: train or test.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
