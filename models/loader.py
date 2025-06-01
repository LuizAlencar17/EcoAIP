import torch.optim as optim
from models.resnet import get_resnet50_classifier
from models.aip_resnet50 import AIPResNet50
from models.improved_aip_resnet50 import ImprovedAIPResNet50


def get_model(config, device, num_classes: int = 2):
    if config.MODEL == "AIPResNet50":
        model = AIPResNet50(num_classes)

    elif config.MODEL == "ImprovedAIPResNet50":
        model = ImprovedAIPResNet50(num_classes)

    elif config.MODEL == "ResNet50":
        model = get_resnet50_classifier(num_classes)

    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE_MODEL, weight_decay=1e-5
    )
    return model.to(device), optimizer
