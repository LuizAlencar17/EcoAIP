import torch
import torch.optim as optim
from models.resnet import get_resnet50, get_resnet50_classifier
from models.aip_resnet50 import AIPResNet50
from models.improved_aip_resnet50 import ImprovedAIPResNet50


def get_model(config, device):
    if config.MODEL == "AIPResNet50":
        model = AIPResNet50()

    elif config.MODEL == "ImprovedAIPResNet50":
        model = ImprovedAIPResNet50()

    elif config.MODEL == "ResNet50":
        model = get_resnet50_classifier()

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_MODEL)
    return model.to(device), optimizer
