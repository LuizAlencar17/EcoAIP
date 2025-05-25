import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def get_resnet50() -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    return model


def get_resnet50_classifier(num_classes: int = 2) -> nn.Module:
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model
