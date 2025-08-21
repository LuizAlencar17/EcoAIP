import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


def get_mobilenet_v2() -> nn.Module:
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze base layers
    return model


def get_mobilenet_v2_classifier(num_classes: int = 2) -> nn.Module:
    model = get_mobilenet_v2()
    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
