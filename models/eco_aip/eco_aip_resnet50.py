import torch
import torch.nn as nn
from models.eco_aip.eco_aip_modules import retinexformer_tiny

from models.baselines.resnet50 import get_resnet50


class EcoAIPResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.aip = retinexformer_tiny()  # leve
        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        x = x.clamp(0, 1)
        x = self.aip(x)  # imagem melhorada
        return self.backbone(x)
