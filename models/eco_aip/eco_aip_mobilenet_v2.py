import torch
import torch.nn as nn
from models.eco_aip.eco_aip_modules import retinexformer_tiny
from models.baselines.mobilenet_v2 import get_mobilenet_v2
from utils.utils import save_side_by_side


class EcoAIPMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.aip = retinexformer_tiny()  # leve
        self.backbone = get_mobilenet_v2()
        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Linear(in_features, num_classes)
        self.idx = 0

    def forward(self, x):
        x = x.clamp(0, 1)
        y = self.aip(x)  # imagem melhorada
        # save_side_by_side("tmp", x, y, self.idx)
        self.idx += 1
        return self.backbone(y)
