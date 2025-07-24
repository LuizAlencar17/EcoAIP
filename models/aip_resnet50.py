import torch
import torch.nn as nn
from models.resnet import get_resnet50
from models.aip_modules import NLPP, DIP


class AIPResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.nlpp = NLPP()
        self.dip = DIP()
        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        params = self.nlpp(x)
        control_signal = torch.sigmoid(params[:, -1])
        if control_signal.mean() > 0.5:  # You can customize threshold
            x = self.dip(x, params)
        logits = self.backbone(x)
        return logits
