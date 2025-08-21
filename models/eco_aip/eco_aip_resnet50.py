import torch
import torch.nn as nn
from models.aip.aip_improved_modules import ZeroDCEPlus

from models.baselines.resnet50 import get_resnet50


class EcoAIPResNet50(nn.Module):
    def __init__(self, num_classes: int = 2):

        super().__init__()
        self.enhancer = ZeroDCEPlus()
        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.idx = 0

    def param_groups(self, lr_backbone=1e-4, lr_enhancer=3e-5, weight_decay=1e-4):
        """Grupos de parâmetros prontos para AdamW/SGD."""
        return [
            {
                "params": self.backbone.parameters(),
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            },
            {
                "params": self.enhancer.parameters(),
                "lr": lr_enhancer,
                "weight_decay": weight_decay,
            },
        ]

    def forward(self, x, return_enhanced: bool = False):
        x = x.float().clamp(0, 1)
        # parte sensível roda bem em AMP; não precisamos desativar autocast
        out = self.enhancer(x, return_aux=return_enhanced)
        if return_enhanced:
            y, aux = out
        else:
            y = out

        # save_side_by_side("./tmp", x, y, self.idx)
        self.idx += 1
        logits = self.backbone(y)
        return (logits, y, aux) if return_enhanced else logits
