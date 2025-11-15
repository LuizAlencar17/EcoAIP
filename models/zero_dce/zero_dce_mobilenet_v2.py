import torch.nn as nn
from models.baselines.mobilenet_v2 import get_mobilenet_v2
from models.zero_dce.zero_dce_modules import ZeroDCE


class ZeroDCEMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Módulo de aprimoramento Zero-DCE
        self.enhancer = ZeroDCE(
            n_iter=8
        )  # n=8 é o valor recomendado no artigo [cite: 1522]

        self.backbone = get_mobilenet_v2()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        final_image = self.enhancer(x)
        logits = self.backbone(final_image)
        return logits
