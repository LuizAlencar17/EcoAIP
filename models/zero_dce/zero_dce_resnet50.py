import torch.nn as nn
from models.baselines.resnet50 import get_resnet50
from models.zero_dce.zero_dce_modules import ZeroDCE


class ZeroDCEResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Módulo de aprimoramento Zero-DCE
        self.enhancer = ZeroDCE(
            n_iter=8
        )  # n=8 é o valor recomendado no artigo [cite: 1522]

        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Adicione o módulo de perda auxiliar
        self.auxiliary_loss_fn = nn.L1Loss()

    def forward(self, x):
        final_image = self.enhancer(x)
        logits = self.backbone(final_image)
        return logits
