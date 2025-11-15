import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baselines.resnet50 import get_resnet50
from models.aip.aip_modules import NLPP, DIP


class AIPResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.nlpp = NLPP()
        self.dip = DIP()
        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Adicione o módulo de perda auxiliar
        self.auxiliary_loss_fn = nn.L1Loss()

    def forward(self, x, target=None, lambda_aux=0.01):
        # Etapa 1: Prediz os parâmetros com o NLPP
        params = self.nlpp(x)

        # Etapa 2: Processa a imagem com o DIP
        enhanced_image = self.dip(x, params)
        control_signal = torch.sigmoid(params[:, -1]).view(-1, 1, 1, 1)
        final_image = control_signal * enhanced_image + (1 - control_signal) * x

        # Classificação final
        logits = self.backbone(final_image)
        return logits
