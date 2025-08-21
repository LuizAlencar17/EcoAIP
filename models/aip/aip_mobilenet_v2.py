import torch
import torch.nn as nn
import torch.nn.functional as F
from models.baselines.mobilenet_v2 import get_mobilenet_v2
from models.aip.aip_modules import NLPP, DIP


class AIPMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.nlpp = NLPP()
        self.dip = DIP()
        self.backbone = get_mobilenet_v2()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

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
