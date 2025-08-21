import torch.nn as nn
from models.baselines.resnet50 import get_resnet50
from models.ia.ia_modules import CNN_PP, DIP

tone_L = 8


class IAResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        num_params = 3 + 1 + 1 + tone_L + 1

        self.cnn_pp = CNN_PP(num_params=num_params)
        self.dip = DIP(tone_L=tone_L)

        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Adicione o módulo de perda auxiliar
        self.auxiliary_loss_fn = nn.L1Loss()

    def forward(self, x):
        # [cite_start]Etapa 1: Prediz os parâmetros de aprimoramento [cite: 1887]
        params = self.cnn_pp(x)

        # [cite_start]Etapa 2: Aprimora a imagem com os parâmetros previstos [cite: 1888]
        final_image = self.dip(x, params)
        logits = self.backbone(final_image)
        return logits
