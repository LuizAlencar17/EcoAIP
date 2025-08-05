import torch
import torch.nn as nn
from models.resnet import get_resnet50
from models.aip_improved_modules import EnhancedNLPP, EnhancedDIP, CBAM


class EcoAIPResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.nlpp = EnhancedNLPP()
        self.dip = EnhancedDIP()
        self.backbone = get_resnet50()
        self.cbam = CBAM(2048)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):

        # Etapa 1: Prediz os parâmetros com o NLPP aprimorado
        params = self.nlpp(x)

        # Etapa 2: Processa a imagem com o DIP aprimorado
        enhanced_image = self.dip(x, params)

        # Etapa 3: "Soft Blending" entre a imagem original e a processada
        # O último parâmetro previsto pelo NLPP atua como um portão (gate)
        control_signal = torch.sigmoid(params[:, -1]).view(-1, 1, 1, 1)
        final_image = control_signal * enhanced_image + (1 - control_signal) * x

        features = self.backbone.conv1(final_image)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        features = self.cbam(features)
        pooled = self.backbone.avgpool(features)
        flat = torch.flatten(pooled, 1)
        out = self.backbone.fc(flat)

        return out
