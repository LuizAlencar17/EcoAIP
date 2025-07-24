import torch
import torch.nn as nn
from models.resnet import get_resnet50
from models.aip_improved_modules import EnhancedNLPP, ImprovedDIP, CBAM


class ImprovedAIPResNet50(nn.Module):
    def __init__(self, num_classes=2, tone_L=8):
        super().__init__()
        self.nlpp = EnhancedNLPP(out_dim=5 + tone_L + 1)
        self.dip = ImprovedDIP(tone_L=tone_L)
        self.backbone = get_resnet50()
        self.cbam = CBAM(2048)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        params = self.nlpp(x)
        control_signal = torch.sigmoid(params[:, -1]).view(-1, 1, 1, 1)
        enhanced = self.dip(x, params)

        # Soft blending
        x = control_signal * enhanced + (1 - control_signal) * x

        features = self.backbone.conv1(x)
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
