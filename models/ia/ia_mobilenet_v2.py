import torch.nn as nn
from models.baselines.mobilenet_v2 import get_mobilenet_v2
from models.ia.ia_modules import CNN_PP, DIP

tone_L = 8


class IAMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        num_params = 3 + 1 + 1 + tone_L + 1

        self.cnn_pp = CNN_PP(num_params=num_params)
        self.dip = DIP(tone_L=tone_L)

        self.backbone = get_mobilenet_v2()
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        params = self.cnn_pp(x)
        final_image = self.dip(x, params)
        logits = self.backbone(final_image)
        return logits
