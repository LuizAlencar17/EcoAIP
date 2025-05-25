import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import get_resnet50


# Non-local block
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.out_conv = nn.Conv2d(in_channels // 2, in_channels, 1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        theta = self.theta(x).view(batch_size, -1, H * W)
        phi = self.phi(x).view(batch_size, -1, H * W)
        g = self.g(x).view(batch_size, -1, H * W)

        attention = torch.bmm(theta.transpose(1, 2), phi)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(g, attention.transpose(1, 2)).view(batch_size, -1, H, W)
        out = self.out_conv(out)
        return out + x


# NLPP: parameter predictor
class NLPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            NonLocalBlock(32),
            nn.MaxPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256))
        x = self.features(x)
        params = self.fc(x)
        return params


# DIP Module
class DIP(nn.Module):
    def __init__(self):
        super().__init__()

    # Advanced: Piecewise linear tone mapping
    def tone_mapping(self, x, tone_params):
        # Example for L = 8 steps
        L = 8
        out = torch.zeros_like(x)
        for k in range(L):
            tk = tone_params[:, k].view(-1, 1, 1, 1)
            mask = ((x >= k / L) & (x < (k + 1) / L)).float()
            out += mask * (x * tk)
        return out

    def forward(self, x, params):
        # params: [gamma, contrast, WB_r, WB_g, WB_b, tone_params(3), sharpen]
        gamma, contrast = params[:, 0], params[:, 1]
        wb = params[:, 2:5].unsqueeze(-1).unsqueeze(-1)
        tone = params[:, 5:8].unsqueeze(-1).unsqueeze(-1)
        # sharpen_factor = params[:, -1]
        sharpen_factor = torch.sigmoid(params[:, -1]) * 1.0

        # Gamma correction
        x = torch.pow(x, gamma.view(-1, 1, 1, 1))

        # Contrast adjustment
        lum = 0.27 * x[:, 0:1, :, :] + 0.67 * x[:, 1:2, :, :] + 0.06 * x[:, 2:3, :, :]
        en = x * (0.5 * (1 - torch.cos(torch.pi * lum)) / (lum + 1e-6))
        x = contrast.view(-1, 1, 1, 1) * en + (1 - contrast.view(-1, 1, 1, 1)) * x

        # White balance
        x = x * wb

        # Tone adjustment (simplified)
        x = self.tone_mapping(x, tone.squeeze(-1).squeeze(-1))

        # Sharpening
        gaussian = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x + sharpen_factor.view(-1, 1, 1, 1) * (x - gaussian)

        return torch.clamp(x, 0, 1)


# Complete model
class AIPResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.nlpp = NLPP()
        self.dip = DIP()
        self.backbone = get_resnet50()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        params = self.nlpp(x)
        # Example: use sigmoid of last param as control
        control_signal = torch.sigmoid(params[:, -1])
        # Apply DIP conditionally
        if control_signal.mean() > 0.5:  # You can customize threshold
            x = self.dip(x, params)
        logits = self.backbone(x)
        return logits
