import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedDIPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sharpen_scales = nn.ModuleList(
            [
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3)
                for _ in range(3)
            ]
        )
        self.init_sharpen_kernels()
        self.hist_eq = ImprovedHistEqualization()

    def init_sharpen_kernels(self):
        laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float()
        for conv in self.sharpen_scales:
            with torch.no_grad():
                k = laplacian.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
                conv.weight.copy_(k)

    def forward(self, x, params):
        gamma, contrast = params[:, 0:1], params[:, 1:2]
        wb = params[:, 2:5].unsqueeze(2).unsqueeze(3)
        tone_curve = params[:, 5:13].view(-1, 8, 1, 1, 1)

        x = self.gamma_adjust(x, gamma)
        x = self.contrast_adjust(x, contrast)
        x = self.white_balance(x, wb)
        x = self.tone_adjust(x, tone_curve)
        x = self.hist_eq(x)

        for conv in self.sharpen_scales:
            x = x + 0.1 * conv(x)
        return x

    def gamma_adjust(self, x, gamma):
        gamma = gamma.view(-1, 1, 1, 1)
        return torch.pow(x.clamp_min(1e-6), gamma)

    def contrast_adjust(self, x, alpha):
        lum = 0.27 * x[:, 0:1] + 0.67 * x[:, 1:2] + 0.06 * x[:, 2:3]
        en = x * 0.5 * (1 - torch.cos(torch.pi * lum)) / (lum + 1e-6)
        alpha = alpha.view(-1, 1, 1, 1)
        return alpha * en + (1 - alpha) * x

    def white_balance(self, x, wb):
        return x * wb

    def tone_adjust(self, x, tone_curve):
        return torch.clamp(x + (torch.sum(tone_curve * x.unsqueeze(1), dim=1)), 0, 1)


class ImprovedHistEqualization(nn.Module):
    def forward(self, x):
        # Simple approximation: normalize to [0,1]
        return (x - x.min()) / (x.max() - x.min() + 1e-6)


class ImprovedViTPredictor(nn.Module):
    def __init__(self, out_dim=13):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64))
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ImprovedAIPModel(nn.Module):
    def __init__(self, backbone, dip, predictor):
        super().__init__()
        self.backbone = backbone
        self.dip = dip
        self.predictor = predictor

    def forward(self, x):
        params = self.predictor(x)
        x_dip = self.dip(x, params)
        out = self.backbone(x_dip)
        return out, x_dip
