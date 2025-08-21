import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_PP(nn.Module):
    """
    Implementa o CNN-based Parameter Predictor (CNN-PP) com saída estabilizada.
    """

    def __init__(self, num_params: int = 14):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_params),
            nn.Tanh(),  # <-- MUDANÇA: Força a saída para o intervalo [-1, 1]
        )

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = self.conv_blocks(x)
        params = self.fc_layers(x)
        return params


# Módulo DIP que reescala os parâmetros para uma faixa segura


class DIP(nn.Module):
    def __init__(self, tone_L: int = 8):
        super().__init__()
        self.tone_L = tone_L

    def tone_mapping(self, x, tone_params):
        # ... (código do tone_mapping)
        out = torch.zeros_like(x)
        for k in range(self.tone_L):
            tk = tone_params[:, k].view(-1, 1, 1, 1)
            mask = ((x >= k / self.tone_L) & (x < (k + 1) / self.tone_L)).float()
            out += mask * (x * tk)
        return out

    def forward(self, x, params):
        # params agora está no intervalo [-1, 1]

        ### MUDANÇA: Reescalonamento de parâmetros para uma faixa operacional segura ###
        # Fórmula: saida = ((entrada + 1) / 2) * (max - min) + min
        wb = (((params[:, 0:3] + 1) / 2) * (1.3 - 0.7) + 0.7).view(-1, 3, 1, 1)
        gamma = (((params[:, 3] + 1) / 2) * (2.5 - 0.4) + 0.4).view(-1, 1, 1, 1)
        contrast = (((params[:, 4] + 1) / 2) * (1.5 - 0.5) + 0.5).view(-1, 1, 1, 1)
        tone = params[:, 5 : 5 + self.tone_L]
        # Sigmoid já força a saída para [0, 1], então reescalonar não é necessário
        sharpen_lambda = torch.sigmoid(params[:, -1]).view(-1, 1, 1, 1)

        # Garante que a entrada para as operações seja positiva
        x = torch.clamp(x, min=1e-6, max=1.0)

        # Aplicação sequencial dos filtros com parâmetros agora estáveis
        x = x * wb
        x = torch.pow(x, gamma)

        lum = 0.27 * x[:, 0:1, :, :] + 0.67 * x[:, 1:2, :, :] + 0.06 * x[:, 2:3, :, :]
        # Adiciona um clamp na luminância para segurança extra
        en = x * (0.5 * (1 - torch.cos(torch.pi * lum)) / (torch.clamp(lum, min=1e-5)))
        x = contrast * en + (1 - contrast) * x

        x = self.tone_mapping(x, tone)

        gaussian_approx = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x + sharpen_lambda * (x - gaussian_approx)

        return torch.clamp(x, 0, 1)
