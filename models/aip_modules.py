import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Bloco Non-Local (Conforme descrito no paper de referência Wang et al., 2018)
# --------------------------------------------------------------------------
class NonLocalBlock(nn.Module):
    """
    Implementa o bloco Non-Local para capturar dependências de longa distância,
    conforme referenciado pelo artigo[cite: 212].
    """

    def __init__(self, in_channels):
        super().__init__()
        inter_channels = in_channels // 2

        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, inter_channels, kernel_size=1)

        self.out_conv = nn.Conv2d(inter_channels, in_channels, kernel_size=1)

        # Inicializa a camada de saída com zeros para estabilidade no início do treino
        nn.init.constant_(self.out_conv.weight, 0)
        nn.init.constant_(self.out_conv.bias, 0)

    def forward(self, x):
        batch_size, C, H, W = x.size()

        # Transforma as entradas
        theta_x = self.theta(x).view(batch_size, -1, H * W)
        phi_x = self.phi(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        g_x = self.g(x).view(batch_size, -1, H * W).permute(0, 2, 1)

        # Calcula a matriz de atenção
        attention = torch.bmm(phi_x, theta_x)
        attention = F.softmax(attention, dim=-1)

        # Aplica a atenção aos features 'g'
        y = torch.bmm(attention, g_x)
        y = y.permute(0, 2, 1).contiguous().view(batch_size, -1, H, W)

        # Projeção de saída + conexão residual
        out = self.out_conv(y)
        return out + x


# --------------------------------------------------------------------------
# NLPP: Preditor de Parâmetros (Conforme Tabela 4 do artigo)
# --------------------------------------------------------------------------
class NLPP(nn.Module):
    def __init__(self, tone_L=8):
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

        # 1(gamma)+1(contrast)+3(wb)+tone_L+1(sharpen)+1(control_signal)
        out_features = 1 + 1 + 3 + tone_L + 1 + 1

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, out_features),
            nn.Tanh(),  # Força a saída para o intervalo estável [-1, 1]
        )

    def forward(self, x):
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = self.features(x)
        return self.fc(x)


# --------------------------------------------------------------------------
# DIP: Módulo de Processamento de Imagem (Conforme Tabela 3 do artigo)
# --------------------------------------------------------------------------
class DIP(nn.Module):
    def __init__(self, tone_L=8):
        super().__init__()
        self.tone_L = tone_L

    def tone_mapping(self, x, tone_params):
        out = torch.zeros_like(x)
        for k in range(self.tone_L):
            tk = tone_params[:, k].view(-1, 1, 1, 1)
            mask = ((x >= k / self.tone_L) & (x < (k + 1) / self.tone_L)).float()
            out += mask * (x * tk)
        return out

    def forward(self, x, params):
        # Re-escala os parâmetros do intervalo [-1, 1] para a faixa operacional
        # Fórmula: saida = ((entrada + 1) / 2) * (max - min) + min
        gamma = ((params[:, 0] + 1) / 2) * (2.5 - 0.4) + 0.4
        contrast = ((params[:, 1] + 1) / 2) * (1.5 - 0.5) + 0.5
        wb = ((params[:, 2:5] + 1) / 2) * (1.3 - 0.7) + 0.7
        tone = params[:, 5 : 5 + self.tone_L]
        sharpen_factor = torch.sigmoid(params[:, -2])  # Penúltimo parâmetro
        # O último parâmetro é o control_signal, usado no modelo principal

        # Aplica as transformações
        x = torch.clamp(x, min=1e-6, max=1.0)
        x = torch.pow(x, gamma.view(-1, 1, 1, 1))

        lum = 0.27 * x[:, 0:1, :, :] + 0.67 * x[:, 1:2, :, :] + 0.06 * x[:, 2:3, :, :]
        en = x * (0.5 * (1 - torch.cos(torch.pi * lum)) / (torch.clamp(lum, min=1e-6)))
        x = contrast.view(-1, 1, 1, 1) * en + (1 - contrast.view(-1, 1, 1, 1)) * x

        x = x * wb.view(-1, 3, 1, 1)
        x = self.tone_mapping(x, tone)

        gaussian = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x + sharpen_factor.view(-1, 1, 1, 1) * (x - gaussian)

        return torch.clamp(x, 0, 1)
