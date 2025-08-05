import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility: CBAM (Convolutional Block Attention Module)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# --------------------------------------------------------------------------
# Sub-módulo: Multi-Head Self-Attention (MHSA) para o EnhancedNLPP
# --------------------------------------------------------------------------
class MHSA(nn.Module):
    """
    Implementa a Multi-Head Self-Attention para melhorar a extração de features globais.
    """

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape  # Batch, SequenceLength, Channels(embed_dim)

        # Gera Q, K, V e os divide em múltiplas cabeças
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calcula a pontuação de atenção
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Aplica a atenção aos valores (V)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


# --------------------------------------------------------------------------
# NLPP Aprimorado com MHSA
# --------------------------------------------------------------------------
class EnhancedNLPP(nn.Module):
    """
    NLPP aprimorado que incorpora Multi-Head Self-Attention (MHSA)
    para uma melhor compreensão do contexto global da imagem.
    """

    def __init__(
        self, out_features: int = 14, embed_dim: int = 256, num_heads: int = 4
    ):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        )

        self.projection = nn.Linear(64, embed_dim)
        self.mhsa = MHSA(embed_dim, num_heads)

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features),
            nn.Tanh(),  # Força a saída para o intervalo estável [-1, 1]
        )

    def forward(self, x):
        # Redimensiona a imagem de entrada para um tamanho fixo
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        # Extrai features iniciais
        x = self.feature_extractor(x)  # Shape: [B, 64, 1, 1]
        x = x.flatten(start_dim=1)  # Shape: [B, 64]

        # Projeta para a dimensão de embedding e aplica MHSA
        x = self.projection(x)  # Shape: [B, embed_dim]
        x = x.unsqueeze(1)  # Shape: [B, 1, embed_dim] - Adiciona dimensão de sequência
        x = self.mhsa(x)
        x = x.squeeze(1)  # Shape: [B, embed_dim]

        # Prediz os parâmetros finais
        params = self.fc(x)
        return params


# --------------------------------------------------------------------------
# DIP Aprimorado com Sharpening Diferenciável
# --------------------------------------------------------------------------
class EnhancedDIP(nn.Module):
    """
    Módulo DIP aprimorado com sharpening via kernel Gaussiano aprendível
    e tone mapping avançado.
    """

    def __init__(self, tone_L: int = 8):
        super().__init__()
        self.tone_L = tone_L

        # Kernel Gaussiano 3x3 para sharpening, não aprendível mas diferenciável
        # Pode ser tornado aprendível removendo requires_grad=False
        gaussian_kernel = torch.tensor(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32
        )
        self.sharpen_kernel = (
            (gaussian_kernel / gaussian_kernel.sum()).unsqueeze(0).unsqueeze(0)
        )
        self.sharpen_kernel = self.sharpen_kernel.repeat(
            3, 1, 1, 1
        )  # Aplica para cada canal de cor

    def tone_mapping(self, x, tone_params):
        """
        Implementa o tone mapping via função linear por partes (piecewise linear).
        """
        out = torch.zeros_like(x)
        for k in range(self.tone_L):
            tk = tone_params[:, k].view(-1, 1, 1, 1)
            mask = ((x >= k / self.tone_L) & (x < (k + 1) / self.tone_L)).float()
            out += mask * (x * tk)
        return out

    def forward(self, x, params):
        # Garante estabilidade na entrada
        x = torch.clamp(x, min=1e-6, max=1.0)

        # Separa os parâmetros
        gamma = params[:, 0].view(-1, 1, 1, 1)
        contrast = params[:, 1].view(-1, 1, 1, 1)
        wb = params[:, 2:5].view(-1, 3, 1, 1)
        tone = params[:, 5 : 5 + self.tone_L]
        sharpen_lambda = params[:, -1].view(-1, 1, 1, 1)

        # 1. Gamma Correction
        x = torch.pow(x, gamma)

        # 2. Contrast Adjustment
        lum = 0.27 * x[:, 0:1, :, :] + 0.67 * x[:, 1:2, :, :] + 0.06 * x[:, 2:3, :, :]
        en = x * (0.5 * (1 - torch.cos(torch.pi * lum)) / (lum + 1e-6))
        x = contrast * en + (1 - contrast) * x

        # 3. White Balance
        x = x * wb

        # 4. Tone Adjustment
        x = self.tone_mapping(x, tone)

        # 5. Sharpening Aprimorado (com kernel Gaussiano)
        blurred = F.conv2d(
            x, self.sharpen_kernel.to(x.device), padding="same", groups=3
        )
        x = x + sharpen_lambda * (x - blurred)

        return torch.clamp(x, 0, 1)
