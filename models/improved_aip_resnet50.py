# Implementation Plan:
# 1. Improved DIP with full 5-function mapping
# 2. Enhanced NLPP with multi-head self-attention (MHSA)
# 3. Backbone ResNet50 + CBAM for attention
# 4. Joint model with soft blending
# 5. Data augmentation pipeline

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import get_resnet50


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


# -----------------------------
# Improved DIP Module
class ImprovedDIP(nn.Module):
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
        B, C, H, W = x.shape
        x = torch.clamp(x, min=1e-6, max=1.0)  # Ensure no zeros/negatives

        # Clamp params for stability
        gamma = torch.clamp(params[:, 0], min=0.5, max=2.0)
        contrast = torch.clamp(params[:, 1], min=0.5, max=1.5)
        wb = torch.clamp(params[:, 2:5], min=0.8, max=1.2).unsqueeze(-1).unsqueeze(-1)
        tone = params[:, 5 : 5 + self.tone_L].unsqueeze(-1).unsqueeze(-1)
        sharpen_factor = torch.sigmoid(params[:, -1]).view(-1, 1, 1, 1)

        # Gamma correction
        x = torch.pow(x, gamma.view(-1, 1, 1, 1))

        # Contrast adjustment
        lum = 0.27 * x[:, 0:1, :, :] + 0.67 * x[:, 1:2, :, :] + 0.06 * x[:, 2:3, :, :]
        en = x * (0.5 * (1 - torch.cos(torch.pi * lum)) / (lum + 1e-3))  # Safer epsilon
        x = contrast.view(-1, 1, 1, 1) * en + (1 - contrast.view(-1, 1, 1, 1)) * x

        # White balance
        x = x * wb

        # Tone adjustment
        x = self.tone_mapping(x, tone.squeeze(-1).squeeze(-1))

        # Differentiable Gaussian sharpening
        kernel = torch.ones((C, 1, 3, 3), device=x.device, dtype=x.dtype) / 9.0
        gaussian = F.conv2d(x, kernel, padding=1, groups=C)
        x = x + sharpen_factor * (x - gaussian)

        return torch.clamp(x, 0, 1)  # Final clamp


# -----------------------------
# Multi-head Self Attention for NLPP
class MHSA(nn.Module):
    def __init__(self, in_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (in_dim // heads) ** -0.5
        self.qkv = nn.Linear(in_dim, in_dim * 3)
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        dots = (q @ k.transpose(-2, -1)) * self.scale
        dots = torch.clamp(dots, min=-50, max=50)  # Prevent overflow
        attn = F.softmax(dots, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.fc(out)


# -----------------------------
# Enhanced NLPP
class EnhancedNLPP(nn.Module):
    def __init__(self, out_dim=16, embed_dim=512, heads=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # (B,32,4,4)
        self.flatten = nn.Flatten()

        self.mhsa = MHSA(in_dim=512, heads=heads)
        self.fc = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64))
        x = self.conv(x)
        x = self.pool(x)
        x = self.flatten(x)  # shape (B, 512)
        x = x.unsqueeze(1)  # (B,1,512)
        x = self.mhsa(x).squeeze(1)  # (B,512)
        return self.fc(x)


# -----------------------------
# Full Model
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
