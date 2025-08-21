# models/aip/zerodce_plus_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# Utilitários
# =========================
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    return (x - IMNET_MEAN.to(x.device, x.dtype)) / IMNET_STD.to(x.device, x.dtype)


def luminance_rgb(x: torch.Tensor) -> torch.Tensor:
    return 0.27 * x[:, 0:1] + 0.67 * x[:, 1:2] + 0.06 * x[:, 2:3]


def safe01(x: torch.Tensor) -> torch.Tensor:
    # evita NaN/Inf sob AMP
    return x.clamp(1e-6, 1.0)


# =========================
# Blocos leves (DW + CBAM)
# =========================
class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=True)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class ChannelAttention(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        m = max(4, c // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, m, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(m, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * (self.fc(self.avg(x)) + self.fc(self.max(x))) * 0.5


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, 1, 3, bias=False)

    def forward(self, x):
        a = x.mean(1, keepdim=True)
        m, _ = x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([a, m], 1)))


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttention(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# =========================
# Spline tonal suave (monotônica "fraca")
# =========================
class SoftSpline(nn.Module):
    def __init__(self, L: int = 16, tau: float = 8.0):
        super().__init__()
        self.L, self.tau = L, tau
        centers = torch.linspace(0.0, 1.0, steps=L).view(1, L, 1, 1)
        self.register_buffer("centers", centers)

    def forward(self, x_gray: torch.Tensor, w_raw: torch.Tensor) -> torch.Tensor:
        # x_gray: [B,1,H,W], w_raw: [B,L]
        B, _, H, W = x_gray.shape
        w = F.softplus(w_raw) + 1e-6
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)  # soma=1 → monotonicidade “fraca”
        d = -self.tau * (x_gray - self.centers).pow(2)  # [B,L,H,W]
        A = F.softmax(d, dim=1)  # base Gaussiana suave
        y = torch.einsum("blhw,bl->bhw", A, w).unsqueeze(1)
        return y.clamp(0, 1)


# =========================
# Parametrizador global (tipo NLPP)
# =========================
class ParamHead(nn.Module):
    """
    Encoder leve (DW+CBAM) → MLP global
    Saída: [gamma, contrast, wb_r, wb_g, wb_b, tone[L], sharpen, exp_map(16*16)]
    """

    def __init__(self, tone_L: int = 16, base_c: int = 24):
        super().__init__()
        self.tone_L = tone_L
        c = base_c
        self.e1 = nn.Sequential(DWConv(3, c), CBAM(c))
        self.e2 = nn.Sequential(DWConv(c, c * 2, s=2), DWConv(c * 2, c * 2))
        self.e3 = nn.Sequential(DWConv(c * 2, c * 4, s=2), DWConv(c * 4, c * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        out_dim = 7 + tone_L + 256
        self.mlp = nn.Sequential(
            nn.Conv2d(c * 4, 128, 1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, out_dim, 1, bias=True),
        )
        # viés pró-clareamento/contraste e tom crescente
        with torch.no_grad():
            b = self.mlp[-1].bias
            if b is not None and b.numel() >= (5 + tone_L):
                b[:1].fill_(-0.5)  # gamma_logit baixo → gamma < 1
                b[1:2].fill_(0.3)  # contrast_logit positivo → >1
                b[2:5].zero_()  # wb ~1
                b[5 : 5 + tone_L].copy_(torch.linspace(0.0, 0.6, steps=tone_L))

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        f = self.e3(self.e2(self.e1(x)))  # [B,4c,H/4,W/4]
        g = self.pool(f)  # [B,4c,1,1]
        p = self.mlp(g).flatten(1)  # [B, 7+L+256]
        return p


# =========================
# DCE backbone (mapa A per-pixel: r1..r8)
# =========================
class DCEBackbone(nn.Module):
    def __init__(self, n_iter=8, c=32):
        super().__init__()
        self.act = nn.ReLU(inplace=False)
        self.e1 = nn.Conv2d(3, c, 3, 1, 1, bias=True)
        self.e2 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.e3 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.e4 = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        self.e5 = nn.Conv2d(c * 2, c, 3, 1, 1, bias=True)
        self.e6 = nn.Conv2d(c * 2, c, 3, 1, 1, bias=True)
        self.e7 = nn.Conv2d(c * 2, 3 * n_iter, 3, 1, 1, bias=True)
        # init estável
        for m in [self.e1, self.e2, self.e3, self.e4, self.e5, self.e6]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.e7.weight)
        nn.init.zeros_(self.e7.bias)

    def forward(self, x):
        x1 = self.act(self.e1(x))
        x2 = self.act(self.e2(x1))
        x3 = self.act(self.e3(x2))
        x4 = self.act(self.e4(x3))
        x5 = self.act(self.e5(torch.cat([x3, x4], 1)))
        x6 = self.act(self.e6(torch.cat([x2, x5], 1)))
        A = torch.tanh(self.e7(torch.cat([x1, x6], 1)))  # [-1,1], [B,3*n,H,W]
        return A


# =========================
# ZeroDCEPlus (aprimorado e estável)
# =========================
class ZeroDCEPlus(nn.Module):
    """
    Aprimorador:
      - compensação de exposição (global + mapa local 16x16)
      - LE iterativa (r1..r8) por-pixel
      - gamma, contraste edge-aware, WB, spline tonal suave, sharpen
    Entrada: x [B,3,H,W] em [0,1]
    Saída: y [B,3,H,W] em [0,1] e aux dict (se return_aux=True)
    """

    def __init__(
        self,
        n_iter=8,
        tone_L=16,
        exp_target=0.60,
        exp_gain_clip=(0.85, 1.40),
        base_c=24,
        dce_c=32,
    ):
        super().__init__()
        self.n_iter = int(n_iter)
        self.tone_L = int(tone_L)
        self.t = float(exp_target)
        self.gmin, self.gmax = float(exp_gain_clip[0]), float(exp_gain_clip[1])
        self.param_head = ParamHead(tone_L=tone_L, base_c=base_c)
        self.spline = SoftSpline(L=tone_L, tau=8.0)
        self.dce = DCEBackbone(n_iter=n_iter, c=dce_c)
        self.register_buffer("box3", torch.ones(3, 1, 3, 3) / 9.0)

    @staticmethod
    def _le_step(x, r):
        # LE curva: x + r * (x^2 - x)
        return x + r * (x * x - x)

    def forward(self, x, return_aux=False):
        x = safe01(x)
        B, C, H, W = x.shape

        # ---- parâmetros globais ----
        p = self.param_head(x)  # [B, 7+L+256]
        i = 0
        gamma = 0.45 + 1.90 * torch.sigmoid(p[:, i])
        i += 1
        contrast = 0.50 + 1.30 * torch.sigmoid(p[:, i])
        i += 1
        wb = 0.80 + 0.40 * torch.sigmoid(p[:, i : i + 3])
        i += 3
        tone_w = p[:, i : i + self.tone_L]
        i += self.tone_L
        sharpen = torch.sigmoid(p[:, i])
        i += 1
        exp_map = torch.tanh(p[:, i : i + 256]).view(B, 1, 16, 16)  # [-1,1]

        gamma = gamma.view(B, 1, 1, 1)
        contrast = contrast.view(B, 1, 1, 1)
        wb = wb.view(B, 3, 1, 1)
        sharpen = sharpen.view(B, 1, 1, 1)

        # ---- compensação global + local ----
        mean_l = luminance_rgb(x).mean(dim=[2, 3], keepdim=True)
        gain_g = (self.t / (mean_l + 1e-6)).clamp(self.gmin, self.gmax)
        gain_l = 1.0 + 0.25 * F.interpolate(
            exp_map, size=(H, W), mode="bilinear", align_corners=False
        )
        gain = (gain_g * gain_l).clamp(self.gmin, self.gmax)
        x0 = (x * gain).clamp(0, 1)

        # ---- mapa A e LE iterativa ----
        A = self.dce(x0)  # [B, 3*n, H, W]
        r_splits = torch.split(A, 3, dim=1)
        y = x0
        for r in r_splits:
            y = self._le_step(y, r).clamp(1e-6, 1.0 - 1e-6)

        # ---- gamma, contraste edge-aware, WB ----
        y = torch.pow(y, gamma)
        lum = luminance_rgb(y).clamp(1e-6, 1.0)
        en = y * (0.5 * (1 - torch.cos(torch.pi * lum)) / (lum + 1e-3))
        y = contrast * en + (1 - contrast) * y
        y = (y * wb).clamp(0, 1)

        # ---- spline tonal suave ----
        g = self.spline(lum, tone_w)  # [B,1,H,W]
        y = ((0.72 + 1.15 * g) * y).clamp(0, 1)

        # ---- sharpen diferenciável ----
        blur = F.conv2d(y, self.box3.to(y.device, y.dtype), padding=1, groups=3)
        y = (y + sharpen * (y - blur)).clamp(0, 1)

        if not return_aux:
            return y
        aux = dict(
            A=A,
            gain=gain,
            exp_map=exp_map,
            gamma=gamma,
            contrast=contrast,
            wb=wb,
            tone_w=tone_w,
            sharpen=sharpen,
        )
        return y, aux


# =========================
# Helpers opcionais (mantendo CE como principal)
# =========================
class ExposureMatchRel(nn.Module):
    """Regularizador leve: mantém a luminância média próxima da original (não força um alvo fixo)."""

    def __init__(self, weight=0.03):
        super().__init__()
        self.weight = float(weight)

    def forward(self, x, y):
        lx = luminance_rgb(x).mean(dim=[1, 2, 3])
        ly = luminance_rgb(y).mean(dim=[1, 2, 3])
        return self.weight * (ly - lx).abs().mean()


class EnhancementMargin(nn.Module):
    """Evita colapso p/ identidade: garante diferença média mínima m_min."""

    def __init__(self, m_min=0.03, weight=0.05):
        super().__init__()
        self.m_min = float(m_min)
        self.weight = float(weight)

    def forward(self, x, y):
        mad = (y - x).abs().mean()
        return self.weight * F.relu(self.m_min - mad)
