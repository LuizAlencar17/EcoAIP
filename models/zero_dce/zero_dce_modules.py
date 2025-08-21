import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# 1) DCE-Net estável
# ----------------------------
class DCE_Net(nn.Module):
    def __init__(self, n_iter=8):
        super().__init__()
        self.n_iter = n_iter
        num_out_channels = 3 * n_iter

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32 * 2, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32 * 2, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32 * 2, num_out_channels, 3, 1, 1)

        self.act = nn.ReLU(inplace=False)  # inplace=False evita problemas de autograd

        # --- inicialização estável ---
        for m in [
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.conv6,
        ]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # última camada = ZERO para saída ~ identidade (alpha≈0)
        nn.init.zeros_(self.conv7.weight)
        nn.init.zeros_(self.conv7.bias)

        # escala aprendível, pequena e limitada (<= 0.25)
        self._raw_scale = nn.Parameter(
            torch.tensor(0.0)
        )  # softplus -> ~0.693 no início
        self.max_scale = 0.25

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(x2))
        x4 = self.act(self.conv4(x3))
        x5 = self.act(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.act(self.conv6(torch.cat([x2, x5], 1)))
        A_raw = self.conv7(torch.cat([x1, x6], 1))  # sem tanh aqui

        # alpha escalado e limitado
        scale = torch.clamp(F.softplus(self._raw_scale), max=self.max_scale)
        A = torch.tanh(A_raw) * scale  # [-max_scale, +max_scale]
        return A


# ----------------------------
# 2) Zero-DCE com guardas
# ----------------------------
class ZeroDCE(nn.Module):
    def __init__(self, n_iter=8, per_iter_clamp=True, eps=1e-6):
        super().__init__()
        self.n_iter = n_iter
        self.dce_net = DCE_Net(n_iter=n_iter)
        self.per_iter_clamp = per_iter_clamp
        self.eps = eps

    def forward(self, x):
        # garanta domínio válido
        x = x.clamp(self.eps, 1.0 - self.eps)

        A = self.dce_net(x)
        enhanced_x = x

        # faça o loop em float32 mesmo sob AMP para evitar FP16 NaN
        use_amp = torch.is_autocast_enabled()
        if use_amp:
            # força precisão 32 para a parte sensível
            with torch.cuda.amp.autocast(enabled=False):
                enhanced_x = enhanced_x.float()
                A = A.float()
                for i in range(self.n_iter):
                    alpha = A[:, i * 3 : (i + 1) * 3]
                    enhanced_x = enhanced_x + alpha * enhanced_x * (1.0 - enhanced_x)
                    if self.per_iter_clamp:
                        enhanced_x = enhanced_x.clamp(self.eps, 1.0 - self.eps)
        else:
            for i in range(self.n_iter):
                alpha = A[:, i * 3 : (i + 1) * 3]
                enhanced_x = enhanced_x + alpha * enhanced_x * (1.0 - enhanced_x)
                if self.per_iter_clamp:
                    enhanced_x = enhanced_x.clamp(self.eps, 1.0 - self.eps)

        return enhanced_x.clamp(0.0, 1.0)
