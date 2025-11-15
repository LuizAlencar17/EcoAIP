# retinexformer.py
# Implementação fiel ao desenho do repo oficial, em um único arquivo, PnP.
# Autor: você :)  Licença: compatível com sua pesquisa

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# Utilidades
# -------------------------------------------------------
def pad_to_factor(
    x: torch.Tensor, factor: int = 4
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    b, c, h, w = x.shape
    H = (h + factor - 1) // factor * factor
    W = (w + factor - 1) // factor * factor
    pad_h, pad_w = H - h, W - w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad(x: torch.Tensor, pads: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pads
    if pad_h or pad_w:
        return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]
    return x


# -------------------------------------------------------
# 1) Illumination Estimator (iguais sinais de entrada/saída do repo)
#    Entrada: img [B,3,H,W] → concat com média canal (B,4,H,W)
#    Saída:   illu_fea_list (multi-escala), illu_map [B,3,H,W]
# -------------------------------------------------------
class Illumination_Estimator(nn.Module):
    """
    Estrutura minimalista inspirada no arquivo oficial:
      1x1 conv → depthwise 5x5 → 1x1 conv.
    """

    def __init__(self, n_fea_middle: int, n_fea_in: int = 4, n_fea_out: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        # groups=n_fea_in no código original; aqui usamos groups=n_fea_middle
        # para preservar a separabilidade respeitando C após conv1.
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            padding=2,
            bias=True,
            groups=n_fea_middle,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        # projeta feições hierárquicas para condicionar o denoiser
        self.pyr2 = nn.Conv2d(n_fea_middle, n_fea_middle, 3, 2, 1)  # H/2
        self.pyr4 = nn.Conv2d(n_fea_middle, n_fea_middle, 3, 2, 1)  # H/4

        self.act = nn.GELU()

    def forward(self, img: torch.Tensor):
        # mean channel
        mean_c = img.mean(dim=1, keepdim=True)
        x = torch.cat([img, mean_c], dim=1)  # [B,4,H,W]

        fea = self.act(self.conv1(x))
        fea = self.act(self.depth_conv(fea))
        illu_map = torch.sigmoid(self.conv2(fea))  # [B,3,H,W] em [0,1]

        # pirâmide simples de condicionamento (multi-escala)
        p2 = self.act(self.pyr2(fea))  # H/2
        p4 = self.act(self.pyr4(p2))  # H/4
        illu_fea_list = [fea, p2, p4]  # fino → grosso

        return illu_fea_list, illu_map


# -------------------------------------------------------
# 2) Blocos Retinexformer (LeWin + Gated FFN) estilo Restormer
# -------------------------------------------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        mean = x.mean(dim=(2, 3), keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class GatedDconvFFN(nn.Module):
    def __init__(self, c: int, expand: int = 2):
        super().__init__()
        hidden = c * expand
        self.pw1 = nn.Conv2d(c, hidden * 2, 1, 1, 0, bias=True)
        self.dw = nn.Conv2d(
            hidden * 2, hidden * 2, 3, 1, 1, groups=hidden * 2, bias=True
        )
        # <<< troque para inplace=False
        self.act = nn.SiLU(inplace=False)
        self.pw2 = nn.Conv2d(hidden, c, 1, 1, 0, bias=True)
        self.norm = LayerNorm2d(c)

    def forward(self, x):
        y = self.norm(x)
        y = self.pw1(y)
        y = self.dw(y)
        a, b = torch.chunk(y, 2, dim=1)  # views
        # opção 1 (mais simples):
        a = self.act(a)  # não-inplace
        y = a * b
        y = self.pw2(y)
        return x + y


def window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    B, C, H, W = x.shape
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, H/ws, W/ws, ws, ws, C
    return x.view(-1, ws * ws, C)  # (num_windows*B), tokens, C


def window_reverse(windows: torch.Tensor, ws: int, B: int, C: int, H: int, W: int):
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, H/ws, ws, W/ws, ws
    return x.view(B, C, H, W)


class LeWinAttention(nn.Module):
    """Self-attention por janelas locais (sem viés relativo para manter leve)."""

    def __init__(self, c: int, heads: int = 4, window_size: int = 8):
        super().__init__()
        self.c = c
        self.h = heads
        self.ws = window_size
        self.qkv = nn.Conv2d(c, c * 3, 1, 1, 0, bias=True)
        self.proj = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.scale = (c // heads) ** -0.5
        self.norm = LayerNorm2d(c)

    def forward(self, x):
        B, C, H, W = x.shape
        if H % self.ws or W % self.ws:
            x, pads = pad_to_factor(x, self.ws)
            B, C, H, W = x.shape
        y = self.norm(x)
        qkv = self.qkv(y)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # [B,C,H,W]
        # partição em janelas
        q = window_partition(q, self.ws)  # [nW*B, ws*ws, C]
        k = window_partition(k, self.ws)
        v = window_partition(v, self.ws)

        # split em cabeças
        def split_heads(t):
            N, T, Cc = t.shape
            d = Cc // self.h
            t = t.view(N, T, self.h, d).permute(0, 2, 1, 3)  # N,h,T,d
            return t

        q, k, v = map(split_heads, (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v  # N,h,T,d
        # junta cabeças
        N, h, T, d = out.shape
        out = out.permute(0, 2, 1, 3).reshape(N, T, h * d)
        # volta para mapa
        out = out.transpose(1, 2).contiguous().view(N, self.c, self.ws, self.ws)
        out = window_reverse(out, self.ws, B, C, H, W)
        out = self.proj(out)
        if "pads" in locals():
            out = unpad(out, pads)
        return x + out


class FiLMCond(nn.Module):
    """Modulação condicional (scale/shift) a partir de feições de iluminação.
    Agora aceita cond_c (canais do condicionador) e projeta para c.
    """

    def __init__(self, c: int, cond_c: Optional[int] = None):
        super().__init__()
        self.c = c
        self.cond_c = cond_c if cond_c is not None else c

        # Projeta cond -> c, se necessário
        if self.cond_c != self.c:
            self.adapt = nn.Conv2d(self.cond_c, self.c, 1, 1, 0, bias=True)
        else:
            self.adapt = nn.Identity()

        self.to_affine = nn.Sequential(
            nn.Conv2d(self.c, self.c, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.Conv2d(self.c, self.c * 2, 1, 1, 0, bias=True),
        )

    def forward(self, x, cond):
        # redimensiona cond para a mesma resolução de x
        if cond is not None and cond.shape[-2:] != x.shape[-2:]:
            cond = F.interpolate(
                cond, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        cond = self.adapt(cond)  # cond_c -> c
        gamma, beta = torch.chunk(self.to_affine(cond), 2, dim=1)
        return x * (1 + torch.tanh(gamma)) + beta


class LeWinBlock(nn.Module):
    def __init__(
        self,
        c: int,
        heads: int = 4,
        window_size: int = 8,
        use_cond: bool = True,
        cond_c: Optional[int] = None,
    ):
        super().__init__()
        self.attn = LeWinAttention(c, heads=heads, window_size=window_size)
        self.ffn = GatedDconvFFN(c)
        self.use_cond = use_cond
        self.film = FiLMCond(c, cond_c=cond_c) if use_cond else None

    def forward(self, x, illu_fea: Optional[torch.Tensor] = None):
        y = self.attn(x)
        y = self.ffn(y)
        if self.use_cond and illu_fea is not None:
            y = self.film(y, illu_fea)
        return y


# -------------------------------------------------------
# 3) Denoiser / Img2Img (encoder-bottleneck-decoder)
# -------------------------------------------------------
class FeaDownSample(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, 2, 1, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.op(x)


class FeaUpSample(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c_in, c_out, 3, 1, 1, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.op(x)


class Fusion(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        return self.op(x)


class RepeatedLeWin(nn.Module):
    def __init__(
        self,
        c: int,
        num_blocks: int,
        heads: int,
        ws: int,
        use_cond: bool,
        cond_c: Optional[int] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                LeWinBlock(
                    c, heads=heads, window_size=ws, use_cond=use_cond, cond_c=cond_c
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, illu: Optional[torch.Tensor] = None):
        for b in self.blocks:
            x = b(x, illu)
        return x


class Denoiser(nn.Module):
    """
    Denoiser/img2img com condicionamento por feições de iluminação, seguindo:
      Embedding → [Encoder: (MSAB → Down)*L] → Bottleneck → [Decoder: (Up→Fusion→LeWin)*L] → Mapping + skip.
    """

    def __init__(
        self,
        in_dim=3,
        out_dim=3,
        dim=31,
        level=2,
        num_blocks=[1, 1, 1],
        heads=4,
        window_size=8,
        illu_c=32,
    ):
        super().__init__()
        assert level >= 1
        self.level = level
        self.dim = dim

        # embedding
        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        # encoder
        enc = []
        c = dim
        for li in range(level):
            nblk = num_blocks[min(li, len(num_blocks) - 1)]
            enc.append(
                nn.ModuleList(
                    [
                        RepeatedLeWin(
                            c, nblk, heads, window_size, use_cond=False
                        ),  # MSAB (sem cond no encoder)
                        FeaDownSample(c, c * 2) if li < level - 1 else nn.Identity(),
                    ]
                )
            )
            if li < level - 1:
                c *= 2
        self.encoder_layers = nn.ModuleList(enc)

        # deepest channel count at bottleneck:
        c = dim * (2 ** (level - 1))

        # bottleneck conditioned with cond_c=illu_c
        self.bottleneck = RepeatedLeWin(
            c,
            num_blocks[min(level - 1, len(num_blocks) - 1)],
            heads,
            window_size,
            use_cond=True,
            cond_c=illu_c,
        )

        # decoder blocks also conditioned with same illu_c (illum features are always n_feat)
        dec = []
        for li in range(level - 1):
            in_c = c
            out_c = c // 2
            dec.append(
                nn.ModuleList(
                    [
                        FeaUpSample(in_c, out_c),
                        Fusion(out_c + out_c, out_c),
                        RepeatedLeWin(
                            out_c, 1, heads, window_size, use_cond=True, cond_c=illu_c
                        ),
                    ]
                )
            )
            c = out_c
        self.decoder_layers = nn.ModuleList(dec)
        # mapping
        self.mapping = nn.Conv2d(dim, out_dim, 3, 1, 1, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(
        self, x: torch.Tensor, illu_fea_list: Optional[List[torch.Tensor]] = None
    ):
        """
        x:  [B,3,H,W]
        illu_fea_list: lista com feições de iluminação do estimador (grossa→fina ou vice-versa).
        """
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        for msab, down in self.encoder_layers:
            fea = msab(fea, None)
            fea_encoder.append(fea)
            fea = down(fea)

        # Bottleneck (condicionado pela feição mais grossa se existir)
        illu_fea = None
        if illu_fea_list is not None:
            illu_fea = illu_fea_list[-1]
        fea = self.bottleneck(fea, illu_fea)

        # Decoder (condiciona do grosso→fino)
        if illu_fea_list is None:
            illu_seq = [None] * (self.level - 1)
        else:
            illu_seq = list(reversed(illu_fea_list[:-1]))  # alinhar aos skips

        for i, (up, fuse, lewin) in enumerate(self.decoder_layers):
            fea = up(fea)
            skip = fea_encoder[self.level - 2 - i]
            fea = fuse(torch.cat([fea, skip], dim=1))
            illu_here = illu_seq[i] if i < len(illu_seq) else None
            fea = lewin(fea, illu_here)

        # Mapping + residual
        out = self.mapping(fea) + x
        return out


# -------------------------------------------------------
# 4) Estágio único e pilha de estágios (oficial)
# -------------------------------------------------------
class RetinexFormer_Single_Stage(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        level=2,
        num_blocks=[1, 1, 1],
        heads=4,
        window_size=8,
    ):
        super().__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(
            in_dim=in_channels,
            out_dim=out_channels,
            dim=n_feat,
            level=level,
            num_blocks=num_blocks,
            heads=heads,
            window_size=window_size,
            illu_c=n_feat,  # <<< HERE
        )

    def forward(self, img: torch.Tensor):
        illu_fea_list, illu_map = self.estimator(img)  # mapas de iluminação
        input_img = img * illu_map + img  # combinação conforme repo
        output_img = self.denoiser(input_img, illu_fea_list)
        return output_img


class RetinexFormer(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        n_feat=31,
        stage=3,
        level=2,
        num_blocks=[1, 1, 1],
        heads=4,
        window_size=8,
    ):
        super().__init__()
        self.body = nn.Sequential(
            *[
                RetinexFormer_Single_Stage(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_feat=n_feat,
                    level=level,
                    num_blocks=num_blocks,
                    heads=heads,
                    window_size=window_size,
                )
                for _ in range(stage)
            ]
        )

    def forward(self, x: torch.Tensor):
        # padding para múltiplos de 4/8 ajuda estabilidade, igual prática do repo
        x_pad, pads = pad_to_factor(x, factor=4)
        y = self.body(x_pad)
        return unpad(y, pads)


# -------------------------------------------------------
# 5) Fábricas convenientes
# -------------------------------------------------------
def retinexformer_tiny(stage=1):
    return RetinexFormer(
        n_feat=32, stage=stage, level=2, num_blocks=[1, 1, 1], heads=2, window_size=8
    )


def retinexformer_small(stage=1):
    # n_feat=40 → Illumination_Estimator outputs 40-ch features
    # illu_c=n_feat=40 flows automatically into Denoiser via Single_Stage
    return RetinexFormer(
        n_feat=40, stage=stage, level=3, num_blocks=[1, 2, 2], heads=4, window_size=8
    )


def retinexformer_base(stage=3):
    return RetinexFormer(
        n_feat=48, stage=stage, level=3, num_blocks=[2, 2, 2], heads=4, window_size=8
    )
