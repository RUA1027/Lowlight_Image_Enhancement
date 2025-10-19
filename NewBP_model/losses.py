import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Dict, Tuple

# Optional dependencies (safe to miss if not used)
try:
    import kornia as K  # type: ignore
    import kornia.color as Kcolor  # type: ignore
    import kornia.losses as Kloss  # type: ignore
except Exception:  # optional
    K = None
    Kcolor = None
    Kloss = None


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss.

    Assumes inputs are sRGB in [0,1]. Applies ImageNet mean/std normalization
    before feeding into VGG19 feature extractor (first 36 layers).
    """
    def __init__(self, device='cuda', use_mse: bool = True, reduction: str = 'mean'):
        super().__init__()
        vgg_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = nn.Sequential(*list(vgg_features.children())[:36]).to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        # ImageNet mean/std (RGB)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.loss_fn = F.mse_loss if use_mse else F.l1_loss
        self.reduction = reduction

    def forward(self, generated_img, target_img):
        xg = (generated_img.clamp(0, 1) - self.mean) / self.std
        xt = (target_img.clamp(0, 1) - self.mean) / self.std
        features_gen = self.vgg(xg)
        features_target = self.vgg(xt)
        return self.loss_fn(features_gen, features_target, reduction=self.reduction)


class HybridLoss(nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_perceptual=0.1, device='cuda'):
    # lambda_l1: L1损失的权重
    # lambda_perceptual: 感知损失的权重
    # device: 设备
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.l1_loss = nn.L1Loss().to(device)
        self.perceptual_loss = PerceptualLoss(device=device)

    def forward(self, generated_img, target_img):
        l1_val = self.l1_loss(generated_img, target_img)
        perceptual_val = self.perceptual_loss(generated_img, target_img)
        total = self.lambda_l1 * l1_val + self.lambda_perceptual * perceptual_val
        return total, l1_val, perceptual_val


class DeltaE00Loss(nn.Module):
    """Compute ΔE00 on sRGB([0,1]) tensors. Requires Kornia."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _ciede2000(Lab1: torch.Tensor, Lab2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # Lab: [N,3,H,W]
        L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
        L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]
        C1 = torch.sqrt(a1 * a1 + b1 * b1 + eps)
        C2 = torch.sqrt(a2 * a2 + b2 * b2 + eps)
        Cbar = 0.5 * (C1 + C2)
        G = 0.5 * (1 - torch.sqrt((Cbar**7) / ((Cbar**7) + (25**7) + eps)))
        a1p = (1 + G) * a1
        a2p = (1 + G) * a2
        C1p = torch.sqrt(a1p * a1p + b1 * b1 + eps)
        C2p = torch.sqrt(a2p * a2p + b2 * b2 + eps)
        h1p = torch.atan2(b1, a1p) % (2 * torch.pi)
        h2p = torch.atan2(b2, a2p) % (2 * torch.pi)
        dLp = L2 - L1
        dCp = C2p - C1p
        dhp = h2p - h1p
        dhp = dhp - (2 * torch.pi) * (dhp > torch.pi) + (2 * torch.pi) * (dhp < -torch.pi)
        dHp = 2 * torch.sqrt(C1p * C2p + eps) * torch.sin(dhp / 2)
        Lbar = 0.5 * (L1 + L2)
        Cbarp = 0.5 * (C1p + C2p)
        hsum = h1p + h2p
        hbarp = hsum / 2 - (torch.pi) * (torch.abs(h1p - h2p) > torch.pi) + (2 * torch.pi) * (hsum < 0)
        T = (
            1
            - 0.17 * torch.cos(hbarp - torch.deg2rad(torch.tensor(30.0, device=hbarp.device)))
            + 0.24 * torch.cos(2 * hbarp)
            + 0.32 * torch.cos(3 * hbarp + torch.deg2rad(torch.tensor(6.0, device=hbarp.device)))
            - 0.20 * torch.cos(4 * hbarp - torch.deg2rad(torch.tensor(63.0, device=hbarp.device)))
        )
        d_ro = 30 * torch.exp(-((torch.rad2deg(hbarp) - 275) / 25) ** 2)
        RC = 2 * torch.sqrt((Cbarp**7) / ((Cbarp**7) + (25**7) + eps))
        SL = 1 + (0.015 * ((Lbar - 50) ** 2)) / torch.sqrt(20 + (Lbar - 50) ** 2 + eps)
        SC = 1 + 0.045 * Cbarp
        SH = 1 + 0.015 * Cbarp * T
        RT = -torch.sin(torch.deg2rad(d_ro)) * RC
        dE = torch.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH) + eps)
        return dE

    def forward(self, gen_srgb01: torch.Tensor, tgt_srgb01: torch.Tensor) -> torch.Tensor:
        assert K is not None and Kcolor is not None, "Install kornia to use DeltaE00Loss (pip install kornia)"
        Lab1 = Kcolor.rgb_to_lab(gen_srgb01.clamp(0, 1))
        Lab2 = Kcolor.rgb_to_lab(tgt_srgb01.clamp(0, 1))
        dE = self._ciede2000(Lab1, Lab2, self.eps)
        return dE.mean()


class SSIMLoss(nn.Module):
    """DSSIM based on Kornia's SSIMLoss. Inputs in [0,1]."""
    def __init__(self, window_size: int = 11, max_val: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        if Kloss is None:
            raise ImportError("Install kornia to use SSIMLoss (pip install kornia)")
        self.loss = Kloss.SSIMLoss(window_size=window_size, max_val=max_val, reduction=reduction)

    def forward(self, gen_srgb01: torch.Tensor, tgt_srgb01: torch.Tensor) -> torch.Tensor:
        return self.loss(gen_srgb01.clamp(0, 1), tgt_srgb01.clamp(0, 1))


class PhysicsConsistencyLoss(nn.Module):
    """|K*Bhat_raw - align(A_raw)|_1 with depthwise conv.

    K: [C,1,kh,kw] or [1,1,kh,kw]; expo_ratio scales A to B.
    """
    def __init__(self, K_kernel: torch.Tensor, device: str = 'cuda', clamp_align: bool = True):
        super().__init__()
        if K_kernel is None:
            raise ValueError("K_kernel must be provided for PhysicsConsistencyLoss")
        self.register_buffer('K', K_kernel.to(device))
        self.K: torch.Tensor
        kh, kw = self.K.shape[-2:]
        self.pad = nn.ReplicationPad2d((kw // 2, kw // 2, kh // 2, kh // 2))
        self.clamp_align = clamp_align

    def forward(self, Bhat_raw: torch.Tensor, A_raw: torch.Tensor, expo_ratio: torch.Tensor) -> torch.Tensor:
        if expo_ratio.dim() == 1:
            expo_ratio = expo_ratio.view(-1, 1, 1, 1)
        A_align = A_raw * expo_ratio
        if self.clamp_align:
            A_align = A_align.clamp(0.0, 1.0)
        x = self.pad(Bhat_raw)
        C = Bhat_raw.shape[1]
        k = self.K
        # If provided kernel is single shared kernel [1,1,kh,kw], broadcast to [C,1,kh,kw]
        if k.shape[0] == 1 and C > 1:
            k = k.expand(C, 1, k.shape[-2], k.shape[-1])
        # groups must match number of input channels for depthwise application
        groups = C if k.shape[0] == C else 1
        # If groups==1, expect k to have in_ch/groups = C. Expand if needed.
        if groups == 1 and k.shape[1] == 1 and C != 1:
            # Expand along in_channel dim for standard conv (not depthwise)
            k = k.expand(k.shape[0], C, k.shape[-2], k.shape[-1])
        Ahat = F.conv2d(x, k, bias=None, stride=1, padding=0, groups=groups)
        return F.l1_loss(Ahat, A_align)


def align_exposure_srgb(a_srgb: torch.Tensor, ratio: torch.Tensor | float) -> torch.Tensor:
    """Align sRGB short exposure to the long exposure scale via scalar ratio, then clamp to [0,1]."""
    if not torch.is_tensor(ratio):
        ratio = torch.tensor(ratio, dtype=a_srgb.dtype, device=a_srgb.device)
    if ratio.dim() == 0:
        ratio = ratio.view(1)
    if ratio.dim() == 1:
        ratio = ratio.view(-1, 1, 1, 1)
    return (a_srgb * ratio).clamp(0.0, 1.0)


class PhysicalConsistencyLossSRGB(nn.Module):
    """Output-side physical consistency on sRGB domain using a PSF module.

    L_phys = || PSF(Bhat_srgb) - Align(A_srgb; ratio) ||_1
    The PSF is a fixed module (register_buffer kernels) and is used ONLY in the loss path.
    """
    def __init__(self, psf_module: nn.Module):
        super().__init__()
        self.psf = psf_module
        self.l1 = nn.L1Loss()

    def forward(self, bhat_srgb: torch.Tensor, a_srgb: torch.Tensor, ratio: torch.Tensor | float) -> torch.Tensor:
        a_align = align_exposure_srgb(a_srgb, ratio)
        a_hat = self.psf(bhat_srgb)
        return self.l1(a_hat, a_align)


class HybridLossPlus(nn.Module):
    """Pluggable hybrid loss combining pixel, perceptual, color, structure, and physics consistency terms.

    set use_uncertainty=True to enable Kendall & Gal homoscedastic uncertainty weighting.
    """
    def __init__(
        self,
        device: str = 'cuda',
        w_l1_raw: float = 1.0,
        w_perc: float = 0.02,
        w_lpips: float = 0.0,
        w_deltaE: float = 0.02,
        w_ssim: float = 0.05,
        w_phys: float = 0.10,
        use_deltaE: bool = True,
        use_ssim: bool = True,
        use_lpips: bool = False,
        use_phys: bool = True,
        use_uncertainty: bool = False,
        physics_kernel: Optional[torch.Tensor] = None,
        physics_psf_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.l1_raw = nn.L1Loss().to(device)
        self.perc = PerceptualLoss(device=device)
        self.deltaE = DeltaE00Loss() if use_deltaE else None
        self.ssim = SSIMLoss() if use_ssim else None
        self.lpips = None
        if use_lpips:
            try:
                import lpips  # type: ignore
                self.lpips = lpips.LPIPS(net='vgg').to(device).eval()
                for p in self.lpips.parameters():
                    p.requires_grad = False
            except Exception:
                self.lpips = None
        # RAW-domain physics via kernels OR sRGB-domain physics via PSF module
        self.phys = PhysicsConsistencyLoss(physics_kernel, device=device) if use_phys and physics_kernel is not None else None
        self.phys_srgb = PhysicalConsistencyLossSRGB(physics_psf_module) if use_phys and physics_psf_module is not None else None

        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_sigma = nn.ParameterDict(
                {
                    'l1': nn.Parameter(torch.zeros(())),
                    'perc': nn.Parameter(torch.zeros(())),
                    'lpips': nn.Parameter(torch.zeros(())),
                    'de': nn.Parameter(torch.zeros(())),
                    'ssim': nn.Parameter(torch.zeros(())),
                    'phys': nn.Parameter(torch.zeros(())),
                }
            )
        else:
            self.w = dict(l1=w_l1_raw, perc=w_perc, lpips=w_lpips, de=w_deltaE, ssim=w_ssim, phys=w_phys)

    def _weighted(self, name: str, val: Optional[torch.Tensor]):
        if val is None:
            return 0.0, torch.tensor(0.0)
        if self.use_uncertainty:
            s = self.log_sigma[name]
            return (val * torch.exp(-2 * s) + s), val.detach()
        else:
            return self.w[name] * val, val.detach()

    def forward(
        self,
        *,
        Bhat_raw: torch.Tensor,
        B_raw: torch.Tensor,
        A_raw: torch.Tensor,
        expo_ratio: torch.Tensor,
        Bhat_srgb01: torch.Tensor,
        B_srgb01: torch.Tensor,
        A_srgb01: Optional[torch.Tensor] = None,
    ):
        logs: Dict[str, torch.Tensor] = {}
        L_total = 0.0

        L_l1 = self.l1_raw(Bhat_raw, B_raw)
        Lw, logs['L1_raw'] = self._weighted('l1', L_l1)
        L_total += Lw

        L_p = self.perc(Bhat_srgb01, B_srgb01)
        Lw, logs['Perc'] = self._weighted('perc', L_p)
        L_total += Lw

        if self.lpips is not None:
            L_lp = self.lpips(Bhat_srgb01, B_srgb01).mean()
            Lw, logs['LPIPS'] = self._weighted('lpips', L_lp)
            L_total += Lw

        if self.deltaE is not None:
            L_de = self.deltaE(Bhat_srgb01, B_srgb01)
            Lw, logs['DeltaE'] = self._weighted('de', L_de)
            L_total += Lw

        if self.ssim is not None:
            L_ss = self.ssim(Bhat_srgb01, B_srgb01)
            Lw, logs['SSIM'] = self._weighted('ssim', L_ss)
            L_total += Lw

        if self.phys is not None:
            L_ph = self.phys(Bhat_raw, A_raw, expo_ratio)
            Lw, logs['Phys'] = self._weighted('phys', L_ph)
            L_total += Lw
        elif self.phys_srgb is not None and A_srgb01 is not None:
            L_phs = self.phys_srgb(Bhat_srgb01, A_srgb01, expo_ratio)
            Lw, logs['Phys'] = self._weighted('phys', L_phs)
            L_total += Lw

        logs['Total'] = torch.as_tensor(L_total).detach()
        return L_total, logs
