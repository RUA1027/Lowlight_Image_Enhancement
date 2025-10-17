# Dependency: lpips (pip install lpips)
"""
通用 LPIPS 评测器（中立、可复用）。

一、指标定义与口径（用于文档与注释）
- LPIPS 定义：在某个感知网络（AlexNet/VGG/SqueezeNet）各层的通道归一化特征做差并加权求和：
  LPIPS(x, y) = Sum_l w_l * MSE( phi_l_hat(x), phi_l_hat(y) )。
  分数越小，感知上越相似。
- 合法输入：形状为 [N,3,H,W] 的 RGB；灰度允许复制到 3 通道后计算；其它通道数禁止。
  取值范围必须归一化到 [-1, 1]（官方要求）。若输入为 [0,1] 或 [0,255]，需在本模块内显式映射。
  尺寸：两图必须等尺寸。默认严格检查并报错；可通过参数开启等比 resize/中心裁剪等策略，并在结果中记录策略口径。
- 骨干与版本：允许 net ∈ { 'alex','vgg','squeeze' }，默认 'alex'；version 默认为 '0.1'。

强制声明：评测时一律 model.eval() 且 torch.no_grad()；默认使用 FP32，可选 AMP 仅作附加口径。

二、模块目标（广泛评测）
- 零耦合：不依赖任何具体模型或训练代码，仅依赖 torch 与 lpips。
- 统一口径：本模块内部统一处理范围/通道/尺寸，确保对照组与实验组在同一口径下可比较、可复现、可审计。
- 统计完备：同时返回逐图分数与汇总统计（mean/std/p50/p95），便于进一步统计分析。

说明：保留原有 LPIPSEvaluator 类以兼容既有代码；新增 LPIPSMetric 与 evaluate_pairs 以满足“通用评测版”需求。
"""

from __future__ import annotations

import contextlib
from typing import Literal, Tuple, List, Dict, Any, cast

import torch
import torch.nn.functional as F
import lpips


class LPIPSEvaluator:
    """Callable helper for computing LPIPS scores over batched image tensors.

    The class encapsulates the underlying LPIPS network so that the model weights are
    loaded a single time during initialization and reused for subsequent evaluations.
    """

    def __init__(
        self,
        net: Literal["alex", "vgg", "squeeze"] = "alex",
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize the LPIPS evaluator.

        Args:
            net: Backbone network used inside LPIPS. Common choices include ``"alex"``
                (default), ``"vgg"``, and ``"squeeze"``.
            device: Optional device identifier. If ``None`` (default), choose CUDA when
                available, otherwise CPU.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.loss_fn = lpips.LPIPS(net=net).to(self.device)
        self.loss_fn.eval()

    def __call__(self, img_true: torch.Tensor, img_pred: torch.Tensor) -> float:
        """Compute the average LPIPS score between two image batches.

        The inputs are expected to be in a standard image range (e.g., ``[0, 1]`` or
        ``[0, 255]``). This method will internally normalize them to ``[-1, 1]`` as
        required by the LPIPS implementation before evaluation.

        Args:
            img_true: Ground-truth image tensor with shape ``(N, C, H, W)`` or ``(C, H, W)``.
            img_pred: Predicted image tensor with the same shape as ``img_true``.

        Returns:
            Average LPIPS distance as a floating-point value (smaller values indicate
            higher perceptual similarity).

        Raises:
            ValueError: If the input tensors do not have matching shapes or if their
                dimensionality is unsupported.
        """
        if img_true.shape != img_pred.shape:
            raise ValueError(
                f"Input shapes must match exactly, got {img_true.shape=} and {img_pred.shape=}."
            )

        if img_true.ndim not in (3, 4):
            raise ValueError(
                f"Inputs must be 3D (C,H,W) or 4D (N,C,H,W) tensors, received ndim={img_true.ndim}."
            )

        if img_true.ndim == 3:
            img_true = img_true.unsqueeze(0)
            img_pred = img_pred.unsqueeze(0)

        img_true = img_true.to(self.device, dtype=torch.float32)
        img_pred = img_pred.to(self.device, dtype=torch.float32)

        # Map inputs from [0,255] or [0,1] to [-1,1] to match official requirements
        def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
            if x.numel() == 0:
                return x
            mx = float(x.max().item())
            mn = float(x.min().item())
            if mx > 1.5:  # treat as [0,255]
                x = x / 255.0
            if mn >= 0.0 and mx <= 1.0:
                x = x * 2.0 - 1.0
            return x

        img_true_norm = _to_minus1_1(img_true)
        img_pred_norm = _to_minus1_1(img_pred)

        with torch.no_grad():
            distances = self.loss_fn(img_pred_norm, img_true_norm)

        return float(distances.mean().item())


class LPIPSMetric:
    """
    通用 LPIPS 评测器（模型无关）。
    - 适配输入范围：支持 [0,255] / [0,1] / [-1,1]，统一映射到 [-1,1] 再计算；
    - 适配通道：支持灰度图自动三通道化，拒绝除 {1,3} 之外的通道；
    - 适配尺寸：默认严格一致，可选 resize / center_crop 策略；
    - 设备与精度：自动选择 GPU/CPU，支持 AMP（可选口径，不默认）；
    - 统计：返回逐图分数与汇总统计（mean/std/p50/p95）。
    """

    def __init__(
        self,
        net: str = "alex",  # {'alex','vgg','squeeze'}
        version: str = "0.1",
        device: str | torch.device | None = None,
        resize_policy: str | None = None,  # {None,'resize','center_crop'}
        resize_mode: str = "bilinear",
        keep_ratio: bool = True,
        use_amp: bool = False,  # 仅作为附加口径，不作为默认
        reduce: str = "mean",  # {'mean','none'}
        eps: float = 1e-12,
    ) -> None:
        self.net = net
        self.version = version
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.resize_policy = resize_policy
        self.resize_mode = resize_mode
        self.keep_ratio = keep_ratio
        self.use_amp = use_amp
        self.reduce = reduce
        self.eps = eps
        self._lpips = None  # lazy init

    # ---------------------------- internal helpers ---------------------------- #
    def _build_lpips(self):
        """Create and freeze the official LPIPS network with the given backbone/version."""
        loss_fn = lpips.LPIPS(net=self.net, version=self.version)
        loss_fn.eval().to(self.device)
        for p in loss_fn.parameters():
            p.requires_grad_(False)
        return loss_fn

    def _resize_or_crop_pair(
        self,
        img_true: torch.Tensor,
        img_pred: torch.Tensor,
        policy: str,
        mode: str = "bilinear",
        keep_ratio: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align sizes according to policy.

        - 'resize': resize prediction to ground-truth size using interpolation.
          If GT and Pred have different aspect ratios, resizing will warp; set
          `policy='center_crop'` if warping is undesired.
        - 'center_crop': crop both to the common centered (min(H), min(W)).
        """
        n, c, ht, wt = img_true.shape
        n2, c2, hp, wp = img_pred.shape
        if (n != n2) or (c != c2):
            # We assume paired batches; if not, raise to avoid silent mismatch
            raise ValueError(
                f"Batch/Channel mismatch for LPIPS: true {img_true.shape} vs pred {img_pred.shape}."
            )

        if policy == "resize":
            if (hp, wp) != (ht, wt):
                if keep_ratio:
                    # Scale pred keeping aspect ratio so that it fully covers GT, then center-crop to GT size
                    scale = max(ht / float(hp), wt / float(wp))
                    new_h = max(1, int(round(hp * scale)))
                    new_w = max(1, int(round(wp * scale)))
                    tmp = F.interpolate(
                        img_pred,
                        size=(new_h, new_w),
                        mode=mode,
                        align_corners=False if mode in {"bilinear", "bicubic"} else None,
                    )
                    # center-crop to (ht, wt)
                    top = max((new_h - ht) // 2, 0)
                    left = max((new_w - wt) // 2, 0)
                    img_pred = tmp[:, :, top:top + ht, left:left + wt]
                else:
                    # Direct warp to GT size (aspect ratio may change)
                    img_pred = F.interpolate(
                        img_pred,
                        size=(ht, wt),
                        mode=mode,
                        align_corners=False if mode in {"bilinear", "bicubic"} else None,
                    )
            return img_true, img_pred

        if policy == "center_crop":
            h = min(ht, hp)
            w = min(wt, wp)
            def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
                _, _, H, W = x.shape
                top = max((H - h) // 2, 0)
                left = max((W - w) // 2, 0)
                return x[:, :, top:top+h, left:left+w]
            return _center_crop(img_true, h, w), _center_crop(img_pred, h, w)

        raise ValueError(f"Unknown resize_policy: {policy}. Use None, 'resize', or 'center_crop'.")

    def _prepare_images(self, img_true: torch.Tensor, img_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 形状检查：接受 [N,C,H,W] 或 [C,H,W]，自动补 batch 维
        if img_true.ndim == 3:
            img_true = img_true.unsqueeze(0)
        if img_pred.ndim == 3:
            img_pred = img_pred.unsqueeze(0)

        if img_true.ndim != 4 or img_pred.ndim != 4:
            raise ValueError("LPIPS expects 4D tensors (N,C,H,W) or 3D that we can batch.")

        # 类型/设备
        dev = self.device
        img_true = img_true.to(dev, dtype=torch.float32)
        img_pred = img_pred.to(dev, dtype=torch.float32)

        # 通道适配：灰度复制到 3 通道；拒绝 C∉{1,3}
        c_true = img_true.shape[1]
        c_pred = img_pred.shape[1]
        if c_true != c_pred:
            raise ValueError(f"LPIPS requires same channels for both inputs. Got C_true={c_true}, C_pred={c_pred}.")
        c = c_true
        if c == 1:
            img_true = img_true.repeat(1, 3, 1, 1)
            img_pred = img_pred.repeat(1, 3, 1, 1)
        elif c != 3:
            raise ValueError(f"LPIPS expects RGB 3-channel (or 1-channel to replicate). Got C={c}.")

        # 尺寸策略
        if self.resize_policy is None:
            if img_true.shape[-2:] != img_pred.shape[-2:]:
                raise ValueError(
                    f"Image size mismatch {img_true.shape[-2:]} vs {img_pred.shape[-2:]}. Enable resize_policy to align."
                )
        else:
            img_true, img_pred = self._resize_or_crop_pair(
                img_true, img_pred, self.resize_policy, mode=self.resize_mode, keep_ratio=self.keep_ratio
            )

        # 归一化口径：把任意输入映射到 [-1,1]
        def _to_minus1_1(x: torch.Tensor) -> torch.Tensor:
            if x.numel() == 0:
                return x
            mx = float(x.max().item())
            mn = float(x.min().item())
            if mx > 1.5:  # treat as [0,255]
                x = x / 255.0
            if mn >= 0.0 and mx <= 1.0:
                x = x * 2.0 - 1.0
            return x

        return _to_minus1_1(img_true), _to_minus1_1(img_pred)

    # ------------------------------- public API ------------------------------- #
    @torch.no_grad()
    def __call__(self, img_true: torch.Tensor, img_pred: torch.Tensor) -> Dict[str, Any]:
        t, p = self._prepare_images(img_true, img_pred)
        lpips_model = self._lpips or self._build_lpips()
        self._lpips = lpips_model

        if self.use_amp and (t.is_cuda or p.is_cuda):
            autocast_ctx = torch.cuda.amp.autocast
        else:
            autocast_ctx = contextlib.nullcontext

        with autocast_ctx():
            # lpips 返回 [N,1,1,1] 或 [N]，统一为 [N]
            # Pylance/Pyright 可能会错误地将 LPIPS.forward 的返回类型推断为元组，
            # 实际上它返回一个张量。我们使用类型断言来纠正这个问题。
            d_raw = lpips_model.forward(p, t)
            d = cast(torch.Tensor, d_raw).view(-1)

        stats: Dict[str, Any] = {
            "per_image": d.detach().cpu().tolist() if self.reduce == "none" else None,
            "mean": float(d.mean().item()),
            "std": float(d.std(unbiased=False).item()) if d.numel() > 1 else 0.0,
            "p50": float(d.median().item()),
            "p95": float(d.kthvalue(max(1, int(0.95 * d.numel())))[0].item()) if d.numel() > 1 else float(d.item()),
            "net": self.net,
            "version": self.version,
            "resize_policy": self.resize_policy,
            "amp": self.use_amp,
            "count": int(d.numel()),
        }
        return stats


def evaluate_pairs(pairs: List[Tuple[torch.Tensor, torch.Tensor]], **kwargs) -> Dict[str, Any]:
    """Evaluate a list of (ground_truth, prediction) pairs under one LPIPS configuration."""
    metric = LPIPSMetric(**kwargs)
    all_scores: List[float] = []
    for gt, pred in pairs:
        s = metric(gt, pred)
        all_scores.append(s["mean"])  # 每张图像（或其内部 batch）的均值
    if len(all_scores) == 0:
        return {
            "per_image": [],
            "mean": float("nan"),
            "std": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "count": 0,
            "net": kwargs.get("net", "alex"),
            "version": kwargs.get("version", "0.1"),
        }
    x = torch.tensor(all_scores, dtype=torch.float32)
    return {
        "per_image": [float(v) for v in x.tolist()],
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "p50": float(x.median().item()),
        "p95": float(x.kthvalue(max(1, int(0.95 * x.numel())))[0].item()) if x.numel() > 1 else float(x.item()),
        "count": int(x.numel()),
        "net": kwargs.get("net", "alex"),
        "version": kwargs.get("version", "0.1"),
    }


if __name__ == "__main__":
    # 示例：伪造两张 [0,255] 灰度图，展示“灰度→RGB复制 + 归一化”的自动流程
    torch.manual_seed(0)

    gt = torch.randint(0, 256, (4, 1, 256, 256), dtype=torch.uint8)
    pred = gt.clone().float() + torch.randn(4, 1, 256, 256) * 5.0
    pred = pred.clamp(0, 255).to(torch.uint8)

    gt = gt.float()
    pred = pred.float()

    metric = LPIPSMetric(net="alex", version="0.1", resize_policy=None, use_amp=False)
    stats = metric(gt, pred)
    print(
        "[LPIPS] mean={mean:.6f}, std={std:.6f}, p50={p50:.6f}, p95={p95:.6f}, net={net}, ver={version}".format(
            **stats
        )
    )

    # 兼容旧接口：LPIPSEvaluator（输入可为 [0,1] 或 [0,255]，内部统一到 [-1,1]）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = LPIPSEvaluator(net="alex", device=device)
    true_images_batch = torch.rand(4, 3, 64, 64)
    pred_images_batch_noisy = torch.clamp(true_images_batch + 0.1 * torch.randn(4, 3, 64, 64), 0.0, 1.0)
    lpips_score = evaluator(true_images_batch, pred_images_batch_noisy)
    print(f"[Compat] LPIPSEvaluator mean: {lpips_score:.6f} ({device})")
