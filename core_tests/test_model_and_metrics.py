"""
综合测试：神经网络模型与评价指标功能验证

- 验证 NewBP-Net 架构能否正常前向推理
- 验证损失函数能否正常计算
- 验证 PSNR、SSIM、LPIPS 等指标能否准确评价

运行方式：
    pytest core_tests/test_model_and_metrics.py
"""
import torch
import pytest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "NAFNet_base")))

from NewBP_model.newbp_net_arch import create_newbp_net, create_crosstalk_psf
from NewBP_model.losses import HybridLoss, HybridLossPlus
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.lpips_metric import LPIPSEvaluator

# 1. 神经网络模型前向推理
@pytest.mark.parametrize("batch_size,channels,height,width", [(2, 3, 128, 128)])
def test_newbp_net_forward(batch_size, channels, height, width):
    model = create_newbp_net(in_channels=channels)
    x = torch.rand(batch_size, channels, height, width)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == batch_size and y.shape[1] == channels

# 2. 损失函数功能
@pytest.mark.parametrize("device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])
def test_hybrid_loss(device):
    batch, c, h, w = 2, 3, 128, 128
    gen = torch.rand(batch, c, h, w, device=device)
    tgt = torch.rand(batch, c, h, w, device=device)
    loss_fn = HybridLoss(device=device)
    total, l1, perc = loss_fn(gen, tgt)
    assert total.item() > 0 and l1.item() > 0 and perc.item() > 0

# 3. 综合损失（HybridLossPlus）
def test_hybrid_loss_plus():
    batch, c, h, w = 2, 3, 128, 128
    Bhat_raw = torch.rand(batch, c, h, w)
    B_raw = torch.rand(batch, c, h, w)
    A_raw = torch.rand(batch, c, h, w)
    expo_ratio = torch.ones(batch)
    Bhat_srgb01 = torch.rand(batch, c, h, w)
    B_srgb01 = torch.rand(batch, c, h, w)
    psf = create_crosstalk_psf(psf_mode='mono', kernel_spec='P2')
    loss_fn = HybridLossPlus(device='cpu', physics_psf_module=psf)
    total, logs = loss_fn(
        Bhat_raw=Bhat_raw, B_raw=B_raw, A_raw=A_raw, expo_ratio=expo_ratio,
        Bhat_srgb01=Bhat_srgb01, B_srgb01=B_srgb01, A_srgb01=B_srgb01
    )
    assert total.item() > 0 and isinstance(logs, dict)

# 4. PSNR 指标
@pytest.mark.parametrize("data_range", [1.0, 255.0])
def test_psnr(data_range):
    img_true = torch.rand(3, 64, 64) * data_range
    img_pred = img_true + torch.randn(3, 64, 64) * 0.05
    img_pred = img_pred.clamp(0, data_range)
    psnr_val = calculate_psnr(img_true, img_pred, data_range)
    assert isinstance(psnr_val, float)

# 5. SSIM 指标
@pytest.mark.parametrize("data_range", [1.0, 255.0])
def test_ssim(data_range):
    img_true = torch.rand(3, 64, 64) * data_range
    img_pred = img_true + torch.randn(3, 64, 64) * 0.05
    img_pred = img_pred.clamp(0, data_range)
    ssim_val = calculate_ssim(img_true, img_pred, data_range)
    assert isinstance(ssim_val, float)

# 6. LPIPS 指标
@pytest.mark.parametrize("net", ["alex", "vgg"])
def test_lpips(net):
    img_true = torch.rand(1, 3, 64, 64)
    img_pred = img_true + torch.randn(1, 3, 64, 64) * 0.05
    img_pred = img_pred.clamp(0, 1)
    evaluator = LPIPSEvaluator(net=net)
    score = evaluator(img_true, img_pred)
    assert isinstance(score, float)

if __name__ == "__main__":
    pytest.main([__file__])
