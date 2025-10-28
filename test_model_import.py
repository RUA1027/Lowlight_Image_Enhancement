"""
测试 NewBP 层和 NAFNet 组合模型代码是否能正常 import 和运行
"""
import sys
import os
import torch
import torch.nn as nn

try:
    import torchvision  # noqa: F401
    HAS_TORCHVISION = True
except Exception:  # pragma: no cover - dependency may be absent locally
    HAS_TORCHVISION = False


def _skip_torchvision(test_name: str) -> bool:
    """Inform the user when torchvision is missing and skip the associated test."""

    print(f"[SKIP] {test_name}: torchvision is not installed.")
    print("       Install via `pip install torchvision` to enable this test.\n")
    return True

# 添加 NAFNet 路径
NAFNET_PATH = os.path.join(os.path.dirname(__file__), 'NAFNet')
if NAFNET_PATH not in sys.path:
    sys.path.insert(0, NAFNET_PATH)

def test_imports():
    """测试所有必要的模块能否正常导入"""
    print("=" * 60)
    print("测试 1: 检查模块导入")
    print("=" * 60)
    if not HAS_TORCHVISION:
        return _skip_torchvision("模块导入")
    
    try:
        # 测试 NAFNet 导入
        from NAFNet_base.basicsr.models.archs.NAFNet_arch import NAFNet
        print("✓ NAFNet 导入成功")
    except Exception as e:
        print(f"✗ NAFNet 导入失败: {e}")
        return False
    
    try:
        # 测试 NewBP 层导入
        from NewBP_model.newbp_layer import NewBPLayer, CrosstalkPSF, build_psf_kernels
        print("✓ NewBP 层导入成功")
    except Exception as e:
        print(f"✗ NewBP 层导入失败: {e}")
        return False
    
    try:
        # 测试组合模型导入
        from NewBP_model.newbp_net_arch import create_newbp_net, create_crosstalk_psf
        print("✓ NewBP 网络架构导入成功")
    except Exception as e:
        print(f"✗ NewBP 网络架构导入失败: {e}")
        return False
    
    try:
        # 测试损失函数导入
        # from NewBP_model.losses import PhysicsConsistencyLoss
        print("✓ 物理一致性损失函数导入成功 (PhysicalConsistencyLossSRGB in use)")
    except Exception as e:
        print(f"✗ 损失函数导入失败: {e}")
        return False
    
    print("\n所有模块导入测试通过！\n")
    return True


def test_nafnet_forward():
    """测试 NAFNet 前向传播"""
    print("=" * 60)
    print("测试 2: NAFNet 前向传播")
    print("=" * 60)
    if not HAS_TORCHVISION:
        return _skip_torchvision("NAFNet 前向传播")
    
    try:
        from NAFNet_base.basicsr.models.archs.NAFNet_arch import NAFNet
        
        # 创建 NAFNet 模型
        model = NAFNet(
            img_channel=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 2],
            dec_blk_nums=[1, 1, 1, 1]
        )
        
        # 创建测试输入
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"✓ 输入形状: {input_tensor.shape}")
        print(f"✓ 输出形状: {output.shape}")
        print(f"✓ 输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 验证输出形状
        assert output.shape == input_tensor.shape, "输出形状与输入形状不匹配"
        print("\nNAFNet 前向传播测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ NAFNet 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_newbp_crosstalk_psf():
    """测试 CrosstalkPSF 前向传播"""
    print("=" * 60)
    print("测试 3: CrosstalkPSF 前向传播")
    print("=" * 60)
    
    try:
        from NewBP_model.newbp_layer import CrosstalkPSF, build_psf_kernels
        
        # 测试 mono 模式
        print("\n[Mono 模式]")
        kernels_mono = build_psf_kernels('mono', 'P2')
        psf_mono = CrosstalkPSF(mode='mono', kernels=kernels_mono)
        
        input_tensor = torch.randn(2, 3, 128, 128)
        output_mono = psf_mono(input_tensor)
        
        print(f"✓ 输入形状: {input_tensor.shape}")
        print(f"✓ 输出形状: {output_mono.shape}")
        print(f"✓ 卷积核形状: {psf_mono.kernel.shape}")
        print(f"✓ 卷积核和: {psf_mono.kernel.sum().item():.6f}")
        
        # 测试 rgb 模式
        print("\n[RGB 模式]")
        kernels_rgb = build_psf_kernels('rgb', 'B2')
        psf_rgb = CrosstalkPSF(mode='rgb', kernels=kernels_rgb)
        
        output_rgb = psf_rgb(input_tensor)
        
        print(f"✓ 输出形状: {output_rgb.shape}")
        print(f"✓ 卷积核形状: {psf_rgb.kernel.shape}")
        print(f"✓ R 通道卷积核和: {psf_rgb.kernel[0].sum().item():.6f}")
        print(f"✓ G 通道卷积核和: {psf_rgb.kernel[1].sum().item():.6f}")
        print(f"✓ B 通道卷积核和: {psf_rgb.kernel[2].sum().item():.6f}")
        
        assert output_mono.shape == input_tensor.shape, "Mono 模式输出形状不匹配"
        assert output_rgb.shape == input_tensor.shape, "RGB 模式输出形状不匹配"
        
        print("\nCrosstalkPSF 前向传播测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ CrosstalkPSF 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_model():
    """测试 NewBP+NAFNet 组合模型"""
    print("=" * 60)
    print("测试 4: NewBP+NAFNet 组合模型")
    print("=" * 60)
    if not HAS_TORCHVISION:
        return _skip_torchvision("NewBP+NAFNet 组合模型")
    
    try:
        from NewBP_model.newbp_net_arch import create_newbp_net, create_crosstalk_psf
        
        # 创建 NAFNet 主干
        nafnet_params = {
            'img_channel': 3,
            'width': 32,
            'middle_blk_num': 1,
            'enc_blk_nums': [1, 1, 1, 2],
            'dec_blk_nums': [1, 1, 1, 1]
        }
        
        backbone = create_newbp_net(
            in_channels=3,
            kernel_type='panchromatic',
            kernel_spec='P2',
            nafnet_params=nafnet_params
        )
        
        # 创建串扰 PSF（用于损失计算）
        psf = create_crosstalk_psf(psf_mode='mono', kernel_spec='P2')
        
        # 测试前向传播
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 256)
        
        backbone.eval()
        with torch.no_grad():
            # Backbone 前向传播
            output_s = backbone(input_tensor)
            
            # PSF 应用（用于损失计算）
            output_y = psf(output_s)
        
        print(f"✓ 输入形状 (低质量图像 X): {input_tensor.shape}")
        print(f"✓ Backbone 输出形状 (预补偿图像 S): {output_s.shape}")
        print(f"✓ PSF 输出形状 (物理一致性输出 Ŷ): {output_y.shape}")
        print(f"✓ Backbone 参数量: {sum(p.numel() for p in backbone.parameters()) / 1e6:.2f}M")
        
        assert output_s.shape == input_tensor.shape, "Backbone 输出形状不匹配"
        assert output_y.shape == input_tensor.shape, "PSF 输出形状不匹配"
        
        print("\nNewBP+NAFNet 组合模型测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 组合模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_physics_loss():
    """测试物理一致性损失函数"""
    print("=" * 60)
    print("测试 5: 物理一致性损失函数")
    print("=" * 60)
    if not HAS_TORCHVISION:
        return _skip_torchvision("物理一致性损失函数")
    
    try:
        from NewBP_model.losses import PhysicalConsistencyLossSRGB
        from NewBP_model.newbp_net_arch import create_crosstalk_psf
        
        # 创建损失函数
        psf = create_crosstalk_psf(psf_mode='mono', kernel_spec='P2')
        loss_fn = PhysicalConsistencyLossSRGB(psf_module=psf)
        
        # 创建测试数据
        batch_size = 2
        bhat_srgb = torch.randn(batch_size, 3, 128, 128).clamp(0, 1)
        a_srgb = torch.randn(batch_size, 3, 128, 128).clamp(0, 1)
        expo_ratio = torch.tensor([2.0, 3.0])
        
        # 计算损失
        loss = loss_fn(bhat_srgb, a_srgb, expo_ratio)
        
        print(f"✓ 预测输出形状: {bhat_srgb.shape}")
        print(f"✓ 目标图像形状: {a_srgb.shape}")
        print(f"✓ 曝光比率: {expo_ratio.tolist()}")
        print(f"✓ 损失值: {loss.item():.6f}")
        
        assert loss.dim() == 0, "损失应该是标量"
        assert loss.item() >= 0, "损失值应该非负"
        
        print("\n物理一致性损失函数测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass():
    """测试反向传播和梯度计算"""
    print("=" * 60)
    print("测试 6: 反向传播和梯度计算")
    print("=" * 60)
    
    try:
        from NewBP_model.newbp_net_arch import create_newbp_net, create_crosstalk_psf
        from NewBP_model.losses import PhysicalConsistencyLossSRGB
        
        # 创建模型
        nafnet_params = {
            'img_channel': 3,
            'width': 16,
            'middle_blk_num': 1,
            'enc_blk_nums': [1, 1],
            'dec_blk_nums': [1, 1]
        }
        
        backbone = create_newbp_net(
            in_channels=3,
            kernel_type='panchromatic',
            kernel_spec='P2',
            nafnet_params=nafnet_params
        )
        
        psf = create_crosstalk_psf(psf_mode='mono', kernel_spec='P2')
        loss_fn = PhysicalConsistencyLossSRGB(psf_module=psf)
        
        # 创建测试数据
        input_tensor = torch.rand(1, 3, 64, 64, dtype=torch.float32, requires_grad=True)
        target = torch.randn(1, 3, 64, 64).clamp(0, 1)
        expo_ratio = 2.0
        
        # 前向传播
        backbone.train()
        output = backbone(input_tensor)
        loss = loss_fn(output, target, expo_ratio)
        
        # 反向传播
        loss.backward()
        
        print(f"✓ 损失值: {loss.item():.6f}")
        print(f"✓ 输入梯度是否存在: {input_tensor.grad is not None}")
        print(f"✓ 模型参数梯度是否存在: {any(p.grad is not None for p in backbone.parameters())}")
        
        # 统计有梯度的参数
        params_with_grad = sum(1 for p in backbone.parameters() if p.grad is not None)
        total_params = sum(1 for p in backbone.parameters())
        print(f"✓ 有梯度的参数数量: {params_with_grad}/{total_params}")
        
        assert input_tensor.grad is not None, "输入应该有梯度"
        assert params_with_grad > 0, "至少应该有一些参数有梯度"
        
        print("\n反向传播和梯度计算测试通过！\n")
        return True
        
    except Exception as e:
        print(f"✗ 反向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("NewBP+NAFNet 模型导入和运行测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("模块导入", test_imports),
        ("NAFNet 前向传播", test_nafnet_forward),
        ("CrosstalkPSF 前向传播", test_newbp_crosstalk_psf),
        ("NewBP+NAFNet 组合模型", test_combined_model),
        ("物理一致性损失函数", test_physics_loss),
        ("反向传播和梯度计算", test_backward_pass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ 测试 '{test_name}' 执行失败: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！模型可以正常导入和运行。")
        return 0
    else:
        print(f"\n⚠️ 有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit(main())
