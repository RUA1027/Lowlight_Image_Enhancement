# 低光照去噪与跨像素串扰建模：一个面向物理一致性的学习框架


## ✨ 我们做了什么（贡献与特性）

1. **物理一致性的训练范式（“Scenario B”）**

   * **不改变网络前向**：输入图像不与 PSF 再卷积，避免“**双重串扰**”。
   * **仅在损失分支引入物理算子**：以**线性、平移不变（LSI）**的退化模型构造约束项，使预测结果与观测（短曝光）在物理上**闭环一致**。
   * 对极弱光/高噪声场景，物理一致性项与感知/像素损失并行，形成**混合损失**。

2. **可控的串扰核（PSF）族**

   * 提供**单色（mono）**与**分色（RGB）**两种模式：`mono` 用同一核广播至三通道，`rgb` 采用逐通道深度卷积。
   * 内置 3×3 的基准核（如 `P2`、`B2`）与**“加压”核族**（R＞G＞B 的波长依赖），便于开展**压力–收益**实验，直观显现算法优势。

3. **从“像素–结构–感知–色彩–边缘”的多维评价体系**

   * 线性域：PSNR / SSIM（严格线性口径）。
   * 显示域：LPIPS（感知相似度），ΔE_00（CIEDE2000 色差），RGB-SSIM；支持**边缘区域**ΔE_00 统计。
   * 通道级：逐通道 PSNR 与 CPSNR。
   * 评测口径（域/范围/单位）在接口与日志中**显式记录**，方便论文与复现实验对齐。
   * LPIPS 与 ΔE_00 的实现均参照权威资料与主流工具链口径。([GitHub][2])

4. **工程化的效率评测**

   * FLOPs 统计提供**多种口径**：`fvcore_fma1`（FMA 记 1 FLOP ≈ MACs）与 `flops_2xmac`（1 MAC=2 FLOPs）。
   * 推理时延采用 CUDA events 严格计时，包含**预热**与**同步**，返回毫秒/张。([detectron2.readthedocs.io][3])

---

## 🧠 方法概述

### 1) 成像与串扰的物理模型（LSI 退化）

低光照相机管线可近似为线性、平移不变的卷积退化：
[
A \approx \mathrm{clip}\big(\rho,(K * B)\big) + n,
]
其中 (B) 为**长曝光/理想场景**，(A) 为**短曝光观测**，(\rho) 为曝光比（可广播到批/通道/空间），(K) 为跨像素串扰 PSF，(n) 为噪声项。训练时我们**不改变**网络前向流，只在损失里把 (\hat{B}) 经 (K) 与 (\rho) 投影回观测域，与 (A) 对齐形成**物理一致性**项。

> 说明：数据若来自 RAW 域（如 SID 数据集），(\rho) 直接来自曝光比；若在 sRGB 域，我们仍可作为**辅助一致性**项使用并与显示域度量（LPIPS/ΔE_00）结合。SID 数据集背景可参考原论文与主页。([CVF开放获取][4])

### 2) 主干网络与“Scenario B”

* 主干以 **NAFNet** 为参考（无激活/简洁高效），适配去噪/去串扰等复原任务；我们**不**在输入端卷 PSF，避免人为加重串扰。([欧盟计算机视觉协会][1])
* 训练时：前向 `A → fθ → \hat{B}`；损失端 `\hat{B} --(K, ρ)--> 观测域` 与 `A` 比较，叠加像素/感知/结构/色彩损失。

### 3) 可控 PSF 与 NewBP 等价性

* **PSF 模式**：

  * `mono`：核形状 `[1,1,kh,kw]`，对 RGB 广播；
  * `rgb`：核形状 `[3,1,kh,kw]`，以 `groups=3` 深度卷积实现通道内串扰；
  * 支持 `P2/B2` 等基准核与“**加压核族**”（R＞G＞B），方便开展敏感性与稳健性分析。
* **反传等价**：若把 PSF 作为固定线性算子，其梯度传递可由 `conv2d/conv_transpose2d` 的对偶关系**无损实现**，确保数值稳定与可解释。

### 4) 混合损失（Hybrid Loss）

* **像素/结构**：`L1/L2 + SSIM`（线性或 sRGB 口径）。
* **感知**：`Perceptual (VGG19)` 与 `LPIPS`。LPIPS 统一把输入归一到 `[-1,1]`，与官方实现口径一致。([GitHub][2])
* **色彩**：`ΔE00`（CIEDE2000），基于 `RGB→Lab(D65/2°)` 的严格变换。([hajim.rochester.edu][5])
* **物理一致性**：`\|\mathcal{P}(\hat{B};K,\rho)-A\|`（Charbonnier/MAE 等变体），其中 (\mathcal{P}) 为物理投影算子。
* 所有项的**域/范围（data_range）**与**度量口径**在日志中显式记录，避免跨方法不可比。

---

## 📦 代码结构

```
.
├── newbp_net_arch.py          # 模型装配与“Scenario B”策略（前向不卷 PSF）
├── newbp_layer.py             # NewBP Function，PSF 模块与核构造/归一化
├── phys_consistency.py        # 物理一致性项（线性/显示域）、曝光比广播规则
├── losses.py                  # 混合损失：L1/L2/SSIM/Perceptual/LPIPS/ΔE00 等
├── ssim.py / psnr.py          # sRGB/线性口径下的结构/信噪度量
├── lpips_metric.py            # LPIPS 评测与批量统计（尺寸对齐策略+元数据）
├── color_error.py             # ΔE00 地图/统计与边缘区域 ΔE00
├── channelwise.py             # 通道级 PSNR、CPSNR、RGB-SSIM
├── linear.py                  # 线性域 PSNR/SSIM（严格口径）
├── flops_utils.py             # FLOPs 统计（fvcore_fma1 / flops_2xmac 等口径）
├── inference_time.py          # CUDA events 推理时延（预热+同步）
├── perceptual.py              # VGG 特征感知损失/评测封装
├── parameter_utils.py         # 参数与核管理的小工具
│
├── 关于混合损失函数的说明.md
├── 物理原理退化过程理解.md
├── 数学原理阐释与项目底层.md
├── 对于串扰的具体分析.md
├── 串扰卷积核推导.md
├── 串扰核的“数值微调”方案（让算法优势更可见）.md
├── 实验设计.md
└── 评价指标.md
```

---

## 🚀 快速开始

### 1) 环境

* Python ≥ 3.8，PyTorch ≥ 1.10，CUDA 可选（建议）。
* 依赖（根据需要取用）：

  ```bash
  pip install -U torch torchvision kornia lpips
  pip install -U fvcore
  ```

  参考：NAFNet 官方实现、LPIPS 官方仓库、Kornia 文档、fvcore 文档。([GitHub][6])

### 2) 最小训练脚手架（示例）

```python
import torch
from newbp_net_arch import create_newbp_net, create_crosstalk_psf
from phys_consistency import phys_cons_srgb  # 或 phys_cons_raw
from losses import HybridLoss

# 1) 模型与 PSF（仅用于损失端）
net = create_newbp_net(in_channels=3, nafnet_params={"img_channel": 3})
psf = create_crosstalk_psf(psf_mode='rgb', kernel_spec='B2')  # 或 psf_mode='mono', 'P2'

# 2) 数据占位（sRGB [0,1]）
A_srgb = torch.rand(2,3,256,256).cuda()   # 短曝光观测
B_gt   = torch.rand(2,3,256,256).cuda()   # 长曝光参考
expo_ratio = 1.0

# 3) 混合损失（像素/感知）+ 物理一致性
hyb = HybridLoss(lambda_l1=1.0, lambda_perceptual=0.02, device='cuda')
B_hat = net(A_srgb)
L_img, L1, L_perc = hyb(B_hat, B_gt)
L_phys = phys_cons_srgb(B_hat, A_srgb, psf.kernel, expo_ratio=expo_ratio)

loss = L_img + 0.05 * L_phys
loss.backward()
```

> 若在 RAW/线性域训练，建议改用 `phys_cons_raw` 与线性域的 PSNR/SSIM；若使用类似 SID 的数据，需要正确设置 `expo_ratio`。([CVF开放获取][4])

### 3) 评测与报告（示例）

```python
# 线性域指标
from linear import psnr_linear, ssim_linear
psnr = psnr_linear(B_hat_raw, B_gt_raw, data_range=4095.0)
ssim = ssim_linear(B_hat_raw, B_gt_raw, data_range=1.0)

# 感知/色彩（显示域）
from lpips_metric import LPIPSMetric
from color_error import deltaE2000_summary
lpips_stats = LPIPSMetric(net="alex")(B_gt_srgb, B_hat_srgb)      # 自动归一到[-1,1]
de_stats    = deltaE2000_summary(B_gt_srgb, B_hat_srgb)           # RGB[0,1]→Lab(D65/2°)

# 通道级
from channelwise import cpsnr
cpsnr_value = cpsnr(B_hat_srgb, B_gt_srgb, data_range=1.0)
```

### 4) FLOPs 与推理时延（示例）

```python
# FLOPs（声明口径）
from flops_utils import FLOPsCounter
counter = FLOPsCounter(convention="fvcore_fma1") # FMA=1 FLOP(≈MAC)
report  = counter(net, (torch.randn(1,3,256,256),))  # GFMA / GMacs
# 或 convention="flops_2xmac"（1 MAC=2 FLOPs）
```

> fvcore 对 FLOP 的定义是“**最佳估计**”，且以 **FMA=1 FLOP** 为口径；不同工具会把 **FLOPs≈2×MACs**。请在论文/报告中**明确标注**口径。([detectron2.readthedocs.io][3])

```python
# 推理时延（CUDA events，含预热与同步）
from inference_time import measure_inference_time
x = torch.randn(1,3,256,256).cuda(); net.cuda().eval()
ms_per_image = measure_inference_time(net, x, num_warmup=20, num_runs=100)
```

> 计时需使用 `torch.cuda.Event(enable_timing=True)` 并在读数前后显式同步，避免异步导致的误差。([PyTorch Docs][7])

---

## 🔬 实验与复现要点

* **域/数据范围（data_range）**

  * 线性域（RAW）：PSNR/SSIM 使用 `psnr_linear/ssim_linear`，`data_range` 与 bit-depth 一致（如 1.0 / 255.0 / 4095.0）。
  * 显示域（sRGB）：LPIPS/ΔE_00/RGB-SSIM；ΔE_00 依赖 **RGB[0,1]→Lab(D65/2°)** 的严格转换。([kornia.readthedocs.io][8])
* **PSF 与模式**：`mono`（广播）与 `rgb`（通道内深度卷积）；核需**能量归一**。
* **曝光比（expo_ratio）**：支持标量/批/通道/空间维的广播；与数据源（如 SID）的曝光信息匹配。([CVF开放获取][4])
* **LPIPS 口径**：输入会统一到 `[-1,1]`，若尺寸不一致可选择 `resize/center_crop` 并在报告中**注明**。([GitHub][2])
* **边缘色差**：`edge_deltaE2000` 以 Sobel 提取高梯度区域，报告 `mean/p95`，更敏感地反映**边缘晕染/串扰**。
* **效率口径**：报告中同时给出 `fvcore_fma1` 和 `flops_2xmac` 以便与不同论文横向可比。([detectron2.readthedocs.io][3])

---

## 🗂️ 数据（可选参考）

* 若使用 SID（See-in-the-Dark）风格的低照度 RAW 数据：包含 5094 张短曝光 RAW 与对应的长曝光参考，并提供曝光比信息，可直接用于我们的物理一致性项。([CVF开放获取][4])

---

## 📈 典型实验（建议）

* **核压力曲线**：使用 `P2/B2` 与其“加压”变体（R＞G＞B），绘制在不同 PSF 压力下的 **PSNR/SSIM/LPIPS/ΔE_00** 曲线，分析“结构保持–色彩保持–物理闭环”的三元权衡。
* **域一致性**：同时报告线性域与显示域的指标，验证“物理一致性”项对真实可复现性的贡献。
* **效率–精度前沿**：NAFNet 尺寸/深度扫描，对比 FLOPs/时延与质量指标的帕累托前沿。([欧盟计算机视觉协会][1])

---

## 📎 引用与致谢（外部资料）

* **NAFNet（ECCV 2022）**：Nonlinear Activation Free Network，简洁高效的图像复原基线。([欧盟计算机视觉协会][1])
* **SSIM（2004）**：结构相似性指数的经典定义与推荐常数。([cns.nyu.edu][9])
* **LPIPS（2018）**：感知相似度指标与官方实现。([GitHub][2])
* **CIEDE2000（ΔE_00）**：实现细节与数学观察（Sharma 等）。([hajim.rochester.edu][5])
* **Kornia：RGB→Lab（D65/2°，输入范围 [0,1]）**。([kornia.readthedocs.io][8])
* **fvcore FlopCountAnalysis：FMA=1 FLOP、FLOP 非严格定义**。([detectron2.readthedocs.io][3])
* **PyTorch CUDA events：精确计时与同步**。([PyTorch Docs][7])
* **SID 数据集（CVPR 2018）**：极低照度 RAW 数据与曝光比定义。([CVF开放获取][4])

---

## 🪪 许可证

请参见仓库中的 `LICENSE` 文件（如尚未添加，请根据需求选择 MIT/BSD-3-Clause/Apache-2.0 等开源协议后更新此处）。

---

## 🤝 贡献

欢迎提交 Issue / PR，建议在 PR 中附上：

* 实验配置（域/范围/核/口径）；
* 质量指标（含 `p95` 等长尾统计）与效率口径；
* 可复现的最小代码片段与日志。

---

> 若你希望，我可以基于本 README 生成一份**模板化报告脚本**（CSV/Markdown 导出 + 指标口径与 FLOPs/时延注脚），并附带**核压力实验**的可复现命令集，直接产出论文风格的表格与曲线。

[1]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136670017.pdf?utm_source=chatgpt.com "Simple Baselines for Image Restoration"
[2]: https://github.com/richzhang/PerceptualSimilarity?utm_source=chatgpt.com "richzhang/PerceptualSimilarity: LPIPS metric. pip install lpips"
[3]: https://detectron2.readthedocs.io/en/stable/_modules/fvcore/nn/flop_count.html?utm_source=chatgpt.com "fvcore.nn.flop_count - detectron2's documentation!"
[4]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf?utm_source=chatgpt.com "Learning to See in the Dark"
[5]: https://hajim.rochester.edu/ece/sites/gsharma/papers/CIEDE2000CRNAFeb05.pdf?utm_source=chatgpt.com "The CIEDE2000 color-difference formula: Implementation ..."
[6]: https://github.com/megvii-research/NAFNet?utm_source=chatgpt.com "megvii-research/NAFNet: The state-of-the-art image ..."
[7]: https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html?utm_source=chatgpt.com "Event - torch.cuda"
[8]: https://kornia.readthedocs.io/en/latest/color.html?utm_source=chatgpt.com "kornia.color - Read the Docs"
[9]: https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf?utm_source=chatgpt.com "Image Quality Assessment: From Error Visibility to Structural ..."
