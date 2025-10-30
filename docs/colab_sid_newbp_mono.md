# Colab 笔记本：NewBP + NAFNet（Mono-PSF 实验组）

> 为避免一次运行时间过长，本笔记本只执行 **Mono-PSF** 实验（NewBP-A）。其他模型请使用相应笔记本。

---

## 第 0 步：挂载 Drive 与设定目录（首次运行必做）

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：克隆仓库（首次运行必做）

```bash
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/megvii-research/NAFNet.git
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```

---

## 第 2 步：安装依赖（首次运行必做）

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless
```

---

## 第 3 步：设置 Python 路径

```python
import sys
paths = [
    "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
]
for p in paths:
    if p not in sys.path:
        sys.path.append(p)
```

---

## 第 4 步：数据准备（若已完成，可跳过）

确保已执行：

1. RAW → PNG：`/content/drive/MyDrive/Lowlight/SID_png/Sony`
2. manifest：`/content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json`
3. LMDB：`/content/drive/MyDrive/Lowlight/SID_lmdb`

命令详见 `docs/colab_sid_unet.md` 第 4 步。

---

## 第 5 步：检查 Mono-PSF 配置

打开 `configs/colab/sid_newbp_mono.yml`，重点确认：

- `network_g.type: NewBPNAFNet`（使用我们封装的 NewBP-NAPNet 架构）。
- `train.hybrid_opt.physics.mode: mono`、`kernel_spec: P2` 对应单波段 PSF。
- `train.enable_amp: true`（混合精度）与 `train.use_grad_clip: true` 均已设置。
- 路径指向统一的数据/日志位置。

如需调整损失权重，可修改 `hybrid_opt` 中各 `w_` 参数，保持与 RGB 实验一致以保证可比性。

---

## 第 6 步：启动 Mono-PSF 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml
```

此配置包含自定义的混合损失（L1 + 感知 + LPIPS + ΔE_00 + 物理一致性），训练时间相对更长，请确保单次运行不会超过 Colab 限制，如有需要可分段续训。

---

## 第 7 步：监控与验证

```python
%load_ext tensorboard
LOG_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${LOG_DIR}
```

评估流程与基线模型相同：可在配置添加 `datasets.test` 节点，指定 `test_short.lmdb` 与 `test_long.lmdb` 后再次运行脚本。

---

## 第 8 步：收尾与注意事项

- 训练产生的权重/日志在 `experiments/SID_NewBP_Mono`（默认名称）下。
- 如 GPU 记忆不足，可同步调低 `batch_size_per_gpu` 与 `patch_size`——务必在所有模型中保持一致。
- 若需恢复训练，将 `path.resume_state` 设置为最新 `*.state` 文件。
- 每次训练结束后，可执行 `import torch; torch.cuda.empty_cache()` 释放显存。

完成以上步骤，即可在独立笔记本内完成 Mono-PSF 实验。***
