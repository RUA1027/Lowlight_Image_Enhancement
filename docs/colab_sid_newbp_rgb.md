# Colab 笔记本：NewBP + NAFNet（RGB-PSF 实验组）

> 本笔记本专注 **RGB-PSF** 变体（NewBP-B）。请确保 Mono-PSF 与 Baseline 模型分开运行，避免超时或冲突。

---

## 第 0 步：挂载 Drive、设定目录（首次运行必做）

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

## 第 3 步：配置 Python 路径

```python
import sys
for p in [
    "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
]:
    if p not in sys.path:
        sys.path.append(p)
```

---

## 第 4 步：数据准备（若已完成，可跳过）

- RAW → PNG
- manifest（固定划分 seed=42）
- LMDB（train/val/test）

命令见 `docs/colab_sid_unet.md` 第 4 步。默认路径：

```
PNG       : /content/drive/MyDrive/Lowlight/SID_png/Sony
manifest  : /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json
LMDB      : /content/drive/MyDrive/Lowlight/SID_lmdb
```

---

## 第 5 步：检查 RGB-PSF 配置

在 `configs/colab/sid_newbp_rgb.yml` 中重点确认：

- `network_g.type: NewBPNAFNet`
- `train.hybrid_opt.physics.mode: rgb`、`kernel_spec: B2`（深度彩色 PSF）
- 混合精度 `enable_amp: true` 已开启
- 数据与日志路径与前述路径一致

如需保持与 Mono-PSF 完全可比，请不要修改损失权重 `w_*` 配置。

---

## 第 6 步：启动 RGB-PSF 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_newbp_rgb.yml
```

> RGB-PSF 计算比 Mono 稍重，建议在训练途中定期保存状态（已默认 `save_checkpoint_freq=5000`）。

---

## 第 7 步：监控与验证

```python
%load_ext tensorboard
LOG_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${LOG_DIR}
```

需要在测试集上评估时，可在配置新增 `datasets.test` 指向 `test_short.lmdb`/`test_long.lmdb`，然后重新运行 `basicsr/train.py`（会执行验证流程，不会重新训练）。

---

## 第 8 步：注意事项

- **显存管理**：若遇 OOM，请与其它模型一致地调低 `batch_size_per_gpu`、`patch_size`。
- **恢复训练**：配置 `path.resume_state` 为最新 `*.state` 文件，即可断点续训。
- **显存释放**：`import torch; torch.cuda.empty_cache()`。
- **结果整理**：所有模型日志/权重位于统一 `experiments` 目录，便于后续汇总。

完成以上步骤，即可在 RGB-PSF 笔记本中完成 NewBP-B 实验。***
