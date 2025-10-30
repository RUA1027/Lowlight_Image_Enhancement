# Colab 笔记本：NewBP + NAFNet（Mono-PSF）全流程指南

> 本笔记本专注 **Mono-PSF** 实验（NewBP-A）。包含从数据下载到训练评估的完整步骤，可直接按顺序运行。

---

## 第 0 步：挂载 Google Drive 并设定目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：下载并解压 SID 数据集

若 Drive 已包含 `SID_raw/Sony/short,long` 可跳过。示例命令：

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

# wget 示例（替换 URL）
!wget -O SID_Sony.zip "https://[SID_DATASET_DOWNLOAD_LINK]"

# gdown 示例（替换 FILE_ID）
# !gdown --id [FILE_ID] -O SID_Sony.zip

!unzip -q SID_Sony.zip
```

---

## 第 2 步：克隆项目及依赖仓库

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/megvii-research/NAFNet.git
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```

---

## 第 3 步：安装依赖

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless
```

---

## 第 4 步：配置 Python 路径

```python
import sys
EXTRA_PATHS = [
    "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
]
for p in EXTRA_PATHS:
    if p not in sys.path:
        sys.path.append(p)
```

---

## 第 5 步：RAW 转 16-bit PNG

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!python tools/convert_sid_raw_to_png.py \
  --raw-root /content/drive/MyDrive/Lowlight/SID_raw/Sony \
  --output-root /content/drive/MyDrive/Lowlight/SID_png/Sony \
  --compress-level 1
```

---

## 第 6 步：生成 manifest（固定数据划分）

```bash
!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1
```

---

## 第 7 步：创建 LMDB

```bash
!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

---

## 第 8 步：校对 Mono-PSF 配置

配置文件：`configs/colab/sid_newbp_mono.yml`。默认参数说明：

- `network_g.type: NewBPNAFNet`
- `train.hybrid_opt.physics.mode: mono`
- `physics.kernel_spec: P2`（单波段 PSF）
- `train.enable_amp: true`

如目录结构不同，请修改 `datasets.*.manifest_path`、`io_backend.db_paths` 及 `path` 段中的路径，确保所有模型共享同一份数据。

---

## 第 9 步：启动 Mono-PSF 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml
```

> 若显存不足，可在配置中统一调低 `batch_size_per_gpu` 与 `patch_size`。

---

## 第 10 步：监控训练并评估指标

### 10.1 TensorBoard 监控
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）

在配置文件中添加 `datasets.test`（结构与 `datasets.val` 相同，指向 `test_*` LMDB），然后重新运行第 9 步。程序会使用统一指标集（线性域 PSNR/SSIM、LPIPS、ΔE_00、Edge-ΔE_00）对测试集进行评估。

---

## 第 11 步：收尾操作

- **断点续训**：设置 `path.resume_state` 为最新 `*.state` 文件。
- **显存清理**：在空闲单元执行 `import torch; torch.cuda.empty_cache()`。
- **结果整理**：权重、日志与 TensorBoard 事件保存在 `experiments/SID_NewBP_Mono`（默认名称），请根据需要备份。

至此，Mono-PSF 实验已在独立笔记本中完成。***
