# Colab 笔记本：U-Net (Baseline) 全流程指南

> 目标：在 **单独 Colab 笔记本** 中完成 Sony（See-in-the-Dark）数据准备、U-Net 训练与评估。所有步骤都给出完整命令，并提供“在线下载”与“手动上传”两种数据获取途径。

---

## 第 0 步：挂载 Google Drive 并建立工作目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：准备 Sony (SID) 数据集

> 如果在线下载失败，请改用手动上传方式。务必保持目录结构为 `SID_raw/Sony/short/*.ARW` 与 `SID_raw/Sony/long/*.ARW`。

### 方式 A（推荐）：手动上传/同步到 Drive
1. 在浏览器中将 `Sony.zip` 或解压后的 `Sony` 文件夹上传至 `MyDrive/Lowlight/SID_raw/`。
2. 上传完成后执行：
   ```bash
   !mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
   %cd /content/drive/MyDrive/Lowlight/SID_raw
   # 若上传的是 ZIP，请手动解压或取消注释下一行
   # !unzip -q Sony.zip
   ```

### 方式 B（尝试）：Colab 内直接下载
```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

# 官方镜像（若链接已失效会提示失败）
!wget --no-verbose -O Sony.zip "https://storage.googleapis.com/isl-datasets/SID/Sony.zip" || echo "官方直链下载失败，可改用 gdown 或手动上传。"

# Google Drive 镜像（如提示权限问题请先确认文件仍公开）
!gdown --id 1P8dkX1bMErx6-8sszv_2i5Vi1Ndo63ok -O Sony.zip || echo "gdown 下载失败，请手动上传。"

!unzip -q Sony.zip
```

### 数据就绪校验
```python
from pathlib import Path
short_dir = Path("/content/drive/MyDrive/Lowlight/SID_raw/Sony/short")
long_dir = Path("/content/drive/MyDrive/Lowlight/SID_raw/Sony/long")

if short_dir.is_dir() and long_dir.is_dir():
    short_cnt = len(list(short_dir.glob("*.ARW")))
    long_cnt = len(list(long_dir.glob("*.ARW")))
    print(f"short RAW 数量: {short_cnt}, long RAW 数量: {long_cnt}")
    if short_cnt == 0 or long_cnt == 0:
        raise SystemExit("检测到 RAW 数量为 0，请确认上传/下载是否完成。")
else:
    raise SystemExit("未找到 Sony RAW 目录，请检查路径或重新上传。")
```

---

## 第 2 步：克隆项目及依赖

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

## 第 5 步：RAW → 16-bit PNG

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

## 第 8 步：确认 U-Net 配置

检查 `configs/colab/sid_unet_baseline.yml`：

- 如目录结构与上述步骤一致，无需修改。
- 否则请调整 `datasets.*.manifest_path`、`datasets.*.io_backend.db_paths` 以及 `path` 段。

---

## 第 9 步：启动 U-Net 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_unet_baseline.yml
```

---

## 第 10 步：监控与测试

### 10.1 TensorBoard
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）
在配置中新增 `datasets.test` 指向 `test_short.lmdb` 与 `test_long.lmdb` 后，重新执行第 9 步即可计算 PSNR、SSIM、LPIPS、ΔE₀₀、Edge-ΔE₀₀。

---

## 第 11 步：收尾

- **断点续训**：设置 `path.resume_state` 为最新 `*.state` 文件。
- **释放显存**：`import torch; torch.cuda.empty_cache()`。
- **备份结果**：权重/日志位于 `experiments/SID_UNet_Baseline`（默认目录）。

完成以上步骤后，即可在独立 Colab 笔记本内完成 U-Net 实验。祝训练顺利！***
