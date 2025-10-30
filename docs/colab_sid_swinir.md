# Colab 笔记本：SwinIR (Baseline) 全流程指南

> 在本笔记本中，从数据准备到训练评估均给出完整命令，并提供“手动上传/同步”与“在线下载”两种方式获取 Sony (See-in-the-Dark) 数据集。

---

## 第 0 步：挂载 Google Drive 并准备工作目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：准备 Sony (SID) 数据集

> 若在线下载失败，请使用手动上传方式，确保最终获得 `SID_raw/Sony/short/*.ARW` 与 `SID_raw/Sony/long/*.ARW`。

### 方式 A：手动上传/同步（推荐）
1. 将 `Sony.zip` 或已解压的 `Sony` 目录上传到 `MyDrive/Lowlight/SID_raw/`。
2. 上传完成后执行：
   ```bash
   !mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
   %cd /content/drive/MyDrive/Lowlight/SID_raw
   # 若上传的是 ZIP，可执行：
   # !unzip -q Sony.zip
   ```

### 方式 B：尝试在线下载
```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

!wget --no-verbose -O Sony.zip "https://storage.googleapis.com/isl-datasets/SID/Sony.zip" || echo "官方直链下载失败，请尝试 gdown 或手动上传。"
!gdown --id 1P8dkX1bMErx6-8sszv_2i5Vi1Ndo63ok -O Sony.zip || echo "gdown 下载失败，请手动上传。"

!unzip -q Sony.zip
```

### 校验数据是否就绪
```python
from pathlib import Path
short_dir = Path("/content/drive/MyDrive/Lowlight/SID_raw/Sony/short")
long_dir = Path("/content/drive/MyDrive/Lowlight/SID_raw/Sony/long")

if short_dir.is_dir() and long_dir.is_dir():
    short_cnt = len(list(short_dir.glob("*.ARW")))
    long_cnt = len(list(long_dir.glob("*.ARW")))
    print(f"short RAW 数量: {short_cnt}, long RAW 数量: {long_cnt}")
    if short_cnt == 0 or long_cnt == 0:
        raise SystemExit("RAW 文件数量为 0，请检查上传/下载。")
else:
    raise SystemExit("未找到 Sony RAW 目录，请重新确认路径。")
```

---

## 第 2 步：克隆项目与依赖仓库

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

## 第 6 步：生成 manifest（固定划分）

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

## 第 8 步：校对 SwinIR 配置

检查 `configs/colab/sid_swinir_baseline.yml` 是否使用正确的 manifest/LMDB 路径，如需自定义日志目录可在 `path` 段修改。

---

## 第 9 步：启动 SwinIR 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_swinir_baseline.yml
```

如显存不足，请同步调低 `batch_size_per_gpu` 与 `patch_size`（并保持所有模型一致）。

---

## 第 10 步：监控与测试评估

### 10.1 TensorBoard
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）
在配置中新增 `datasets.test` 指向 `test_short.lmdb` 与 `test_long.lmdb`，保存后重复第 9 步即可评估 PSNR、SSIM、LPIPS、ΔE₀₀、Edge-ΔE₀₀。

---

## 第 11 步：收尾

- **断点续训**：设置 `path.resume_state` 为最新 `*.state` 文件。
- **释放显存**：`import torch; torch.cuda.empty_cache()`。
- **备份成果**：权重与日志位于 `experiments/SID_SwinIR_Baseline`。

至此，SwinIR Baseline 实验在独立笔记本中完成。***
