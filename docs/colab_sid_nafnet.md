# Colab 笔记本：NAFNet (Baseline) 全流程指南

> 本笔记本在独立环境中完成 Sony (See-in-the-Dark) 数据准备与 NAFNet Baseline 训练，包含“手动上传/同步”和“在线下载”两种数据获取方式。

---

## 第 0 步：挂载 Google Drive 并初始化目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：准备 Sony (SID) 数据集

### 方式 A：手动上传或同步到 Drive（推荐）
1. 将 `Sony.zip` 或解压后的 `Sony` 文件夹上传到 `MyDrive/Lowlight/SID_raw/`。
2. 上传完成后执行：
   ```bash
   !mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
   %cd /content/drive/MyDrive/Lowlight/SID_raw
   # 若上传的是 ZIP，需要解压时可取消注释下行：
   # !unzip -q Sony.zip
   ```

### 方式 B：尝试在 Colab 中直接下载
> 数据集托管链接偶尔会失效，请根据运行结果决定是否改用方式 A。

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

!wget --no-verbose -O Sony.zip "https://storage.googleapis.com/isl-datasets/SID/Sony.zip" || echo "官方直链下载失败，请改用 gdown 或手动上传。"
!gdown --id 1P8dkX1bMErx6-8sszv_2i5Vi1Ndo63ok -O Sony.zip || echo "gdown 下载失败，请手动上传。"

!unzip -q Sony.zip
```

### 校验 RAW 数据是否就绪
```python
from pathlib import Path
short_dir = Path("/content/drive/MyDrive/Lowlight/SID_raw/Sony/short")
long_dir = Path("/content/drive/MyDrive/Lowlight/SID_raw/Sony/long")

if short_dir.is_dir() and long_dir.is_dir():
    short_cnt = len(list(short_dir.glob("*.ARW")))
    long_cnt = len(list(long_dir.glob("*.ARW")))
    print(f"short RAW: {short_cnt} 份, long RAW: {long_cnt} 份")
    if short_cnt == 0 or long_cnt == 0:
        raise SystemExit("RAW 数量为 0，请确认上传/下载流程。")
else:
    raise SystemExit("未检测到 Sony RAW 目录，请检查路径或重新上传。")
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
for p in [
    "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
]:
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

## 第 8 步：确认 NAFNet 配置

在 `configs/colab/sid_nafnet_baseline.yml` 中确认：

- `datasets.*.manifest_path` 与 `io_backend.db_paths` 指向上述路径；
- 如需自定义日志/模型输出，请修改 `path` 段。

---

## 第 9 步：启动 NAFNet 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_nafnet_baseline.yml
```

若显存不足，可统一调低 `batch_size_per_gpu` 与 `patch_size`（并在其他模型里保持一致）。

---

## 第 10 步：监控与测试

### 10.1 TensorBoard
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）
在配置中新增 `datasets.test` 指向 `test_short.lmdb` / `test_long.lmdb`，保存后重新运行第 9 步即可计算全套指标（PSNR、SSIM、LPIPS、ΔE₀₀、Edge-ΔE₀₀）。

---

## 第 11 步：收尾

- **断点续训**：配置 `path.resume_state` 为最新 `*.state` 文件。
- **释放显存**：`import torch; torch.cuda.empty_cache()`。
- **备份结果**：输出位于 `experiments/SID_NAFNet_Baseline`（默认命名）。

完成以上步骤，即可在独立笔记本中完成 NAFNet Baseline 实验。祝实验顺利！***
