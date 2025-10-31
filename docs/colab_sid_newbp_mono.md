# Colab 笔记本：NewBP + NAFNet（Mono-PSF）全流程指南

> 本指南用于在独立 Colab 笔记本中完成 Mono-PSF（NewBP-A）实验，提供手动上传与在线下载两种数据导入方式，以及数据校验命令。

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

### 方式 A：手动上传/同步（推荐且稳定）
1. 在浏览器将 `Sony.zip` 或解压后的 `Sony` 文件夹上传至 `MyDrive/Lowlight/SID_raw/`。
2. 上传完毕后执行以下命令（若已上传解压版，请跳过解压）：
   ```bash
   !mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
   %cd /content/drive/MyDrive/Lowlight/SID_raw
   # 若上传的是 ZIP，请手动解压或取消注释
   # !unzip -q Sony.zip
   ```

### 方式 B：尝试在线下载（若失败请改用方式 A）
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
        raise SystemExit("RAW 文件数量为 0，请确认上传/下载是否成功。")
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

## 第 3 步：安装依赖（务必包含 torchmetrics）

```bash
!pip install -r requirements.txt
# 保险起见，单独补充关键依赖，避免 Colab 预装版本不一致导致缺失
!pip install --upgrade rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics
```

### 依赖自检（推荐）

```python
import importlib, sys
mods = ['torch', 'torchvision', 'kornia', 'lpips', 'lmdb', 'torchmetrics']
for m in mods:
  try:
    importlib.import_module(m)
    print(f'[OK] {m}')
  except Exception as e:
    print(f'[X] {m}:', e)
    raise SystemExit('依赖未安装完整，请先修复后再继续。')
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

## 第 8 步：核对 Mono-PSF 配置文件

配置文件：`configs/colab/sid_newbp_mono.yml`。确保以下路径与前述步骤一致，如需自定义日志/模型目录可在 `path` 段修改：

- `datasets.*.manifest_path`
- `datasets.*.io_backend.db_paths`

---

## 第 9 步：启动 Mono-PSF 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml
```

若显存不足，请与其他模型一致地调低 `batch_size_per_gpu` 与 `patch_size`。

---

## 第 10 步：监控与测试评估

### 10.1 TensorBoard

```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）

在配置文件新增 `datasets.test` 指向 `test_short.lmdb` / `test_long.lmdb` 后，重新执行第 9 步即可评估 PSNR、SSIM、LPIPS、ΔE₀₀、Edge-ΔE₀₀。

---

## 第 11 步：收尾

- **断点续训**：设置 `path.resume_state` 为最新 `*.state` 文件。
- **释放显存**：`import torch; torch.cuda.empty_cache()`。
- **备份成果**：模型/日志默认位于 `experiments/SID_NewBP_Mono`。

至此，Mono-PSF 实验在独立笔记本中全部完成。***
