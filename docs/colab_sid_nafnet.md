# Colab 笔记本：NAFNet (Baseline) 全流程指南

> 该笔记本独立完成 Sony SID 数据准备、NAFNet Baseline 训练与评估。每个步骤均提供完整命令，可直接复制执行。

---

## 第 0 步：挂载 Google Drive 并设置根目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：下载并解压 SID 数据集

若 Drive 中已存在 RAW 数据可跳过，下述命令仅为模板，请按需替换链接或文件 ID。

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

# 示例：wget 方式
!wget -O SID_Sony.zip "https://[SID_DATASET_DOWNLOAD_LINK]"

# 示例：gdown 方式
# !gdown --id [FILE_ID] -O SID_Sony.zip

!unzip -q SID_Sony.zip
```

---

## 第 2 步：克隆主项目与依赖仓库

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

## 第 6 步：生成 manifest 并划分数据集

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

## 第 8 步：核对 NAFNet 配置

配置文件：`configs/colab/sid_nafnet_baseline.yml`。默认路径与上述步骤一致，无需改动；如目录不同，请在以下字段同步修改：

- `datasets.train/val.manifest_path`
- `datasets.*.io_backend.db_paths`
- `path` 段（若需自定义输出位置）

确保所有模型共享相同的 manifest 与 LMDB 目录以保证公平对比。

---

## 第 9 步：启动 NAFNet 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_nafnet_baseline.yml
```

若出现显存不足，可在配置中统一调低 `batch_size_per_gpu` 与 `patch_size`（并在其他模型配置中保持一致）。

---

## 第 10 步：监控与评估

### 10.1 TensorBoard 监控
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）

在配置文件中新增 `datasets.test`（结构与 `datasets.val` 相同，指向 `test_*` LMDB），保存后再次执行第 9 步。脚本会按照 `val.metrics` 计算 PSNR、SSIM、LPIPS、ΔE_00、Edge-ΔE_00。

---

## 第 11 步：收尾与维护

- **断点续训**：在 YAML 中设置 `path.resume_state` 为最新 `*.state` 文件。
- **释放显存**：`import torch; torch.cuda.empty_cache()`。
- **备份成果**：模型权重、日志、TensorBoard 事件位于 `experiments/SID_NAFNet_Baseline`（默认名称）。

至此，NAFNet Baseline 在独立笔记本中的所有步骤已经完成。***
