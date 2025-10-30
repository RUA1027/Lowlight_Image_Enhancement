# Colab 笔记本：U-Net (Baseline) 全流程指南

> 目标：在 **单独的 Colab 笔记本** 内完成 Sony SID 数据准备、U-Net 训练与评估，不依赖其他文件。按顺序复制代码块执行即可。

---

## 第 0 步：挂载 Google Drive 并创建工作目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：在云端下载并解压 SID 数据集

> 如果 Drive 中已存在 `SID_raw/Sony/short` 与 `SID_raw/Sony/long` 文件夹，可跳过本步骤。示例命令提供两种下载方式，请根据实际情况替换链接或文件 ID。

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

# 方法 A：wget（请替换为有效的公开下载链接）
!wget -O SID_Sony.zip "https://[SID_DATASET_DOWNLOAD_LINK]"

# 方法 B：gdown（请替换 FILE_ID）
# !gdown --id [FILE_ID] -O SID_Sony.zip

!unzip -q SID_Sony.zip
# 解压完成后应存在 Sony/short/*.ARW 与 Sony/long/*.ARW
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

## 第 3 步：安装 Python 依赖

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

## 第 5 步：RAW → PNG（16-bit）转换

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!python tools/convert_sid_raw_to_png.py \
  --raw-root /content/drive/MyDrive/Lowlight/SID_raw/Sony \
  --output-root /content/drive/MyDrive/Lowlight/SID_png/Sony \
  --compress-level 1
```

执行完毕后，`SID_png/Sony/short` 与 `SID_png/Sony/long` 中应当包含 16-bit PNG 文件。

---

## 第 6 步：生成 manifest（数据划分）

```bash
!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1
```

可自定义 `val-ratio` / `test-ratio` 或通过 `--split-file` 指定固定列表，但需保证所有模型共享同一 manifest。

---

## 第 7 步：创建 LMDB（加速 I/O）

```bash
!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

完成后应生成 `train_short.lmdb`、`val_long.lmdb`、`test_short.lmdb` 等目录。

---

## 第 8 步：校对 U-Net 专用配置

配置文件 `configs/colab/sid_unet_baseline.yml` 默认使用上述路径。如目录结构与示例不同，请在运行训练前修改以下字段：

- `datasets.train/val.manifest_path`
- `io_backend.db_paths`
- `path` 段（若想自定义日志/模型输出位置）

所有占位路径均在 `/content/drive/MyDrive/Lowlight/...` 下，若你按建议结构操作则无需修改。

---

## 第 9 步：启动 U-Net 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_unet_baseline.yml
```

训练默认启用 AMP，梯度裁剪已在配置中开启。日志会写入 `experiments` 目录，TensorBoard 可直接读取。

---

## 第 10 步：监控与评估

### 10.1 TensorBoard 监控
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）

若要在测试集上计算指标，可在配置文件新增 `datasets.test` 节点：

```yaml
  test:
    name: SID-test
    type: SonySIDLMDBDataset
    phase: test
    manifest_path: /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json
    subset: test
    random_crop: false
    samples_per_pair: 1
    batch_size_per_gpu: 1
    num_worker_per_gpu: 1
    pin_memory: false
    io_backend:
      type: lmdb
      db_paths:
        - /content/drive/MyDrive/Lowlight/SID_lmdb/test_short.lmdb
        - /content/drive/MyDrive/Lowlight/SID_lmdb/test_long.lmdb
      client_keys:
        - short
        - long
```

保存后重新执行第 9 步命令；脚本会在指定频率对测试集计算你定义的全部指标（PSNR/SSIM/LPIPS/ΔE00/Edge-ΔE00）。

---

## 第 11 步：收尾与管理

- **断点续训**：在配置 `path.resume_state` 填入 `experiments/.../*.state` 文件即可。
- **显存清理**：`import torch; torch.cuda.empty_cache()`。
- **结果整理**：权重、日志、TensorBoard 事件均位于 `experiments` 子目录，可在本地或云端备份。

至此，U-Net 基线模型的完整流程已在独立笔记本中完成，可移步到其他模型的对应指南继续实验。***
