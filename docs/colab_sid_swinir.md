# Colab 笔记本：SwinIR (Baseline) 全流程指南

> 本指南用于在 **单独的 Colab 笔记本** 内完成 Sony SID 数据准备与 SwinIR 训练。所有步骤均给出完整命令，无需参考其他文件。

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

## 第 1 步：下载并解压 SID 数据集

若你已手动上传或准备好 `SID_raw/Sony/short, long`，可跳过本步骤。以下命令提供通用模板：

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

# 方式 A：wget（替换 URL）
!wget -O SID_Sony.zip "https://[SID_DATASET_DOWNLOAD_LINK]"

# 方式 B：gdown（替换 FILE_ID）
# !gdown --id [FILE_ID] -O SID_Sony.zip

!unzip -q SID_Sony.zip
```

确保解压后存在 `Sony/short/*.ARW` 与 `Sony/long/*.ARW`。

---

## 第 2 步：克隆主项目及所需依赖

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

## 第 3 步：安装依赖包

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless
```

---

## 第 4 步：设置 Python 路径（含 SwinIR）

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

## 第 5 步：RAW → 16-bit PNG 转换

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!python tools/convert_sid_raw_to_png.py \
  --raw-root /content/drive/MyDrive/Lowlight/SID_raw/Sony \
  --output-root /content/drive/MyDrive/Lowlight/SID_png/Sony \
  --compress-level 1
```

---

## 第 6 步：生成 manifest（确保数据划分一致）

```bash
!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1
```

---

## 第 7 步：创建 LMDB 加速数据读取

```bash
!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

---

## 第 8 步：校验 SwinIR 配置文件

配置文件位置：`configs/colab/sid_swinir_baseline.yml`。若使用默认路径，可直接跳过修改。否则请检查并替换以下路径：

- `datasets.train/val.manifest_path`
- `datasets.*.io_backend.db_paths`
- 日志/模型输出（可在 `path` 段自定义）

SwinIR 相关关键参数已在配置中设置（如 `window_size`, `embed_dim` 等），无需额外改动。

---

## 第 9 步：启动 SwinIR 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_swinir_baseline.yml
```

若显存不足，可在同一配置文件中统一调整 `batch_size_per_gpu` 与 `patch_size`（请在所有模型中保持一致以维持公平性）。

---

## 第 10 步：指标监控与可视化

### 10.1 实时监控（TensorBoard）
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）

若需在测试集上验证，请在 `sid_swinir_baseline.yml` 中新增如下节点，然后重新执行第 9 步：

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

脚本会在验证频率（默认 5000 iteration）自动计算 PSNR/SSIM/LPIPS/ΔE00/Edge-ΔE00。

---

## 第 11 步：收尾工作

- **断点续训**：将配置 `path.resume_state` 指向最新的 `*.state` 文件即可恢复。
- **显存释放**：使用 `import torch; torch.cuda.empty_cache()`。
- **成果整理**：训练权重、日志、TensorBoard 事件保存在 `experiments/SID_SwinIR_Baseline`（默认名称），可按需备份或迁移。

完成以上步骤，即可在独立笔记本内完成 SwinIR Baseline 实验。***
