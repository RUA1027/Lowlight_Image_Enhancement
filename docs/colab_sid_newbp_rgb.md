# Colab 笔记本：NewBP + NAFNet（RGB-PSF）全流程指南

> 本指南适用于 **RGB-PSF** 实验（NewBP-B）。按顺序执行，即可在独立 Colab 笔记本中完成全部步骤。

---

## 第 0 步：挂载 Google Drive 并创建工作区

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：下载并解压 SID 数据集

若 Drive 中已有 `SID_raw/Sony/short,long`，可跳过此步。示例命令（需自行替换链接或文件 ID）：

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw

# wget 示例
!wget -O SID_Sony.zip "https://[SID_DATASET_DOWNLOAD_LINK]"

# gdown 示例
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

## 第 4 步：配置 Python 依赖路径

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

## 第 6 步：生成 manifest（确保划分一致）

```bash
!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1
```

---

## 第 7 步：创建 LMDB 数据库

```bash
!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

---

## 第 8 步：检查 RGB-PSF 配置文件

配置文件位于 `configs/colab/sid_newbp_rgb.yml`，默认设置如下：

- `network_g.type: NewBPNAFNet`
- `train.hybrid_opt.physics.mode: rgb`
- `physics.kernel_spec: B2`（RGB PSF）
- `train.enable_amp: true`

若路径与默认不符，请修改以下字段以匹配当前目录：

- `datasets.train/val.manifest_path`
- `datasets.*.io_backend.db_paths`
- （可选）`path` 段中日志、模型输出位置

请确保 Mono 与 RGB 实验共享相同的 manifest/LMDB，以便公平对比。

---

## 第 9 步：启动 RGB-PSF 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_newbp_rgb.yml
```

训练过程中会启用混合损失（L1 + 感知 + LPIPS + ΔE_00 + RGB 物理一致性）及 AMP。若显存不足，请同步调低 `batch_size_per_gpu` 与 `patch_size`。

---

## 第 10 步：监控与指标评估

### 10.1 TensorBoard
```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

### 10.2 测试集评估（可选）

在配置文件中新增 `datasets.test` 节点（结构同 `val`，指向 `test_*` LMDB），然后重新运行第 9 步命令即可对测试集计算全套指标（线性域 PSNR/SSIM、LPIPS、ΔE_00、Edge-ΔE_00）。

---

## 第 11 步：收尾与维护

- **断点续训**：设置 `path.resume_state` 为最新 `*.state` 文件即可续训。
- **显存清理**：`import torch; torch.cuda.empty_cache()`。
- **成果管理**：训练权重与日志保存在 `experiments/SID_NewBP_RGB`（默认名称），建议及时备份。

至此，RGB-PSF 实验已全部完成，可与其他模型结果进行对比分析。***
