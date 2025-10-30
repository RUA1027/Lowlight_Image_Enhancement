# Colab 上运行 Sony SID 低照度实验全流程指南

本文档以 Colab + Google Drive 为环境，串联数据准备、LMDB 构建、模型训练与评估的全部步骤。所有命令均可直接复制到 Colab 代码单元执行。请确保在开始前拥有充足的 Google Drive 存储空间（建议 ≥400 GB）。

---

## 0. 基础环境设置

```python
# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 建議的工作目錄（可按需調整）
%cd /content/drive/MyDrive
!mkdir -p Lowlight/SID_experiments
%cd Lowlight/SID_experiments
```

依次克隆所需倉庫：

```bash
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/megvii-research/NAFNet.git
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```

安装依赖（包含 rawpy/kornia/lpips/lmdb 等）：

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm
```

确保 Python 路径包含 SwinIR 及 NAFNet（新增到 `PYTHONPATH`）：

```python
import sys
sys.path.append('/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR')
sys.path.append('/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet')
```

---

## 1. 数据下载与解压

示例命令（请替换为实际的 SID 数据下载链接或 gdown ID）：

```bash
!mkdir -p /content/drive/MyDrive/Lowlight/SID_raw
%cd /content/drive/MyDrive/Lowlight/SID_raw
!wget -O SID_Sony.zip "https://[SID_DATASET_LINK]"
!unzip -q SID_Sony.zip
```

解压后确认目录结构为：

```
SID_raw/
├── Sony/
    ├── long/*.ARW
    └── short/*.ARW
```

---

## 2. RAW → 16-bit PNG 转换

使用仓库提供的脚本将 RAW 转为 16-bit PNG（保留曝光比信息）：

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!python tools/convert_sid_raw_to_png.py \
  --raw-root /content/drive/MyDrive/Lowlight/SID_raw/Sony \
  --output-root /content/drive/MyDrive/Lowlight/SID_png/Sony \
  --compress-level 1
```

生成目录示例：

```
SID_png/Sony/
├── long/*.png
└── short/*.png
```

---

## 3. 构建清单（Manifest）并划分数据集

```bash
!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1
```

说明：
- 保持 seed=42，可确保不同模型使用完全一致的 train/val/test 划分。
- 如需自定义固定划分，可传入 `--split-file path/to/custom_split.json`，文件需包含 `{"train": [...], "val": [...], "test": [...]}` 的 pair_id 列表。

---

## 4. 构建 LMDB 以优化 I/O

```bash
!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

执行后会生成：

```
SID_lmdb/
├── train_short.lmdb
├── train_long.lmdb
├── val_short.lmdb
├── val_long.lmdb
├── test_short.lmdb
└── test_long.lmdb
```

LMDB 的 key 与 manifest 中 `short_key/long_key` 一致，供 dataloader 直接调用。

---

## 5. 配置文件路径更新

进入仓库根目录，使用编辑器或 `sed` 将 `configs/colab/*.yml` 中的占位路径替换为实际路径，推荐执行：

```bash
CONFIG_ROOT=/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/configs/colab
DATA_ROOT=/content/drive/MyDrive/Lowlight

for cfg in sid_unet_baseline.yml sid_swinir_baseline.yml sid_nafnet_baseline.yml sid_newbp_mono.yml sid_newbp_rgb.yml; do
  cfg_path="${CONFIG_ROOT}/${cfg}"
  sed -i "s#/content/drive/MyDrive/Lowlight/SID#${DATA_ROOT}/SID#g" "${cfg_path}"
  sed -i "s#/content/drive/MyDrive/Lowlight/SID_assets#${DATA_ROOT}/SID_assets#g" "${cfg_path}"
  sed -i "s#/content/drive/MyDrive/Lowlight/SID_lmdb#${DATA_ROOT}/SID_lmdb#g" "${cfg_path}"
done
```

如需要将实验输出保存到自定义目录，可在每个配置的 `path` 段补充：

```yaml
path:
  experiments_root: /content/drive/MyDrive/Lowlight/experiments
  log: /content/drive/MyDrive/Lowlight/experiments/logs
  visualization: /content/drive/MyDrive/Lowlight/experiments/vis
```

---

## 6. 统一受控变量检查清单

- 数据集：使用 manifest + LMDB 中的 Sony SID，分割 seed=42。
- 预处理：`datasets/sony_sid_lmdb_dataset.py` 统一 RAW→16bit→比值对齐流程。
- Patch Size：全部模型均使用 `512×512`。
- Batch Size：全部配置 `batch_size_per_gpu: 4`。
- 总迭代：统一 `300k`，scheduler 覆盖全程。
- 评估指标：`linear_psnr/linear_ssim/lpips/deltaE_mean/deltaE_edge`；`use_image: false` 确保直接在张量上计算。
- AMP：可在 Colab 中通过 `torch.cuda.amp` 自动启用（项目默认支持）。

---

## 7. 启动训练（同命令适用于 T4/V100）

所有训练均在仓库根目录执行，确保 `PYTHONPATH` 已包含外部模型：

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base

# U-Net Baseline
!python basicsr/train.py -opt ../configs/colab/sid_unet_baseline.yml

# SwinIR Baseline
!python basicsr/train.py -opt ../configs/colab/sid_swinir_baseline.yml

# NAFNet Baseline
!python basicsr/train.py -opt ../configs/colab/sid_nafnet_baseline.yml

# NewBP + Mono-PSF
!python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml

# NewBP + RGB-PSF
!python basicsr/train.py -opt ../configs/colab/sid_newbp_rgb.yml
```

提示：
- 首次运行请关注 `basicsr/train.py` 输出，确认 dataloader 能正确读取 LMDB。
- 训练日志与 TensorBoard 事件将保存在配置 `path` 段指定位置。
- 若中途中断，可在 `path.resume_state` 指向最新的 `state` 文件继续训练。

---

## 8. 验证与测试

训练脚本本身会在 `val_freq` 间隔进行验证。若需额外在测试集评估，可执行：

```bash
!python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml --launcher none --input_path "" --output_path ""
```

或编写单独推理脚本（如 `basicsr/test.py`），将 `datasets/test` 修改为 `manifest` 中的 `test` 子集并复用相同配置。

---

## 9. 结果收集与对比

1. 统计 `experiments/<exp_name>/logs` 中的 TensorBoard 指标：`psnr_linear`, `ssim_linear`, `lpips_vgg`, `deltaE_mean`, `deltaE_edge`。
2. 可使用以下示例生成对比表：

```python
import pandas as pd
import json, glob

def load_metrics(log_dir):
    with open(log_dir, 'r') as f:
        return json.load(f)

# 自行实现读取 tensorboard 或日志的逻辑
```

3. 确保以统一的迭代轮数或最佳验证指标对比五种模型。

---

## 10. 后续建议

- 若 GPU 内存允许，可尝试增大 `samples_per_pair` 或 patch 数量加速收敛。
- 利用 `tools/prepare_sid_manifest.py` 的 `--split-file` 参数，可固定第一轮实验的最佳迭代点再复用于第二轮。
- 若需进一步节省 I/O，可将 LMDB 和实验日志目录迁移到 Colab 临时 SSD (`/content`)；但需额外脚本同步回 Drive。

---

完成上述步骤后，即可在 Colab 环境内完整复现与扩展 Sony SID 的五模型对比实验。祝实验顺利！
