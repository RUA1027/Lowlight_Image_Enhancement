# Colab 笔记本：U-Net (Baseline) 训练流程

> 说明：本笔记本 **只负责 U-Net** 模型训练与评估。若在其他笔记本已经完成「环境/数据准备」，可跳过重复步骤，直接从“第 3 步”开始。

---

## 第 0 步（首次运行必做）：挂载 Drive 并设定工作目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步（首次运行必做）：克隆代码仓库与依赖项目

```bash
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/megvii-research/NAFNet.git
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```

---

## 第 2 步（首次运行必做）：安装依赖

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless
```

> 若已在任一模型笔记本运行过上述命令，可跳过。

---

## 第 3 步（首次运行必做）：统一设置 Python 路径

```python
import sys
ROOT = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement"
if ROOT not in sys.path:
    sys.path.append(ROOT)

SWINIR_ROOT = "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR"
if SWINIR_ROOT not in sys.path:
    sys.path.append(SWINIR_ROOT)
```

---

## 第 4 步（若未完成，请执行）：数据准备

> 如果之前任一笔记本已经完成 RAW→PNG、manifest、LMDB 的制作，直接跳到“第 5 步”。

1. **RAW → 16-bit PNG**

```bash
!python tools/convert_sid_raw_to_png.py \
  --raw-root /content/drive/MyDrive/Lowlight/SID_raw/Sony \
  --output-root /content/drive/MyDrive/Lowlight/SID_png/Sony \
  --compress-level 1
```

2. **生成 manifest（固定划分）**

```bash
!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1
```

3. **创建 LMDB（train/val/test）**

```bash
!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

---

## 第 5 步：确认配置文件（U-Net 专用）

`configs/colab/sid_unet_baseline.yml` 已默认设置以下路径，如与自身目录一致，则无需修改：

- manifest：`/content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json`
- LMDB：`/content/drive/MyDrive/Lowlight/SID_lmdb/train_*.lmdb`、`val_*.lmdb`
- 日志/模型默认写入 `experiments_root`（如需自定义，可在配置 `path` 段自行增加）

如目录结构不同，请使用 `sed` 或手动编辑替换为你的实际路径。

---

## 第 6 步：启动 U-Net 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_unet_baseline.yml
```

训练过程默认启用 AMP，日志会写入 TensorBoard（`logger.use_tb_logger=true`）。

---

## 第 7 步：监控与评估

1. **查看 TensorBoard**
   ```python
   %load_ext tensorboard
   LOG_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
   tensorboard --logdir ${LOG_DIR}
   ```
2. **测试集评估（可选）**  
   若需在测试集上验证，可复制训练命令，更换 `opt` 配置中的 `datasets.test`（参考 `datasets.val` 配置）后再运行一次 `basicsr/train.py`，程序会进入验证模式。

---

## 第 8 步：收尾

- 训练结束后，可在 `path.models`、`path.log` 等目录中查看权重与日志。
- 若需恢复训练，请将 `train.resume_state` 指向最新的 `*.state` 文件。
- 清理 GPU 内存：`import torch; torch.cuda.empty_cache()`（可在 Colab 控制台运行）。

到此，U-Net 实验在独立 Colab 笔记本中完成。下一模型请启动对应的笔记本，重复执行“第 0 ～ 第 7 步”中尚未执行的部分。***
