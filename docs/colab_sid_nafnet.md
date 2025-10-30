# Colab 笔记本：NAFNet (Baseline) 训练流程

> 本笔记本仅运行 **NAFNet Baseline**。与其他模型相同的准备工作，只需在任一笔记本中执行一次；若已完成可跳过对应步骤。

---

## 第 0 步：挂载 Drive & 设置目录（首次运行必做）

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步：克隆仓库（首次运行必做）

```bash
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/megvii-research/NAFNet.git
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```

---

## 第 2 步：安装依赖（首次运行必做）

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless
```

---

## 第 3 步：配置 Python 路径

```python
import sys
ROOT = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement"
for path in (
    ROOT,
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
):
    if path not in sys.path:
        sys.path.append(path)
```

---

## 第 4 步：数据准备（若已完成，可跳过）

同其它笔记本：确保 RAW→PNG、manifest、LMDB 均位于

- PNG：`/content/drive/MyDrive/Lowlight/SID_png/Sony`
- manifest：`/content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json`
- LMDB：`/content/drive/MyDrive/Lowlight/SID_lmdb`

执行命令见 `docs/colab_sid_unet.md` 第 4 步。

---

## 第 5 步：检查 NAFNet 配置

打开 `configs/colab/sid_nafnet_baseline.yml`：

- `network_g.type: NAFNet`，参数 `width/enc_blk_nums/middle_blk_num` 与论文设置一致。
- `train.enable_amp: true` 已开启混合精度。
- 数据路径与日志输出目录默认使用上述统一结构，如需修改请同步调整。

---

## 第 6 步：启动 NAFNet Baseline 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_nafnet_baseline.yml
```

> 若显存不足，可在配置里同步调低 `patch_size` 和 `batch_size_per_gpu`，所有模型需保持一致以满足对比原则。

---

## 第 7 步：监控与评估

```python
%load_ext tensorboard
LOG_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${LOG_DIR}
```

如需测试集评估，可参照配置 `datasets.val`，新增 `datasets.test` 指向 `test_short.lmdb` 与 `test_long.lmdb` 后重新运行 `basicsr/train.py`。

---

## 第 8 步：后续操作

- **恢复训练**：在配置 `path.resume_state` 填入最新 `*.state`。
- **清理显存**：`import torch; torch.cuda.empty_cache()`。
- **日志管理**：输出权重和 logs 默认保存在 `experiments` 子目录，建议定期同步至本地或云端备份。

至此，NAFNet Baseline 在独立 Colab 笔记本的训练流程完成。***
