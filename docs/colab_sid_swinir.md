# Colab 笔记本：SwinIR (Baseline) 训练流程

> 说明：本笔记本 **仅负责 SwinIR** 模型。若已在其他模型笔记本完成环境或数据准备，可直接跳到“第 5 步”或之后。

---

## 第 0 步（首次运行必做）：挂载 Drive 与设定工作目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments"
!mkdir -p {BASE_DIR}
%cd {BASE_DIR}
```

---

## 第 1 步（首次运行必做）：克隆仓库

若此前未执行，请运行：

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

---

## 第 3 步：设置 Python 路径（确保 SwinIR 可被导入）

```python
import sys
ROOT = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement"
SWINIR_ROOT = "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR"
for path in (ROOT, SWINIR_ROOT):
    if path not in sys.path:
        sys.path.append(path)
```

---

## 第 4 步：数据准备（若已完成，可跳过）

与其他模型一致，依次执行 RAW→PNG、生成 manifest、创建 LMDB（详见 U-Net 文档 `docs/colab_sid_unet.md` 第 4 步）。路径保持：

- RAW：`/content/drive/MyDrive/Lowlight/SID_raw/Sony`
- PNG：`/content/drive/MyDrive/Lowlight/SID_png/Sony`
- manifest：`/content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json`
- LMDB：`/content/drive/MyDrive/Lowlight/SID_lmdb`

---

## 第 5 步：校验 SwinIR 配置

打开 `configs/colab/sid_swinir_baseline.yml`，确认如下键值：

- `network_g.type: SwinIRRestoration`（由我们提供的封装，确保和本项目兼容）
- 数据路径与 manifest/LMDB 指向第 4 步所述位置
- `train.enable_amp: true` 已默认开启混合精度

若路径或文件夹命名不同，请调整配置。

---

## 第 6 步：启动 SwinIR 训练

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
!python basicsr/train.py -opt ../configs/colab/sid_swinir_baseline.yml
```

> SwinIR 依赖较大的窗口注意显存。如遇 OOM，可在配置中同步调低 `batch_size_per_gpu` 与 `patch_size`。

---

## 第 7 步：监控与验证

1. **TensorBoard**
   ```python
   %load_ext tensorboard
   LOG_DIR = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
   tensorboard --logdir ${LOG_DIR}
   ```

2. **测试集验证（可选）**  
   在配置中新增 `datasets.test` 节点，指向 `test_short.lmdb` / `test_long.lmdb`，然后再次运行 `basicsr/train.py` 进入验证流程。

---

## 第 8 步：常见额外操作

- **恢复训练**：在配置 `path.resume_state` 写入最新 `*.state` 文件路径。
- **释放显存**：`import torch; torch.cuda.empty_cache()`。
- **保留日志**：模型权重与日志默认写入 `experiments` 目录，每个模型都有独立子文件夹，方便整理。

至此，在独立 Colab 笔记本中完成 SwinIR 训练任务。***
