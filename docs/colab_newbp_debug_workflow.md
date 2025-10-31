# Colab 调试流程：NewBP Mono & RGB

> 依照导师的“先小后大”原则，这份指南从环境准备到单批次过拟合，再到完整训练，按顺序列出 **NewBP-Mono** 与 **NewBP-RGB** 两条 Colab 调试链路，确保路径、依赖、数据、外部模型全部就位。

---

## 0. 挂载 Drive 并核对目录

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight"
!ls -R {BASE_DIR} | head -n 40  # 确认截图中的 SID_assets / SID_lmdb 等目录存在

EXPERIMENT_ROOT = f"{BASE_DIR}/SID_experiments"
!mkdir -p {EXPERIMENT_ROOT}
%cd {EXPERIMENT_ROOT}
```

若未看到 `SID_assets`, `SID_lmdb`, `SID_png`, `SID_raw`, `SID_experiments/external` 等目录，请先在 Drive 中整理出与截图一致的结构。

---

## 1. 克隆项目与外部对比模型

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/megvii-research/NAFNet.git
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/jacobgil/pytorch-unet.git  # 若需对比 UNet，可保留
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```

确保仓库路径与配置文件中引用的路径完全一致，否则 import 会失败。

---

## 2. 安装依赖并自检

```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless  # 再次确认关键库
```

```python
import importlib
for pkg in ["torch", "torchvision", "rawpy", "kornia", "lpips", "lmdb", "torchmetrics"]:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except Exception as exc:
        raise SystemExit(f"[X] {pkg} 未正确安装：{exc}")
```

---

## 3. 配置 Python 搜索路径

```python
import sys
extra_paths = [
    "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
    "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
]
for p in extra_paths:
    if p not in sys.path:
        sys.path.append(p)
print("sys.path 已补齐。")
```

---

## 4. 数据准备（RAW → PNG → Manifest → LMDB）

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!python tools/convert_sid_raw_to_png.py \
  --raw-root /content/drive/MyDrive/Lowlight/SID_raw/Sony \
  --output-root /content/drive/MyDrive/Lowlight/SID_png/Sony \
  --compress-level 1

!python tools/prepare_sid_manifest.py \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --seed 42 --val-ratio 0.1 --test-ratio 0.1

!python tools/create_sid_lmdb.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --output-root /content/drive/MyDrive/Lowlight/SID_lmdb \
  --compress-level 1
```

快速验证：

```python
from datasets.sony_sid_dataset import SonySIDDataset
ds = SonySIDDataset(
    root_dir="/content/drive/MyDrive/Lowlight/SID_raw",
    camera="Sony",
    patch_size=None,
    samples_per_pair=1,
)
print("RAW 样本:", len(ds))
```

---

## 5. 单批次过拟合检查（NewBP Mono / RGB）

在正式训练前，用合成数据验证模型与损失数值是否稳定。

### 5.1 Mono（Panchromatic / P2）
```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
python tools/debug_overfit.py --device cuda --iters 200 --height 64 --width 64
python tools/debug_overfit.py --device cuda --iters 400 --loss hybrid --enable-phys
```

### 5.2 RGB（Trichromatic / B2）
```bash
python tools/debug_overfit.py --device cuda --iters 200 --kernel-type rgb --kernel-spec B2
python tools/debug_overfit.py --device cuda --iters 400 --kernel-type rgb --kernel-spec B2 \
       --loss hybrid --enable-phys
```

若任一步出现 `NaN` / `Inf`，请根据报错的损失项（L1_raw、Perc、Phys 等）回溯数据或超参。

---

## 6. 小数据集端到端调试

> 先为 `configs/debug/*.yml` 填入 **单对图像** 的 LMDB/Manifest 路径。

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
python basicsr/train.py -opt ../configs/debug/sid_newbp_mono_debug.yml
python basicsr/train.py -opt ../configs/debug/sid_newbp_rgb_debug.yml
```

期望数百 iter 内 loss 下降且无 NaN；若失败，请回退至第 5 步重新定位。

---

## 7. 完整训练脚本

确认 `configs/colab/sid_newbp_mono.yml` 与 `configs/colab/sid_newbp_rgb.yml` 中的路径与 Drive 目录一致（`manifest_path`, `db_paths`, `path.*`）。

```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml
python basicsr/train.py -opt ../configs/colab/sid_newbp_rgb.yml
```

如需限制显存，可在 YAML 中同步调整 `patch_size`, `batch_size_per_gpu`, `samples_per_pair`, `num_worker_per_gpu`，并把 `prefetch_mode` 改为 `null` 或 `cpu`。

---

## 8. 可视化与日志

```python
%load_ext tensorboard
TENSORBOARD_LOG = "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/experiments"
tensorboard --logdir ${TENSORBOARD_LOG}
```

模型、日志、可视化默认保存在：
```
/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base/experiments
```

---

## 9. 常见问题排查表

| 症状 | 排查要点 |
| ---- | ---- |
| `ModuleNotFoundError` | 确认已执行第 3 步 `sys.path` 补充，且 external 仓库已克隆。 |
| `manifest` 相关报错 | 检查 `SID_assets/manifest_sid.json` 是否存在并与 YAML 中路径一致。 |
| 读取 LMDB 失败 | 确认 `SID_lmdb` 下的 `.lmdb` 目录已生成且 Colab 权限允许访问。 |
| 训练中出现 NaN | 先运行第 5 步脚本锁定问题损失项，再检查曝光比、物理核和学习率。 |
| OOM / 资源不足 | 减小 `patch_size`、`batch_size_per_gpu`、禁用 `enable_amp`、调低 `num_worker_per_gpu`。 |
| 依赖未找到 | 重跑第 2 步 pip 安装，并留意 `rawpy`, `kornia`, `lpips` 这三个常缺库。 |

---

遵循以上顺序，可先确保 NewBP-Mono 与 NewBP-RGB 在最小规模上稳定，再逐步扩展到完整实验。
