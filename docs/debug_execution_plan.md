# NewBP 项目分阶段调试计划（本地 + Colab）

> 核心目标：把“是否能跑”与“是否正确”分离。先在本地 CPU 上完成最小可验证单元，再在 Colab GPU 上逐步放大。每个步骤都给出执行位置、命令与判定标准，确保 debug 覆盖数据、模型、损失、训练框架等关键节点。

---

## 阶段 A：本地（CPU）快速自检

> 这些步骤无需 GPU，推荐在仓库根目录（例如 `D:\Lowlight_image_enhancement`）执行。

### A0. 环境准备
- **命令：**
  ```powershell
  py -3 -m venv .venv
  .\.venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip install rawpy kornia lpips lmdb tqdm opencv-python-headless
  ```
- **通过标准：** 安装过程中无报错；执行 `python -c "import torch, rawpy, kornia, lmdb"` 无异常。

### A1. 代码结构快照
- **命令：**
  ```powershell
  Get-ChildItem -Recurse -Depth 1
  ```
- **检查点：** 看到 `configs/`, `NewBP_model/`, `tools/`、`docs/` 等目录存在；确认 `tools/debug_overfit.py`、`tools/debug_dataset.py`、`tools/debug_losses.py` 已就绪。

### A2. Python 搜索路径验证
- **命令：**
  ```powershell
  python - <<'PY'
  import sys
  target = [
      "NewBP_model", "NAFNet_base/basicsr", "datasets", "tools"
  ]
  for t in target:
      print(t, "->", any(t in p.replace("\\\\", "/") for p in sys.path))
  PY
  ```
- **通过标准：** 所有目标路径均返回 `True`。

### A3. 数据/Manifest 自查（使用示例或真实数据）
- **命令：**
  ```powershell
  python tools/debug_dataset.py ^
    --manifest data/debug_sid/manifest_sid_debug.json ^
    --short-root data/debug_sid/short ^
    --long-root data/debug_sid/long ^
    --limit 2 --inspect
  ```
  若已经准备真实 RAW/PNG，请将路径替换为实际目录。
- **通过标准：** 输出中“校验完成”且无缺失文件、曝光比异常等错误。

### A4. 自定义损失稳定性测试
- **命令：**
  ```powershell
  python tools/debug_losses.py --device cpu --steps 2 --height 64 --width 64
  ```
- **通过标准：** 控制台打印每一步的 loss 和 grad_norm，且“所有损失项检查通过”。若出现 `RuntimeError`，说明 HybridLossPlus 内部仍有 NaN/Inf，需优先修复。

### A5. 单批次过拟合（最小模型）
- **命令：**
  ```powershell
  python tools/debug_overfit.py --device cpu --iters 80 --height 64 --width 64 --log-interval 20
  ```
- **通过标准：** loss 在几十步内稳定下降、梯度范数为有限值；脚本未抛出异常。若 CPU 太慢，可将 `--iters` 降为 40 仅作冒烟。

> **本地阶段总结：** 只有当 A3~A5 全部通过，才能确信数据、损失、网络的核心逻辑已经打通。

---

## 阶段 B：Colab（GPU）扩展验证

> 下面命令在 Colab Notebook 中逐格执行；默认工作目录 `/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement`。

### B0. 挂载与目录核对
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

BASE_DIR = "/content/drive/MyDrive/Lowlight"
!ls -R {BASE_DIR} | head -n 40
```
- **通过标准：** 输出包含 `SID_assets/`, `SID_lmdb/`, `SID_png/`, `SID_raw/`, `SID_experiments/` 等目录。

### B1. 克隆仓库与外部模型（首次运行需要）
```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments
!git clone https://github.com/RUA1027/Lowlight_Image_Enhancement.git
!mkdir -p external
%cd external
!git clone https://github.com/megvii-research/NAFNet.git
!git clone https://github.com/JingyunLiang/SwinIR.git
!git clone https://github.com/jacobgil/pytorch-unet.git
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
```
- **通过标准：** 各仓库克隆成功，无 404 或权限错误。

### B2. 安装依赖 & 导入测试
```bash
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics
```
```python
import importlib
for pkg in ["torch","torchvision","rawpy","kornia","lpips","lmdb","torchmetrics"]:
    importlib.import_module(pkg)
print("依赖导入成功")
```
- **通过标准：** 全部模块导入成功，无报错。

### B3. 补充 Python 搜索路径
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
print("paths ready")
```
- **通过标准：** 输出 `paths ready`，后续 import 无问题。

### B4. 数据全链路检查（RAW/PNG/Manifest/LMDB）
```bash
python tools/debug_dataset.py \
  --manifest /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json \
  --short-root /content/drive/MyDrive/Lowlight/SID_png/Sony/short \
  --long-root  /content/drive/MyDrive/Lowlight/SID_png/Sony/long \
  --lmdb-short /content/drive/MyDrive/Lowlight/SID_lmdb/train_short.lmdb \
  --lmdb-long  /content/drive/MyDrive/Lowlight/SID_lmdb/train_long.lmdb \
  --limit 3
```
- **通过标准：** 每个 pair 显示图像存在且 LMDB key 命中；无异常提示。

### B5. GPU 上的单批次过拟合
```bash
python tools/debug_overfit.py --device cuda --iters 200 --height 64 --width 64
python tools/debug_overfit.py --device cuda --iters 400 --kernel-type rgb --kernel-spec B2
python tools/debug_overfit.py --device cuda --iters 400 --loss hybrid --enable-phys
```
- **通过标准：** 三次运行均无 NaN，loss 下降到 ~1e-3 或更低。

### B6. 真实数据小集合（调试配置）
```bash
%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base
python basicsr/train.py -opt ../configs/debug/sid_newbp_mono_debug.yml
python basicsr/train.py -opt ../configs/debug/sid_newbp_rgb_debug.yml
```
- **通过标准：** 几百 iter 内 loss 有下降趋势，无 NaN；日志中的梯度或损失未飘逸。

### B7. 基线模型确认
```bash
python basicsr/train.py -opt ../configs/colab/sid_nafnet_baseline.yml
```
- **通过标准：** 1~2 个 epoch 即可看到稳定下降的 Baseline loss，证明数据与框架无结构性问题。

### B8. 完整实验
```bash
python basicsr/train.py -opt ../configs/colab/sid_newbp_mono.yml
python basicsr/train.py -opt ../configs/colab/sid_newbp_rgb.yml
```
- **通过标准：** 训练过程中无崩溃，TensorBoard 中指标逐渐稳定；迭代时间与 GPU 资源预期一致。

---

## 阶段 C：问题定位参考

| 问题表现 | 重点排查步骤 |
| -------- | ------------ |
| `ModuleNotFoundError` | 重新执行步骤 A2/B3 补齐 `sys.path`，确认 external 仓库存在。 |
| 曝光或文件缺失 | 重跑步骤 A3/B4 的 `tools/debug_dataset.py`，核对 manifest 与目录。 |
| `HybridLossPlus detected non-finite` | 使用步骤 A4 脚本重现，记录抛错项（如 `Phys_raw`），再回查对应张量。 |
| 单批次 loss 不下降 | 在步骤 A5/B5 中调低学习率或缩小网络宽度；检查模型 forward。 |
| 调试配置 NaN | 回到步骤 B6，检查 `physics` 配置是否与 `kernel_type` 对应（Mono-P2 / RGB-B2）。 |
| Baseline 无法收敛 | 说明训练框架或数据仍异常，必须在 B7 处修复后再执行新模型实验。 |

---

依照上述流程，可以先在本地低成本验证各组件，再在 Colab 中逐级放大，确保“能运行”与“是对的”两个目标都得到验证。若任一步失败，请回到最近一次通过的阶段修复后再前进。
