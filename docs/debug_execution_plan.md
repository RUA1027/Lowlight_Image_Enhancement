# NewBP 项目分阶段调试与验证计划（本地 PowerShell + Colab）

> 核心目标：严格区分“能运行”(运行链路可通) 与 “是正确”(数值/逻辑稳定)。先在本地 CPU + PowerShell 环境完成最小闭环，再迁移到 Colab GPU 放大。文档按依赖梯度递增编排：环境 → 结构 → 路径 → 数据 → 损失 → 过拟合 → GPU 放大 → 全量训练。所有命令已针对本地 Windows PowerShell (`pwsh`) 语法适配；Colab 部分保持 Notebook/bash 形式。

---

## 阶段 A：本地（CPU, PowerShell）快速自检

> 建议在仓库根目录 `D:\Lowlight_image_enhancement` 中执行。若使用 VS Code 终端，确认当前 shell 为 `pwsh`。阶段内任何一步失败，都不要继续后续步骤，先修复再前进。

### A0. 环境准备（虚拟环境 + 依赖）

- **命令（PowerShell）：**

  ```powershell
  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  python -m pip install rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics pyyaml pillow
  ```

- **补充校验：**

  ```powershell
  python - <<'PY'
  import torch, sys
  print('Python', sys.version)
  print('Torch version:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
  PY
  ```

- **通过标准：** 无安装错误；上述脚本输出版本信息且无异常；允许 CUDA 显示为 False（本地 CPU 调试）。

### A0.1 可选：基础随机性与确定性冒烟

```powershell
python - <<'PY'
import torch, random, numpy as np
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
print('Seeded tensor mean:', torch.randn(1000).mean().item())
PY
```

通过标准：脚本运行无报错；为后续排查非收敛问题提供基准（记录该 mean 便于异常比对）。

### A1. 代码结构快照

- **命令：**

  ```powershell
  Get-ChildItem -Recurse -Depth 1
  ```

- **检查点：** 看到 `configs/`, `NewBP_model/`, `tools/`、`docs/` 等目录存在；确认 `tools/debug_overfit.py`、`tools/debug_dataset.py`、`tools/debug_losses.py` 已就绪。

### A2. Python 搜索路径验证（import 视角而非环境变量）

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

- **若有 False：** 临时补救（当前 session 生效）：

  ```powershell
  $repo = (Get-Location).Path
  $extra = @(
    "$repo", "$repo\NAFNet_base\basicsr", "$repo\datasets", "$repo\tools"
  )
  $env:PYTHONPATH = ($extra -join ';')
  ```

  永久方案：在仓库根创建 `sitecustomize.py`：

  ```python
  # sitecustomize.py
  import sys, pathlib
  root = pathlib.Path(__file__).resolve().parent
  for rel in ["NAFNet_base/basicsr", "datasets", "tools", "NewBP_model"]:
      p = str(root / rel)
      (p not in sys.path) and sys.path.insert(0, p)
  ```

### A2.1 可选：列出潜在冲突包

```powershell
python - <<'PY'
import pkgutil, re
hits = [n for _, n, _ in pkgutil.iter_modules() if re.search(r'(lpips|rawpy|kornia)', n)]
print('Found related packages:', hits)
PY
```

用于确认是否存在多版本冲突（通常仅显示 1 次）。

### A3. 调试数据与 LMDB（本地小样）

- 建议目录：`data/debug_sid/` 包含以下内容：
  - `manifest_sid_debug.json`
  - `short/`, `long/` 示例 PNG（若没有，可先放入 3~5 对小样）
  - 如需生成 LMDB（推荐，便于与测试对齐）：

  ```powershell
  python NAFNet_base\tools\create_sid_lmdb.py `
    --manifest data/debug_sid/manifest_sid_debug.json `
    --short-root data/debug_sid/short `
    --long-root  data/debug_sid/long `
    --output-root data/debug_sid `
    --compress-level 0
  ```

- 说明：该脚本会按照 manifest 中的 `subset` 字段生成 `{subset}_short.lmdb/` 与 `{subset}_long.lmdb/`。
  - 例如当前 `manifest_sid_debug.json` 含有 `train_small`/`val_small` 两个 subset，则会生成：
    - `data/debug_sid/train_small_short.lmdb/` 与 `data/debug_sid/train_small_long.lmdb/`
    - `data/debug_sid/val_small_short.lmdb/` 与 `data/debug_sid/val_small_long.lmdb/`
  - 集成测试现已支持自动发现 subset 命名（如 `train_small_short.lmdb/` 和 `val_small_short.lmdb/`），无需强制重命名。
    如你更偏好旧名（`train_short_debug.lmdb/` 与 `train_long_debug.lmdb/`），可按下述方式复制一份（可选）：

  ```powershell
  # 以 train_small 为例，重命名/复制为测试期望的名称（任选其一）
  Copy-Item data/debug_sid/train_small_short.lmdb data/debug_sid/train_short_debug.lmdb -Recurse
  Copy-Item data/debug_sid/train_small_long.lmdb  data/debug_sid/train_long_debug.lmdb  -Recurse
  ```

  若执行脚本时报 `ModuleNotFoundError: basicsr` 或 `NewBP_model`，请先执行一次路径引导脚本（已集成在工具中）或确认在仓库根目录激活虚拟环境后再运行；新版脚本已自动注入 `NAFNet_base/basicsr` 与项目根到 `sys.path`。

### A3.1 数据链路快速检查

```powershell
# 方式一：使用“测试期望旧名”（若已复制）
python tools/debug_dataset.py `
  --manifest data/debug_sid/manifest_sid_debug.json `
  --short-root data/debug_sid/short `
  --long-root  data/debug_sid/long `
  --lmdb-short data/debug_sid/train_short_debug.lmdb `
  --lmdb-long  data/debug_sid/train_long_debug.lmdb `
  --limit 3

# 方式二：直接使用按 subset 生成的 LMDB（无需复制）
python tools/debug_dataset.py `
  --manifest data/debug_sid/manifest_sid_debug.json `
  --short-root data/debug_sid/short `
  --long-root  data/debug_sid/long `
  --lmdb-short data/debug_sid/train_small_short.lmdb `
  --lmdb-long  data/debug_sid/train_small_long.lmdb `
  --limit 3
```

通过标准：每对样本均报告文件存在、曝光比例合理、LMDB key 命中；无异常。若你只生成了训练 subset（无 val），本项目测试也能自动使用训练 subset 作为验证（已在测试逻辑中兼容）。

### A4. 损失函数冒烟（HybridLossPlus）

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

### A6. 冒烟测试（pytest 精简用例，非性能）

```powershell
python -m pytest tests/test_data_pipeline_and_training.py -k smoke --maxfail=1 --disable-warnings
```

通过标准：关键用例通过（若有 skip 按提示补齐数据）。此步骤让后续迁移 Colab 前就发现入口/配置层面问题。

> **阶段 A 完成标准：** A0~A5 必须通过；A6 推荐执行。否则进入 GPU 阶段风险高。

### A7. 最小闭环（一次跑通即可）

按顺序执行以下 4 条命令，确保本地链路“能跑通”：

```powershell
# 1) 损失冒烟（数值稳定）
python tools/debug_losses.py --device cpu --steps 2 --height 64 --width 64

# 2) 单批过拟合（学习过程可行）
python tools/debug_overfit.py --device cpu --iters 60 --height 64 --width 64 --log-interval 20

# 3) 若尚未创建调试 LMDB，则创建
python NAFNet_base\tools\create_sid_lmdb.py `
  --manifest data/debug_sid/manifest_sid_debug.json `
  --short-root data/debug_sid/short `
  --long-root  data/debug_sid/long `
  --output-root data/debug_sid `
  --compress-level 0

# 4) 用例冒烟（入口/配置正确）
python -m pytest tests/test_data_pipeline_and_training.py -k smoke --maxfail=1 --disable-warnings
```

全部成功后再进入阶段 B 放大到 GPU。

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
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics pillow
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

**通过标准：** 每个 pair 显示图像存在且 LMDB key 命中；无异常提示。

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

### B9. 可选：AMP 与多损失组合稳定性回归

```bash
python tools/debug_overfit.py --device cuda --iters 120 --loss hybrid --amp --enable-phys
```

通过标准：AMP 下 loss 单调或阶梯式下降，无 `non-finite` 报错。

---

## 阶段 C：问题定位参考

| 问题表现 | 重点排查步骤 |
| -------- | ------------ |
| `ModuleNotFoundError` | 回到 A2（本地）或 B3（Colab）补齐路径；检查是否遗漏 `sitecustomize.py`。 |
| 曝光或文件缺失 | 重跑 A3/B4 数据检查；复核 manifest 字段与真实文件名大小写。 |
| 无 LMDB 被跳过 | 执行 A3.1 或在 Colab 生成临时小样本 LMDB。 |
| `HybridLossPlus detected non-finite` | 在 A4/B5 添加 `--enable-phys` 缩小尺寸复现；打印触发分支张量统计。 |
| 单批次 loss 不下降 | 检查学习率、随机种子、模型初始化；使用 A0.1 记录的随机张量特征对比是否异常。 |
| 调试配置 NaN | 回到 B6，核对 `kernel_type` 与 `physics` 参数是否匹配（Mono-P2 / RGB-B2）。 |
| Baseline 无法收敛 | 优先执行 B7 以验证框架基础；若仍失败检查数据加载批次是否随机化。 |
| AMP 报错或非收敛 | 运行 B9；确认 PyTorch 版本与 Colab CUDA 对应；尝试关闭某单项物理损失定位。 |
| Windows 下路径过长 | 减少嵌套或在仓库根启用 `git config core.longpaths true`。 |
| 虚拟环境未激活导致依赖缺失 | 重新执行 A0 并确认提示 `( .venv )` 前缀。 |

---

## 附录：本地 CPU vs Colab GPU 测试矩阵（建议）

- 本地 CPU（优先）：
  - tools/debug_losses.py（A4）
  - tools/debug_overfit.py（A5）
  - tools/debug_dataset.py（A3.1）
  - tests/test_data_pipeline_and_training.py -k smoke（A6）
- Colab GPU（放大与对齐）：
  - B5 单批次过拟合（cuda/amp/物理项组合）
  - B6 调试配置训练（debug yaml）
  - B7 Baseline 训练以验证框架稳定
  - B8 完整实验
  - GPU/AMP 回归：standard_tests 与 core_tests（在 Colab 运行，确保 parity 与稳定性）

---

## 语法与环境差异提示

1. 本地 PowerShell 使用反引号 `` ` `` 作为行继续符；原文中的 `^` 已改为 `` ` ``。不要混用。
2. Colab Notebook 中保留 `!`、`%cd`、反斜杠续行（bash），不需要改写。
3. 若未来迁移到 WSL 或 GitHub Actions，可直接采用 B 阶段 bash 语法。

---

依照上述调整后的流程，先在本地低成本验证，再在 Colab 中逐级放大，最大化减少“跑满大实验才发现基础损坏”的风险。出现异常时就近回溯到最近完全通过的步骤重新定位。
