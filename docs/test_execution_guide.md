# 测试执行指南：数据管线 / 配置 / 训练入口 & GPU Parity 用例

## 1. 本地新增集成测试（CPU 环境，PowerShell）

### 1.0 建议的虚拟环境与依赖安装

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics pyyaml pillow
```

验证：

```powershell
python - <<'PY'
import torch, rawpy, kornia, lmdb, yaml, PIL
print('Torch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
PY
```

### 1.1 前置条件

- 已安装 `requirements.txt`，并额外安装 `rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics pyyaml pillow`。
- 仓库下存在调试数据：`data/debug_sid/`，包含：
  - `manifest_sid_debug.json`
  - `short/`, `long/` 目录（示例 PNG）
  - subset LMDB：运行下方脚本后会生成形如 `train_small_short.lmdb/`, `train_small_long.lmdb/`（以及可选的 `val_small_*`）。测试代码会自动发现并优先使用包含 "train" 的 subset；若存在验证 subset 也将自动绑定，无需手动重命名。

    ```powershell
    python NAFNet_base\tools\create_sid_lmdb.py `
      --manifest data/debug_sid/manifest_sid_debug.json `
      --short-root data/debug_sid/short `
      --long-root  data/debug_sid/long `
      --output-root data/debug_sid `
      --compress-level 0
    ```

  （可选兼容旧名：若你仍希望使用历史固定名称 `train_short_debug.lmdb/` 与 `train_long_debug.lmdb/`，可自行复制一份。）
  若出现 `ModuleNotFoundError: basicsr`，请确认在仓库根执行并已安装依赖；新版脚本已自动添加路径，仍报错时可手动运行：

  ```powershell
  $root = (Get-Location).Path
  $paths = @($root, "$root\NAFNet_base", "$root\NAFNet_base\basicsr", "$root\NewBP_model")
  foreach ($p in $paths) { if (-not ($env:PYTHONPATH -like "*$p*")) { $env:PYTHONPATH = "$p;$env:PYTHONPATH" } }
  ```

### 1.2 运行测试

```powershell
python -m pytest tests/test_data_pipeline_and_training.py --disable-warnings
```

### 1.3 判定标准

- `test_dataset_loader_debug_assets` 通过：说明 YAML 配置可解析，调试数据能被 DataLoader 正常读取。
- `test_training_entrypoint_smoke` 通过：表示 `create_model` + 训练循环可跑至少 5 个 iteration，核心日志项返回正常。
- 若出现 `pytest.skip`，根据提示补齐调试 LMDB 或依赖后重试；任何失败都应先修复再进入实际训练。

---

### 1.4 最小闭环（本地快速执行顺序）

按顺序运行以下命令，确保本地链路“能跑通”：

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

## 2. Colab（GPU）执行整体测试与 GPU parity 用例

### 2.1 环境准备

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics pillow
```

确保 `sys.path` 补充：

```python
import sys
extra = [
  "/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement",
  "/content/drive/MyDrive/Lowlight/SID_experiments/external/NAFNet",
  "/content/drive/MyDrive/Lowlight/SID_experiments/external/SwinIR",
]
for p in extra:
  if p not in sys.path:
    sys.path.append(p)
```

### 2.2 数据准备（一次性执行）

若尚无调试 LMDB，可通过小样本生成：

```bash
python NAFNet_base/tools/create_sid_lmdb.py \
  --manifest data/debug_sid/manifest_sid_debug.json \
  --short-root data/debug_sid/short \
  --long-root  data/debug_sid/long \
  --output-root data/debug_sid \
  --compress-level 0
```

### 2.3 运行新增集成测试

```bash
pytest tests/test_data_pipeline_and_training.py --disable-warnings
```

在 GPU 环境执行可确认 CUDA + 调试数据同样稳定。

### 2.4 GPU parity / AMP 用例

依次运行此前在 CPU 环境被跳过的测试，验证 CPU/GPU 结果一致性：

```bash
pytest standard_tests/test_channelwise.py::test_cpu_cuda_parity --disable-warnings
pytest standard_tests/test_color_error.py::test_deltae_cpu_cuda_parity --disable-warnings
pytest standard_tests/test_phys_consistency.py::test_cpu_cuda_parity --disable-warnings
pytest standard_tests/test_lpips_wrapper.py::test_amp_autocast_stability --disable-warnings
pytest standard_tests/test_lpips_wrapper.py::test_cpu_cuda_parity --disable-warnings
pytest core_tests/test_integration_forward_amp.py --disable-warnings
```


### 2.5 完整回归（可选）

```bash
pytest standard_tests --disable-warnings
pytest core_tests --disable-warnings
```

- 输出中应无 `skipped`（表示 GPU parity 已执行），所有测试通过即可证明指标、损失、Scenario-B、AMP 逻辑在 GPU 下与 CPU 保持一致。

---

## 3. 常见问题

| 问题 | 处理建议 |
| ---- | -------- |
| 缺少 LMDB 导致测试跳过 | 使用调试 PNG + `NAFNet_base/tools/create_sid_lmdb.py` 生成小型 LMDB；或在自定义配置中改为磁盘读取。 |


## 附录：本地 CPU 与 Colab GPU 测试执行矩阵

| 目标 | 本地 CPU（PowerShell） | Colab GPU |
| ---- | --------------------- | --------- |
| 数据与路径冒烟 | `python tools/debug_dataset.py`（小样 + LMDB） | 大全集合参数（B4） |
| 损失稳定性 | `python tools/debug_losses.py --device cpu` | `python tools/debug_losses.py --device cuda --amp`（可选） |
| 单批过拟合 | `python tools/debug_overfit.py --device cpu` | `python tools/debug_overfit.py --device cuda --enable-phys` |
| 集成入口冒烟 | `pytest tests/test_data_pipeline_and_training.py -k smoke` | 同脚本 + CUDA 验证 |
| 指标/物理 parity | （跳过或受限） | `pytest standard_tests/* parity 用例` |
| AMP forward | （无 GPU） | `pytest core_tests/test_integration_forward_amp.py` |
| 完整训练 | 不执行 | `basicsr/train.py -opt configs/colab/*.yml` |

最小闭环顺序（本地）：依赖安装 → 数据小样 + LMDB → debug_losses → debug_overfit → smoke pytest。全部通过后再迁移 GPU。
| `ModuleNotFoundError` | 重新执行依赖安装步骤，确认 `sys.path` 已补充外部仓库。 |
| AMP 测试报错 | 确保 Notebook 选择的是 GPU Runtime，并安装匹配版本的 PyTorch / CUDA。 |
| 日志路径权限问题 | `tests/test_data_pipeline_and_training.py` 会将 `experiments_root` 指向 `logs/pytest_smoke_*`，若需自定义目录可在 `_patch_debug_paths` 里调整。 |
