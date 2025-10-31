# 测试执行指南：数据管线 / 配置 / 训练入口 & GPU Parity 用例

## 1. 本地新增集成测试（CPU 环境）

### 1.1 前置条件
- 已安装 `requirements.txt`，并额外安装 `rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics pyyaml`。
- 仓库下存在调试数据：`data/debug_sid/`，包含
  - `manifest_sid_debug.json`
  - `short/`, `long/` 目录（示例 PNG）
  - `train_short_debug.lmdb`, `train_long_debug.lmdb` （若尚未生成，可运行：  
    ```powershell
    python tools/create_sid_lmdb.py ^
      --manifest data/debug_sid/manifest_sid_debug.json ^
      --short-root data/debug_sid/short ^
      --long-root data/debug_sid/long ^
      --output-root data/debug_sid ^
      --compress-level 0
    ```
    将会创建 `train_short_debug.lmdb/`, `train_long_debug.lmdb/`。)

### 1.2 运行测试
```powershell
pytest tests/test_data_pipeline_and_training.py --disable-warnings
```

### 1.3 判定标准
- `test_dataset_loader_debug_assets` 通过：说明 YAML 配置可解析，调试数据能被 DataLoader 正常读取。
- `test_training_entrypoint_smoke` 通过：表示 `create_model` + 训练循环可跑至少 5 个 iteration，核心日志项返回正常。
- 若出现 `pytest.skip`，根据提示补齐调试 LMDB 或依赖后重试；任何失败都应先修复再进入实际训练。

---

## 2. Colab（GPU）执行整体测试与 GPU parity 用例

### 2.1 环境准备
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

%cd /content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement
!pip install -r requirements.txt
!pip install rawpy kornia lpips lmdb tqdm opencv-python-headless torchmetrics
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
python tools/create_sid_lmdb.py \
  --manifest data/debug_sid/manifest_sid_debug.json \
  --short-root data/debug_sid/short \
  --long-root data/debug_sid/long \
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
| 缺少 LMDB 导致测试跳过 | 使用调试 PNG + `tools/create_sid_lmdb.py` 生成小型 LMDB；或在自定义配置中改为磁盘读取。 |
| `ModuleNotFoundError` | 重新执行依赖安装步骤，确认 `sys.path` 已补充外部仓库。 |
| AMP 测试报错 | 确保 Notebook 选择的是 GPU Runtime，并安装匹配版本的 PyTorch / CUDA。 |
| 日志路径权限问题 | `tests/test_data_pipeline_and_training.py` 会将 `experiments_root` 指向 `logs/pytest_smoke_*`，若需自定义目录可在 `_patch_debug_paths` 里调整。 |
