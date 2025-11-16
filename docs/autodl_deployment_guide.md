# AutoDL GPU 部署指南

本文档详细说明如何在 AutoDL 平台（或其他 Linux GPU 环境）上部署和运行 Lowlight Image Enhancement 项目。

## 目录
- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [环境配置](#环境配置)
- [模型训练](#模型训练)
- [常见问题](#常见问题)

---

## 环境要求

### 硬件要求
- GPU: RTX 5090 或其他支持 CUDA 的 GPU
- 内存: 至少 32GB RAM
- 存储: 至少 100GB 可用空间（用于数据集和模型）

### 软件要求
- 操作系统: Linux (Ubuntu 20.04+ 推荐)
- Python: 3.8 或 3.9
- CUDA: 11.8 或更高版本
- PyTorch: 2.0 或更高版本

---

## 数据准备

### 1. 上传数据到 AutoDL

将以下文件夹上传到 AutoDL 实例：

```
/root/autodl-tmp/Lowlight/
├── SID_assets/
│   ├── manifest_sid.json
│   └── manifest_sid_small.json
├── SID_lmdb/
│   ├── train_short.lmdb/
│   ├── train_long.lmdb/
│   ├── val_short.lmdb/
│   ├── val_long.lmdb/
│   ├── test_short.lmdb/
│   └── test_long.lmdb/
├── SID_raw/  # 可选，如果需要重新生成 LMDB
│   └── Sony/
│       ├── long/
│       └── short/
└── SID_experiments/
    └── Lowlight_Image_Enhancement/
```

**建议路径**: `/root/autodl-tmp/Lowlight`（AutoDL 默认数据目录）

### 2. 验证数据完整性

```bash
# 检查 LMDB 文件
ls -lh /root/autodl-tmp/Lowlight/SID_lmdb/

# 验证 manifest 文件
cat /root/autodl-tmp/Lowlight/SID_assets/manifest_sid.json | head -20
```

---

## 环境配置

### 1. 设置环境变量

在 `~/.bashrc` 或训练脚本开头添加：

```bash
# 设置项目根目录
export SID_ROOT="/root/autodl-tmp/Lowlight"

# 设置 Python 路径（用于导入自定义模块）
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement:${PYTHONPATH}"
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:${PYTHONPATH}"

# 使环境变量生效
source ~/.bashrc
```

### 2. 安装依赖

#### 对于 NewBP 模型（Lowlight_Image_Enhancement）

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement

# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 安装 RAW 图像处理库（如果需要）
pip install rawpy

# 安装 LMDB
pip install lmdb
```

#### 对于 NAFNet 基线模型

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/external/NAFNet

# 安装依赖
pip install -r requirements.txt
pip install lmdb opencv-python
```

### 3. 验证环境

```bash
# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查环境变量
echo $SID_ROOT
echo $PYTHONPATH

# 测试数据加载
python -c "
import os
from pathlib import Path
manifest_path = Path(os.environ['SID_ROOT']) / 'SID_assets' / 'manifest_sid.json'
print(f'Manifest exists: {manifest_path.exists()}')
"
```

---

## 模型训练

### NewBP Mono 模型

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement

# 单 GPU 训练
python NAFNet_base/basicsr/train.py \
    -opt configs/colab/sid_newbp_mono.yml \
    --launcher none

# 多 GPU 训练（如果有多个 GPU）
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    NAFNet_base/basicsr/train.py \
    -opt configs/colab/sid_newbp_mono.yml \
    --launcher pytorch
```

### NewBP RGB 模型

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement

# 单 GPU 训练
python NAFNet_base/basicsr/train.py \
    -opt configs/colab/sid_newbp_rgb.yml \
    --launcher none
```

### NAFNet 基线模型

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/external/NAFNet

# 训练
python basicsr/train.py \
    -opt options/train/SID/NAFNet-SID-baseline.yml \
    --launcher none
```

### 监控训练进度

```bash
# 使用 tensorboard 查看训练曲线
tensorboard --logdir=/root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement/logs --port=6006

# 在本地浏览器访问：
# http://<your-autodl-instance-ip>:6006
```

---

## 测试和评估

### NewBP 模型测试

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement

# 测试模型
python NAFNet_base/basicsr/test.py \
    -opt configs/colab/sid_newbp_mono.yml
```

### NAFNet 基线测试

```bash
cd /root/autodl-tmp/Lowlight/SID_experiments/external/NAFNet

# 测试
python basicsr/test.py \
    -opt options/test/SID/NAFNet-SID-baseline.yml
```

---

## 常见问题

### 1. 路径相关错误

**错误**: `FileNotFoundError: Manifest file not found`

**解决方案**:
```bash
# 确认环境变量设置正确
echo $SID_ROOT

# 如果未设置，手动设置
export SID_ROOT="/root/autodl-tmp/Lowlight"

# 验证文件存在
ls -l $SID_ROOT/SID_assets/manifest_sid.json
```

### 2. LMDB 读取错误

**错误**: `lmdb.Error: /path/to/database.lmdb: No such file or directory`

**解决方案**:
```bash
# 检查 LMDB 文件权限
chmod -R 755 $SID_ROOT/SID_lmdb/

# 验证 LMDB 文件完整性
python -c "
import lmdb
env = lmdb.open('$SID_ROOT/SID_lmdb/train_short.lmdb', readonly=True, lock=False)
with env.begin() as txn:
    print(f'Total keys: {txn.stat()[\"entries\"]}')
env.close()
"
```

### 3. GPU 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
在配置文件中调整以下参数：
- 减小 `batch_size_per_gpu`: `2` → `1`
- 减小 `patch_size`: `384` → `256`
- 关闭混合精度训练: `enable_amp: false`

### 4. 模块导入错误

**错误**: `ModuleNotFoundError: No module named 'basicsr'`

**解决方案**:
```bash
# 设置 PYTHONPATH
export PYTHONPATH="/root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:${PYTHONPATH}"

# 或在 Python 脚本开头添加
import sys
sys.path.insert(0, '/root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base')
```

### 5. 权限问题

**错误**: `Permission denied`

**解决方案**:
```bash
# 修改文件所有权
sudo chown -R $(whoami):$(whoami) /root/autodl-tmp/Lowlight

# 添加执行权限
chmod +x /root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base/basicsr/train.py
```

---

## 高级配置

### 使用自定义路径

如果数据存储在不同位置，修改环境变量：

```bash
# 自定义路径
export SID_ROOT="/path/to/your/data"
export MANIFEST_PATH="${SID_ROOT}/SID_assets/manifest_sid.json"
export LMDB_ROOT="${SID_ROOT}/SID_lmdb"
```

### 断点续训

在配置文件中设置：

```yaml
path:
  resume_state: /path/to/checkpoint/training_state.pth
```

或在命令行中：

```bash
python NAFNet_base/basicsr/train.py \
    -opt configs/colab/sid_newbp_mono.yml \
    --resume_state /path/to/checkpoint.pth
```

---

## 性能优化建议

1. **使用 CUDA Prefetcher**: 在配置中设置 `prefetch_mode: cuda`
2. **增加 Workers**: 根据 CPU 核心数调整 `num_worker_per_gpu`
3. **混合精度训练**: 保持 `enable_amp: true`（节省显存）
4. **梯度累积**: 如果显存不足，可以使用梯度累积技术

---

## 联系与支持

如遇到其他问题，请检查：
1. 项目 README.md
2. 相关配置文件注释
3. 日志文件：`logs/SID_NewBP_Mono/train_*.log`

---

**最后更新**: 2025年11月16日
