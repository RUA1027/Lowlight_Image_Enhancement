# 跨平台部署适配说明

本文档说明了为支持 Linux GPU 环境（如 AutoDL）所做的修改和适配工作。

## 修改概览

### 1. NewBP 模型配置文件更新

**修改的文件**:
- `configs/colab/sid_newbp_mono.yml`
- `configs/colab/sid_newbp_rgb.yml`
- `configs/colab/sid_nafnet_baseline.yml`
- `configs/colab/sid_unet_baseline.yml`
- `configs/colab/sid_swinir_baseline.yml`

**修改内容**:
所有硬编码的路径（如 `/content/drive/MyDrive/Lowlight/...`）已替换为环境变量 `${SID_ROOT}`，支持灵活的路径配置。

**修改前**:
```yaml
manifest_path: /content/drive/MyDrive/Lowlight/SID_assets/manifest_sid.json
db_paths:
  - /content/drive/MyDrive/Lowlight/SID_lmdb/train_short.lmdb
  - /content/drive/MyDrive/Lowlight/SID_lmdb/train_long.lmdb
```

**修改后**:
```yaml
manifest_path: ${SID_ROOT}/SID_assets/manifest_sid.json
db_paths:
  - ${SID_ROOT}/SID_lmdb/train_short.lmdb
  - ${SID_ROOT}/SID_lmdb/train_long.lmdb
```

### 2. NewBP 数据集模块更新

**修改的文件**:
- `NAFNet_base/basicsr/data/sony_sid_lmdb_dataset.py`

**修改内容**:
- 添加 `os.path.expandvars()` 支持环境变量扩展
- 使用 `pathlib.Path` 确保跨平台路径兼容
- 所有路径操作已适配 Linux 环境

**关键代码**:
```python
# 支持环境变量
manifest_path_str = opt["manifest_path"]
manifest_path_str = os.path.expandvars(manifest_path_str)
manifest_path = Path(manifest_path_str)

# 扩展 LMDB 路径中的环境变量
db_paths = [os.path.expandvars(str(p)) for p in db_paths]
```

### 3. NAFNet 基线模型适配

**新增文件**:
- `SID_experiments/external/NAFNet/basicsr/data/sony_sid_lmdb_dataset.py`
- `SID_experiments/external/NAFNet/options/train/SID/NAFNet-SID-baseline.yml`
- `SID_experiments/external/NAFNet/options/test/SID/NAFNet-SID-baseline.yml`

**修改内容**:
为原始 NAFNet 创建了完整的 SID 数据集适配器，支持：
- 从 LMDB 读取 RAW 格式图像（已转换为 16-bit PNG）
- 自动曝光对齐
- 随机裁剪和数据增强
- 与 BasicSR 训练框架集成

### 4. 部署文档和脚本

**新增文件**:
- `docs/autodl_deployment_guide.md` - 详细的 AutoDL 部署指南
- `setup_autodl.sh` - 自动化环境配置脚本
- `docs/path_configuration_example.sh` - 路径配置示例

---

## 跨平台兼容性保证

### 路径处理
所有路径相关代码已确保：
1. ✅ 使用 `pathlib.Path` 进行路径操作（自动处理 `/` vs `\`）
2. ✅ 支持环境变量扩展（`${VAR}` 和 `$VAR`）
3. ✅ 使用 `os.path.expandvars()` 和 `Path.expanduser()`
4. ✅ 避免硬编码的绝对路径

### 数据加载
1. ✅ LMDB 数据库路径支持环境变量
2. ✅ Manifest JSON 文件路径支持环境变量
3. ✅ 所有文件 I/O 使用二进制模式，避免行结束符问题
4. ✅ 使用 UTF-8 编码读取文本文件

### 模块导入
1. ✅ PYTHONPATH 环境变量配置
2. ✅ 相对导入路径已验证
3. ✅ 所有自定义模块可正常导入

---

## 使用指南

### 在 AutoDL 上部署

#### 快速开始（推荐）

```bash
# 1. 上传数据到 AutoDL
# 将整个 Lowlight 文件夹上传到 /root/autodl-tmp/

# 2. 运行自动配置脚本
cd /root/autodl-tmp/Lowlight/SID_experiments/Lowlight_Image_Enhancement
chmod +x setup_autodl.sh
bash setup_autodl.sh

# 3. 激活环境
source ~/.bashrc

# 4. 开始训练
python NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_mono.yml --launcher none
```

#### 手动配置

```bash
# 1. 设置环境变量
export SID_ROOT="/root/autodl-tmp/Lowlight"
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement:${PYTHONPATH}"
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:${PYTHONPATH}"

# 2. 写入 bashrc（可选，用于持久化）
echo 'export SID_ROOT="/root/autodl-tmp/Lowlight"' >> ~/.bashrc
echo 'export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement:${PYTHONPATH}"' >> ~/.bashrc
source ~/.bashrc

# 3. 安装依赖
cd $SID_ROOT/SID_experiments/Lowlight_Image_Enhancement
pip install -r requirements.txt

# 4. 验证环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import os; print(f'SID_ROOT: {os.environ[\"SID_ROOT\"]}')"

# 5. 开始训练
python NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_mono.yml --launcher none
```

### 在其他 Linux 服务器上部署

只需修改 `SID_ROOT` 环境变量为实际的数据路径：

```bash
export SID_ROOT="/path/to/your/data"
```

其他步骤保持一致。

---

## 训练命令参考

### NewBP Mono 模型

```bash
# 单 GPU
python NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_mono.yml --launcher none

# 多 GPU
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 \
    NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_mono.yml --launcher pytorch
```

### NewBP RGB 模型

```bash
python NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_rgb.yml --launcher none
```

### NAFNet 基线

```bash
cd $SID_ROOT/SID_experiments/external/NAFNet
python basicsr/train.py -opt options/train/SID/NAFNet-SID-baseline.yml --launcher none
```

---

## 验证清单

在开始训练前，请确认：

- [ ] `$SID_ROOT` 环境变量已正确设置
- [ ] `$PYTHONPATH` 包含项目路径
- [ ] LMDB 文件存在且可访问
- [ ] manifest_sid.json 文件存在
- [ ] PyTorch 和 CUDA 正常工作
- [ ] 所有依赖包已安装
- [ ] 有足够的磁盘空间（至少 100GB）
- [ ] GPU 可用且显存充足（建议 24GB+）

---

## 常见问题

### 1. FileNotFoundError: Manifest file not found

**原因**: 环境变量未设置或路径错误

**解决**:
```bash
export SID_ROOT="/root/autodl-tmp/Lowlight"
ls -l $SID_ROOT/SID_assets/manifest_sid.json
```

### 2. ModuleNotFoundError

**原因**: PYTHONPATH 未正确设置

**解决**:
```bash
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement:${PYTHONPATH}"
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:${PYTHONPATH}"
```

### 3. LMDB 读取错误

**原因**: 数据库文件损坏或权限问题

**解决**:
```bash
chmod -R 755 $SID_ROOT/SID_lmdb/
# 验证数据库
python -c "import lmdb; env = lmdb.open('$SID_ROOT/SID_lmdb/train_short.lmdb', readonly=True); print('OK'); env.close()"
```

---

## 技术细节

### 环境变量扩展机制

配置文件中的 `${VAR}` 会在运行时被 Python 的 `os.path.expandvars()` 函数替换：

```python
# 在 sony_sid_lmdb_dataset.py 中
manifest_path_str = opt["manifest_path"]  # "${SID_ROOT}/SID_assets/manifest_sid.json"
manifest_path_str = os.path.expandvars(manifest_path_str)  # "/root/autodl-tmp/Lowlight/SID_assets/manifest_sid.json"
```

### 跨平台路径处理

使用 `pathlib.Path` 确保路径分隔符自动适配：

```python
from pathlib import Path
manifest_path = Path(manifest_path_str)  # 自动处理 / 和 \
```

### LMDB 路径兼容

```python
db_paths = [os.path.expandvars(str(p)) for p in db_paths]
```

---

## 更新历史

- **2025-11-16**: 初始版本
  - 更新所有配置文件使用环境变量
  - 为 NAFNet 添加 SID 数据集支持
  - 创建 AutoDL 部署指南和自动化脚本
  - 确保所有路径处理跨平台兼容

---

## 相关文档

- [AutoDL 部署指南](docs/autodl_deployment_guide.md)
- [路径配置示例](docs/path_configuration_example.sh)
- [NewBP Mono 训练文档](docs/colab_sid_newbp_mono.md)
- [NewBP RGB 训练文档](docs/colab_sid_newbp_rgb.md)
- [NAFNet 基线文档](docs/colab_sid_nafnet.md)

---

**作者**: AI Assistant  
**日期**: 2025年11月16日
