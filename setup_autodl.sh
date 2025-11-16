#!/bin/bash

# AutoDL 环境初始化脚本
# 用于快速配置 Lowlight Image Enhancement 项目环境

set -e  # 遇到错误立即退出

echo "============================================"
echo "Lowlight Image Enhancement 环境初始化"
echo "============================================"

# 1. 设置环境变量
echo ""
echo "[1/5] 设置环境变量..."

# 检测项目根目录
if [ -d "/root/autodl-tmp/Lowlight" ]; then
    SID_ROOT="/root/autodl-tmp/Lowlight"
elif [ -d "$HOME/Lowlight" ]; then
    SID_ROOT="$HOME/Lowlight"
else
    echo "错误: 未找到项目目录，请指定 SID_ROOT 路径"
    echo "用法: SID_ROOT=/your/path bash setup_autodl.sh"
    exit 1
fi

echo "项目根目录: $SID_ROOT"

# 导出环境变量
export SID_ROOT="$SID_ROOT"
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement:${PYTHONPATH}"
export PYTHONPATH="${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:${PYTHONPATH}"
export PYTHONPATH="${SID_ROOT}/SID_experiments/external/NAFNet:${PYTHONPATH}"

# 写入 bashrc 以便持久化
if ! grep -q "SID_ROOT" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Lowlight Image Enhancement 环境变量" >> ~/.bashrc
    echo "export SID_ROOT=\"$SID_ROOT\"" >> ~/.bashrc
    echo "export PYTHONPATH=\"\${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement:\${PYTHONPATH}\"" >> ~/.bashrc
    echo "export PYTHONPATH=\"\${SID_ROOT}/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:\${PYTHONPATH}\"" >> ~/.bashrc
    echo "export PYTHONPATH=\"\${SID_ROOT}/SID_experiments/external/NAFNet:\${PYTHONPATH}\"" >> ~/.bashrc
    echo "环境变量已写入 ~/.bashrc"
else
    echo "环境变量已存在于 ~/.bashrc"
fi

# 2. 验证数据完整性
echo ""
echo "[2/5] 验证数据完整性..."

# 检查必需的目录和文件
REQUIRED_PATHS=(
    "$SID_ROOT/SID_assets/manifest_sid.json"
    "$SID_ROOT/SID_lmdb/train_short.lmdb"
    "$SID_ROOT/SID_lmdb/train_long.lmdb"
    "$SID_ROOT/SID_lmdb/val_short.lmdb"
    "$SID_ROOT/SID_lmdb/val_long.lmdb"
    "$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement"
)

for path in "${REQUIRED_PATHS[@]}"; do
    if [ -e "$path" ]; then
        echo "✓ $path"
    else
        echo "✗ 缺失: $path"
        echo "警告: 部分数据文件缺失，可能影响训练"
    fi
done

# 3. 检查 Python 环境
echo ""
echo "[3/5] 检查 Python 环境..."

python --version
echo "CUDA 可用性检查..."
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 4. 安装缺失的依赖
echo ""
echo "[4/5] 检查依赖包..."

REQUIRED_PACKAGES=(
    "torch"
    "torchvision"
    "numpy"
    "opencv-python"
    "pillow"
    "pyyaml"
    "lmdb"
    "tqdm"
    "tensorboard"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    python -c "import $package" 2>/dev/null && echo "✓ $package" || echo "✗ 需要安装: $package"
done

echo ""
read -p "是否安装缺失的依赖? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "安装依赖中..."
    cd "$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement"
    pip install -r requirements.txt -q
    echo "依赖安装完成"
fi

# 5. 创建必要的目录
echo ""
echo "[5/5] 创建工作目录..."

WORK_DIRS=(
    "$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement/logs"
    "$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement/results"
    "$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement/experiments"
    "$SID_ROOT/SID_experiments/external/NAFNet/experiments"
)

for dir in "${WORK_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "✓ 创建: $dir"
    else
        echo "✓ 已存在: $dir"
    fi
done

# 完成
echo ""
echo "============================================"
echo "环境初始化完成！"
echo "============================================"
echo ""
echo "下一步操作："
echo "1. 激活环境变量:"
echo "   source ~/.bashrc"
echo ""
echo "2. 开始训练 NewBP Mono 模型:"
echo "   cd \$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement"
echo "   python NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_mono.yml --launcher none"
echo ""
echo "3. 开始训练 NewBP RGB 模型:"
echo "   cd \$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement"
echo "   python NAFNet_base/basicsr/train.py -opt configs/colab/sid_newbp_rgb.yml --launcher none"
echo ""
echo "4. 开始训练 NAFNet 基线:"
echo "   cd \$SID_ROOT/SID_experiments/external/NAFNet"
echo "   python basicsr/train.py -opt options/train/SID/NAFNet-SID-baseline.yml --launcher none"
echo ""
echo "5. 查看训练日志:"
echo "   tensorboard --logdir=\$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement/logs --port=6006"
echo ""
echo "更多信息请参考: docs/autodl_deployment_guide.md"
echo "============================================"
