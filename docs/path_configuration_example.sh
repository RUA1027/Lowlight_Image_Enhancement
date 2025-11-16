# 路径配置示例
# 根据您的实际部署环境修改这些路径

# ============================================
# AutoDL 平台配置（Linux）
# ============================================
# 默认数据目录在 /root/autodl-tmp
SID_ROOT=/root/autodl-tmp/Lowlight

# Python 模块路径
PYTHONPATH=$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement:$PYTHONPATH
PYTHONPATH=$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement/NAFNet_base:$PYTHONPATH
PYTHONPATH=$SID_ROOT/SID_experiments/external/NAFNet:$PYTHONPATH


# ============================================
# Google Colab 配置
# ============================================
# SID_ROOT=/content/drive/MyDrive/Lowlight
# PYTHONPATH=/content/drive/MyDrive/Lowlight/SID_experiments/Lowlight_Image_Enhancement:$PYTHONPATH


# ============================================
# 本地 Windows 配置（仅用于开发）
# ============================================
# 注意: Windows 路径在实际运行时会被转换
# SID_ROOT=G:\我的云端硬盘\Lowlight
# PYTHONPATH=G:\我的云端硬盘\Lowlight\SID_experiments\Lowlight_Image_Enhancement


# ============================================
# 其他 Linux 服务器配置
# ============================================
# SID_ROOT=/data/username/Lowlight
# PYTHONPATH=/data/username/Lowlight/SID_experiments/Lowlight_Image_Enhancement:$PYTHONPATH


# ============================================
# 数据集路径（自动从 SID_ROOT 派生）
# ============================================
# Manifest 文件
# MANIFEST_PATH=$SID_ROOT/SID_assets/manifest_sid.json

# LMDB 数据库路径
# TRAIN_SHORT_LMDB=$SID_ROOT/SID_lmdb/train_short.lmdb
# TRAIN_LONG_LMDB=$SID_ROOT/SID_lmdb/train_long.lmdb
# VAL_SHORT_LMDB=$SID_ROOT/SID_lmdb/val_short.lmdb
# VAL_LONG_LMDB=$SID_ROOT/SID_lmdb/val_long.lmdb
# TEST_SHORT_LMDB=$SID_ROOT/SID_lmdb/test_short.lmdb
# TEST_LONG_LMDB=$SID_ROOT/SID_lmdb/test_long.lmdb

# RAW 数据路径（可选）
# RAW_DATA_PATH=$SID_ROOT/SID_raw/Sony


# ============================================
# 使用说明
# ============================================
# 
# 1. 在 Linux/Mac 上设置环境变量:
#    export SID_ROOT=/root/autodl-tmp/Lowlight
#    export PYTHONPATH=$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement:$PYTHONPATH
#
# 2. 永久设置（添加到 ~/.bashrc 或 ~/.bash_profile）:
#    echo 'export SID_ROOT=/root/autodl-tmp/Lowlight' >> ~/.bashrc
#    echo 'export PYTHONPATH=$SID_ROOT/SID_experiments/Lowlight_Image_Enhancement:$PYTHONPATH' >> ~/.bashrc
#    source ~/.bashrc
#
# 3. 在 Python 脚本中使用:
#    import os
#    sid_root = os.environ.get('SID_ROOT', '/default/path')
#    manifest_path = os.path.join(sid_root, 'SID_assets', 'manifest_sid.json')
#
# 4. 在 YAML 配置文件中使用:
#    manifest_path: ${SID_ROOT}/SID_assets/manifest_sid.json
#
# 注意: 
# - 所有配置文件已更新为使用 ${SID_ROOT} 环境变量
# - 数据加载模块已支持自动路径扩展（os.path.expandvars）
# - 跨平台兼容性已通过 pathlib.Path 确保
