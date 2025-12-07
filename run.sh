#!/usr/bin/env bash
set -euo pipefail

# 环境变量设置
export PYTHONPATH=/data/zengzhiw/DeepProg-master  # Python 模块搜索路径
export TF_DETERMINISTIC_OPS=1                     # TensorFlow 确定性操作，保证可复现性
export PYTHONHASHSEED=0                           # Python hash 随机种子，保证可复现性
export OMP_NUM_THREADS=1                          # OpenMP 线程数，避免多线程冲突
export CUBLAS_WORKSPACE_CONFIG=:4096:8            # CUDA cuBLAS 工作空间配置
export CUDNN_DETERMINISTIC=1                      # cuDNN 确定性操作
export CUDNN_BENCHMARK=0                          # 禁用 cuDNN 基准测试（确保确定性）

# 直接运行，输出显示在终端（不使用后台运行或日志重定向）
# -u 参数：unbuffered 模式，确保输出实时显示
/data/zengzhiw/conda_envs/deep_new/bin/python -u simdeep/tools/external_validation.py \
    --data-root data \
    --mods "RNA METH MIR" \
    --mirna-map data/mir_to_gene.tsv \
    --k 5 \
    --d-gene 64 \
    --epochs 10 \
    --batch-size 64 \
    --nb-features 50 \
    --nb-threads 1 \
    --nb-it 10 \
    --boost-epochs 20 \
    --seed 42 \
    "$@"

# ============================================================================
# 训练参数说明
# ============================================================================
# --data-root data                    # 数据根目录，包含 survival.tsv、rna.tsv、meth.tsv、mir.tsv 等文件
# --mods "RNA METH MIR"               # 使用的组学数据类型：RNA（转录组）、METH（甲基化）、MIR（miRNA）
# --mirna-map data/mir_to_gene.tsv    # miRNA 到基因的映射文件，用于将 miRNA 表达映射到基因空间
# --k 5                               # 每个基因的 PCA 主成分数量，影响特征维度（k 越大特征越多，但计算量也越大）
# --d-gene 64                         # Gene-level autoencoder 的隐向量维度，控制基因嵌入的维度
# --epochs 10                         # Gene-level autoencoder 的训练轮数，控制特征提取的迭代次数
# --batch-size 64                     # 训练时的批次大小，影响内存使用和训练速度
# --nb-features 50                    # 从所有特征中选择的 top 特征数量，用于最终分类（特征选择数量）
# --nb-threads 1                      # 并行线程数，用于 Cox-PH 生存分析的并行计算
# --nb-it 10                          # SimDeepBoosting 中集成的模型数量（boosting 迭代次数）
# --boost-epochs 20                   # 每个 SimDeep 模型的训练轮数，控制单个模型的训练深度
# --seed 42                           # 随机种子，用于保证数据划分和模型训练的可复现性
# "$@"                                # 允许从命令行传入额外的参数，覆盖上述默认值
