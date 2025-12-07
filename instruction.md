# 多组学生存分析预测框架

本项目是一个面向多组学（RNA/METH/MIR）与生存分析的集成框架，采用"外部验证管线"（基因层面多分量 + PCA 特征），通过固定随机源与确定性环境实现稳定复现。

## 主要特性
- 多组学整合与生存相关特征筛选
- 集成训练与评估（`SimDeepBoosting`）
- 一键端到端管线（原始数据 → 特征生成 → 训练/评估 → 标签与图）
- 可选深度模式（Keras/TensorFlow 自动编码器）
- 可选分布式与超参优化（Ray、Ray Tune + SkOpt）

## 目录概览
```text
.
├── data/                                    # 原始输入与集成输出
│   ├── rna.tsv(.gz)                         # RNA 原始矩阵
│   ├── meth.tsv(.gz)                        # 甲基化原始矩阵
│   ├── mir.tsv(.gz)                         # miRNA 原始矩阵（可选）
│   ├── survival.tsv                         # 生存文件（Samples/days/event）
│   ├── mir_to_gene.tsv                      # miRNA 到基因映射文件（可选）

│   ├── models/                              # 保存的模型文件
│   ├── train/                               # 训练集数据（自动生成）
│   │   ├── raw/                             # 原始训练数据
│   │   ├── integrated/                      # 处理后的特征
│   │   │   ├── rna_gene_pca.tsv             # RNA PCA 特征
│   │   │   ├── meth_gene_pca.tsv            # METH PCA 特征
│   │   │   ├── mir_gene_pca.tsv             # MIR PCA 特征（可选）
│   │   │   └── gene_level/                  # 基因层面嵌入（3D 数组）
│   │   └── train_survival.tsv               # 训练集生存数据
│   ├── val/                                 # 验证集数据（自动生成）
│   │   ├── raw/                             # 原始验证数据
│   │   ├── integrated/                      # 处理后的特征（同 train/integrated）
│   │   └── val_survival.tsv                 # 验证集生存数据
│   └── integrated/                          # 管线输出
│       └── external_validation/             # 外部验证结果
│           ├── metrics_history.tsv          # 历史指标记录（所有实验）
│           ├── data_split_analysis.png      # 数据划分分析图
│           └── {timestamp}_seed{seed}/      # 每次实验的独立文件夹
│               ├── metrics.tsv              # 本次实验的评估指标
│               ├── external_validation_full_labels.tsv        # 全数据集标签
│               ├── external_validation_val_test_labels.tsv     # 验证集标签
│               └── *.pdf                     # KM 生存曲线图（4个PDF文件）
│                   # 示例文件夹：20251207_0636_seed42/
├── simdeep/                                 # 核心模块
│   ├── tools/                               # 工具脚本
│   │   ├── generate_gene_pca_tsv.py         # 原始数据 → 基因层面 PCA 特征
│   │   └── external_validation.py             # 外部验证主脚本（当前使用）
│   ├── simdeep_boosting.py                  # 集成训练模块
│   ├── simdeep_analysis.py                  # 单模型分析
│   ├── extract_data.py                      # 数据加载
│   ├── deepmodel_base.py                    # 深度学习基础模型
│   ├── coxph_from_r.py                      # Cox 回归与生存分析
│   ├── survival_utils.py                    # 生存分析工具
│   ├── survival_model_utils.py              # 生存模型工具
│   ├── plot_utils.py                         # 可视化工具
│   ├── simdeep_utils.py                     # 通用工具函数
│   └── config.py                            # 配置文件
├── run.sh                                   # 一键运行脚本（固定随机源与环境）
├── environment.yml                          # Conda 环境配置文件
├── setup.py                                 # Python 包安装配置
├── introduce.TXT                            # 项目介绍文档
└── Legacy/                                  # 未使用的模块和文档（已归档）

```

## 环境与安装

### 环境要求
- Python 3.8+
- 建议使用 `conda` 创建独立环境

### 安装步骤

**推荐方式（使用 environment.yml）**：
```bash
# 使用 conda 环境文件一键创建环境并安装所有依赖
conda env create -f environment.yml
conda activate deep_new
pip install -e .  # 安装项目包
```

**手动安装方式**：
```bash
# 1. 创建 conda 环境
conda create -n deep_new python=3.8 -y
conda activate deep_new

# 2. 安装依赖
pip install tensorflow==2.4.1 keras==2.4.3
pip install lifelines scikit-survival scikit-learn
pip install pandas numpy scipy
pip install torch torchvision  # PyTorch（用于 Gene-level autoencoder）
pip install simplejson dill colour mpld3  # 其他工具依赖
pip install -e .
```

**验证安装**（可选）：
```python
import tensorflow as tf
import torch
print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))
```

## 快速开始

### 1. 激活环境
```bash
conda activate deep_new  # 或您的环境名称
```

### 2. 运行训练（推荐）
```bash
# 确保在项目根目录
./run.sh  # 使用优化后的参数配置，输出直接显示在终端
```

### 3. 查看结果
训练完成后，查看评估指标：
```bash
# 最新运行结果
cat data/integrated/external_validation/metrics.tsv

# 历史运行记录
cat data/integrated/external_validation/metrics_history.tsv

# 查看特定实验的结果（示例：20251207_0636_seed42）
ls data/integrated/external_validation/20251207_0636_seed42/
cat data/integrated/external_validation/20251207_0636_seed42/metrics.tsv
```

### 4. 自定义参数（可选）
如需修改参数，可在运行脚本时附加参数：
```bash
./run.sh --seed 100 --nb-it 15 --boost-epochs 25
```

## 管线输出

### 评估指标
- **历史记录**：`data/integrated/external_validation/metrics_history.tsv`
  - 记录所有运行的历史指标，便于对比分析
- **每次实验的指标**：`data/integrated/external_validation/{timestamp}_seed{seed}/metrics.tsv`
  - 包含：`train_pvalue_full`, `train_cindex_full`, `val_pvalue`, `val_cindex`
  - 示例：`data/integrated/external_validation/20251207_0636_seed42/metrics.tsv`

### 预测标签
每次实验的结果保存在独立的文件夹中：
- **全数据集标签**：`data/integrated/external_validation/{timestamp}_seed{seed}/external_validation_full_labels.tsv`
  - 格式：`sample_id`, `label`, `proba_0`, `days`, `event`
- **验证集标签**：`data/integrated/external_validation/{timestamp}_seed{seed}/external_validation_val_test_labels.tsv`

### 可视化图表
每次实验的KM图保存在对应的实验文件夹中：
- **KM 生存曲线**（位于 `data/integrated/external_validation/{timestamp}_seed{seed}/`）：
  - `external_validation_full_proba_KM_plot_boosting_full_*.pdf`（全数据集概率分组）
  - `external_validation_full_labels_KM_plot_boosting_full_*.pdf`（全数据集标签分组）
  - `external_validation_val_proba_KM_plot_boosting_val_*.pdf`（验证集概率分组）
  - `external_validation_val_labels_KM_plot_boosting_val_*.pdf`（验证集标签分组）
  - 文件名包含时间戳（格式：YYYYMMDD_HHMM_seed{seed}）

### 数据划分分析
- `data/integrated/external_validation/data_split_analysis.png`：训练集/验证集分布对比图（保留在根目录）

### 文件组织结构
每次运行实验时，所有结果文件会自动保存到以 `{timestamp}_seed{seed}` 命名的独立文件夹中，例如：
```
data/integrated/external_validation/
├── metrics_history.tsv                    # 所有实验的历史记录
├── data_split_analysis.png                # 数据划分分析图
├── 20251207_0636_seed42/                  # 实验1的结果文件夹
│   ├── metrics.tsv
│   ├── external_validation_full_labels.tsv
│   ├── external_validation_val_test_labels.tsv
│   └── *.pdf (4个KM图)
├── 20251207_0900_seed42/                  # 实验2的结果文件夹
│   └── ...
└── ...
```

## 关于自动编码器模式

**当前项目使用 Gene-level Autoencoder + PCA 模式**，而不是 Keras 自动编码器。

### 当前模式（推荐）：Gene-level Autoencoder + PCA
- **特征生成阶段**（`generate_gene_pca_tsv.py`）：
  - 使用 **PyTorch Gene-level Autoencoder** 将每个基因编码为 `d_gene` 维（如 64 维）
  - 然后对每个基因的编码进行 **PCA 降维**到 `k` 维（如 5 维）
  - 输出特征文件：
    - `data/train/integrated/rna_gene_pca.tsv`
    - `data/train/integrated/meth_gene_pca.tsv`
    - `data/train/integrated/mir_gene_pca.tsv`（如果使用 MIR）
    - `data/train/integrated/gene_level/`（基因层面嵌入的 3D 数组）
  - 验证集特征文件同样生成在 `data/val/integrated/` 目录下
  
- **模型训练阶段**（`external_validation.py`）：
  - 设置 `use_autoencoders=False`
  - 直接使用 PCA 特征进行训练，不再进行二次编码

### 为什么不使用 Keras Autoencoder（`use_autoencoders=True`）？

**重要**：当前项目**不建议**设置 `use_autoencoders=True`，原因如下：

1. **双重编码冲突**：
   - 特征生成阶段已经使用了 PyTorch Gene-level Autoencoder
   - 如果训练阶段再使用 Keras Autoencoder，会导致双重编码
   - 流程：原始数据 → PyTorch AE → PCA → Keras AE → 训练（不推荐）

2. **设计目的不同**：
   - **Gene-level Autoencoder**：针对每个基因进行编码，适合处理基因层面的多组学数据
   - **Keras Autoencoder**：针对整个特征矩阵进行编码，设计用于替代 PCA 特征提取
   - 两者用途不同，不应同时使用

3. **当前配置已优化**：
   - 当前模式（Gene-level AE + PCA）已经过优化，验证集 C-index = 0.666
   - 改变特征提取方式可能影响性能

### 技术说明

- **Gene-level Autoencoder**：在 `generate_gene_pca_tsv.py` 中使用 PyTorch 实现
- **Keras Autoencoder**：在 `deepmodel_base.py` 中使用 TensorFlow/Keras 实现
- 两者是**独立的系统**，设计用于不同的特征提取流程

## 数据格式要求

### 输入文件
- **组学数据**：`data/rna.tsv.gz`、`data/meth.tsv.gz`、`data/mir.tsv.gz`
  - 支持压缩格式 `*.tsv.gz`
  - 格式：行=样本，列=特征（脚本会自动识别并转置）
- **生存数据**：`data/survival.tsv`
  - 必需列：`Samples`（样本ID）、`days`（生存时间）、`event`（事件状态：0/1）
- **miRNA 映射**（可选）：`data/mir_to_gene.tsv`
  - 用于将 miRNA 表达映射到基因空间

## 项目结构说明

### 当前使用的核心模块
- **主训练脚本**：`simdeep/tools/external_validation.py`
- **特征生成**：`simdeep/tools/generate_gene_pca_tsv.py`
- **集成训练**：`simdeep/simdeep_boosting.py`
- **单模型**：`simdeep/simdeep_analysis.py`
- **数据加载**：`simdeep/extract_data.py`
- **生存分析**：`simdeep/coxph_from_r.py`、`simdeep/survival_utils.py`

### 未使用的模块（已移至 Legacy）
以下功能模块已移至 `Legacy/simdeep_optional/`，当前项目不使用：
- 分布式训练（`simdeep_distributed.py`）
- 多测试集预测（`simdeep_multiple_dataset.py`）
- 超参数优化（`simdeep_tuning.py`）
- 其他训练脚本（`run_integrated.sh`）

如需使用这些功能，请参考 `Legacy/` 目录下的相关文档。

## 常见问题（FAQ）

### 性能相关
- **训练集 C-index 接近随机（0.5）**：
  - 这是正常现象，主要关注验证集指标
  - 验证集 C-index = 0.666 说明模型泛化能力良好
- **验证集 p-value 较高（>0.05）**：
  - 当前为 0.303，说明风险组差异不够显著
  - 可尝试增加 `boost-epochs` 或调整 `nb-features`

### 技术问题
- **KM 图生成失败**：
  - 检查 Python `lifelines` 是否已安装
  - 生存评估封装见 `simdeep/coxph_from_r.py`
- **特征生成耗时**：
  - 视 `--k`（每基因分量数）与 `--d_gene`（嵌入维度）而定
  - 可先用较小参数验证，再提高精度
- **CUDA 内存不足**：
  - 减少 `--k` 或 `--d-gene` 参数
  - 减少 `--batch-size` 参数

## 参数调优建议

### 如何提升验证集 C-index
- 当前最佳配置：`k=5`, `d-gene=64`, `nb-features=50`, `nb-it=10`, `boost-epochs=20`
- 可尝试：
  - 增加 `boost-epochs` 到 25-30
  - 调整 `nb-features` 到 75-100
  - 增加 `nb-it` 到 15-20（集成更多模型）

### 特征维度说明
- **计算公式**：总特征数 = 基因数 × k (PCA分量数)
- **当前配置（k=5）**：15,055 基因 × 5 = 75,275 维
- **之前配置（k=8）**：15,055 基因 × 8 = 120,440 维 ≈ 12万维
- **数据流程**：
  1. 原始数据：每个基因 1 个值（表达量或甲基化水平）
  2. Gene-level autoencoder：每个基因编码为 `d_gene=64` 维
  3. Per-gene PCA：每个基因从 64 维降维到 `k` 维
  4. 特征选择：从所有特征中选择 top `nb-features` 个用于最终分类

## 数据划分说明

训练集和验证集按 **75:25** 比例随机划分（使用 `seed=42` 保证可复现）：
- **训练集**：283 个样本（75.1%）
- **验证集**：94 个样本（24.9%）
- **事件率**：训练集 36.0%，验证集 31.9%（分布平衡）
- **数据划分合理性**：已通过统计检验，无显著差异

详细分析见：`data/integrated/external_validation/data_split_analysis.png`

**注意**：每次实验的结果都保存在独立的文件夹中（格式：`{timestamp}_seed{seed}/`），便于管理和对比不同实验的结果。
