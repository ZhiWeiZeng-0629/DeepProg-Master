# self-MultiOmics: 多组学生存分析预测框架

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)

> 面向多组学（RNA/METH/MIR）与生存分析的集成框架，基于 DeepProg 方法学，采用基因层面自动编码器与集成学习进行生存预测。

## 📋 目录

- [项目概述](#项目概述)
- [主要特性](#主要特性)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [使用方法](#使用方法)
- [输出结果](#输出结果)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [引用](#引用)
- [许可证](#许可证)

## 🎯 项目概述

self-MultiOmics 是一个面向多组学生存分析的集成框架，整合 RNA-Seq、DNA 甲基化和 miRNA 数据，用于预测患者生存结局。本框架基于 **DeepProg** 方法学开发，采用**基因层面自动编码器 + PCA** 方法进行特征提取，使用**SimDeepBoosting**进行集成学习，通过固定随机种子和确定性环境确保结果可复现。

### 核心亮点

- 🔬 **多组学整合**：无缝整合 RNA、甲基化和 miRNA 数据
- 🧬 **基因层面特征提取**：基于 PyTorch 的逐基因嵌入
- 📊 **集成学习**：SimDeepBoosting 实现稳健的生存预测
- 🔄 **可复现流程**：确定性环境与固定随机种子
- 📈 **全面评估**：C-index、p 值和 KM 生存曲线

## ✨ 主要特性

- **多组学整合**：整合 RNA-Seq、DNA 甲基化和 miRNA 数据
- **基因层面自动编码器**：基于 PyTorch 的逐基因嵌入（64 维）与 PCA 降维
- **集成训练**：可配置迭代次数的 SimDeepBoosting
- **端到端流程**：从原始数据到预测结果与可视化
- **外部验证**：训练/验证集划分与全面评估指标
- **可复现结果**：固定随机种子与确定性操作
- **GPU 支持**：CUDA 加速自动编码器训练

## 🚀 安装指南

### 环境要求

- Python 3.8+
- Conda（推荐）
- 支持 CUDA 的 GPU（可选，用于加速训练）

### 快速安装

**推荐方式：使用 environment.yml**

```bash
# 克隆仓库
git clone https://github.com/ZhiWeiZeng-0629/self-MultiOmics.git
cd self-MultiOmics

# 创建 conda 环境并安装依赖
conda env create -f environment.yml
conda activate deep_new

# 安装项目包
pip install -e .
```

**手动安装**

```bash
# 创建 conda 环境
conda create -n deep_new python=3.8 -y
conda activate deep_new

# 安装核心依赖
pip install tensorflow==2.4.1 keras==2.4.3
pip install torch torchvision
pip install lifelines scikit-survival scikit-learn
pip install pandas numpy scipy
pip install simplejson dill colour mpld3

# 安装项目包
pip install -e .
```

### 验证安装

```python
import tensorflow as tf
import torch
print("TensorFlow:", tf.__version__)
print("PyTorch:", torch.__version__)
print("CUDA 可用:", tf.test.is_built_with_cuda())
```

## 🏃 快速开始

### 1. 准备数据

将数据文件放置在 `data/` 目录下：

```
data/
├── rna.tsv.gz          # RNA-Seq 表达矩阵
├── meth.tsv.gz         # DNA 甲基化矩阵
├── mir.tsv.gz          # miRNA 表达矩阵（可选）
├── survival.tsv        # 生存数据（Samples, days, event）
└── mir_to_gene.tsv     # miRNA 到基因映射文件（可选）
```

**数据格式要求：**
- **组学数据**：TSV 格式（行=样本，列=特征）或压缩格式 `.tsv.gz`
- **生存数据**：必须包含列：`Samples`、`days`、`event`（0/1）

### 2. 运行流程

```bash
# 激活环境
conda activate deep_new

# 使用默认参数运行
./run.sh

# 或使用自定义参数
./run.sh --seed 100 --nb-it 15 --boost-epochs 25 --k 5 --d-gene 64
```

### 3. 查看结果

```bash
# 历史指标记录（所有实验）
cat data/integrated/external_validation/metrics_history.tsv

# 查看特定实验的结果（示例：20251207_0636_seed42）
ls data/integrated/external_validation/20251207_0636_seed42/
cat data/integrated/external_validation/20251207_0636_seed42/metrics.tsv
cat data/integrated/external_validation/20251207_0636_seed42/external_validation_full_labels.tsv
```

## 📁 项目结构

```
.
├── data/                                    # 数据目录
│   ├── rna.tsv(.gz)                         # RNA-Seq 数据
│   ├── meth.tsv(.gz)                        # 甲基化数据
│   ├── mir.tsv(.gz)                         # miRNA 数据（可选）
│   ├── survival.tsv                         # 生存数据
│   ├── train/                               # 训练集（自动生成）
│   │   ├── integrated/                      # 处理后的特征
│   │   │   ├── rna_gene_pca.tsv
│   │   │   ├── meth_gene_pca.tsv
│   │   │   └── mir_gene_pca.tsv
│   │   └── train_survival.tsv
│   ├── val/                                 # 验证集（自动生成）
│   └── integrated/external_validation/   # 输出结果
│       ├── metrics_history.tsv              # 历史指标记录（所有实验）
│       ├── data_split_analysis.png          # 数据划分分析图
│       └── {timestamp}_seed{seed}/         # 每次实验的独立文件夹
│           ├── metrics.tsv                  # 本次实验的评估指标
│           ├── external_validation_full_labels.tsv
│           ├── external_validation_val_test_labels.tsv
│           └── *.pdf                        # KM 生存曲线（4个PDF文件）
├── simdeep/                                 # 核心模块
│   ├── tools/
│   │   ├── generate_gene_pca_tsv.py         # 特征生成
│   │   └── external_validation.py           # 主训练脚本
│   ├── simdeep_boosting.py                  # 集成训练
│   ├── simdeep_analysis.py                  # 单模型
│   ├── extract_data.py                      # 数据加载
│   ├── coxph_from_r.py                      # 生存分析
│   └── ...
├── run.sh                                   # 主执行脚本
├── environment.yml                          # Conda 环境配置
└── setup.py                                 # 包安装配置
```

## 💻 使用方法

### 命令行参数

```bash
python simdeep/tools/external_validation.py \
    --data-root data \
    --mods "RNA METH MIR" \
    --k 5 \
    --d-gene 64 \
    --epochs 10 \
    --batch-size 64 \
    --nb-features 50 \
    --nb-it 10 \
    --boost-epochs 20 \
    --seed 42
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-root` | 数据文件根目录 | `data` |
| `--mods` | 使用的组学数据类型 | `"RNA METH MIR"` |
| `--k` | 每个基因的 PCA 主成分数 | `5` |
| `--d-gene` | 基因嵌入维度 | `64` |
| `--epochs` | 自动编码器训练轮数 | `10` |
| `--nb-features` | 选择的特征数量 | `50` |
| `--nb-it` | 集成迭代次数 | `10` |
| `--boost-epochs` | 每个模型的训练轮数 | `20` |
| `--seed` | 随机种子 | `42` |

## 📊 输出结果

### 评估指标

- **历史记录**：`data/integrated/external_validation/metrics_history.tsv`
  - 记录所有实验的历史指标
- **每次实验的指标**：`data/integrated/external_validation/{timestamp}_seed{seed}/metrics.tsv`
  - `train_pvalue_full`：训练集 p 值
  - `train_cindex_full`：训练集 C-index
  - `val_pvalue`：验证集 p 值
  - `val_cindex`：验证集 C-index

### 预测结果

每次实验的结果保存在独立的文件夹中（格式：`{timestamp}_seed{seed}/`）：
- **`external_validation_full_labels.tsv`**：全数据集预测标签
  - 列：`sample_id`, `label`, `proba_0`, `days`, `event`
- **`external_validation_val_test_labels.tsv`**：验证集预测标签

### 可视化结果

每次实验的KM图保存在对应的实验文件夹中：
- **KM 生存曲线**：带时间戳的 PDF 文件（4个PDF文件）
  - `external_validation_full_proba_KM_plot_boosting_full_*.pdf`（全数据集概率分组）
  - `external_validation_full_labels_KM_plot_boosting_full_*.pdf`（全数据集标签分组）
  - `external_validation_val_proba_KM_plot_boosting_val_*.pdf`（验证集概率分组）
  - `external_validation_val_labels_KM_plot_boosting_val_*.pdf`（验证集标签分组）
- **数据划分分析**：`data/integrated/external_validation/data_split_analysis.png`（保留在根目录）

### 文件组织结构

每次运行实验时，所有结果文件会自动保存到独立的文件夹中：
```
data/integrated/external_validation/
├── metrics_history.tsv                    # 所有实验的历史记录
├── data_split_analysis.png                # 数据划分分析图
├── 20251207_0636_seed42/                  # 实验文件夹（格式：{timestamp}_seed{seed}）
│   ├── metrics.tsv                        # 本次实验的评估指标
│   ├── external_validation_full_labels.tsv
│   ├── external_validation_val_test_labels.tsv
│   └── *.pdf (4个KM图)
└── ...
```

## ⚙️ 配置说明

### 基因层面自动编码器 + PCA 模式（推荐）

框架采用两阶段特征提取：

1. **基因层面自动编码器**（PyTorch）：将每个基因编码为 `d_gene` 维（如 64 维）
2. **逐基因 PCA**：将每个基因的嵌入降维到 `k` 维（如 5 维）

**为什么不使用 Keras 自动编码器？**

- 基因层面自动编码器已完成特征提取
- 使用 Keras 自动编码器会导致双重编码
- 当前配置在验证集上达到 C-index = 0.666

### 特征维度计算

- **总特征数** = 基因数 × `k`（PCA 主成分数）
- **示例**：15,055 个基因 × 5 = 75,275 维
- **特征选择**：选择 top `nb-features` 个特征用于最终分类

## ❓ 常见问题

### 性能相关

**问：为什么训练集 C-index 接近 0.5？**  
答：这是正常现象。重点关注验证集指标。验证集 C-index = 0.666 说明模型泛化能力良好。

**问：如何提升验证集 C-index？**  
答：可以尝试增加 `--boost-epochs`（25-30）、调整 `--nb-features`（75-100）或增加 `--nb-it`（15-20）。

### 技术问题

**问：KM 图生成失败**  
答：确保已安装 `lifelines`：`pip install lifelines`

**问：CUDA 内存不足**  
答：减少 `--k`、`--d-gene` 或 `--batch-size` 参数

**问：特征生成速度慢**  
答：先用较小参数测试（`--k 3 --d-gene 32 --epochs 1`）

### 数据相关

**问：需要什么数据格式？**  
答：TSV 文件，行为样本，列为特征。脚本会自动处理转置。

**问：训练/验证集如何划分？**  
答：默认 75:25 划分，使用 `seed=42` 保证可复现。划分分析保存在 `data_split_analysis.png`。

## 📚 引用

如果您在研究中使用了本框架，请引用：

```bibtex
@software{selfmultiomics,
  title = {self-MultiOmics: 多组学生存分析预测框架},
  author = {ZhiWeiZeng-0629},
  year = {2025},
  url = {https://github.com/ZhiWeiZeng-0629/self-MultiOmics}
}
```

## 📄 许可证

本项目基于 DeepProg 方法学开发。原始项目信息请参考 `Legacy/` 目录下的相关文档。

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## 📧 联系方式

如有问题或建议，请在 GitHub 上提交 [Issue](https://github.com/ZhiWeiZeng-0629/self-MultiOmics/issues) 或联系维护者。

---

**注意**：本框架基于 DeepProg 方法学开发。DeepProg 原始项目信息请参考 `Legacy/` 目录下的相关文档。详细使用说明请参阅 `instruction.txt`。
