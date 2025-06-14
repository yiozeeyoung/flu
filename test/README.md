# PyTorch Geometric 分子预测项目指南

本目录包含了使用PyTorch Geometric构建分子预测程序的完整教程和示例代码。

## 📁 文件结构

### 1. `pyg_quickstart.py` - 快速入门指南
**适合人群**: 完全初学者
**内容概览**:
- PyTorch Geometric基础概念
- 图数据结构解释
- 从SMILES创建分子图
- 简单GNN模型构建
- 基础训练流程
- 实用技巧和最佳实践

**主要功能**:
```python
# 检查安装
check_installations()

# 创建简单图
create_simple_graph()

# 从SMILES创建分子
create_molecule_from_smiles()

# 演示模型使用
demonstrate_model()

# 简单训练演示
simple_training_demo()
```

### 2. `molecular_prediction_project.py` - 完整项目示例
**适合人群**: 有一定基础的开发者
**内容概览**:
- 完整的分子溶解度预测项目
- 高级模型架构设计
- 专业的训练和评估流程
- 结果可视化
- 性能分析

**项目结构**:
```python
# 数据处理模块
class MolecularDataset

# 高级模型定义
class AdvancedMolecularGNN

# 训练器类
class ModelTrainer

# 可视化函数
plot_training_history()
plot_predictions()
```

### 3. `pytorch_geometric_tutorial.py` - 详细教程
**适合人群**: 想深入了解PyG的用户
**内容概览**:
- PyTorch Geometric核心概念详解
- 分子图数据处理技术
- 多种GNN架构比较
- 实际应用场景分析
- 实现路线图

**教学模块**:
```python
# 基础概念
explain_pyg_basics()

# 分子图创建
demonstrate_molecular_graph_creation()

# 特征分析
analyze_molecular_features()

# 模型设计
MolecularPropertyPredictor()

# 训练流程
demonstrate_training_process()
```

## 🚀 快速开始

### 第1步: 环境设置
```bash
# 安装PyTorch (根据你的CUDA版本选择)
pip install torch

# 安装PyTorch Geometric
pip install torch-geometric

# 安装分子处理库
pip install rdkit-pypi

# 安装可视化库
pip install matplotlib seaborn

# 安装科学计算库
pip install numpy pandas scikit-learn
```

### 第2步: 选择入门文件
- **新手**: 从 `pyg_quickstart.py` 开始
- **有经验**: 直接看 `molecular_prediction_project.py`
- **想详细了解**: 阅读 `pytorch_geometric_tutorial.py`

### 第3步: 运行示例
```bash
# 进入test目录
cd d:\code\flu\test

# 运行快速入门示例
python pyg_quickstart.py

# 或运行完整项目示例
python molecular_prediction_project.py
```

## 📊 PyTorch Geometric 核心概念

### 图数据结构
```python
from torch_geometric.data import Data

# 节点特征 (原子特征)
x = torch.tensor([[原子类型, 电荷, 度数, ...], ...])

# 边索引 (化学键)
edge_index = torch.tensor([[源原子索引], [目标原子索引]])

# 图数据对象
data = Data(x=x, edge_index=edge_index, y=target_property)
```

### 从SMILES创建分子图
```python
from torch_geometric.utils import from_smiles

# 自动从SMILES创建图
data = from_smiles('CCO')  # 乙醇分子
print(f"原子数: {data.num_nodes}")
print(f"键数: {data.num_edges // 2}")
```

### 图神经网络层
```python
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

# 图卷积层
conv = GCNConv(in_channels=16, out_channels=32)

# 图注意力层
gat = GATConv(in_channels=16, out_channels=32, heads=4)

# 图级池化
graph_features = global_mean_pool(node_features, batch)
```

## 🧪 分子预测应用场景

### 1. 药物发现
- **目标**: 预测小分子的生物活性、毒性、ADMET性质
- **数据集**: ChEMBL, DrugBank, Tox21
- **评估指标**: IC50, EC50, LD50

### 2. 材料科学
- **目标**: 预测材料的物理化学性质
- **数据集**: Materials Project, NOMAD
- **评估指标**: 带隙、形成能、弹性模量

### 3. 环境化学
- **目标**: 评估化学品的环境影响
- **数据集**: EPA CompTox, REACH
- **评估指标**: 生物降解性、生态毒性

## 🔧 模型架构选择指南

### 简单任务 (数据集 < 1000)
```python
class SimpleGNN(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(input_dim, 32)
        self.conv2 = GCNConv(32, 16)
        self.predictor = nn.Linear(16, 1)
```

### 中等复杂度 (数据集 1000-10000)
```python
class MediumGNN(nn.Module):
    def __init__(self):
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 64)
        self.gat = GATConv(64, 64, heads=4)
        self.predictor = nn.Sequential(...)
```

### 复杂任务 (数据集 > 10000)
```python
class AdvancedGNN(nn.Module):
    def __init__(self):
        self.convs = nn.ModuleList([...])  # 多层GCN
        self.gat = GATConv(...)            # 注意力机制
        self.batch_norms = nn.ModuleList([...])  # 批归一化
        self.attention_pool = AttentionPooling(...)  # 注意力池化
```

## 📈 性能优化技巧

### 1. 数据优化
- 预计算分子特征
- 使用高效的数据加载器
- 合理设置batch_size

### 2. 模型优化
- 梯度裁剪防止梯度爆炸
- 学习率调度
- 早停机制

### 3. 硬件优化
- 使用GPU加速
- 混合精度训练
- 数据并行处理

## 🎯 最佳实践

### 数据预处理
1. **质量检查**: 移除无效SMILES和异常值
2. **标准化**: 对目标值进行归一化
3. **划分**: 合理划分训练/验证/测试集
4. **增强**: 考虑数据增强技术

### 模型设计
1. **从简单开始**: 先用基础GCN验证可行性
2. **逐步复杂化**: 根据需要添加注意力、残差连接等
3. **正则化**: 使用dropout、批归一化防止过拟合
4. **集成学习**: 考虑多模型融合

### 训练策略
1. **监控过程**: 记录训练和验证损失
2. **超参调优**: 系统性地调整学习率、隐藏维度等
3. **交叉验证**: 使用k折交叉验证评估泛化能力
4. **结果分析**: 分析错误案例，改进模型

## 🔗 进阶学习资源

### 官方文档
- [PyTorch Geometric 官方文档](https://pytorch-geometric.readthedocs.io/)
- [RDKit 文档](https://www.rdkit.org/docs/)

### 学术论文
- "Semi-Supervised Classification with Graph Convolutional Networks" (GCN)
- "Graph Attention Networks" (GAT)
- "How Powerful are Graph Neural Networks?" (GIN)

### 开源项目
- DeepChem: 深度学习在化学中的应用
- DGL-LifeSci: 图神经网络在生命科学中的应用
- ChemBERTa: 化学领域的预训练模型

## ❓ 常见问题

### Q: 如何处理大分子?
A: 考虑使用图采样技术或分层图神经网络

### Q: 模型训练不收敛怎么办?
A: 检查数据质量、调整学习率、添加正则化

### Q: 如何选择合适的池化策略?
A: 尝试mean、max、attention池化，根据验证集性能选择

### Q: 内存不足怎么办?
A: 减小batch_size、使用梯度累积、考虑模型压缩

---

## 🎉 开始你的分子预测之旅！

现在你已经掌握了使用PyTorch Geometric进行分子预测的基础知识。建议按照以下步骤开始实践：

1. 运行快速入门示例
2. 收集你感兴趣的分子数据
3. 根据数据规模选择合适的模型
4. 进行系统性的实验和优化
5. 分享你的结果和经验

祝你在分子机器学习的道路上取得成功！🚀
