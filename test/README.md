# 分子属性预测项目 (flu)

本项目旨在使用图神经网络（GNN）和PyTorch Geometric库来预测分子的各种化学性质。项目探索了从SMILES字符串处理分子数据，构建图表示，以及训练GNN模型进行预测的完整流程。

## 🌟 项目特点

*   **SMILES 处理**: 使用RDKit将SMILES字符串转换为分子对象，并进一步转换为图数据结构。
*   **PyTorch Geometric 集成**: 利用PyTorch Geometric高效地构建和训练图神经网络模型。
*   **特征工程**: 自动从分子结构中提取原子和键的特征。
*   **含氟分子分析**: 特别关注了含氟分子的处理和特征表示，并进行了验证。
*   **自定义数据集**: 演示了如何在PyTorch中创建和使用自定义数据集，以适应特定的数据格式和任务需求。
*   **模块化代码**: 项目代码结构清晰，包含数据处理、模型定义、训练和评估等模块。

## 📁 目录结构

*   `dataset/`: 可能包含数据集文件或数据处理脚本。
    *   `smiles_to_adjacency.py`: 将SMILES转换为邻接矩阵的示例脚本。
*   `model/`: 可能包含模型定义脚本 (目前为空)。
*   `test/`: 包含各种测试、演示和教程脚本。
    *   `pyg_quickstart.py`: PyTorch Geometric的快速入门示例。
    *   `molecular_prediction_project.py`: 一个更完整的分子预测项目框架。
    *   `pytorch_geometric_tutorial.py`: PyTorch Geometric的详细教程脚本。
    *   `fluorine_molecules_demo.py`: 演示如何处理多种含氟分子并提取其图表示。
    *   `fluorine_features_analysis.py`: 分析含氟分子原子特征的脚本。
    *   `fluorine_features_detailed.py`: 更详细地输出含氟分子原子和键特征的脚本。
    *   `energy_test/`:
        *   `custom_data.csv`: 自定义数据集的示例CSV文件。
        *   `custom_dataset_demo.py`: 演示如何加载和使用自定义数据集的PyTorch脚本。
*   `requirements.txt`: 项目的Python依赖列表。
*   `fluorine_analysis_complete.py`: (可能是一个整合了含氟分子分析的脚本，根据之前的对话推断)
*   `README.md`: (本项目的主README文件，你正在阅读这个！)
*   `test/README.md`: `test`目录下的详细说明文件。

## 🚀 如何开始

1.  **克隆仓库**:
    ```bash
    git clone https://github.com/yiozeeyoung/flu.git
    cd flu
    ```

2.  **创建虚拟环境并安装依赖**:
    建议使用conda或venv创建虚拟环境。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows

    pip install -r requirements.txt
    ```
    *注意：安装 `torch` 和 `torch_geometric` 时，请参考其官方文档，确保版本与您的CUDA版本（如果使用GPU）兼容。*

3.  **运行示例脚本**:
    你可以进入 `test/` 目录并运行其中的Python脚本来查看不同的功能演示。
    例如，运行自定义数据集演示：
    ```bash
    cd test/energy_test
    python custom_dataset_demo.py
    ```
    或者运行含氟分子处理演示：
    ```bash
    cd ../  # 返回到 test 目录
    python fluorine_molecules_demo.py
    ```

## 🔬 主要探索和功能

*   **PyTorch Geometric对任意SMILES的处理能力**: 验证了PyG可以处理有效的SMILES字符串，包括复杂的有机分子和含氟化合物。
*   **含氟分子的特征表示**: 详细分析了PyG为含氟分子生成的原子特征（通常是9维）和键特征（通常是3维），确认氟原子的独特性质被正确编码。
*   **自定义数据集实现**: 在 `test/energy_test/` 目录下提供了一个完整的PyTorch自定义数据集加载和使用的示例，展示了如何从CSV文件读取数据并将其整合到PyTorch的 `Dataset` 和 `DataLoader` 中。

## 💡 未来工作和扩展方向

*   实现一个端到端的分子属性（如能量、溶解度等）预测模型。
*   集成更先进的GNN架构。
*   支持更多类型的分子特征和数据格式。
*   进行超参数调优和模型评估。

##🤝 贡献

欢迎通过Pull Request或Issues来贡献代码或提出建议。

---

(以下是 `test/README.md` 的原始内容，为了完整性保留，但主要信息已整合到上方)

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
