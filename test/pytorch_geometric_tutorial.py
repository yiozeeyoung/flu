"""
PyTorch Geometric 教程 - 分子预测应用指南

本文件详细介绍PyTorch Geometric (PyG)的作用，以及如何使用它来构建
化学分子性质预测的深度学习程序。

作者: AI Assistant
日期: 2025-06-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import from_smiles
import matplotlib.pyplot as plt

# ============================================================================
# 第1部分: PyTorch Geometric 简介和核心概念
# ============================================================================

def explain_pyg_basics():
    """
    解释PyTorch Geometric的基本概念和作用
    """
    print("=" * 60)
    print("PyTorch Geometric (PyG) 核心作用:")
    print("=" * 60)
    
    print("1. 图数据结构标准化")
    print("   - 提供统一的图数据格式 (Data对象)")
    print("   - 自动处理节点特征、边索引、图级标签等")
    
    print("\n2. 高效的图神经网络层")
    print("   - 预实现多种GNN层: GCN, GAT, GraphSAGE, GIN等")
    print("   - 优化的稀疏矩阵操作")
    print("   - 支持大规模图处理")
    
    print("\n3. 分子化学特化功能")
    print("   - 直接从SMILES字符串创建分子图")
    print("   - 自动提取原子和化学键特征")
    print("   - 内置分子数据集 (MoleculeNet)")
    
    print("\n4. 批处理和数据加载")
    print("   - 自动处理不同大小图的批处理")
    print("   - 高效的数据加载器")
    
    print("\n5. 丰富的工具函数")
    print("   - 图变换、池化、采样等操作")
    print("   - 与NetworkX、RDKit等库的集成")

def demonstrate_basic_graph_creation():
    """
    演示基本的图数据创建
    """
    print("\n" + "=" * 60)
    print("基本图数据创建示例")
    print("=" * 60)
    
    # 创建一个简单的三角形图
    # 节点特征: 每个节点2维特征
    x = torch.tensor([[1.0, 2.0],   # 节点0
                     [2.0, 1.0],   # 节点1  
                     [3.0, 3.0]], dtype=torch.float)  # 节点2
    
    # 边索引: 表示连接关系 (无向图，每条边需要两个方向)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0],  # 源节点
                              [1, 0, 2, 1, 0, 2]], dtype=torch.long)  # 目标节点
    
    # 图级标签 (例如：图分类任务的标签)
    y = torch.tensor([1], dtype=torch.long)
    
    # 创建PyG数据对象
    data = Data(x=x, edge_index=edge_index, y=y)
    
    print(f"节点数量: {data.num_nodes}")
    print(f"边数量: {data.num_edges}")
    print(f"节点特征维度: {data.x.shape}")
    print(f"是否有孤立节点: {data.has_isolated_nodes()}")
    print(f"是否有自环: {data.has_self_loops()}")
    
    return data

# ============================================================================
# 第2部分: 分子图数据处理
# ============================================================================

def demonstrate_molecular_graph_creation():
    """
    演示如何从SMILES字符串创建分子图
    """
    print("\n" + "=" * 60)
    print("分子图创建示例")
    print("=" * 60)
    
    # 不同复杂度的分子示例
    molecules = {
        '乙醇': 'CCO',
        '苯': 'c1ccccc1', 
        '阿司匹林': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        '咖啡因': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    }
    
    for name, smiles in molecules.items():
        try:
            # 从SMILES创建分子图
            data = from_smiles(smiles)
            
            print(f"\n{name} ({smiles}):")
            print(f"  原子数量: {data.num_nodes}")
            print(f"  化学键数量: {data.num_edges // 2}")  # 无向图，每条边计算两次
            print(f"  原子特征维度: {data.x.shape[1]}")
            print(f"  包含边特征: {data.edge_attr is not None}")
            
            if data.edge_attr is not None:
                print(f"  边特征维度: {data.edge_attr.shape[1]}")
                
        except Exception as e:
            print(f"处理 {name} 时出错: {e}")
    
    return data

def analyze_molecular_features():
    """
    分析PyG自动提取的分子特征
    """
    print("\n" + "=" * 60)
    print("分子特征分析")
    print("=" * 60)
    
    # 创建阿司匹林分子图
    aspirin_data = from_smiles('CC(=O)OC1=CC=CC=C1C(=O)O')
    
    print("PyG自动提取的原子特征包括:")
    print("1. 原子类型 (C, N, O等)")
    print("2. 原子度数 (连接的原子数)")
    print("3. 形式电荷")
    print("4. 杂化类型 (sp, sp2, sp3)")
    print("5. 是否在芳香环中")
    print("6. 氢原子数量")
    print("7. 化合价")
    print("8. 质量数")
    print("9. 是否为手性中心")
    
    print(f"\n阿司匹林原子特征矩阵形状: {aspirin_data.x.shape}")
    print(f"每个原子的特征维度: {aspirin_data.x.shape[1]}")
    
    # 显示前3个原子的特征
    print(f"\n前3个原子的特征:")
    for i in range(min(3, aspirin_data.num_nodes)):
        print(f"原子 {i}: {aspirin_data.x[i].numpy()}")

# ============================================================================
# 第3部分: 构建分子预测模型
# ============================================================================

class MolecularPropertyPredictor(nn.Module):
    """
    分子性质预测模型
    结合多种GNN层来学习分子表示
    """
    def __init__(self, num_features, hidden_dim=64, num_classes=1, dropout=0.2):
        super().__init__()
        
        # 图卷积层
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 图注意力层 (可选)
        self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, num_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, num_edge_features] (可选)
            batch: 批次索引 [num_nodes] (批处理时使用)
        
        返回:
            分子级别的预测结果
        """
        # 第一层图卷积
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层图卷积
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第三层图卷积
        x = F.relu(self.conv3(x, edge_index))
        
        # 图注意力层
        x = F.relu(self.gat(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 图级池化：将节点特征聚合为图特征
        if batch is None:
            # 单个图的情况
            x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        else:
            # 批处理的情况
            x = global_mean_pool(x, batch)
        
        # 最终预测
        return self.classifier(x)

def demonstrate_model_usage():
    """
    演示模型的使用方法
    """
    print("\n" + "=" * 60)
    print("分子预测模型使用示例")
    print("=" * 60)
    
    # 创建一些示例分子
    molecules = ['CCO', 'CC(C)O', 'CCCO', 'CC(C)(C)O']  # 不同的醇类
    data_list = []
    
    for smiles in molecules:
        try:
            data = from_smiles(smiles)
            # 添加虚拟标签 (实际应用中应该是真实的性质数据)
            data.y = torch.randn(1)  # 例如：溶解度、毒性等
            data_list.append(data)
        except:
            continue
    
    if not data_list:
        print("无法创建分子数据，请检查RDKit安装")
        return
    
    # 创建数据加载器
    loader = DataLoader(data_list, batch_size=2, shuffle=True)
    
    # 获取特征维度
    num_features = data_list[0].x.shape[1]
    
    # 创建模型
    model = MolecularPropertyPredictor(num_features=num_features, hidden_dim=32, num_classes=1)
    
    print(f"模型输入特征维度: {num_features}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 前向传播示例
    model.eval()
    for batch in loader:
        with torch.no_grad():
            predictions = model(batch.x, batch.edge_index, batch=batch.batch)
            print(f"批次大小: {batch.num_graphs}")
            print(f"预测结果形状: {predictions.shape}")
            print(f"预测值: {predictions.squeeze().tolist()}")
        break

# ============================================================================
# 第4部分: 训练流程示例
# ============================================================================

def demonstrate_training_process():
    """
    演示完整的训练流程
    """
    print("\n" + "=" * 60)
    print("模型训练流程示例")
    print("=" * 60)
    
    # 生成模拟数据集
    molecules = [
        'CCO', 'CC(C)O', 'CCCO', 'CC(C)(C)O', 'CCCCO',  # 醇类
        'CC(=O)C', 'CCC(=O)C', 'CC(=O)CC',  # 酮类
        'CC(=O)O', 'CCC(=O)O', 'CC(C)C(=O)O'  # 酸类
    ]
    
    # 模拟性质数据 (例如沸点)
    properties = [78.4, 82.3, 97.2, 82.4, 117.7,  # 醇类沸点
                  56.1, 79.6, 79.6,  # 酮类沸点  
                  118.1, 141.1, 154.5]  # 酸类沸点
    
    # 创建数据集
    data_list = []
    for smiles, prop in zip(molecules, properties):
        try:
            data = from_smiles(smiles)
            # 标准化性质值
            data.y = torch.tensor([(prop - 100) / 50], dtype=torch.float)  # 简单标准化
            data_list.append(data)
        except:
            continue
    
    if len(data_list) < 5:
        print("数据集太小，无法进行有效训练演示")
        return
    
    # 划分训练集和测试集
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    test_data = data_list[train_size:]
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)
    
    # 创建模型、损失函数和优化器
    num_features = data_list[0].x.shape[1]
    model = MolecularPropertyPredictor(num_features=num_features, hidden_dim=32, num_classes=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print("开始训练...")
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch.x, batch.edge_index, batch=batch.batch)
            loss = criterion(predictions.squeeze(), batch.y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                predictions = model(batch.x, batch.edge_index, batch=batch.batch)
                test_loss += criterion(predictions.squeeze(), batch.y).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Test Loss: {test_loss/len(test_loader):.4f}")

# ============================================================================
# 第5部分: 实际应用场景
# ============================================================================

def explain_real_world_applications():
    """
    解释PyG在分子预测中的实际应用场景
    """
    print("\n" + "=" * 60)
    print("PyG在分子预测中的实际应用")
    print("=" * 60)
    
    applications = {
        "药物发现": {
            "描述": "预测小分子的生物活性、毒性、副作用",
            "数据集": "ChEMBL, DrugBank, Tox21",
            "预测目标": "IC50, EC50, LD50, ADMET性质"
        },
        
        "材料科学": {
            "描述": "预测材料的物理化学性质",
            "数据集": "Materials Project, NOMAD",
            "预测目标": "带隙、形成能、弹性模量"
        },
        
        "环境化学": {
            "描述": "评估化学品的环境影响",
            "数据集": "EPA CompTox, REACH",
            "预测目标": "生物降解性、生物累积性、生态毒性"
        },
        
        "催化剂设计": {
            "描述": "预测催化剂的活性和选择性",
            "数据集": "NIST, Catalysis-Hub",
            "预测目标": "转化率、选择性、稳定性"
        }
    }
    
    for app, details in applications.items():
        print(f"\n{app}:")
        print(f"  应用描述: {details['描述']}")
        print(f"  常用数据集: {details['数据集']}")
        print(f"  预测目标: {details['预测目标']}")

def provide_implementation_roadmap():
    """
    提供实现分子预测程序的路线图
    """
    print("\n" + "=" * 60)
    print("分子预测程序实现路线图")
    print("=" * 60)
    
    steps = [
        {
            "步骤": "1. 环境设置",
            "任务": [
                "安装PyTorch和PyTorch Geometric",
                "安装RDKit用于分子处理",
                "配置CUDA环境（如有GPU）"
            ]
        },
        {
            "步骤": "2. 数据准备",
            "任务": [
                "收集SMILES字符串和对应的性质数据",
                "数据清洗和预处理",
                "划分训练集、验证集、测试集"
            ]
        },
        {
            "步骤": "3. 特征工程",
            "任务": [
                "使用PyG的from_smiles自动提取特征",
                "或自定义原子和化学键特征",
                "特征标准化和归一化"
            ]
        },
        {
            "步骤": "4. 模型设计",
            "任务": [
                "选择合适的GNN架构（GCN, GAT, GIN等）",
                "设计图级池化策略",
                "添加正则化和dropout"
            ]
        },
        {
            "步骤": "5. 训练和评估",
            "任务": [
                "实现训练循环",
                "监控训练过程和验证性能",
                "使用适当的评估指标"
            ]
        },
        {
            "步骤": "6. 模型优化",
            "任务": [
                "超参数调优",
                "模型集成",
                "性能分析和改进"
            ]
        }
    ]
    
    for step in steps:
        print(f"\n{step['步骤']}:")
        for task in step['任务']:
            print(f"  • {task}")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：运行所有示例和教程
    """
    print("PyTorch Geometric 分子预测教程")
    print("=" * 60)
    
    try:
        # 基础概念介绍
        explain_pyg_basics()
        
        # 基本图数据创建
        demonstrate_basic_graph_creation()
        
        # 分子图创建
        demonstrate_molecular_graph_creation()
        
        # 分子特征分析
        analyze_molecular_features()
        
        # 模型使用示例
        demonstrate_model_usage()
        
        # 训练流程示例
        demonstrate_training_process()
        
        # 实际应用场景
        explain_real_world_applications()
        
        # 实现路线图
        provide_implementation_roadmap()
        
        print("\n" + "=" * 60)
        print("教程完成！")
        print("建议下一步：")
        print("1. 安装必要的依赖包")
        print("2. 收集你感兴趣的分子数据集")
        print("3. 根据具体任务调整模型架构")
        print("4. 开始你的分子预测项目！")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("\n请安装以下包:")
        print("pip install torch torch-geometric rdkit-pypi matplotlib")
    except Exception as e:
        print(f"运行错误: {e}")

if __name__ == "__main__":
    main()
