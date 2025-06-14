"""
PyTorch Geometric 快速入门指南

这个文件提供了PyTorch Geometric的基础使用方法，
专门针对分子预测任务的简单示例。

适合初学者，包含最少的代码和清晰的注释。

作者: AI Assistant
日期: 2025-06-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ============================================================================
# 第1步: 安装和导入
# ============================================================================

def check_installations():
    """
    检查必要的包是否已安装
    """
    try:
        import torch_geometric
        print("✓ PyTorch Geometric 已安装")
        print(f"  版本: {torch_geometric.__version__}")
    except ImportError:
        print("✗ 需要安装 PyTorch Geometric")
        print("  安装命令: pip install torch-geometric")
        return False
    
    try:
        from rdkit import Chem
        print("✓ RDKit 已安装")
    except ImportError:
        print("✗ 需要安装 RDKit")
        print("  安装命令: pip install rdkit-pypi")
        return False
    
    print("✓ 所有依赖包都已安装！")
    return True

# ============================================================================
# 第2步: 理解PyG的数据格式
# ============================================================================

def create_simple_graph():
    """
    创建一个简单的图来理解PyG的数据格式
    
    我们创建一个三原子分子 H-O-H (水分子)
    """
    print("\n" + "="*50)
    print("创建简单图示例 - 水分子 (H-O-H)")
    print("="*50)
    
    # 节点特征: [原子序数, 是否为氧原子, 是否为氢原子]
    # 原子 0: H (氢)
    # 原子 1: O (氧) 
    # 原子 2: H (氢)
    node_features = torch.tensor([
        [1, 0, 1],  # H: 原子序数=1, 不是氧, 是氢
        [8, 1, 0],  # O: 原子序数=8, 是氧, 不是氢
        [1, 0, 1]   # H: 原子序数=1, 不是氧, 是氢
    ], dtype=torch.float)
    
    # 边索引: 表示哪些原子之间有化学键
    # 连接: H(0)-O(1), O(1)-H(2)
    # 注意: PyG使用无向图，所以每条边需要两个方向
    edge_index = torch.tensor([
        [0, 1, 1, 2],  # 源节点
        [1, 0, 2, 1]   # 目标节点
    ], dtype=torch.long)
    
    # 创建PyG数据对象
    data = Data(x=node_features, edge_index=edge_index)
    
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"节点特征形状: {data.x.shape}")
    print(f"边索引形状: {data.edge_index.shape}")
    print(f"\n节点特征:")
    print(data.x)
    print(f"\n边索引:")
    print(data.edge_index)
    
    return data

# ============================================================================
# 第3步: 从SMILES创建分子图
# ============================================================================

def create_molecule_from_smiles():
    """
    从SMILES字符串创建分子图
    这是PyG的强大功能 - 自动从化学表示创建图
    """
    print("\n" + "="*50)
    print("从SMILES创建分子图")
    print("="*50)
    
    try:
        from torch_geometric.utils import from_smiles
        
        # 不同复杂度的分子
        molecules = {
            '甲烷': 'C',
            '乙醇': 'CCO', 
            '苯': 'c1ccccc1',
            '水': 'O'
        }
        
        for name, smiles in molecules.items():
            data = from_smiles(smiles)
            print(f"\n{name} ({smiles}):")
            print(f"  原子数: {data.num_nodes}")
            print(f"  键数: {data.num_edges // 2}")  # 除以2因为是无向图
            print(f"  特征维度: {data.x.shape[1]}")
            
        return data  # 返回最后一个分子的数据
        
    except ImportError:
        print("需要安装 RDKit 才能使用 from_smiles 功能")
        return create_simple_graph()  # 回退到手动创建的图

# ============================================================================
# 第4步: 创建简单的图神经网络
# ============================================================================

class SimpleGNN(nn.Module):
    """
    简单的图神经网络模型
    用于分子性质预测
    """
    
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super().__init__()
        
        # 两层图卷积
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # 最终的预测层
        self.predictor = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, batch=None):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 (多个图时使用)
        """
        # 第一层图卷积 + 激活函数
        x = F.relu(self.conv1(x, edge_index))
        
        # 第二层图卷积 + 激活函数
        x = F.relu(self.conv2(x, edge_index))
        
        # 图级池化: 将所有原子的特征聚合成一个分子特征
        if batch is None:
            # 单个分子的情况
            x = torch.mean(x, dim=0, keepdim=True)
        else:
            # 多个分子批处理的情况
            x = global_mean_pool(x, batch)
        
        # 最终预测
        return self.predictor(x)

def demonstrate_model():
    """
    演示模型的使用
    """
    print("\n" + "="*50)
    print("演示图神经网络模型")
    print("="*50)
    
    # 创建示例数据
    data = create_molecule_from_smiles()
    
    # 创建模型
    input_dim = data.x.shape[1]  # 输入特征维度
    model = SimpleGNN(input_dim=input_dim, hidden_dim=16, output_dim=1)
    
    print(f"模型输入维度: {input_dim}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        prediction = model(data.x, data.edge_index)
        print(f"预测输出: {prediction.item():.4f}")
    
    return model, data

# ============================================================================
# 第5步: 简单的训练示例
# ============================================================================

def create_toy_dataset():
    """
    创建一个玩具数据集用于训练演示
    """
    print("\n" + "="*50)
    print("创建玩具数据集")
    print("="*50)
    
    # 简单的分子和虚构的性质值
    molecules = [
        ('C', 1.0),       # 甲烷 - 低极性
        ('O', 5.0),       # 水 - 高极性
        ('CCO', 3.0),     # 乙醇 - 中等极性
        ('CC', 1.5),      # 乙烷 - 低极性
        ('CO', 4.0),      # 甲醇 - 较高极性
    ]
    
    data_list = []
    
    try:
        from torch_geometric.utils import from_smiles
        
        for smiles, property_value in molecules:
            data = from_smiles(smiles)
            if data is not None:
                data.y = torch.tensor([property_value], dtype=torch.float)
                data_list.append(data)
                
    except ImportError:
        print("无法创建SMILES数据集，使用手动创建的数据")
        # 创建手动数据
        for i in range(3):
            x = torch.randn(3, 4)  # 3个原子，4维特征
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            y = torch.tensor([float(i)], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
    
    print(f"数据集大小: {len(data_list)}")
    return data_list

def simple_training_demo():
    """
    简单的训练演示
    """
    print("\n" + "="*50)
    print("简单训练演示")
    print("="*50)
    
    # 创建数据集
    data_list = create_toy_dataset()
    
    if not data_list:
        print("无法创建数据集")
        return
    
    # 创建数据加载器
    loader = DataLoader(data_list, batch_size=2, shuffle=True)
    
    # 创建模型
    input_dim = data_list[0].x.shape[1]
    model = SimpleGNN(input_dim=input_dim, hidden_dim=8, output_dim=1)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练几个epoch
    print("开始训练...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(pred.squeeze(), batch.y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    print("训练完成！")
    
    # 测试预测
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_list[:3]):
            pred = model(data.x, data.edge_index)
            true_val = data.y.item() if hasattr(data, 'y') else 0
            print(f"分子 {i+1}: 真实值 = {true_val:.2f}, 预测值 = {pred.item():.2f}")

# ============================================================================
# 第6步: 实用技巧和最佳实践
# ============================================================================

def practical_tips():
    """
    PyTorch Geometric 使用技巧
    """
    print("\n" + "="*50)
    print("PyTorch Geometric 使用技巧")
    print("="*50)
    
    tips = [
        {
            "标题": "1. 数据预处理",
            "内容": [
                "• 使用 from_smiles() 自动从SMILES创建图",
                "• 检查数据质量，移除无效分子",
                "• 标准化目标值以提高训练稳定性",
                "• 考虑数据增强（添加噪声、旋转等）"
            ]
        },
        {
            "标题": "2. 模型设计",
            "内容": [
                "• 从简单模型开始（如GCN），逐步增加复杂度",
                "• 使用适当的池化策略（mean, max, attention）",
                "• 添加批归一化和dropout提高泛化能力",
                "• 考虑残差连接处理深层网络"
            ]
        },
        {
            "标题": "3. 训练技巧",
            "内容": [
                "• 使用学习率调度器",
                "• 实施早停防止过拟合",
                "• 梯度裁剪防止梯度爆炸",
                "• 交叉验证评估模型性能"
            ]
        },
        {
            "标题": "4. 性能优化",
            "内容": [
                "• 使用GPU加速训练",
                "• 合理设置batch_size平衡内存和性能",
                "• 使用数据并行处理大数据集",
                "• 预计算分子特征节省时间"
            ]
        },
        {
            "标题": "5. 调试技巧",
            "内容": [
                "• 检查数据形状和类型",
                "• 可视化分子图结构",
                "• 监控训练和验证损失",
                "• 分析预测错误的分子"
            ]
        }
    ]
    
    for tip in tips:
        print(f"\n{tip['标题']}")
        for item in tip['内容']:
            print(f"  {item}")

def next_steps():
    """
    下一步建议
    """
    print("\n" + "="*50)
    print("下一步建议")
    print("="*50)
    
    steps = [
        "1. 掌握基础后，尝试更复杂的GNN架构 (GAT, GIN, GraphSAGE)",
        "2. 学习使用预训练的分子模型（如ChemBERTa）",
        "3. 探索不同的分子特征工程技术",
        "4. 实践多任务学习（同时预测多个性质）",
        "5. 学习分子生成和优化技术",
        "6. 参与开源项目和学术研究",
        "7. 关注最新的图神经网络和化学信息学进展"
    ]
    
    for step in steps:
        print(f"  {step}")
    
    print(f"\n推荐资源:")
    print(f"  • PyTorch Geometric 官方文档")
    print(f"  • RDKit 教程")
    print(f"  • DeepChem 库")
    print(f"  • 分子机器学习相关论文")

# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    运行完整的入门教程
    """
    print("PyTorch Geometric 快速入门指南")
    print("="*50)
    
    # 检查安装
    if not check_installations():
        print("\n请先安装必要的依赖包，然后重新运行。")
        return
    
    # 运行所有示例
    create_simple_graph()
    create_molecule_from_smiles()
    demonstrate_model()
    simple_training_demo()
    practical_tips()
    next_steps()
    
    print("\n" + "="*50)
    print("恭喜！你已经完成了PyTorch Geometric的基础学习")
    print("现在可以开始构建你自己的分子预测模型了！")
    print("="*50)

if __name__ == "__main__":
    main()
