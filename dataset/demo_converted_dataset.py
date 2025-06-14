"""
FairChem 转换数据集的使用示例
演示如何在 PyTorch Geometric 中使用转换后的数据集进行训练
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
import os

class MolecularEnergyPredictor(torch.nn.Module):
    """
    简单的分子能量预测模型
    """
    def __init__(self, num_features=1, hidden_dim=64, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # GCN 层
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 输出层
        self.predictor = torch.nn.Linear(hidden_dim, 1)
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        # GCN 层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # 全局池化
        x = global_mean_pool(x, batch)
        
        # 预测
        return self.predictor(x)

class LoadedDataset(InMemoryDataset):
    """加载已转换的数据集"""
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        return ['data.pt']

def demonstrate_usage():
    """演示如何使用转换后的数据集"""
    print("=== FairChem 转换数据集使用演示 ===\n")
    
    # 数据路径
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    
    # 检查数据是否存在
    processed_file = os.path.join(output_dir, "processed", "data.pt")
    if not os.path.exists(processed_file):
        print("未找到转换后的数据，请先运行转换脚本")
        return
    
    # 加载数据集
    print("1. 加载数据集...")
    dataset = LoadedDataset(output_dir)
    print(f"   数据集包含 {len(dataset)} 个样本")
    
    # 分析第一个样本
    print("\n2. 样本分析...")
    sample = dataset[0]
    print(f"   样本结构: {sample}")
    print(f"   原子数量: {sample.x.shape[0]}")
    print(f"   特征维度: {sample.x.shape[1]}")
    print(f"   能量值: {sample.y.item():.6f}")
    print(f"   是否有边: {hasattr(sample, 'edge_index')}")
    print(f"   原子类型范围: {sample.x.min().item():.0f} - {sample.x.max().item():.0f}")
    
    # 创建数据加载器
    print("\n3. 创建数据加载器...")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"   训练集: {len(train_dataset)} 个样本")
    print(f"   测试集: {len(test_dataset)} 个样本")
    
    # 创建模型
    print("\n4. 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MolecularEnergyPredictor(num_features=1, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   使用设备: {device}")
    
    # 简单训练演示
    print("\n5. 训练演示（1个epoch）...")
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        loss = criterion(out.squeeze(), batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches >= 3:  # 只训练3个批次作为演示
            break
    
    avg_loss = total_loss / num_batches
    print(f"   平均训练损失: {avg_loss:.6f}")
    
    # 简单测试演示
    print("\n6. 测试演示...")
    model.eval()
    test_loss = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.squeeze(), batch.y)
            test_loss += loss.item()
            num_test_batches += 1
            
            if num_test_batches >= 2:  # 只测试2个批次作为演示
                break
    
    avg_test_loss = test_loss / num_test_batches
    print(f"   平均测试损失: {avg_test_loss:.6f}")
    
    print("\n=== 演示完成 ===")
    print("您现在可以:")
    print("1. 扩展训练更多epochs")
    print("2. 调整模型架构")
    print("3. 尝试不同的优化器和学习率")
    print("4. 处理更多数据（调整 max_samples 参数）")
    print("5. 添加验证集和更详细的评估指标")

if __name__ == "__main__":
    demonstrate_usage()
