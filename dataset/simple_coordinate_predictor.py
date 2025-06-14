"""
简化版3D分子坐标预测模型
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader, InMemoryDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class LoadedDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        return ['data.pt']

class SimpleCoordinate3DPredictor(torch.nn.Module):
    """
    简化版3D坐标预测模型
    """
    
    def __init__(self, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        # 原子特征嵌入
        self.atom_embedding = torch.nn.Embedding(100, hidden_dim)
        
        # GCN层
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 坐标预测网络
        self.pos_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 3)  # 输出 x, y, z 坐标
        )
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 原子序数嵌入
        x = self.atom_embedding(x.squeeze(-1))
        
        # GCN层
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 预测3D坐标
        predicted_pos = self.pos_predictor(x)
        
        return predicted_pos

def coordinate_loss(pred_pos, true_pos):
    """计算坐标预测损失"""
    return F.mse_loss(pred_pos, true_pos)

def distance_loss(pred_pos, true_pos, edge_index, weight=0.5):
    """距离保持损失"""
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    
    row, col = edge_index
    pred_dist = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    true_dist = torch.norm(true_pos[row] - true_pos[col], dim=1)
    
    return weight * F.mse_loss(pred_dist, true_dist)

def evaluate_model(model, data_loader, device):
    """评估模型性能"""
    model.eval()
    total_rmse = 0
    total_mae = 0
    num_atoms = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred_pos = model(batch)
            true_pos = batch.pos
            
            mse = F.mse_loss(pred_pos, true_pos, reduction='none').sum(dim=-1)
            rmse = torch.sqrt(mse)
            mae = F.l1_loss(pred_pos, true_pos, reduction='none').sum(dim=-1)
            
            total_rmse += rmse.sum().item()
            total_mae += mae.sum().item()
            num_atoms += pred_pos.size(0)
    
    return total_rmse / num_atoms, total_mae / num_atoms

def train_simple_coordinate_predictor():
    """训练简化版坐标预测模型"""
    print("=== 简化版3D分子坐标预测模型训练 ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    dataset = LoadedDataset(output_dir)
    print(f"数据集大小: {len(dataset)} 个分子")
    
    # 坐标归一化
    print("进行坐标归一化...")
    all_positions = torch.cat([data.pos for data in dataset], dim=0)
    pos_mean = all_positions.mean(dim=0)
    pos_std = all_positions.std(dim=0)
    
    print(f"位置均值: {pos_mean}")
    print(f"位置标准差: {pos_std}")
    
    for data in dataset:
        data.pos = (data.pos - pos_mean) / pos_std
    
    # 数据集划分
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 模型
    model = SimpleCoordinate3DPredictor(
        hidden_dim=64,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 训练
    num_epochs = 100
    best_val_rmse = float('inf')
    train_losses = []
    val_rmses = []
    
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_pos = model(batch)
            
            # 组合损失
            coord_loss = coordinate_loss(pred_pos, batch.pos)
            dist_loss = distance_loss(pred_pos, batch.pos, batch.edge_index)
            total_loss = coord_loss + dist_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证
        val_rmse, val_mae = evaluate_model(model, val_loader, device)
        val_rmses.append(val_rmse)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.6f} | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}')
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'pos_mean': pos_mean,
                'pos_std': pos_std,
                'val_rmse': val_rmse,
                'epoch': epoch
            }, 'simple_coordinate_model.pth')
    
    # 测试
    print(f"\n最佳验证 RMSE: {best_val_rmse:.6f}")
    
    # 加载最佳模型进行测试
    checkpoint = torch.load('simple_coordinate_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_rmse, test_mae = evaluate_model(model, test_loader, device)
    print(f'测试 RMSE: {test_rmse:.6f}')
    print(f'测试 MAE: {test_mae:.6f}')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_rmses)
    plt.title('Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_coordinate_training.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, checkpoint

def demo_simple_prediction():
    """演示简单坐标预测"""
    print("\n=== 简单坐标预测演示 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists('simple_coordinate_model.pth'):
        checkpoint = torch.load('simple_coordinate_model.pth')
        pos_mean = checkpoint['pos_mean']
        pos_std = checkpoint['pos_std']
        
        model = SimpleCoordinate3DPredictor(
            hidden_dim=64,
            num_layers=3,
            dropout=0.1
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 加载数据
        output_dir = r"D:\code\flu\dataset\train_4M_pyg"
        dataset = LoadedDataset(output_dir)
          # 归一化
        for data in dataset:
            data.pos = (data.pos - pos_mean) / pos_std
        
        # 预测示例
        sample = dataset[0].to(device)
          with torch.no_grad():
            # 使用DataLoader包装单个样本
            loader = DataLoader([sample], batch_size=1)
            batch = next(iter(loader))
            pred_pos = model(batch)
            
            # 反归一化
            pred_pos = pred_pos * pos_std.to(device) + pos_mean.to(device)
            true_pos = sample.pos * pos_std.to(device) + pos_mean.to(device)
            
            rmse = torch.sqrt(F.mse_loss(pred_pos.squeeze(0), true_pos))
            mae = F.l1_loss(pred_pos.squeeze(0), true_pos)
            
            print(f"预测结果:")
            print(f"  原子数量: {sample.x.size(0)}")
            print(f"  RMSE: {rmse.item():.6f} Å")
            print(f"  MAE: {mae.item():.6f} Å")
            
            # 显示原子类型分布
            atomic_nums = sample.x.squeeze(-1).cpu().numpy()
            from collections import Counter
            atom_counts = Counter(atomic_nums)
            
            atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S'}
            print(f"\n  分子组成:")
            for atomic_num, count in sorted(atom_counts.items()):
                symbol = atomic_symbols.get(atomic_num, f'Z{atomic_num}')
                print(f"    {symbol}: {count} 个原子")
            
            print(f"\n前5个原子的坐标比较:")
            print("原子 |    真实坐标 (Å)    |    预测坐标 (Å)    |  误差 (Å)")
            print("-" * 70)            for i in range(min(5, sample.x.size(0))):
                true_coord = true_pos[i]
                pred_coord = pred_pos.squeeze(0)[i]
                error = torch.norm(true_coord - pred_coord).item()
                atomic_num = int(sample.x[i].item())
                symbol = atomic_symbols.get(atomic_num, f'Z{atomic_num}')
                print(f"{symbol:2s}{i+1:2d} | {true_coord[0].item():6.3f},{true_coord[1].item():6.3f},{true_coord[2].item():6.3f} | {pred_coord[0].item():6.3f},{pred_coord[1].item():6.3f},{pred_coord[2].item():6.3f} | {error:6.3f}")
    
    else:
        print("未找到训练好的模型")

if __name__ == "__main__":
    # 训练模型
    model, checkpoint = train_simple_coordinate_predictor()
    
    # 演示预测
    demo_simple_prediction()
