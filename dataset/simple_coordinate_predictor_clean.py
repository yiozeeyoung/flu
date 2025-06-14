import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import DataLoader
import numpy as np
import os
from pathlib import Path

# 导入数据集
from fairchem_to_pyg_final import FairChemToGeometricDataset

class SimpleCoordinatePredictor(nn.Module):
    """简化的3D坐标预测模型"""
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=3):
        super().__init__()
        
        # 图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # 预测坐标
        pred_pos = self.predictor(x)
        
        return pred_pos

def train_simple_coordinate_predictor():
    """训练简化的坐标预测模型"""
    print("=== 简化版3D分子坐标预测模型训练 ===")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")    # 加载数据集
    data_dir = Path("train_4M_pyg")
    
    # 如果已经有处理好的数据，直接加载
    if (data_dir / "processed" / "data.pt").exists():
        from torch_geometric.data import InMemoryDataset
        import torch
        
        class LoadedDataset(InMemoryDataset):
            def __init__(self, root):
                super().__init__(root)
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            
            @property
            def processed_file_names(self):
                return ['data.pt']
        
        dataset = LoadedDataset(data_dir)
    else:
        # 如果没有处理好的数据，从原始数据创建
        aselmdb_dir = Path("train_4M")
        aselmdb_paths = list(aselmdb_dir.glob("*.aselmdb"))
        dataset = FairChemToGeometricDataset(data_dir, aselmdb_paths, max_samples=100)
    
    # 为演示使用小样本
    dataset = dataset[:50]
    print(f"数据集大小: {len(dataset)} 个分子")
    
    # 坐标归一化
    print("进行坐标归一化...")
    all_pos = torch.cat([data.pos for data in dataset], dim=0)
    pos_mean = all_pos.mean(dim=0)
    pos_std = all_pos.std(dim=0)
    print(f"位置均值: {pos_mean}")
    print(f"位置标准差: {pos_std}")
    
    # 归一化
    for data in dataset:
        data.pos = (data.pos - pos_mean) / pos_std
    
    # 数据集分割
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 模型
    model = SimpleCoordinatePredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 训练
    print("开始训练...")
    best_val_rmse = float('inf')
    
    for epoch in range(100):
        # 训练
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_pos = model(batch)
            loss = F.mse_loss(pred_pos, batch.pos)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        if epoch % 10 == 0:
            model.eval()
            val_rmse = 0
            val_mae = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_pos = model(batch)
                    
                    rmse = torch.sqrt(F.mse_loss(pred_pos, batch.pos))
                    mae = F.l1_loss(pred_pos, batch.pos)
                    
                    val_rmse += rmse.item()
                    val_mae += mae.item()
                    val_count += 1
            
            val_rmse /= val_count
            val_mae /= val_count
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss/len(train_loader):.6f} | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}")
            
            # 保存最佳模型
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'pos_mean': pos_mean,
                    'pos_std': pos_std,
                    'val_rmse': val_rmse
                }, 'best_simple_coordinate_model.pth')
    
    print(f"最佳验证 RMSE: {best_val_rmse:.6f}")
    
    # 测试
    model.eval()
    test_rmse = 0
    test_mae = 0
    test_count = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred_pos = model(batch)
            
            rmse = torch.sqrt(F.mse_loss(pred_pos, batch.pos))
            mae = F.l1_loss(pred_pos, batch.pos)
            
            test_rmse += rmse.item()
            test_mae += mae.item()
            test_count += 1
    
    test_rmse /= test_count
    test_mae /= test_count
    
    print(f"测试 RMSE: {test_rmse:.6f}")
    print(f"测试 MAE: {test_mae:.6f}")
    
    return model, {'pos_mean': pos_mean, 'pos_std': pos_std}

def demo_simple_prediction():
    """演示简单坐标预测"""
    print("=== 简单坐标预测演示 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查是否有训练好的模型
    if os.path.exists('best_simple_coordinate_model.pth'):
        # 加载模型
        checkpoint = torch.load('best_simple_coordinate_model.pth', weights_only=True)
        model = SimpleCoordinatePredictor().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        pos_mean = checkpoint['pos_mean'].to(device)
        pos_std = checkpoint['pos_std'].to(device)        # 加载数据
        data_dir = Path("train_4M_pyg")
        
        # 如果已经有处理好的数据，直接加载
        if (data_dir / "processed" / "data.pt").exists():
            from torch_geometric.data import InMemoryDataset
            import torch
            
            class LoadedDataset(InMemoryDataset):
                def __init__(self, root):
                    super().__init__(root)
                    self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
                
                @property
                def processed_file_names(self):
                    return ['data.pt']
            
            dataset = LoadedDataset(data_dir)
        else:
            # 如果没有处理好的数据，从原始数据创建
            aselmdb_dir = Path("train_4M")
            aselmdb_paths = list(aselmdb_dir.glob("*.aselmdb"))
            dataset = FairChemToGeometricDataset(data_dir, aselmdb_paths, max_samples=100)
        dataset = dataset[:50]
        
        # 归一化
        for data in dataset:
            data.pos = (data.pos - pos_mean.cpu()) / pos_std.cpu()
        
        # 预测示例
        sample = dataset[0].to(device)
        
        with torch.no_grad():
            # 使用DataLoader包装单个样本
            loader = DataLoader([sample], batch_size=1)
            batch = next(iter(loader))
            pred_pos = model(batch)
            
            # 反归一化
            pred_pos = pred_pos * pos_std + pos_mean
            true_pos = sample.pos * pos_std + pos_mean
            
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
                symbol = atomic_symbols.get(int(atomic_num), f'Z{int(atomic_num)}')
                print(f"    {symbol}: {count} 个原子")
            
            print(f"\n前5个原子的坐标比较:")
            print("原子 |    真实坐标 (Å)    |    预测坐标 (Å)    |  误差 (Å)")
            print("-" * 70)
            
            pred_pos_squeezed = pred_pos.squeeze(0)
            for i in range(min(5, sample.x.size(0))):
                true_coord = true_pos[i]
                pred_coord = pred_pos_squeezed[i]
                error = torch.norm(true_coord - pred_coord).item()
                atomic_num = int(sample.x[i].item())
                symbol = atomic_symbols.get(atomic_num, f'Z{atomic_num}')
                
                tc = true_coord.cpu()
                pc = pred_coord.cpu()
                print(f"{symbol:2s}{i+1:2d} | {tc[0].item():6.3f},{tc[1].item():6.3f},{tc[2].item():6.3f} | {pc[0].item():6.3f},{pc[1].item():6.3f},{pc[2].item():6.3f} | {error:6.3f}")
    
    else:
        print("未找到训练好的模型")

if __name__ == "__main__":
    # 训练模型
    model, checkpoint = train_simple_coordinate_predictor()
    
    # 演示预测
    demo_simple_prediction()
