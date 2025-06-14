"""
3D分子坐标预测模型
使用 FairChem 数据集训练预测原子3D坐标的模型
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, TransformerConv
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

class Coordinate3DPredictor(torch.nn.Module):
    """
    3D坐标预测模型
    
    输入: 原子类型、初始坐标(可选)、分子图结构
    输出: 预测的3D坐标
    """
    
    def __init__(self, num_atom_features=1, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 原子特征嵌入
        self.atom_embedding = torch.nn.Embedding(100, hidden_dim)  # 支持原子序数1-99
        
        # 图神经网络层
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False))
        
        # 位置预测头
        self.pos_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 3)  # 输出 x, y, z 坐标
        )
        
        # 层归一化
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 原子序数嵌入
        x = self.atom_embedding(x.squeeze(-1))
        
        # 图神经网络前向传播
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            # 残差连接
            residual = x
            x = conv(x, edge_index)
            x = norm(x + residual)  # 残差连接 + 层归一化
            x = F.relu(x)
            x = self.dropout(x)
        
        # 预测3D坐标
        predicted_pos = self.pos_predictor(x)
        
        return predicted_pos

class EnhancedCoordinate3DPredictor(torch.nn.Module):
    """
    增强版3D坐标预测模型，包含距离约束
    """
    
    def __init__(self, num_atom_features=1, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        
        # 原子特征嵌入
        self.atom_embedding = torch.nn.Embedding(100, hidden_dim)
        
        # 边特征处理（距离信息）
        self.edge_embedding = torch.nn.Linear(1, hidden_dim // 4)
          # Transformer-based 图神经网络
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # 第一层需要处理嵌入维度
                self.convs.append(TransformerConv(
                    hidden_dim, 
                    hidden_dim, 
                    heads=8, 
                    dropout=dropout,
                    edge_dim=hidden_dim // 4
                ))
            else:
                self.convs.append(TransformerConv(
                    hidden_dim, 
                    hidden_dim, 
                    heads=8, 
                    dropout=dropout,
                    edge_dim=hidden_dim // 4
                ))
        
        # 多头注意力机制用于坐标预测
        self.attention = torch.nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # 坐标预测网络
        self.pos_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 3)
        )
        
        # 层归一化
        self.layer_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 原子序数嵌入
        x = self.atom_embedding(x.squeeze(-1))
        
        # 边特征嵌入
        if edge_attr is not None and edge_attr.size(0) > 0:
            edge_attr = self.edge_embedding(edge_attr)
        else:
            edge_attr = None
          # 图神经网络层
        for i, (conv, norm) in enumerate(zip(self.convs, self.layer_norms)):
            residual = x
            x = conv(x, edge_index, edge_attr)
            
            # 检查维度是否匹配再做残差连接
            if x.shape == residual.shape:
                x = norm(x + residual)
            else:
                x = norm(x)
            
            x = F.gelu(x)
            x = self.dropout(x)
        
        # 自注意力机制（在批次内进行）
        # 这里需要将图数据重组为序列
        batch_size = batch.max().item() + 1
        max_nodes = torch.bincount(batch).max().item()
        
        # 创建填充的序列
        padded_x = torch.zeros(batch_size, max_nodes, x.size(-1), device=x.device)
        mask = torch.ones(batch_size, max_nodes, device=x.device, dtype=torch.bool)
        
        for i in range(batch_size):
            node_mask = (batch == i)
            num_nodes = node_mask.sum().item()
            padded_x[i, :num_nodes] = x[node_mask]
            mask[i, :num_nodes] = False
        
        # 应用自注意力
        attended_x, _ = self.attention(padded_x, padded_x, padded_x, key_padding_mask=mask)
        
        # 重新展平
        output_x = torch.zeros_like(x)
        for i in range(batch_size):
            node_mask = (batch == i)
            num_nodes = node_mask.sum().item()
            output_x[node_mask] = attended_x[i, :num_nodes]
        
        # 预测3D坐标
        predicted_pos = self.pos_predictor(output_x)
        
        return predicted_pos

def coordinate_loss(pred_pos, true_pos, reduce_mean=True):
    """
    计算坐标预测损失
    
    Args:
        pred_pos: 预测的坐标 [N, 3]
        true_pos: 真实坐标 [N, 3]
        reduce_mean: 是否计算平均值
    """
    # 均方误差损失
    mse_loss = F.mse_loss(pred_pos, true_pos, reduction='none')
    
    if reduce_mean:
        return mse_loss.mean()
    else:
        return mse_loss.sum(dim=-1)  # 每个原子的损失

def distance_loss(pred_pos, true_pos, edge_index, weight=1.0):
    """
    距离保持损失 - 确保预测的原子间距离与真实距离接近
    """
    if edge_index.size(1) == 0:
        return torch.tensor(0.0, device=pred_pos.device)
    
    # 计算预测距离
    row, col = edge_index
    pred_dist = torch.norm(pred_pos[row] - pred_pos[col], dim=1)
    true_dist = torch.norm(true_pos[row] - true_pos[col], dim=1)
    
    # 距离损失
    dist_loss = F.mse_loss(pred_dist, true_dist)
    
    return weight * dist_loss

def angle_loss(pred_pos, true_pos, edge_index, weight=0.5):
    """
    角度保持损失 - 保持分子几何结构
    """
    if edge_index.size(1) < 3:
        return torch.tensor(0.0, device=pred_pos.device)
    
    # 这里可以实现角度损失，暂时返回0
    return torch.tensor(0.0, device=pred_pos.device)

def combined_loss(pred_pos, true_pos, edge_index, coord_weight=1.0, dist_weight=0.5, angle_weight=0.1):
    """
    组合损失函数
    """
    coord_loss_val = coordinate_loss(pred_pos, true_pos)
    dist_loss_val = distance_loss(pred_pos, true_pos, edge_index, dist_weight)
    angle_loss_val = angle_loss(pred_pos, true_pos, edge_index, angle_weight)
    
    total_loss = coord_weight * coord_loss_val + dist_loss_val + angle_loss_val
    
    return total_loss, coord_loss_val, dist_loss_val, angle_loss_val

def evaluate_model(model, data_loader, device):
    """
    评估模型性能
    """
    model.eval()
    total_rmse = 0
    total_mae = 0
    num_atoms = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            pred_pos = model(batch)
            true_pos = batch.pos
            
            # 计算RMSE和MAE
            mse = F.mse_loss(pred_pos, true_pos, reduction='none').sum(dim=-1)
            rmse = torch.sqrt(mse)
            mae = F.l1_loss(pred_pos, true_pos, reduction='none').sum(dim=-1)
            
            total_rmse += rmse.sum().item()
            total_mae += mae.sum().item()
            num_atoms += pred_pos.size(0)
    
    avg_rmse = total_rmse / num_atoms
    avg_mae = total_mae / num_atoms
    
    return avg_rmse, avg_mae

def train_coordinate_predictor():
    """
    训练3D坐标预测模型
    """
    print("=== 3D分子坐标预测模型训练 ===\n")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据集
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    dataset = LoadedDataset(output_dir)
    print(f"数据集大小: {len(dataset)} 个分子")
    
    # 数据预处理 - 坐标归一化
    print("进行坐标归一化...")
    all_positions = torch.cat([data.pos for data in dataset], dim=0)
    pos_mean = all_positions.mean(dim=0)
    pos_std = all_positions.std(dim=0)
    
    print(f"位置均值: {pos_mean}")
    print(f"位置标准差: {pos_std}")
    
    # 归一化数据集
    for data in dataset:
        data.pos = (data.pos - pos_mean) / pos_std
    
    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"训练集: {len(train_dataset)}")
    print(f"验证集: {len(val_dataset)}")
    print(f"测试集: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 创建模型
    model = EnhancedCoordinate3DPredictor(
        hidden_dim=128,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
      # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10
    )
    
    # 训练循环
    num_epochs = 50
    best_val_rmse = float('inf')
    train_losses = []
    val_rmses = []
    
    print("\n开始训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            pred_pos = model(batch)
            
            # 计算损失
            total_loss, coord_loss_val, dist_loss_val, angle_loss_val = combined_loss(
                pred_pos, batch.pos, batch.edge_index
            )
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            train_pbar.set_postfix({
                'Loss': f'{total_loss.item():.6f}',
                'Coord': f'{coord_loss_val.item():.6f}',
                'Dist': f'{dist_loss_val.item():.6f}'
            })
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        val_rmse, val_mae = evaluate_model(model, val_loader, device)
        val_rmses.append(val_rmse)
        
        scheduler.step(val_rmse)
        
        print(f'Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}')
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pos_mean': pos_mean,
                'pos_std': pos_std,
                'val_rmse': val_rmse,
                'epoch': epoch
            }, 'best_coordinate_model.pth')
            print(f'保存最佳模型 (Val RMSE: {val_rmse:.6f})')
    
    # 测试阶段
    print("\n测试最佳模型...")
    checkpoint = torch.load('best_coordinate_model.pth')
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
    plt.savefig('coordinate_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, checkpoint

def demo_coordinate_prediction():
    """
    演示坐标预测
    """
    print("\n=== 坐标预测演示 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    if os.path.exists('best_coordinate_model.pth'):
        checkpoint = torch.load('best_coordinate_model.pth')
        pos_mean = checkpoint['pos_mean']
        pos_std = checkpoint['pos_std']
        
        model = EnhancedCoordinate3DPredictor(
            hidden_dim=128,
            num_layers=4,
            dropout=0.1
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 加载测试数据
        output_dir = r"D:\code\flu\dataset\train_4M_pyg"
        dataset = LoadedDataset(output_dir)
        
        # 归一化测试数据
        for data in dataset:
            data.pos = (data.pos - pos_mean) / pos_std
        
        # 选择一个样本进行预测
        sample = dataset[0].to(device)
        
        with torch.no_grad():
            pred_pos = model(sample.unsqueeze(0))
            
            # 反归一化
            pred_pos = pred_pos * pos_std + pos_mean
            true_pos = sample.pos * pos_std + pos_mean
            
            # 计算误差
            rmse = torch.sqrt(F.mse_loss(pred_pos, true_pos.unsqueeze(0)))
            mae = F.l1_loss(pred_pos, true_pos.unsqueeze(0))
            
            print(f"样本预测结果:")
            print(f"  原子数量: {sample.x.size(0)}")
            print(f"  RMSE: {rmse.item():.6f}")
            print(f"  MAE: {mae.item():.6f}")
            
            # 显示前几个原子的预测结果
            print(f"\n前5个原子的坐标比较:")
            print("原子 |    真实坐标    |    预测坐标    |   误差")
            print("-" * 55)
            for i in range(min(5, sample.x.size(0))):
                true_coord = true_pos[i]
                pred_coord = pred_pos[0, i]
                error = torch.norm(true_coord - pred_coord).item()
                print(f"{i+1:3d}  | {true_coord[0]:6.3f},{true_coord[1]:6.3f},{true_coord[2]:6.3f} | {pred_coord[0]:6.3f},{pred_coord[1]:6.3f},{pred_coord[2]:6.3f} | {error:6.3f}")
    
    else:
        print("未找到训练好的模型，请先运行训练")

if __name__ == "__main__":
    # 训练模型
    model, checkpoint = train_coordinate_predictor()
    
    # 演示预测
    demo_coordinate_prediction()
