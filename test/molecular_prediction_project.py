"""
PyTorch Geometric 实战项目：分子溶解度预测

这个文件展示了如何使用PyTorch Geometric构建一个完整的
分子性质预测项目，以溶解度预测为例。

项目结构：
1. 数据加载和预处理
2. 模型定义
3. 训练循环
4. 评估和预测
5. 结果可视化

作者: AI Assistant  
日期: 2025-06-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import from_smiles
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 数据处理模块
# ============================================================================

class MolecularDataset:
    """
    分子数据集类，用于处理SMILES和性质数据
    """
    
    def __init__(self, smiles_list: List[str], properties: List[float], 
                 transform=None, target_transform=None):
        """
        初始化数据集
        
        参数:
            smiles_list: SMILES字符串列表
            properties: 对应的性质值列表
            transform: 图数据变换函数
            target_transform: 目标值变换函数
        """
        self.smiles_list = smiles_list
        self.properties = properties
        self.transform = transform
        self.target_transform = target_transform
        
        # 预处理数据
        self.data_list = self._process_molecules()
        
    def _process_molecules(self) -> List[Data]:
        """
        将SMILES转换为PyG数据对象
        """
        data_list = []
        failed_indices = []
        
        for i, (smiles, prop) in enumerate(zip(self.smiles_list, self.properties)):
            try:
                # 从SMILES创建图
                data = from_smiles(smiles)
                
                if data is None or data.x.size(0) == 0:
                    failed_indices.append(i)
                    continue
                
                # 添加目标属性
                if self.target_transform:
                    prop = self.target_transform(prop)
                data.y = torch.tensor([prop], dtype=torch.float)
                
                # 添加SMILES信息（用于调试）
                data.smiles = smiles
                
                # 应用变换
                if self.transform:
                    data = self.transform(data)
                
                data_list.append(data)
                
            except Exception as e:
                print(f"处理分子 {smiles} 时出错: {e}")
                failed_indices.append(i)
        
        if failed_indices:
            print(f"共有 {len(failed_indices)} 个分子处理失败")
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def get_statistics(self):
        """
        获取数据集统计信息
        """
        if not self.data_list:
            return {}
        
        # 节点数量统计
        num_nodes = [data.num_nodes for data in self.data_list]
        num_edges = [data.num_edges for data in self.data_list]
        properties = [data.y.item() for data in self.data_list]
        
        stats = {
            'num_molecules': len(self.data_list),
            'avg_num_atoms': np.mean(num_nodes),
            'max_num_atoms': np.max(num_nodes),
            'min_num_atoms': np.min(num_nodes),
            'avg_num_bonds': np.mean(num_edges) / 2,  # 无向图
            'feature_dim': self.data_list[0].x.shape[1],
            'property_mean': np.mean(properties),
            'property_std': np.std(properties),
            'property_range': (np.min(properties), np.max(properties))
        }
        
        return stats

def create_sample_dataset():
    """
    创建一个示例数据集用于演示
    包含一些常见分子及其水溶解度数据 (log S)
    """
    # 示例数据：分子SMILES和对应的水溶解度 (log S)
    molecules_data = [
        # 简单醇类
        ('C', -2.9),      # 甲醇
        ('CC', -3.1),     # 乙醇  
        ('CCC', -3.3),    # 丙醇
        ('CCCC', -3.7),   # 丁醇
        ('CCCCC', -4.1),  # 戊醇
        
        # 支链醇
        ('CC(C)O', -3.0),     # 异丙醇
        ('CC(C)(C)O', -2.8),  # 叔丁醇
        
        # 二醇
        ('OCC', -1.8),     # 乙二醇
        ('OCCC', -2.1),    # 丙二醇
        
        # 芳香族化合物
        ('c1ccccc1', -5.2),        # 苯
        ('Cc1ccccc1', -5.7),       # 甲苯
        ('c1ccccc1O', -3.5),       # 苯酚
        ('Cc1ccccc1O', -4.0),      # 甲酚
        
        # 羧酸
        ('CC(=O)O', -1.2),         # 乙酸
        ('CCC(=O)O', -1.5),        # 丙酸
        ('c1ccccc1C(=O)O', -2.8),  # 苯甲酸
        
        # 酯类
        ('CC(=O)OC', -2.5),        # 乙酸甲酯
        ('CC(=O)OCC', -2.8),       # 乙酸乙酯
        
        # 胺类
        ('CN', -1.0),              # 甲胺
        ('CCN', -1.3),             # 乙胺
        ('c1ccccc1N', -3.2),       # 苯胺
        
        # 醚类
        ('COC', -2.8),             # 二甲醚
        ('CCOC', -3.1),            # 甲乙醚
        ('CCOCC', -3.5),           # 二乙醚
        
        # 酮类
        ('CC(=O)C', -2.4),         # 丙酮
        ('CC(=O)CC', -2.7),        # 丁酮
        
        # 含卤素化合物
        ('CCCl', -3.8),            # 氯乙烷
        ('CCBr', -4.1),            # 溴乙烷
        ('CCF', -2.9),             # 氟乙烷
    ]
    
    smiles_list = [mol[0] for mol in molecules_data]
    solubility_list = [mol[1] for mol in molecules_data]
    
    return smiles_list, solubility_list

# ============================================================================
# 模型定义模块
# ============================================================================

class AttentionPooling(nn.Module):
    """
    注意力池化层，用于将节点特征聚合为图特征
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, batch):
        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(x), dim=0)
        
        # 加权求和
        if batch is None:
            return torch.sum(attn_weights * x, dim=0, keepdim=True)
        else:
            # 批处理情况下的实现会更复杂，这里简化
            return global_mean_pool(x, batch)

class AdvancedMolecularGNN(nn.Module):
    """
    高级分子图神经网络模型
    结合多种技术：图卷积、图注意力、残差连接、注意力池化
    """
    
    def __init__(self, num_features, hidden_dim=128, num_layers=3, 
                 dropout=0.2, pool_type='mean', use_attention=True):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool_type = pool_type
        self.use_attention = use_attention
        
        # 图卷积层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(num_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 图注意力层
        if use_attention:
            self.gat = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        
        # 批归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # 池化层
        if pool_type == 'attention':
            self.pool = AttentionPooling(hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # 保存输入用于残差连接
        identity = None
        
        # 图卷积层
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.batch_norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # 残差连接（从第二层开始）
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new
        
        # 图注意力层
        if self.use_attention:
            x_att = self.gat(x, edge_index)
            x = x + x_att  # 残差连接
        
        # 图级池化
        if self.pool_type == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pool_type == 'max':
            x = global_max_pool(x, batch)
        elif self.pool_type == 'attention':
            x = self.pool(x, batch)
        elif self.pool_type == 'concat':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
            # 更新分类器输入维度
            if not hasattr(self, '_updated_classifier'):
                self.classifier[0] = nn.Linear(x.size(1), self.classifier[0].out_features)
                self._updated_classifier = True
        
        # 最终预测
        return self.classifier(x)

# ============================================================================
# 训练和评估模块
# ============================================================================

class ModelTrainer:
    """
    模型训练器类
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in train_loader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # 前向传播
            pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(pred.squeeze(), batch.y)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(pred.squeeze().detach().cpu().numpy())
            targets.extend(batch.y.detach().cpu().numpy())
        
        # 计算指标
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return total_loss / len(train_loader), {'mse': mse, 'mae': mae, 'r2': r2}
    
    def evaluate(self, data_loader, criterion):
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                pred = self.model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(pred.squeeze(), batch.y)
                
                total_loss += loss.item()
                predictions.extend(pred.squeeze().cpu().numpy())
                targets.extend(batch.y.cpu().numpy())
        
        # 计算指标
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return total_loss / len(data_loader), {'mse': mse, 'mae': mae, 'r2': r2}, predictions, targets
    
    def train(self, train_loader, val_loader, num_epochs=100, lr=0.001, 
              weight_decay=1e-5, patience=20, min_delta=1e-6):
        """
        完整训练流程
        """
        # 优化器和学习率调度器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.MSELoss()
        
        # 早停相关
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练...")
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_metrics, _, _ = self.evaluate(val_loader, criterion)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            # 打印进度
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, R²: {train_metrics['r2']:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, R²: {val_metrics['r2']:.4f}")
                print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("训练完成！")

# ============================================================================
# 可视化模块
# ============================================================================

def plot_training_history(trainer: ModelTrainer):
    """
    绘制训练历史
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    epochs = range(1, len(trainer.train_losses) + 1)
    
    # 损失曲线
    axes[0, 0].plot(epochs, trainer.train_losses, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, trainer.val_losses, 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # R²曲线
    train_r2 = [m['r2'] for m in trainer.train_metrics]
    val_r2 = [m['r2'] for m in trainer.val_metrics]
    axes[0, 1].plot(epochs, train_r2, 'b-', label='Train R²')
    axes[0, 1].plot(epochs, val_r2, 'r-', label='Val R²')
    axes[0, 1].set_title('R² Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # MAE曲线
    train_mae = [m['mae'] for m in trainer.train_metrics]
    val_mae = [m['mae'] for m in trainer.val_metrics]
    axes[1, 0].plot(epochs, train_mae, 'b-', label='Train MAE')
    axes[1, 0].plot(epochs, val_mae, 'r-', label='Val MAE')
    axes[1, 0].set_title('Mean Absolute Error')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # MSE曲线
    train_mse = [m['mse'] for m in trainer.train_metrics]
    val_mse = [m['mse'] for m in trainer.val_metrics]
    axes[1, 1].plot(epochs, train_mse, 'b-', label='Train MSE')
    axes[1, 1].plot(epochs, val_mse, 'r-', label='Val MSE')
    axes[1, 1].set_title('Mean Squared Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_predictions(targets, predictions, title="预测结果"):
    """
    绘制预测值 vs 真实值的散点图
    """
    plt.figure(figsize=(8, 6))
    
    # 散点图
    plt.scatter(targets, predictions, alpha=0.6, s=50)
    
    # 完美预测线
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # 计算指标
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{title}\nR² = {r2:.4f}, MAE = {mae:.4f}, MSE = {mse:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 等比例坐标轴
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# ============================================================================
# 主函数和完整流程
# ============================================================================

def main():
    """
    主函数：完整的分子预测流程演示
    """
    print("=" * 80)
    print("PyTorch Geometric 分子溶解度预测项目")
    print("=" * 80)
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    smiles_list, solubility_list = create_sample_dataset()
    
    print(f"数据集大小: {len(smiles_list)} 个分子")
    print(f"溶解度范围: {min(solubility_list):.2f} 到 {max(solubility_list):.2f} log S")
    
    # 创建数据集
    dataset = MolecularDataset(smiles_list, solubility_list)
    stats = dataset.get_statistics()
    
    print(f"成功处理: {stats['num_molecules']} 个分子")
    print(f"平均原子数: {stats['avg_num_atoms']:.1f}")
    print(f"特征维度: {stats['feature_dim']}")
    
    # 2. 数据划分
    print("\n2. 数据划分...")
    train_data, test_data = train_test_split(dataset.data_list, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.6, 0.2, 0.2
    
    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    
    print(f"训练集: {len(train_data)} 个分子")
    print(f"验证集: {len(val_data)} 个分子")
    print(f"测试集: {len(test_data)} 个分子")
    
    # 3. 创建模型
    print("\n3. 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = AdvancedMolecularGNN(
        num_features=stats['feature_dim'],
        hidden_dim=64,
        num_layers=3,
        dropout=0.3,
        pool_type='concat',
        use_attention=True
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 训练模型
    print("\n4. 训练模型...")
    trainer = ModelTrainer(model, device)
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=50,
            lr=0.001,
            weight_decay=1e-5,
            patience=15
        )
        
        # 5. 评估模型
        print("\n5. 评估模型...")
        criterion = nn.MSELoss()
        
        # 训练集评估
        train_loss, train_metrics, train_preds, train_targets = trainer.evaluate(train_loader, criterion)
        print(f"训练集 - Loss: {train_loss:.4f}, R²: {train_metrics['r2']:.4f}, MAE: {train_metrics['mae']:.4f}")
        
        # 验证集评估
        val_loss, val_metrics, val_preds, val_targets = trainer.evaluate(val_loader, criterion)
        print(f"验证集 - Loss: {val_loss:.4f}, R²: {val_metrics['r2']:.4f}, MAE: {val_metrics['mae']:.4f}")
        
        # 测试集评估
        test_loss, test_metrics, test_preds, test_targets = trainer.evaluate(test_loader, criterion)
        print(f"测试集 - Loss: {test_loss:.4f}, R²: {test_metrics['r2']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        # 6. 可视化结果
        print("\n6. 可视化结果...")
        
        # 绘制训练历史
        plot_training_history(trainer)
        
        # 绘制预测结果
        plot_predictions(test_targets, test_preds, "测试集预测结果")
        
        # 7. 单分子预测示例
        print("\n7. 单分子预测示例...")
        test_molecules = {
            '水': 'O',
            '甲苯': 'Cc1ccccc1',
            '乙酸': 'CC(=O)O',
            '咖啡因': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
        }
        
        model.eval()
        with torch.no_grad():
            for name, smiles in test_molecules.items():
                try:
                    data = from_smiles(smiles)
                    if data is not None:
                        data = data.to(device)
                        pred = model(data.x, data.edge_index)
                        print(f"{name} ({smiles}): 预测溶解度 = {pred.item():.2f} log S")
                except:
                    print(f"无法预测 {name} ({smiles})")
        
        print("\n" + "=" * 80)
        print("项目演示完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        print("可能的原因:")
        print("1. 数据集太小，导致训练不稳定")
        print("2. 缺少必要的依赖包 (torch-geometric, rdkit-pypi)")
        print("3. 内存不足")

if __name__ == "__main__":
    main()
