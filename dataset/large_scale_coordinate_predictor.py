import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.data import DataLoader, InMemoryDataset
import numpy as np
import os
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedCoordinatePredictor(nn.Module):
    """高级3D坐标预测模型 - 用于大规模训练"""
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=3, num_layers=6, model_type='gat'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        # 原子嵌入层
        self.atom_embedding = nn.Embedding(100, hidden_dim)  # 支持原子序数到100
        
        # 图神经网络层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if model_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif model_type == 'gat':
                conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=0.1)
            elif model_type == 'transformer':
                conv = TransformerConv(hidden_dim, hidden_dim, heads=8, dropout=0.1)
            else:
                conv = GCNConv(hidden_dim, hidden_dim)
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # 坐标预测头
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 原子嵌入
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        x = self.atom_embedding(x.long().squeeze(-1))
        
        # 残差连接的图卷积
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            
            # 残差连接（第一层后开始）
            if i > 0:
                x = x + x_residual
            
            x = self.dropout(x)
        
        # 预测坐标
        pred_pos = self.coord_predictor(x)
        
        return pred_pos

class LoadedDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        return ['data.pt']

class LargeScaleTrainer:
    """大规模训练器"""
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # 训练记录
        self.train_history = {
            'train_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'learning_rates': []
        }
        
    def load_dataset(self):
        """加载数据集"""
        logger.info("加载数据集...")
        data_dir = Path(self.config['data_dir'])
        
        if not (data_dir / "processed" / "data.pt").exists():
            logger.error(f"数据集未找到: {data_dir}")
            raise FileNotFoundError(f"请先运行数据转换脚本生成 {data_dir}/processed/data.pt")
        
        dataset = LoadedDataset(data_dir)
        logger.info(f"数据集大小: {len(dataset)} 个分子")
        
        # 限制数据大小（如果指定）
        if self.config.get('max_samples') and self.config['max_samples'] < len(dataset):
            dataset = dataset[:self.config['max_samples']]
            logger.info(f"使用数据: {len(dataset)} 个分子")
        
        return dataset
    
    def normalize_coordinates(self, dataset):
        """坐标归一化"""
        logger.info("计算坐标归一化参数...")
        
        # 采样一部分数据计算统计量（避免内存不足）
        sample_size = min(10000, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        
        all_pos = []
        for idx in sample_indices:
            all_pos.append(dataset[idx].pos)
        
        all_pos = torch.cat(all_pos, dim=0)
        pos_mean = all_pos.mean(dim=0)
        pos_std = all_pos.std(dim=0)
        
        logger.info(f"位置均值: {pos_mean}")
        logger.info(f"位置标准差: {pos_std}")
        
        # 防止除零
        pos_std = torch.where(pos_std < 1e-6, torch.ones_like(pos_std), pos_std)
        
        return pos_mean, pos_std
    
    def create_data_loaders(self, dataset, pos_mean, pos_std):
        """创建数据加载器"""
        logger.info("创建数据加载器...")
        
        # 数据集分割
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        # 随机打乱
        indices = torch.randperm(total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = [dataset[i] for i in train_indices]
        val_dataset = [dataset[i] for i in val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        
        logger.info(f"训练集: {len(train_dataset)}")
        logger.info(f"验证集: {len(val_dataset)}")
        logger.info(f"测试集: {len(test_dataset)}")
        
        # 归一化
        for data in train_dataset + val_dataset + test_dataset:
            data.pos = (data.pos - pos_mean) / pos_std
        
        # DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self):
        """创建模型"""
        model = AdvancedCoordinatePredictor(
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            model_type=self.config['model_type']
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数数量: {total_params:,}")
        
        return model
    
    def create_optimizer_scheduler(self, model):
        """创建优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.7,
            patience=5
        )
        
        return optimizer, scheduler
    
    def validate(self, model, val_loader):
        """验证模型"""
        model.eval()
        total_rmse = 0
        total_mae = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                pred_pos = model(batch)
                
                rmse = torch.sqrt(F.mse_loss(pred_pos, batch.pos))
                mae = F.l1_loss(pred_pos, batch.pos)
                
                total_rmse += rmse.item()
                total_mae += mae.item()
                total_batches += 1
        
        return total_rmse / total_batches, total_mae / total_batches
    
    def save_checkpoint(self, model, optimizer, epoch, pos_mean, pos_std, val_rmse):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'pos_mean': pos_mean,
            'pos_std': pos_std,
            'val_rmse': val_rmse,
            'config': self.config,
            'train_history': self.train_history
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        best_path = self.output_dir / 'best_model.pth'
        if not best_path.exists() or val_rmse < torch.load(best_path, weights_only=True)['val_rmse']:
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型，验证RMSE: {val_rmse:.6f}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 5))
        
        # 训练损失
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        
        # 验证RMSE
        plt.subplot(1, 3, 2)
        plt.plot(self.train_history['val_rmse'])
        plt.title('Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE (Å)')
        
        # 学习率
        plt.subplot(1, 3, 3)
        plt.plot(self.train_history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300)
        plt.close()
    
    def train(self):
        """主训练流程"""
        logger.info("=== 大规模3D分子坐标预测训练开始 ===")
        start_time = time.time()
        
        # 1. 加载数据集
        dataset = self.load_dataset()
        
        # 2. 坐标归一化
        pos_mean, pos_std = self.normalize_coordinates(dataset)
        
        # 3. 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders(dataset, pos_mean, pos_std)
        
        # 4. 创建模型
        model = self.create_model()
        
        # 5. 创建优化器
        optimizer, scheduler = self.create_optimizer_scheduler(model)
        
        # 6. 训练循环
        best_val_rmse = float('inf')
        patience_counter = 0
        
        logger.info("开始训练...")
        for epoch in range(self.config['max_epochs']):
            # 训练
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                pred_pos = model(batch)
                loss = F.mse_loss(pred_pos, batch.pos)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # 打印进度
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.config['max_epochs']}, "
                              f"Batch {batch_idx}/{len(train_loader)}, "
                              f"Loss: {loss.item():.6f}")
            
            avg_train_loss = total_loss / num_batches
            
            # 验证
            val_rmse, val_mae = self.validate(model, val_loader)
            
            # 更新学习率
            scheduler.step(val_rmse)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(avg_train_loss)
            self.train_history['val_rmse'].append(val_rmse)
            self.train_history['val_mae'].append(val_mae)
            self.train_history['learning_rates'].append(current_lr)
            
            # 打印结果
            logger.info(f"Epoch {epoch+1:3d} | "
                       f"Train Loss: {avg_train_loss:.6f} | "
                       f"Val RMSE: {val_rmse:.6f} | "
                       f"Val MAE: {val_mae:.6f} | "
                       f"LR: {current_lr:.2e}")
            
            # 保存检查点
            if (epoch + 1) % self.config['save_every'] == 0:
                self.save_checkpoint(model, optimizer, epoch + 1, pos_mean, pos_std, val_rmse)
            
            # 早停
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.config['patience']:
                logger.info(f"早停在第 {epoch+1} 轮，最佳验证RMSE: {best_val_rmse:.6f}")
                break
            
            # 绘制训练曲线
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()
        
        # 最终测试
        logger.info("进行最终测试...")
        test_rmse, test_mae = self.validate(model, test_loader)
        logger.info(f"最终测试结果 - RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}")
        
        # 保存最终结果
        results = {
            'best_val_rmse': best_val_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'training_time': time.time() - start_time,
            'config': self.config
        }
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.plot_training_curves()
        
        logger.info(f"训练完成！总时间: {time.time() - start_time:.2f}秒")
        logger.info(f"结果保存在: {self.output_dir}")
        
        return model, results

def main():
    """主函数"""
    # 大规模训练配置
    config = {
        # 数据相关
        'data_dir': 'train_4M_pyg',
        'max_samples': None,  # None表示使用全部数据，或设置如100000限制样本数
        
        # 模型相关
        'model_type': 'gat',  # 'gcn', 'gat', 'transformer'
        'hidden_dim': 256,
        'num_layers': 8,
        
        # 训练相关
        'batch_size': 32,  # 根据GPU内存调整
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_epochs': 200,
        'patience': 20,
        
        # 系统相关
        'num_workers': 8,  # 数据加载并行数
        'output_dir': f'large_scale_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'save_every': 10,  # 每10轮保存一次
    }
    
    # 创建训练器
    trainer = LargeScaleTrainer(config)
    
    # 开始训练
    model, results = trainer.train()
    
    print("\n" + "="*60)
    print("🎉 大规模训练完成！")
    print(f"最佳验证RMSE: {results['best_val_rmse']:.6f} Å")
    print(f"测试RMSE: {results['test_rmse']:.6f} Å")
    print(f"测试MAE: {results['test_mae']:.6f} Å")
    print(f"训练时间: {results['training_time']:.2f} 秒")
    print("="*60)

if __name__ == "__main__":
    main()
