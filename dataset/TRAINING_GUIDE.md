# 大规模分子坐标预测训练配置指南

## 🚀 快速开始

### 1. 基础配置（适合单GPU，8GB显存）
```python
config = {
    'data_dir': 'train_4M_pyg',
    'max_samples': 50000,  # 限制5万个分子
    'model_type': 'gcn',
    'hidden_dim': 128,
    'num_layers': 4,
    'batch_size': 16,
    'learning_rate': 1e-4,
    'max_epochs': 100,
    'num_workers': 4,
}
```

### 2. 中等配置（适合高端GPU，16GB显存）
```python
config = {
    'data_dir': 'train_4M_pyg',
    'max_samples': 200000,  # 20万个分子
    'model_type': 'gat',
    'hidden_dim': 256,
    'num_layers': 6,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'max_epochs': 150,
    'num_workers': 8,
}
```

### 3. 高端配置（适合服务器级GPU，32GB+显存）
```python
config = {
    'data_dir': 'train_4M_pyg',
    'max_samples': None,  # 使用全部数据
    'model_type': 'transformer',
    'hidden_dim': 512,
    'num_layers': 12,
    'batch_size': 64,
    'learning_rate': 1e-4,
    'max_epochs': 300,
    'num_workers': 16,
}
```

## 📊 模型架构选择

### GCN (Graph Convolutional Network)
- **优点**: 简单高效，训练速度快
- **适用**: 基础实验，资源有限
- **参数**: `model_type: 'gcn'`

### GAT (Graph Attention Network) 
- **优点**: 注意力机制，表达能力强
- **适用**: 平衡性能和效果
- **参数**: `model_type: 'gat'`

### Transformer
- **优点**: 最强表达能力，SOTA性能
- **适用**: 高端硬件，追求最佳效果
- **参数**: `model_type: 'transformer'`

## 💾 内存和显存优化

### 降低显存使用：
- 减小 `batch_size`
- 减小 `hidden_dim` 
- 减少 `num_layers`
- 设置 `max_samples` 限制数据量

### 提升训练速度：
- 增加 `num_workers` (但不超过CPU核心数)
- 使用更大的 `batch_size`
- 启用混合精度训练（代码中可添加）

## 🔄 训练策略

### 学习率调度：
- 使用 `ReduceLROnPlateau` 自动调整
- 初始学习率 `1e-4` 适合大多数情况
- 如果损失下降慢，可尝试 `5e-4`

### 早停策略：
- `patience: 20` 表示20轮无改善则停止
- 适合大数据集，防止过拟合

### 正则化：
- `weight_decay: 1e-5` 权重衰减
- Dropout: 0.1 (模型中已设置)
- 梯度裁剪: max_norm=1.0

## 📈 预期训练时间

| 配置级别 | 数据量 | 单轮时间 | 预计总时间 |
|---------|--------|----------|-----------|
| 基础     | 5万    | 5分钟    | 8-10小时  |
| 中等     | 20万   | 15分钟   | 1-2天     |
| 高端     | 全量   | 30分钟   | 3-5天     |

## 🎯 性能指标参考

| 模型 | 隐藏维度 | 层数 | 预期RMSE | 参数量 |
|------|----------|------|----------|--------|
| GCN  | 128      | 4    | 8-12 Å   | ~100K  |
| GAT  | 256      | 6    | 6-10 Å   | ~500K  |
| Transformer | 512 | 12  | 4-8 Å    | ~2M    |

## 🛠 故障排除

### 显存不足 (CUDA out of memory)：
```python
# 减小batch_size
config['batch_size'] = 8

# 或减小模型大小
config['hidden_dim'] = 64
config['num_layers'] = 3
```

### 训练太慢：
```python
# 增加workers
config['num_workers'] = min(8, CPU核心数)

# 或限制数据量快速测试
config['max_samples'] = 10000
```

### 模型不收敛：
```python
# 调整学习率
config['learning_rate'] = 5e-5

# 或增加正则化
config['weight_decay'] = 1e-4
```

## 📝 使用说明

1. **修改配置**: 编辑 `large_scale_coordinate_predictor.py` 中的 config 字典
2. **运行训练**: `python large_scale_coordinate_predictor.py`
3. **监控进度**: 查看控制台输出和生成的图表
4. **查看结果**: 训练完成后检查输出目录中的结果文件

## 📂 输出文件说明

- `best_model.pth`: 最佳模型权重
- `training_curves.png`: 训练曲线图
- `training_results.json`: 训练结果统计
- `checkpoint_epoch_*.pth`: 定期保存的检查点

训练结果将自动保存在时间戳命名的目录中，方便管理多次实验。
