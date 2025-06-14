"""
大规模分子坐标预测 - 配置选择器
根据你的硬件资源自动选择最佳配置
"""

import torch
import psutil
import json
from pathlib import Path

def get_system_info():
    """获取系统信息"""
    info = {
        'cpu_cores': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'cuda_available': torch.cuda.is_available(),
        'gpu_memory_gb': 0,
        'gpu_name': 'None'
    }
    
    if torch.cuda.is_available():
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return info

def suggest_config(system_info):
    """根据系统信息推荐配置"""
    configs = {
        'light': {
            'description': '轻量级配置 - 适合入门和快速测试',
            'data_dir': 'train_4M_pyg',
            'max_samples': 10000,
            'model_type': 'gcn',
            'hidden_dim': 64,
            'num_layers': 3,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 50,
            'patience': 10,
            'num_workers': min(4, system_info['cpu_cores']),
            'expected_time': '1-2小时',
            'expected_rmse': '10-15 Å'
        },
        
        'medium': {
            'description': '中等配置 - 平衡性能与效果',
            'data_dir': 'train_4M_pyg',
            'max_samples': 100000,
            'model_type': 'gat',
            'hidden_dim': 128,
            'num_layers': 6,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 100,
            'patience': 15,
            'num_workers': min(8, system_info['cpu_cores']),
            'expected_time': '8-12小时',
            'expected_rmse': '6-10 Å'
        },
        
        'heavy': {
            'description': '重型配置 - 追求最佳效果',
            'data_dir': 'train_4M_pyg',
            'max_samples': 500000,
            'model_type': 'gat',
            'hidden_dim': 256,
            'num_layers': 8,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 200,
            'patience': 20,
            'num_workers': min(12, system_info['cpu_cores']),
            'expected_time': '1-2天',
            'expected_rmse': '4-8 Å'
        },
        
        'extreme': {
            'description': '极限配置 - 服务器级硬件',
            'data_dir': 'train_4M_pyg',
            'max_samples': None,
            'model_type': 'transformer',
            'hidden_dim': 512,
            'num_layers': 12,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 300,
            'patience': 25,
            'num_workers': min(16, system_info['cpu_cores']),
            'expected_time': '3-5天',
            'expected_rmse': '2-6 Å'
        }
    }
    
    # 根据GPU内存推荐配置
    gpu_memory = system_info['gpu_memory_gb']
    
    if not system_info['cuda_available']:
        return 'light', configs['light']
    elif gpu_memory < 6:
        return 'light', configs['light']
    elif gpu_memory < 12:
        return 'medium', configs['medium']
    elif gpu_memory < 20:
        return 'heavy', configs['heavy']
    else:
        return 'extreme', configs['extreme']

def interactive_config_selection():
    """交互式配置选择"""
    print("🚀 大规模分子坐标预测 - 配置选择器")
    print("="*60)
    
    # 获取系统信息
    system_info = get_system_info()
    print(f"📊 系统信息:")
    print(f"  CPU核心数: {system_info['cpu_cores']}")
    print(f"  内存: {system_info['memory_gb']:.1f} GB")
    print(f"  GPU: {system_info['gpu_name']}")
    print(f"  GPU内存: {system_info['gpu_memory_gb']:.1f} GB" if system_info['cuda_available'] else "  GPU: 未检测到CUDA")
    print()
    
    # 推荐配置
    recommended_name, recommended_config = suggest_config(system_info)
    print(f"💡 推荐配置: {recommended_name.upper()}")
    print(f"  {recommended_config['description']}")
    print(f"  预期训练时间: {recommended_config['expected_time']}")
    print(f"  预期RMSE: {recommended_config['expected_rmse']}")
    print()
    
    # 显示所有配置选项
    configs = {
        'light': {'description': '轻量级配置 - 适合入门和快速测试', 'expected_time': '1-2小时'},
        'medium': {'description': '中等配置 - 平衡性能与效果', 'expected_time': '8-12小时'},
        'heavy': {'description': '重型配置 - 追求最佳效果', 'expected_time': '1-2天'},
        'extreme': {'description': '极限配置 - 服务器级硬件', 'expected_time': '3-5天'}
    }
    
    print("📋 可选配置:")
    for i, (name, info) in enumerate(configs.items(), 1):
        marker = " ⭐" if name == recommended_name else ""
        print(f"  {i}. {name.upper()}{marker}")
        print(f"     {info['description']}")
        print(f"     预期时间: {info['expected_time']}")
        print()
    
    # 用户选择
    while True:
        try:
            choice = input(f"请选择配置 (1-4, 默认使用推荐配置 {recommended_name}): ").strip()
            
            if not choice:
                selected_name = recommended_name
                break
            
            choice_map = {
                '1': 'light',
                '2': 'medium', 
                '3': 'heavy',
                '4': 'extreme'
            }
            
            if choice in choice_map:
                selected_name = choice_map[choice]
                break
            else:
                print("❌ 无效选择，请输入1-4的数字")
        except KeyboardInterrupt:
            print("\n👋 已取消")
            return None
    
    return selected_name

def create_config_file(config_name):
    """创建配置文件"""
    system_info = get_system_info()
    
    all_configs = {
        'light': {
            'data_dir': 'train_4M_pyg',
            'max_samples': 10000,
            'model_type': 'gcn',
            'hidden_dim': 64,
            'num_layers': 3,
            'batch_size': 8,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 50,
            'patience': 10,
            'num_workers': min(4, system_info['cpu_cores']),            'output_dir': f'training_light',
            'save_every': 5,
        },
        
        'medium': {
            'data_dir': 'train_4M_pyg',
            'max_samples': 100000,
            'model_type': 'gat',
            'hidden_dim': 128,
            'num_layers': 6,
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 100,
            'patience': 15,
            'num_workers': min(8, system_info['cpu_cores']),
            'output_dir': f'training_medium',
            'save_every': 10,
        },
        
        'heavy': {
            'data_dir': 'train_4M_pyg',
            'max_samples': 500000,
            'model_type': 'gat',
            'hidden_dim': 256,
            'num_layers': 8,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 200,
            'patience': 20,
            'num_workers': min(12, system_info['cpu_cores']),
            'output_dir': f'training_heavy',
            'save_every': 10,
        },
        
        'extreme': {
            'data_dir': 'train_4M_pyg',
            'max_samples': None,
            'model_type': 'transformer',
            'hidden_dim': 512,
            'num_layers': 12,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 300,
            'patience': 25,
            'num_workers': min(16, system_info['cpu_cores']),
            'output_dir': f'training_extreme',
            'save_every': 20,
        }
    }
    
    return all_configs[config_name]

def main():
    """主函数"""
    print("🔧 正在分析你的系统配置...")
    
    # 检查数据集是否存在
    data_path = Path("train_4M_pyg/processed/data.pt")
    if not data_path.exists():
        print("❌ 错误: 未找到处理后的数据集")
        print(f"   请确保 {data_path} 存在")
        print("   你可能需要先运行数据转换脚本")
        return
    
    print("✅ 数据集检查通过")
    print()
    
    # 交互式选择配置
    selected_config = interactive_config_selection()
    
    if selected_config is None:
        return
      # 生成配置
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = create_config_file(selected_config)
    config['output_dir'] = f'training_{selected_config}_{timestamp}'
    
    # 确认开始训练
    print()
    print("🎯 最终配置:")
    print(f"  配置级别: {selected_config.upper()}")
    print(f"  数据量: {config['max_samples'] if config['max_samples'] else '全量'}")
    print(f"  模型类型: {config['model_type'].upper()}")
    print(f"  隐藏维度: {config['hidden_dim']}")
    print(f"  网络层数: {config['num_layers']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  最大轮数: {config['max_epochs']}")
    print()
    
    confirm = input("🚀 确认开始训练? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes', '是']:
        print("👋 已取消训练")
        return
    
    # 导入并运行训练
    print("\n" + "="*60)
    print("🎯 开始大规模分子坐标预测训练!")
    print("="*60)
    
    try:
        from large_scale_coordinate_predictor import LargeScaleTrainer
        
        trainer = LargeScaleTrainer(config)
        model, results = trainer.train()
        
        print("\n" + "="*60)
        print("🎉 训练完成!")
        print(f"最佳验证RMSE: {results['best_val_rmse']:.6f} Å")
        print(f"测试RMSE: {results['test_rmse']:.6f} Å")
        print(f"测试MAE: {results['test_mae']:.6f} Å")
        print(f"训练时间: {results['training_time']:.2f} 秒")
        print(f"结果保存在: {config['output_dir']}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        print("请检查配置和数据集是否正确")

if __name__ == "__main__":
    main()
