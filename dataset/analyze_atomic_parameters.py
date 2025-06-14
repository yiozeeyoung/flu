"""
分析 FairChem 数据集中的原子参数
"""

import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
from collections import Counter
import os

class LoadedDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        return ['data.pt']

def analyze_atomic_parameters():
    """分析数据集中的原子参数"""
    print("=== FairChem 数据集原子参数分析 ===\n")
    
    # 加载数据集
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    dataset = LoadedDataset(output_dir)
    
    print(f"数据集大小: {len(dataset)} 个分子样本\n")
    
    # 统计原子类型
    print("1. 原子类型统计:")
    all_atomic_numbers = []
    all_atom_counts = []
    all_energies = []
    force_magnitudes = []
    
    for i, sample in enumerate(dataset):
        # 收集原子序数
        atomic_numbers = sample.x.flatten().tolist()
        all_atomic_numbers.extend(atomic_numbers)
        all_atom_counts.append(len(atomic_numbers))
        
        # 收集能量
        if hasattr(sample, 'y'):
            all_energies.append(sample.y.item())
        
        # 收集力的大小
        if hasattr(sample, 'force'):
            forces = sample.force
            force_mag = torch.norm(forces, dim=1)
            force_magnitudes.extend(force_mag.tolist())
    
    # 原子类型统计
    atomic_counter = Counter(all_atomic_numbers)
    print("   原子序数 -> 元素符号 (数量)")
    
    # 原子序数到元素符号的映射
    atomic_symbols = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr'
    }
    
    total_atoms = len(all_atomic_numbers)
    for atomic_num in sorted(atomic_counter.keys()):
        symbol = atomic_symbols.get(atomic_num, f'Z{atomic_num}')
        count = atomic_counter[atomic_num]
        percentage = (count / total_atoms) * 100
        print(f"   {atomic_num:2d} -> {symbol:2s} ({count:6d} 原子, {percentage:5.1f}%)")
    
    print(f"\n   总原子数: {total_atoms:,}")
    print(f"   不同元素种类: {len(atomic_counter)}")
    
    # 分子大小统计
    print("\n2. 分子大小统计:")
    atom_counts = np.array(all_atom_counts)
    print(f"   每个分子原子数 - 最小: {atom_counts.min()}")
    print(f"   每个分子原子数 - 最大: {atom_counts.max()}")
    print(f"   每个分子原子数 - 平均: {atom_counts.mean():.1f}")
    print(f"   每个分子原子数 - 中位数: {np.median(atom_counts):.1f}")
    
    # 能量统计
    print("\n3. 能量统计:")
    energies = np.array(all_energies)
    print(f"   能量范围: {energies.min():.2f} 到 {energies.max():.2f}")
    print(f"   平均能量: {energies.mean():.2f}")
    print(f"   能量标准差: {energies.std():.2f}")
    
    # 力统计
    print("\n4. 力统计:")
    forces = np.array(force_magnitudes)
    print(f"   力大小范围: {forces.min():.4f} 到 {forces.max():.4f}")
    print(f"   平均力大小: {forces.mean():.4f}")
    print(f"   力标准差: {forces.std():.4f}")
    
    # 边统计
    print("\n5. 分子图结构统计:")
    edge_counts = []
    for sample in dataset:
        if hasattr(sample, 'edge_index'):
            edge_count = sample.edge_index.shape[1]
            edge_counts.append(edge_count)
    
    if edge_counts:
        edge_counts = np.array(edge_counts)
        print(f"   每个分子边数 - 最小: {edge_counts.min()}")
        print(f"   每个分子边数 - 最大: {edge_counts.max()}")
        print(f"   每个分子边数 - 平均: {edge_counts.mean():.1f}")
        print(f"   总边数: {edge_counts.sum():,}")
    
    # 详细的样本示例
    print("\n6. 样本示例:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        atomic_numbers = sample.x.flatten()
        elements = [atomic_symbols.get(num.item(), f'Z{num.item()}') for num in atomic_numbers]
        element_count = Counter(elements)
        
        print(f"   样本 {i+1}:")
        print(f"     分子式: {dict(element_count)}")
        print(f"     总原子数: {len(atomic_numbers)}")
        print(f"     能量: {sample.y.item():.2f}")
        if hasattr(sample, 'edge_index'):
            print(f"     边数: {sample.edge_index.shape[1]}")
        print()

def estimate_full_dataset_size():
    """估算完整数据集的大小"""
    print("\n=== 完整数据集规模估算 ===")
    
    # 基于当前 50 个样本的统计
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    dataset = LoadedDataset(output_dir)
    
    current_samples = len(dataset)
    total_files = 80
    samples_per_file = 49835
    total_samples = total_files * samples_per_file
    
    # 计算当前样本的平均原子数
    total_atoms_current = sum(sample.x.shape[0] for sample in dataset)
    avg_atoms_per_sample = total_atoms_current / current_samples
    
    # 估算完整数据集
    estimated_total_atoms = int(total_samples * avg_atoms_per_sample)
    
    print(f"当前处理样本数: {current_samples:,}")
    print(f"完整数据集样本数: {total_samples:,}")
    print(f"平均每个样本原子数: {avg_atoms_per_sample:.1f}")
    print(f"估算完整数据集总原子数: {estimated_total_atoms:,}")
    print(f"约等于: {estimated_total_atoms/1e6:.1f} 百万个原子")
    
    # 数据文件大小估算
    current_file_size = os.path.getsize(os.path.join(output_dir, "processed", "data.pt"))
    estimated_full_size = current_file_size * (total_samples / current_samples)
    
    print(f"\n存储大小估算:")
    print(f"当前转换数据大小: {current_file_size / (1024**2):.1f} MB")
    print(f"完整数据集预估大小: {estimated_full_size / (1024**3):.1f} GB")

if __name__ == "__main__":
    try:
        analyze_atomic_parameters()
        estimate_full_dataset_size()
    except Exception as e:
        print(f"分析时出错: {e}")
        import traceback
        traceback.print_exc()
