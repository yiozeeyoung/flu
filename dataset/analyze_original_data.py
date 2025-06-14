"""
分析 FairChem 原始数据中的原子参数
"""

try:
    from fairchem.core.datasets import AseDBDataset
    FAIRCHEM_AVAILABLE = True
except ImportError:
    FAIRCHEM_AVAILABLE = False

import numpy as np
from collections import Counter

def analyze_original_fairchem_data():
    """分析原始 FairChem 数据的详细参数"""
    if not FAIRCHEM_AVAILABLE:
        print("FairChem 不可用")
        return
    
    print("=== 原始 FairChem 数据详细分析 ===\n")
    
    # 加载第一个文件进行详细分析
    file_path = r"D:\code\flu\dataset\train_4M\data0000.aselmdb"
    dataset = AseDBDataset({"src": file_path})
    
    print(f"分析文件: data0000.aselmdb")
    print(f"样本总数: {len(dataset):,}\n")
    
    # 分析前100个样本的详细信息
    sample_count = min(100, len(dataset))
    print(f"详细分析前 {sample_count} 个样本:\n")
    
    # 收集原子类型信息
    all_atomic_numbers = []
    molecule_sizes = []
    energies = []
    forces_info = []
    
    print("样本详情:")
    print("ID  | 原子数 | 主要元素组成                    | 能量         | 力范围")
    print("-" * 80)
    
    for i in range(sample_count):
        try:
            sample = dataset[i]
            
            # 获取原子信息
            atomic_numbers = sample.atomic_numbers.cpu().numpy()
            all_atomic_numbers.extend(atomic_numbers)
            molecule_sizes.append(len(atomic_numbers))
            
            # 获取能量
            energy = sample.energy.item() if hasattr(sample, 'energy') else None
            if energy is not None:
                energies.append(energy)
            
            # 获取力信息
            if hasattr(sample, 'forces'):
                forces = sample.forces.cpu().numpy()
                force_magnitudes = np.linalg.norm(forces, axis=1)
                forces_info.extend(force_magnitudes)
                force_range = f"{force_magnitudes.min():.3f}-{force_magnitudes.max():.3f}"
            else:
                force_range = "N/A"
            
            # 统计元素组成
            atomic_symbols = {
                1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 11: 'Na', 12: 'Mg', 13: 'Al',
                14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
                22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni',
                29: 'Cu', 30: 'Zn', 35: 'Br'
            }
            
            element_counts = Counter([atomic_symbols.get(num, f'Z{num}') for num in atomic_numbers])
            
            # 只显示前5个最多的元素
            top_elements = element_counts.most_common(5)
            composition = ', '.join([f"{elem}:{count}" for elem, count in top_elements])
            
            energy_str = f"{energy:12.2f}" if energy is not None else "N/A".rjust(12)
            
            print(f"{i+1:3d} | {len(atomic_numbers):6d} | {composition:30s} | {energy_str} | {force_range}")
            
            if i >= 20:  # 只显示前20行详情，避免输出过长
                break
                
        except Exception as e:
            print(f"{i+1:3d} | ERROR: {e}")
            continue
    
    if sample_count > 20:
        print(f"... (省略了 {sample_count - 20} 个样本的详情)")
    
    # 统计分析
    print(f"\n总体统计 (基于前 {sample_count} 个样本):")
    print(f"总原子数: {len(all_atomic_numbers):,}")
    
    # 原子类型分布
    atomic_counter = Counter(all_atomic_numbers)
    print(f"\n原子类型分布:")
    atomic_symbols_full = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi'
    }
    
    total_atoms = len(all_atomic_numbers)
    for atomic_num in sorted(atomic_counter.keys()):
        symbol = atomic_symbols_full.get(atomic_num, f'Z{atomic_num}')
        count = atomic_counter[atomic_num]
        percentage = (count / total_atoms) * 100
        print(f"  {atomic_num:2d} ({symbol:2s}): {count:8,} 原子 ({percentage:5.1f}%)")
    
    # 分子大小分布
    molecules = np.array(molecule_sizes)
    print(f"\n分子大小统计:")
    print(f"  最小分子: {molecules.min()} 原子")
    print(f"  最大分子: {molecules.max()} 原子")
    print(f"  平均大小: {molecules.mean():.1f} 原子")
    print(f"  中位数: {np.median(molecules):.1f} 原子")
    
    # 能量分布
    if energies:
        energies_arr = np.array(energies)
        print(f"\n能量统计:")
        print(f"  能量范围: {energies_arr.min():.2f} 到 {energies_arr.max():.2f}")
        print(f"  平均能量: {energies_arr.mean():.2f}")
        print(f"  能量标准差: {energies_arr.std():.2f}")
    
    # 力的分布
    if forces_info:
        forces_arr = np.array(forces_info)
        print(f"\n力统计:")
        print(f"  力大小范围: {forces_arr.min():.4f} 到 {forces_arr.max():.4f}")
        print(f"  平均力大小: {forces_arr.mean():.4f}")
        print(f"  力标准差: {forces_arr.std():.4f}")

def analyze_atomic_parameters_structure():
    """分析单个 AtomicData 对象的详细结构"""
    if not FAIRCHEM_AVAILABLE:
        print("FairChem 不可用")
        return
    
    print("\n=== AtomicData 对象结构分析 ===\n")
    
    file_path = r"D:\code\flu\dataset\train_4M\data0000.aselmdb"
    dataset = AseDBDataset({"src": file_path})
    
    # 获取第一个样本
    sample = dataset[0]
    
    print("AtomicData 对象的所有属性:")
    attributes = [attr for attr in dir(sample) if not attr.startswith('_')]
    
    for attr in sorted(attributes):
        try:
            value = getattr(sample, attr)
            if hasattr(value, 'shape'):
                print(f"  {attr:20s}: {type(value).__name__} {value.shape} - {value.dtype}")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                print(f"  {attr:20s}: {type(value).__name__} (长度: {len(value)})")
            else:
                print(f"  {attr:20s}: {type(value).__name__} - {value}")
        except Exception as e:
            print(f"  {attr:20s}: 无法访问 ({e})")
    
    print(f"\n详细的原子和物理参数:")
    print(f"  原子序数: {sample.atomic_numbers}")
    print(f"  原子位置: {sample.pos.shape} - {sample.pos.dtype}")
    print(f"  晶胞: {sample.cell}")
    print(f"  周期性边界: {sample.pbc}")
    print(f"  能量: {sample.energy}")
    print(f"  力: {sample.forces.shape}")
    
    if hasattr(sample, 'charge'):
        print(f"  电荷: {sample.charge}")
    if hasattr(sample, 'spin'):
        print(f"  自旋: {sample.spin}")

if __name__ == "__main__":
    analyze_original_fairchem_data()
    analyze_atomic_parameters_structure()
