"""
使用 FairChem 内置数据集类的转换器
"""

import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import numpy as np
from ase.neighborlist import neighbor_list

try:
    from fairchem.core.datasets import AseDBDataset, create_dataset
    print("成功导入 FairChem 数据集类")
    FAIRCHEM_AVAILABLE = True
except ImportError as e:
    print(f"FairChem 导入失败: {e}")
    FAIRCHEM_AVAILABLE = False

def atoms_to_pyg_data(atoms, energy=None, forces=None, **kwargs):
    """
    将 ASE Atoms 对象转换为 PyTorch Geometric Data 对象
    """
    # 获取原子位置
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
    
    # 获取原子类型/原子序数
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    
    # 构建边 (使用邻居列表)
    cutoff = 5.0  # Angstrom
    
    try:
        i, j, d = neighbor_list('ijd', atoms, cutoff)
        edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
        edge_attr = torch.tensor(d, dtype=torch.float).view(-1, 1)
    except:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.float).view(-1, 1)
    
    # 创建 PyG Data 对象
    data = Data(
        x=atomic_numbers.view(-1, 1),  # 节点特征 (原子序数)
        pos=pos,                       # 节点位置
        edge_index=edge_index,         # 边索引
        edge_attr=edge_attr           # 边属性 (距离)
    )
    
    # 添加目标值
    if energy is not None:
        data.y = torch.tensor([energy], dtype=torch.float)
    
    if forces is not None:
        data.force = torch.tensor(forces, dtype=torch.float)
    
    # 添加其他属性
    for key, value in kwargs.items():
        if isinstance(value, (int, float)):
            setattr(data, key, torch.tensor([value], dtype=torch.float))
        elif isinstance(value, (list, tuple, np.ndarray)):
            setattr(data, key, torch.tensor(value, dtype=torch.float))
    
    return data

class FairChemDatasetWrapper:
    """
    FairChem 数据集的包装器，尝试多种方法访问数据
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.length = 0
        
        # 尝试使用 FairChem 的 create_dataset 函数
        if FAIRCHEM_AVAILABLE:
            try:
                config = {
                    "dataset": {
                        "format": "ase_db",
                        "src": data_path,
                    }
                }
                self.dataset = create_dataset(config["dataset"])
                self.length = len(self.dataset)
                print(f"使用 FairChem create_dataset 成功，数据集长度: {self.length}")
                return
            except Exception as e:
                print(f"FairChem create_dataset 失败: {e}")
        
        # 如果上面失败，尝试直接使用 AseDBDataset
        if FAIRCHEM_AVAILABLE:
            try:
                self.dataset = AseDBDataset({"src": data_path})
                self.length = len(self.dataset)
                print(f"使用 AseDBDataset 成功，数据集长度: {self.length}")
                return
            except Exception as e:
                print(f"AseDBDataset 失败: {e}")
        
        print("所有 FairChem 方法都失败了")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.dataset is None:
            raise RuntimeError("数据集未初始化")
        return self.dataset[idx]

def test_single_file(file_path):
    """测试单个文件的访问"""
    print(f"测试文件: {os.path.basename(file_path)}")
    
    try:
        wrapper = FairChemDatasetWrapper(file_path)
        
        if len(wrapper) > 0:
            print(f"成功打开文件，包含 {len(wrapper)} 个样本")
            
            # 尝试读取第一个样本
            sample = wrapper[0]
            print(f"第一个样本类型: {type(sample)}")
            
            # 检查样本结构
            if hasattr(sample, '__dict__'):
                attrs = list(sample.__dict__.keys())
                print(f"样本属性: {attrs}")
            elif isinstance(sample, dict):
                attrs = list(sample.keys())
                print(f"样本键: {attrs}")
            
            # 检查是否有原子信息
            atoms = None
            if hasattr(sample, 'atoms'):
                atoms = sample.atoms
            elif isinstance(sample, dict) and 'atoms' in sample:
                atoms = sample['atoms']
            
            if atoms is not None:
                print(f"原子数量: {len(atoms)}")
                print(f"原子种类: {set(atoms.get_chemical_symbols())}")
                
                # 尝试转换为 PyG 格式
                energy = getattr(sample, 'energy', sample.get('energy', None) if isinstance(sample, dict) else None)
                forces = getattr(sample, 'forces', sample.get('forces', None) if isinstance(sample, dict) else None)
                
                pyg_data = atoms_to_pyg_data(atoms, energy, forces)
                print(f"转换为 PyG 格式成功:")
                print(f"  节点数: {pyg_data.x.shape[0]}")
                print(f"  边数: {pyg_data.edge_index.shape[1]}")
                print(f"  包含能量: {hasattr(pyg_data, 'y')}")
                print(f"  包含力: {hasattr(pyg_data, 'force')}")
                
                return True
            else:
                print("未找到原子信息")
        else:
            print("文件为空或无法访问")
            
    except Exception as e:
        print(f"测试文件时出错: {e}")
        import traceback
        traceback.print_exc()
    
    return False

def convert_with_fairchem(train_4m_dir, output_dir, max_samples=100):
    """
    使用 FairChem 转换数据集
    """
    if not FAIRCHEM_AVAILABLE:
        print("FairChem 不可用，无法进行转换")
        return None
    
    # 获取所有 .aselmdb 文件
    aselmdb_files = [
        os.path.join(train_4m_dir, f) 
        for f in os.listdir(train_4m_dir) 
        if f.endswith('.aselmdb')
    ]
    
    print(f"找到 {len(aselmdb_files)} 个 .aselmdb 文件")
    
    # 先测试第一个文件
    if aselmdb_files:
        print("测试第一个文件...")
        success = test_single_file(aselmdb_files[0])
        if not success:
            print("第一个文件测试失败，尝试其他文件...")
            for file_path in aselmdb_files[1:3]:  # 测试前3个文件
                if test_single_file(file_path):
                    success = True
                    break
        
        if not success:
            print("所有测试文件都失败了")
            return None
    else:
        print("未找到 .aselmdb 文件")
        return None
    
    print("文件访问测试成功，开始转换...")
    # 这里可以继续实现完整的转换逻辑
    
if __name__ == "__main__":
    train_4m_dir = r"D:\code\flu\dataset\train_4M"
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    
    print("=== 使用 FairChem 进行数据转换测试 ===\n")
    
    convert_with_fairchem(train_4m_dir, output_dir)
