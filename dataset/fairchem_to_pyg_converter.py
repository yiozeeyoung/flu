"""
FairChem 到 PyTorch Geometric 数据转换器

这个脚本从 .aselmdb 文件中读取分子数据，并将其转换为 PyTorch Geometric 兼容的格式。
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from tqdm import tqdm
import lmdb
import pickle
import ase
from ase import Atoms
from ase.neighborlist import neighbor_list

try:
    from fairchem.core.datasets import LmdbDataset
    print("成功导入 fairchem.core.datasets.LmdbDataset")
except ImportError:
    try:
        from fairchem.data.oc.core.dataset import LmdbDataset
        print("成功导入 fairchem.data.oc.core.dataset.LmdbDataset")
    except ImportError:
        print("警告: 无法导入 fairchem 的 LmdbDataset，将使用自定义 LMDB 读取器")
        LmdbDataset = None


class ASELMDBReader:
    """自定义 ASELMDB 文件读取器"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.env = lmdb.open(
            db_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1
        )
        self.txn = self.env.begin()
        
        # 获取数据库中的样本数量
        self.length = int(self.txn.get(b"length").decode("ascii"))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """根据索引获取数据样本"""
        try:
            # 从 LMDB 获取序列化的数据
            data_bytes = self.txn.get(f"{idx}".encode("ascii"))
            if data_bytes is None:
                raise KeyError(f"索引 {idx} 在数据库中不存在")
            
            # 反序列化数据
            data_object = pickle.loads(data_bytes)
            return data_object
        except Exception as e:
            print(f"读取索引 {idx} 时出错: {e}")
            return None
    
    def close(self):
        """关闭数据库连接"""
        self.env.close()


def atoms_to_pyg_data(atoms, energy=None, forces=None):
    """
    将 ASE Atoms 对象转换为 PyTorch Geometric Data 对象
    
    Args:
        atoms: ASE Atoms 对象
        energy: 能量值 (如果有)
        forces: 力向量 (如果有)
    
    Returns:
        torch_geometric.data.Data 对象
    """
    # 获取原子位置
    pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
    
    # 获取原子类型/原子序数
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    
    # 构建边 (使用邻居列表)
    # 设置截断半径，这里用一个合理的默认值
    cutoff = 5.0  # Angstrom
    
    try:
        i, j, d = neighbor_list('ijd', atoms, cutoff)
        edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
        edge_attr = torch.tensor(d, dtype=torch.float).view(-1, 1)
    except:
        # 如果邻居列表构建失败，创建一个空的边
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
    
    return data


class FairChemToGeometricDataset(InMemoryDataset):
    """
    从 FairChem ASELMDB 数据创建 PyTorch Geometric 数据集
    """
    
    def __init__(self, root, aselmdb_paths, transform=None, pre_transform=None, 
                 max_samples=None):
        """
        Args:
            root: 保存处理后数据的根目录
            aselmdb_paths: .aselmdb 文件路径列表
            max_samples: 最大处理样本数 (用于测试)
        """
        self.aselmdb_paths = aselmdb_paths
        self.max_samples = max_samples
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [os.path.basename(path) for path in self.aselmdb_paths]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # 这里不需要下载，数据已经在本地
        pass
    
    def process(self):
        """处理原始数据并转换为 PyG 格式"""
        data_list = []
        total_processed = 0
        
        print(f"开始处理 {len(self.aselmdb_paths)} 个 ASELMDB 文件...")
        
        for aselmdb_path in self.aselmdb_paths:
            print(f"处理文件: {os.path.basename(aselmdb_path)}")
            
            try:
                # 尝试使用 FairChem 的数据集类
                if LmdbDataset is not None:
                    try:
                        dataset = LmdbDataset({"src": aselmdb_path})
                        reader = dataset
                    except:
                        reader = ASELMDBReader(aselmdb_path)
                else:
                    reader = ASELMDBReader(aselmdb_path)
                
                # 确定处理的样本数
                file_length = len(reader)
                if self.max_samples:
                    samples_to_process = min(self.max_samples - total_processed, file_length)
                else:
                    samples_to_process = file_length
                
                print(f"  文件包含 {file_length} 个样本，将处理 {samples_to_process} 个")
                
                # 处理样本
                for i in tqdm(range(samples_to_process), desc=f"处理 {os.path.basename(aselmdb_path)}"):
                    try:
                        sample = reader[i]
                        
                        # 从样本中提取信息
                        if hasattr(sample, 'atoms'):
                            atoms = sample.atoms
                        elif isinstance(sample, dict) and 'atoms' in sample:
                            atoms = sample['atoms']
                        elif isinstance(sample, Atoms):
                            atoms = sample
                        else:
                            print(f"  跳过索引 {i}: 无法识别的数据格式")
                            continue
                        
                        # 提取能量和力（如果有）
                        energy = None
                        forces = None
                        
                        if hasattr(sample, 'energy'):
                            energy = sample.energy
                        elif isinstance(sample, dict):
                            energy = sample.get('energy', sample.get('y', None))
                        
                        if hasattr(sample, 'forces'):
                            forces = sample.forces
                        elif isinstance(sample, dict):
                            forces = sample.get('forces', None)
                        
                        # 转换为 PyG 格式
                        data = atoms_to_pyg_data(atoms, energy, forces)
                        
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        
                        data_list.append(data)
                        total_processed += 1
                        
                        # 如果达到最大样本数，停止处理
                        if self.max_samples and total_processed >= self.max_samples:
                            break
                            
                    except Exception as e:
                        print(f"  处理索引 {i} 时出错: {e}")
                        continue
                
                # 关闭读取器
                if hasattr(reader, 'close'):
                    reader.close()
                
                # 如果达到最大样本数，停止处理文件
                if self.max_samples and total_processed >= self.max_samples:
                    break
                    
            except Exception as e:
                print(f"处理文件 {aselmdb_path} 时出错: {e}")
                continue
        
        print(f"总共处理了 {total_processed} 个样本")
        
        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def convert_fairchem_to_pyg(train_4m_dir, output_dir, max_samples=1000):
    """
    将 FairChem train_4M 数据集转换为 PyTorch Geometric 格式
    
    Args:
        train_4m_dir: 包含 .aselmdb 文件的目录
        output_dir: 输出转换后数据的目录
        max_samples: 最大处理样本数 (用于测试，设置为 None 处理所有数据)
    """
    # 获取所有 .aselmdb 文件
    aselmdb_files = [
        os.path.join(train_4m_dir, f) 
        for f in os.listdir(train_4m_dir) 
        if f.endswith('.aselmdb')
    ]
    
    print(f"找到 {len(aselmdb_files)} 个 .aselmdb 文件")
    
    if max_samples:
        print(f"将处理最多 {max_samples} 个样本进行测试")
    
    # 创建数据集
    dataset = FairChemToGeometricDataset(
        root=output_dir,
        aselmdb_paths=aselmdb_files,
        max_samples=max_samples
    )
    
    print(f"转换完成! 数据集包含 {len(dataset)} 个样本")
    print(f"数据保存在: {output_dir}")
    
    # 显示第一个样本的信息
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n示例样本信息:")
        print(f"  节点数: {sample.x.shape[0]}")
        print(f"  边数: {sample.edge_index.shape[1]}")
        print(f"  节点特征维度: {sample.x.shape[1]}")
        print(f"  是否包含位置: {hasattr(sample, 'pos')}")
        print(f"  是否包含目标值: {hasattr(sample, 'y')}")
        print(f"  是否包含力: {hasattr(sample, 'force')}")
    
    return dataset


if __name__ == "__main__":
    # 设置路径
    train_4m_dir = r"D:\code\flu\dataset\train_4M"
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    
    # 转换数据 (首先用少量样本测试)
    print("开始转换 FairChem 数据到 PyTorch Geometric 格式...")
    dataset = convert_fairchem_to_pyg(
        train_4m_dir=train_4m_dir,
        output_dir=output_dir,
        max_samples=100  # 先处理100个样本进行测试
    )
    
    print("转换完成!")
