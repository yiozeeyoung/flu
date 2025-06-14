"""
FairChem AtomicData 到 PyTorch Geometric 的最终转换器
"""

import os
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import numpy as np

try:
    from fairchem.core.datasets import AseDBDataset
    FAIRCHEM_AVAILABLE = True
    print("✓ FairChem 可用")
except ImportError as e:
    FAIRCHEM_AVAILABLE = False
    print(f"✗ FairChem 不可用: {e}")

def atomic_data_to_pyg_data(atomic_data):
    """
    将 FairChem 的 AtomicData 转换为 PyTorch Geometric Data 对象
    
    Args:
        atomic_data: FairChem AtomicData 对象
    
    Returns:
        torch_geometric.data.Data 对象
    """
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    
    # AtomicData 已经包含了大部分我们需要的信息
    data = Data()
    
    # 基本原子信息
    data.x = atomic_data.atomic_numbers.view(-1, 1)  # 原子序数作为节点特征
    data.pos = atomic_data.pos  # 原子位置
    
    # 构建边信息（如果原始数据没有边或边为空）
    if not hasattr(atomic_data, 'edge_index') or atomic_data.edge_index is None or atomic_data.edge_index.numel() == 0:
        # 使用 ASE 构建邻居列表
        try:
            # 创建 ASE Atoms 对象
            atoms = Atoms(
                numbers=atomic_data.atomic_numbers.cpu().numpy(),
                positions=atomic_data.pos.cpu().numpy()
            )
            
            # 使用合理的截断半径构建邻居列表
            cutoff = 5.0  # Angstrom
            i, j, d = neighbor_list('ijd', atoms, cutoff)
            
            if len(i) > 0:
                edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
                edge_attr = torch.tensor(d, dtype=torch.float).view(-1, 1)
                data.edge_index = edge_index
                data.edge_attr = edge_attr
            else:
                # 如果没有边，创建空的边索引
                data.edge_index = torch.tensor([[], []], dtype=torch.long)
                data.edge_attr = torch.tensor([], dtype=torch.float).view(-1, 1)
                
        except Exception as e:
            print(f"构建边时出错: {e}")
            # 创建空的边索引作为后备
            data.edge_index = torch.tensor([[], []], dtype=torch.long)
            data.edge_attr = torch.tensor([], dtype=torch.float).view(-1, 1)
    else:
        # 使用原始边信息
        data.edge_index = atomic_data.edge_index
        
        # 计算边属性（距离）
        if hasattr(atomic_data, 'pos'):
            row, col = atomic_data.edge_index
            edge_vec = atomic_data.pos[row] - atomic_data.pos[col]
            edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)
            data.edge_attr = edge_dist
    
    # 目标值
    if hasattr(atomic_data, 'energy'):
        data.y = atomic_data.energy.view(-1) if atomic_data.energy.dim() > 0 else atomic_data.energy.view(1)
    
    # 力信息
    if hasattr(atomic_data, 'forces'):
        data.force = atomic_data.forces
    
    # 其他有用的属性
    if hasattr(atomic_data, 'natoms'):
        data.natoms = atomic_data.natoms
    
    if hasattr(atomic_data, 'sid'):
        data.sid = atomic_data.sid
        
    if hasattr(atomic_data, 'cell'):
        data.cell = atomic_data.cell
        
    if hasattr(atomic_data, 'pbc'):
        data.pbc = atomic_data.pbc
    
    return data

class FairChemToGeometricDataset(InMemoryDataset):
    """
    从 FairChem ASELMDB 数据创建 PyTorch Geometric 数据集
    """
    
    def __init__(self, root, aselmdb_paths, transform=None, pre_transform=None, 
                 max_samples=None, samples_per_file=None):
        """
        Args:
            root: 保存处理后数据的根目录
            aselmdb_paths: .aselmdb 文件路径列表
            max_samples: 最大处理样本数 (用于测试)
            samples_per_file: 每个文件处理的样本数 (如果为None，则处理所有)        """
        self.aselmdb_paths = aselmdb_paths
        self.max_samples = max_samples
        self.samples_per_file = samples_per_file
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        return [os.path.basename(path) for path in self.aselmdb_paths]
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        pass
    
    def process(self):
        """处理原始数据并转换为 PyG 格式"""
        if not FAIRCHEM_AVAILABLE:
            raise RuntimeError("FairChem 不可用，无法处理数据")
        
        data_list = []
        total_processed = 0
        
        print(f"开始处理 {len(self.aselmdb_paths)} 个 ASELMDB 文件...")
        
        for file_idx, aselmdb_path in enumerate(self.aselmdb_paths):
            print(f"\n处理文件 {file_idx + 1}/{len(self.aselmdb_paths)}: {os.path.basename(aselmdb_path)}")
            
            try:
                # 使用 FairChem 的 AseDBDataset
                dataset = AseDBDataset({"src": aselmdb_path})
                file_length = len(dataset)
                
                # 确定要处理的样本数
                if self.samples_per_file:
                    samples_to_process = min(self.samples_per_file, file_length)
                elif self.max_samples:
                    samples_to_process = min(self.max_samples - total_processed, file_length)
                else:
                    samples_to_process = file_length
                
                print(f"  文件包含 {file_length} 个样本，将处理 {samples_to_process} 个")
                
                # 处理样本
                for i in tqdm(range(samples_to_process), desc=f"处理 {os.path.basename(aselmdb_path)}"):
                    try:
                        atomic_data = dataset[i]
                        
                        # 转换为 PyG 格式
                        data = atomic_data_to_pyg_data(atomic_data)
                        
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
                
                # 如果达到最大样本数，停止处理文件
                if self.max_samples and total_processed >= self.max_samples:
                    break
                    
            except Exception as e:
                print(f"处理文件 {aselmdb_path} 时出错: {e}")
                continue
        
        print(f"\n总共处理了 {total_processed} 个样本")
        
        if len(data_list) == 0:
            raise RuntimeError("没有成功处理任何样本")
        
        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def convert_fairchem_to_pyg(train_4m_dir, output_dir, max_samples=1000, samples_per_file=100):
    """
    将 FairChem train_4M 数据集转换为 PyTorch Geometric 格式
    
    Args:
        train_4m_dir: 包含 .aselmdb 文件的目录
        output_dir: 输出转换后数据的目录
        max_samples: 最大处理样本数 (设置为 None 处理所有数据)
        samples_per_file: 每个文件处理的样本数
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
    
    if max_samples:
        print(f"将处理最多 {max_samples} 个样本进行测试")
    if samples_per_file:
        print(f"每个文件最多处理 {samples_per_file} 个样本")
    
    # 创建数据集
    dataset = FairChemToGeometricDataset(
        root=output_dir,
        aselmdb_paths=aselmdb_files,
        max_samples=max_samples,
        samples_per_file=samples_per_file
    )
    
    print(f"\n转换完成! 数据集包含 {len(dataset)} 个样本")
    print(f"数据保存在: {output_dir}")
    
    # 显示第一个样本的信息
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n示例样本信息:")
        print(f"  节点数: {sample.x.shape[0]}")
        print(f"  边数: {sample.edge_index.shape[1] if hasattr(sample, 'edge_index') and sample.edge_index is not None else 0}")
        print(f"  节点特征维度: {sample.x.shape[1]}")
        print(f"  是否包含位置: {hasattr(sample, 'pos')}")
        print(f"  是否包含目标值: {hasattr(sample, 'y')}")
        print(f"  是否包含力: {hasattr(sample, 'force')}")
        
        # 显示更多细节
        if hasattr(sample, 'y'):
            print(f"  能量值范围: {sample.y.item():.6f}")
        if hasattr(sample, 'force'):
            print(f"  力的形状: {sample.force.shape}")
        if hasattr(sample, 'natoms'):
            print(f"  原子数量: {sample.natoms.item()}")
    
    return dataset

def demo_usage():
    """演示如何使用转换后的数据集"""
    print("\n=== 数据集使用演示 ===")
    
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    
    # 检查是否有已转换的数据
    processed_file = os.path.join(output_dir, "processed", "data.pt")
    if os.path.exists(processed_file):
        print("发现已转换的数据，加载中...")
        
        # 创建一个空的数据集实例来加载数据
        class LoadedDataset(InMemoryDataset):
            def __init__(self, root):
                super().__init__(root)
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            
            @property
            def processed_file_names(self):
                return ['data.pt']
        
        try:
            dataset = LoadedDataset(output_dir)
            print(f"成功加载数据集，包含 {len(dataset)} 个样本")
            
            # 展示一些统计信息
            sample = dataset[0]
            print(f"样本示例: {sample}")
            
            # 可以进一步分析
            return dataset
            
        except Exception as e:
            print(f"加载数据集时出错: {e}")
    else:
        print("未找到已转换的数据")
    
    return None

if __name__ == "__main__":
    # 设置路径
    train_4m_dir = r"D:\code\flu\dataset\train_4M"
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    
    print("=== FairChem 到 PyTorch Geometric 转换器 ===\n")
    
    # 先进行小规模测试
    print("开始小规模转换测试...")
    dataset = convert_fairchem_to_pyg(
        train_4m_dir=train_4m_dir,
        output_dir=output_dir,
        max_samples=50,  # 先处理50个样本进行测试
        samples_per_file=25  # 每个文件最多25个样本
    )
    
    if dataset:
        print("\n转换测试成功!")
        demo_usage()
    else:
        print("\n转换测试失败!")
