"""
检查转换后数据的详细信息
"""

import torch
from torch_geometric.data import InMemoryDataset
import os

class LoadedDataset(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def processed_file_names(self):
        return ['data.pt']

def check_data_details():
    """详细检查数据格式"""
    print("=== 数据详细检查 ===\n")
    
    output_dir = r"D:\code\flu\dataset\train_4M_pyg"
    dataset = LoadedDataset(output_dir)
    
    sample = dataset[0]
    print(f"样本类型: {type(sample)}")
    print(f"样本属性: {dir(sample)}")
    
    # 检查每个属性
    for attr in ['x', 'pos', 'edge_index', 'y', 'force', 'natoms', 'sid', 'cell', 'pbc']:
        if hasattr(sample, attr):
            value = getattr(sample, attr)
            print(f"\n{attr}:")
            print(f"  类型: {type(value)}")
            print(f"  形状: {value.shape if hasattr(value, 'shape') else 'No shape'}")
            print(f"  数据类型: {value.dtype if hasattr(value, 'dtype') else 'No dtype'}")
            print(f"  值: {value}")
        else:
            print(f"\n{attr}: 不存在")

if __name__ == "__main__":
    check_data_details()
