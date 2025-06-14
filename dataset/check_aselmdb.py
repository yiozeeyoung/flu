"""
简单的 FairChem .aselmdb 文件检查工具
"""

import os
import lmdb
import pickle
import sys

def check_aselmdb_file(file_path):
    """检查单个 .aselmdb 文件的内容"""
    print(f"检查文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return False
    
    print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    
    try:
        # 尝试打开 LMDB 数据库
        env = lmdb.open(
            file_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1
        )
        
        with env.begin() as txn:
            # 检查是否有 'length' 键
            length_bytes = txn.get(b"length")
            if length_bytes:
                length = int(length_bytes.decode("ascii"))
                print(f"数据库包含 {length} 个样本")
                
                # 尝试读取第一个样本
                if length > 0:
                    try:
                        first_sample_bytes = txn.get(b"0")
                        if first_sample_bytes:
                            first_sample = pickle.loads(first_sample_bytes)
                            print(f"第一个样本类型: {type(first_sample)}")
                            
                            # 检查样本内容
                            if hasattr(first_sample, '__dict__'):
                                print(f"样本属性: {list(first_sample.__dict__.keys())}")
                            elif isinstance(first_sample, dict):
                                print(f"样本键: {list(first_sample.keys())}")
                            
                            # 如果有 atoms 属性
                            if hasattr(first_sample, 'atoms'):
                                atoms = first_sample.atoms
                                print(f"原子数量: {len(atoms)}")
                                print(f"原子类型: {atoms.get_chemical_symbols()}")
                            elif isinstance(first_sample, dict) and 'atoms' in first_sample:
                                atoms = first_sample['atoms']
                                print(f"原子数量: {len(atoms)}")
                                print(f"原子类型: {atoms.get_chemical_symbols()}")
                                
                        else:
                            print("无法读取第一个样本")
                    except Exception as e:
                        print(f"读取第一个样本时出错: {e}")
                
            else:
                print("未找到 'length' 键，尝试其他方法...")
                
                # 列出所有键
                cursor = txn.cursor()
                keys = [key for key, _ in cursor]
                print(f"找到 {len(keys)} 个键")
                if keys:
                    print(f"前几个键: {keys[:5]}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"打开数据库时出错: {e}")
        return False

def test_fairchem_import():
    """测试 FairChem 导入"""
    print("测试 FairChem 导入...")
    
    try:
        import fairchem
        print("✓ fairchem 导入成功")
        
        # 尝试导入 LmdbDataset
        try:
            from fairchem.core.datasets import LmdbDataset
            print("✓ 从 fairchem.core.datasets 导入 LmdbDataset 成功")
            return LmdbDataset
        except ImportError:
            try:
                from fairchem.data.oc.core.dataset import LmdbDataset
                print("✓ 从 fairchem.data.oc.core.dataset 导入 LmdbDataset 成功")
                return LmdbDataset
            except ImportError:
                print("✗ 无法导入 LmdbDataset")
                return None
                
    except ImportError as e:
        print(f"✗ fairchem 导入失败: {e}")
        return None

def test_fairchem_dataset(file_path, LmdbDataset):
    """使用 FairChem 的 LmdbDataset 测试文件"""
    if LmdbDataset is None:
        print("LmdbDataset 不可用，跳过测试")
        return
    
    print(f"\n使用 FairChem LmdbDataset 测试: {file_path}")
    
    try:
        dataset = LmdbDataset({"src": file_path})
        print(f"数据集长度: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"第一个样本类型: {type(sample)}")
            
            if hasattr(sample, '__dict__'):
                print(f"样本属性: {list(sample.__dict__.keys())}")
            elif isinstance(sample, dict):
                print(f"样本键: {list(sample.keys())}")
                
    except Exception as e:
        print(f"使用 FairChem LmdbDataset 时出错: {e}")

if __name__ == "__main__":
    # 设置文件路径
    test_file = r"D:\code\flu\dataset\train_4M\data0000.aselmdb"
    
    print("=== FairChem ASELMDB 文件检查工具 ===\n")
    
    # 测试 FairChem 导入
    LmdbDataset = test_fairchem_import()
    
    print("\n" + "="*50)
    
    # 检查文件
    success = check_aselmdb_file(test_file)
    
    if success and LmdbDataset:
        print("\n" + "="*50)
        test_fairchem_dataset(test_file, LmdbDataset)
