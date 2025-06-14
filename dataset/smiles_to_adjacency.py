"""
SMILES to Adjacency Matrix Converter
将SMILES字符串转换为邻接矩阵的脚本

Dependencies:
- rdkit: 用于分子处理
- numpy: 用于数组操作

Usage:
    python smiles_to_adjacency.py
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import sys


def smiles_to_adjacency_matrix(smiles, include_hydrogen=False):
    """
    将SMILES字符串转换为邻接矩阵
    
    Args:
        smiles (str): SMILES字符串
        include_hydrogen (bool): 是否包含氢原子，默认为False
    
    Returns:
        tuple: (邻接矩阵 (numpy.ndarray), 原子列表 (list))
        如果SMILES无效，返回 (None, None)
    """
    try:
        # 从SMILES创建分子对象
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print(f"错误: 无法解析SMILES字符串 '{smiles}'")
            return None, None
        
        # 如果需要包含氢原子，则添加氢原子
        if include_hydrogen:
            mol = Chem.AddHs(mol)
        
        # 获取原子数量
        num_atoms = mol.GetNumAtoms()
        
        # 创建邻接矩阵 (初始化为0)
        adj_matrix = np.zeros((num_atoms, num_atoms), dtype=int)
        
        # 获取原子信息
        atoms = []
        for atom in mol.GetAtoms():
            atoms.append(atom.GetSymbol())
        
        # 填充邻接矩阵
        for bond in mol.GetBonds():
            start_atom = bond.GetBeginAtomIdx()
            end_atom = bond.GetEndAtomIdx()
            bond_value = 1  # 其他类型的键默认为1
            
            # 邻接矩阵是对称的
            adj_matrix[start_atom, end_atom] = bond_value
            adj_matrix[end_atom, start_atom] = bond_value
        
        return adj_matrix, atoms
    
    except Exception as e:
        print(f"处理SMILES '{smiles}' 时发生错误: {str(e)}")
        return None, None

def matrix_generator(adj_matrix):
    """
    为邻接矩阵添加自环并进行归一化
    
    Args:
        adj_matrix (numpy.ndarray): 输入的邻接矩阵
    
    Returns:
        numpy.ndarray: 归一化后的邻接矩阵
    """
    A_tilde = adj_matrix + np.eye(adj_matrix.shape[0])
    D_tilde_inv_sqrt = np.diag(1 / np.sqrt(np.sum(A_tilde, axis=1))) 
    normalized_A = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
    return normalized_A


def print_molecule_info(smiles, adj_matrix, atoms, show_normalized=True):
    """
    打印分子信息
    
    Args:
        smiles (str): SMILES字符串
        adj_matrix (numpy.ndarray): 邻接矩阵
        atoms (list): 原子列表
        show_normalized (bool): 是否显示归一化邻接矩阵
    """
    print(f"\nSMILES: {smiles}")
    print(f"原子数量: {len(atoms)}")
    print(f"原子类型: {atoms}")
    print(f"邻接矩阵形状: {adj_matrix.shape}")
    print("\n原始邻接矩阵:")
    print(adj_matrix)
    
    if show_normalized:
        normalized_matrix = matrix_generator(adj_matrix)
        print("\n归一化邻接矩阵:")
        print(normalized_matrix)
        print(f"归一化矩阵形状: {normalized_matrix.shape}")
    
    # 打印原子索引对应关系
    print("\n原子索引对应关系:")
    for i, atom in enumerate(atoms):
        print(f"索引 {i}: {atom}")


def save_adjacency_matrix(adj_matrix, filename):
    """
    保存邻接矩阵到文件
    
    Args:
        adj_matrix (numpy.ndarray): 邻接矩阵
        filename (str): 文件名
    """
    try:
        np.save(filename, adj_matrix)
        print(f"\n邻接矩阵已保存到: {filename}")
    except Exception as e:
        print(f"保存文件时发生错误: {str(e)}")


def load_adjacency_matrix(filename):
    """
    从文件加载邻接矩阵
    
    Args:
        filename (str): 文件名
    
    Returns:
        numpy.ndarray: 邻接矩阵
    """
    try:
        adj_matrix = np.load(filename)
        print(f"邻接矩阵已从 {filename} 加载")
        return adj_matrix
    except Exception as e:
        print(f"加载文件时发生错误: {str(e)}")
        return None


def batch_process_smiles(smiles_list, include_hydrogen=False):
    """
    批量处理SMILES字符串
    
    Args:
        smiles_list (list): SMILES字符串列表
        include_hydrogen (bool): 是否包含氢原子
    
    Returns:
        list: 包含(smiles, adj_matrix, atoms)的元组列表
    """
    results = []
    
    for smiles in smiles_list:
        adj_matrix, atoms = smiles_to_adjacency_matrix(smiles, include_hydrogen)
        if adj_matrix is not None:
            results.append((smiles, adj_matrix, atoms))
        else:
            print(f"跳过无效的SMILES: {smiles}")
    
    return results


def main():
    """主函数"""
    # 示例SMILES字符串
    example_smiles = [
        "CCO",           # 乙醇
        "c1ccccc1",      # 苯
        "CC(=O)O",       # 乙酸
        "CCN(CC)CC",     # 三乙胺
        "C1=CC=CC=C1O",  # 苯酚
    ]
    
    print("SMILES到邻接矩阵转换器")
    print("=" * 50)
    
    # 处理示例分子
    for smiles in example_smiles:
        adj_matrix, atoms = smiles_to_adjacency_matrix(smiles)
        
        if adj_matrix is not None:
            print_molecule_info(smiles, adj_matrix, atoms)
            print("-" * 50)
    
    # 交互式输入
    print("\n您可以输入自己的SMILES字符串进行转换:")
    print("输入 'quit' 或 'q' 退出程序")
    print("输入 'batch' 进行批量处理")
    
    while True:
        user_input = input("\n请输入SMILES字符串: ").strip()
        
        if user_input.lower() in ['quit', 'q', 'exit']:
            print("程序结束")
            break
        
        if user_input.lower() == 'batch':
            print("批量处理模式")
            batch_input = input("请输入多个SMILES字符串，用逗号分隔: ").strip()
            if batch_input:
                smiles_list = [s.strip() for s in batch_input.split(',')]
                include_h = input("是否包含氢原子? (y/n, 默认n): ").strip().lower() == 'y'
                show_norm = input("是否显示归一化矩阵? (y/n, 默认y): ").strip().lower() != 'n'
                
                results = batch_process_smiles(smiles_list, include_h)
                
                for smiles, adj_matrix, atoms in results:
                    print_molecule_info(smiles, adj_matrix, atoms, show_norm)
                    print("-" * 30)
            continue
        
        if not user_input:
            continue
        
        # 检查是否要包含氢原子
        include_h = input("是否包含氢原子? (y/n, 默认n): ").strip().lower() == 'y'
        show_norm = input("是否显示归一化矩阵? (y/n, 默认y): ").strip().lower() != 'n'
        
        adj_matrix, atoms = smiles_to_adjacency_matrix(user_input, include_hydrogen=include_h)
        
        if adj_matrix is not None:
            print_molecule_info(user_input, adj_matrix, atoms, show_norm)
            
            # 询问是否保存
            save_choice = input("\n是否保存邻接矩阵到文件? (y/n): ").strip().lower()
            if save_choice == 'y':
                matrix_type = input("保存哪种矩阵? (1: 原始, 2: 归一化, 默认1): ").strip()
                filename = input("请输入文件名 (不含扩展名): ").strip()
                if filename:
                    if matrix_type == '2':
                        normalized_matrix = matrix_generator(adj_matrix)
                        save_adjacency_matrix(normalized_matrix, f"{filename}_normalized.npy")
                    else:
                        save_adjacency_matrix(adj_matrix, f"{filename}.npy")


if __name__ == "__main__":
    main()
