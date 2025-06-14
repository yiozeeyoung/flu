"""
详细分析PyTorch Geometric中含氟分子的特征维度

这个脚本专门分析PyG从含氟分子中提取的所有特征
"""

import torch
from torch_geometric.utils import from_smiles
import numpy as np

def analyze_fluorine_molecule_features():
    """
    详细分析含氟分子的特征维度和内容
    """
    print("=" * 80)
    print("PyTorch Geometric 含氟分子特征详细分析")
    print("=" * 80)
    
    # 选择不同复杂度的含氟分子进行分析
    test_molecules = {
        "氟甲烷": "CF",
        "氟苯": "c1ccc(F)cc1", 
        "三氟甲苯": "Cc1cc(F)c(F)c(F)c1",
        "氟西汀": "CCCOC(c1ccc(C(F)(F)F)cc1)c2cccnc2"
    }
    
    for name, smiles in test_molecules.items():
        print(f"\n{'='*60}")
        print(f"分析分子: {name} ({smiles})")
        print(f"{'='*60}")
        
        data = from_smiles(smiles)
        if data is None:
            print("❌ 无法解析此SMILES")
            continue
            
        # 基本信息
        print(f"原子数量: {data.num_nodes}")
        print(f"化学键数量: {data.num_edges // 2}")
        
        # 节点特征 (原子特征)
        print(f"\n📊 节点特征 (原子特征):")
        print(f"   形状: {data.x.shape}")
        print(f"   维度: {data.x.shape[1]} 个特征")
          print(f"\n   各维度特征值范围:")
        for i in range(data.x.shape[1]):
            values = data.x[:, i].float()  # 转换为float类型
            print(f"   特征 {i:2d}: [{values.min().item():6.2f}, {values.max().item():6.2f}] "
                  f"(均值: {values.mean().item():6.2f})")
        
        # 边特征 (化学键特征)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            print(f"\n📊 边特征 (化学键特征):")
            print(f"   形状: {data.edge_attr.shape}")
            print(f"   维度: {data.edge_attr.shape[1]} 个特征")
              print(f"\n   各维度特征值范围:")
            for i in range(data.edge_attr.shape[1]):
                values = data.edge_attr[:, i].float()  # 转换为float类型
                print(f"   特征 {i:2d}: [{values.min().item():6.2f}, {values.max().item():6.2f}] "
                      f"(均值: {values.mean().item():6.2f})")
        else:
            print(f"\n📊 边特征: 无")
        
        # 分析氟原子的特征
        analyze_fluorine_atoms(data, name)

def analyze_fluorine_atoms(data, molecule_name):
    """
    专门分析氟原子的特征
    """
    print(f"\n🔬 氟原子特征分析:")
    
    # 找到氟原子 (原子序数为9)
    atom_numbers = data.x[:, 0]  # 第一个特征通常是原子序数
    fluorine_indices = (atom_numbers == 9).nonzero(as_tuple=True)[0]
    
    if len(fluorine_indices) == 0:
        print("   ⚠️  未检测到氟原子")
        return
    
    print(f"   检测到 {len(fluorine_indices)} 个氟原子")
    print(f"   氟原子索引: {fluorine_indices.tolist()}")
    
    # 显示氟原子的特征
    for i, f_idx in enumerate(fluorine_indices):
        print(f"\n   氟原子 {i+1} (索引 {f_idx.item()}) 的特征:")
        fluorine_features = data.x[f_idx]
        for j, feature_val in enumerate(fluorine_features):
            print(f"      特征 {j:2d}: {feature_val.item():8.3f}")

def explain_feature_meanings():
    """
    解释每个特征的含义
    """
    print(f"\n{'='*80}")
    print("PyTorch Geometric 原子特征含义解释")
    print(f"{'='*80}")
    
    feature_explanations = [
        "特征 0: 原子序数 (Atomic Number)",
        "特征 1: 原子度数 (Degree) - 连接的原子数", 
        "特征 2: 形式电荷 (Formal Charge)",
        "特征 3: 杂化状态 (Hybridization) - SP, SP2, SP3等",
        "特征 4: 是否为芳香原子 (Is Aromatic)",
        "特征 5: 隐式氢原子数 (Num Implicit Hs)",
        "特征 6: 是否在环中 (In Ring)",
        "特征 7: 手性标记 (Chirality)",
        "特征 8: 原子质量 (Mass) / 其他特征"
    ]
    
    print("标准的9维原子特征包括:")
    for i, explanation in enumerate(feature_explanations):
        print(f"   {explanation}")
    
    print(f"\n边特征 (化学键特征) 通常包括:")
    edge_explanations = [
        "特征 0: 键类型 (Bond Type) - 单键/双键/三键/芳香键",
        "特征 1: 键的立体化学 (Stereo)",
        "特征 2: 是否在环中 (In Ring)"
    ]
    
    for explanation in edge_explanations:
        print(f"   {explanation}")

def compare_fluorine_vs_other_atoms():
    """
    比较氟原子与其他原子的特征差异
    """
    print(f"\n{'='*80}")
    print("氟原子 vs 其他原子的特征对比")
    print(f"{'='*80}")
    
    # 创建包含多种原子的分子
    comparison_molecule = "c1ccc(C(F)(F)F)c(O)c1N"  # 含F, C, O, N的分子
    data = from_smiles(comparison_molecule)
    
    if data is None:
        print("❌ 无法解析对比分子")
        return
    
    print(f"对比分子: {comparison_molecule}")
    print(f"原子数量: {data.num_nodes}")
    
    # 识别不同类型的原子
    atom_numbers = data.x[:, 0]
    atom_types = {
        6: "碳 (C)",
        7: "氮 (N)", 
        8: "氧 (O)",
        9: "氟 (F)"
    }
    
    print(f"\n各类型原子的特征对比:")
    for atomic_num, atom_name in atom_types.items():
        indices = (atom_numbers == atomic_num).nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            print(f"\n{atom_name} 原子 (共{len(indices)}个):")
            # 显示第一个该类型原子的特征
            first_atom_features = data.x[indices[0]]
            for i, feature_val in enumerate(first_atom_features):
                print(f"   特征 {i}: {feature_val.item():8.3f}")

def feature_statistics_summary():
    """
    特征统计总结
    """
    print(f"\n{'='*80}")
    print("特征维度统计总结")
    print(f"{'='*80}")
    
    molecules = [
        ("简单含氟", "CF"),
        ("复杂含氟", "c1ccc(C(F)(F)F)cc1"),
        ("药物分子", "CCCOC(c1ccc(C(F)(F)F)cc1)c2cccnc2")
    ]
    
    print("分子类型                | 原子特征维度 | 边特征维度 | 氟原子数")
    print("-" * 65)
    
    for mol_type, smiles in molecules:
        data = from_smiles(smiles)
        if data is not None:
            node_features = data.x.shape[1]
            edge_features = data.edge_attr.shape[1] if hasattr(data, 'edge_attr') and data.edge_attr is not None else 0
            fluorine_count = (data.x[:, 0] == 9).sum().item()
            
            print(f"{mol_type:20} | {node_features:10d} | {edge_features:8d} | {fluorine_count:6d}")

if __name__ == "__main__":
    analyze_fluorine_molecule_features()
    explain_feature_meanings()
    compare_fluorine_vs_other_atoms()
    feature_statistics_summary()
    
    print(f"\n{'='*80}")
    print("🎯 总结回答:")
    print("PyTorch Geometric 处理含氟分子时:")
    print("   📊 原子特征: 9维 (每个原子包括氟原子)")
    print("   📊 边特征: 3维 (每条化学键包括C-F键)")
    print("   🔬 氟原子被完整表示，包含其独特的化学性质")
    print("   ✅ 特征足够丰富，可用于各种机器学习任务")
    print(f"{'='*80}")
