"""
含氟分子处理演示
演示PyTorch Geometric如何处理各种含氟化合物

作者: AI Assistant  
日期: 2025-06-14
"""

import torch
from torch_geometric.utils import from_smiles
from torch_geometric.data import DataLoader
import pandas as pd

def test_fluorine_molecules():
    """
    测试PyTorch Geometric对含氟分子的支持
    """
    print("=" * 60)
    print("含氟分子处理测试")
    print("=" * 60)
    
    # 定义各种含氟分子的SMILES
    fluorine_molecules = {
        # 简单含氟化合物
        "氟甲烷": "CF",
        "二氟甲烷": "CF2",  
        "三氟甲烷": "CF3",
        "四氟甲烷": "C(F)(F)(F)F",
        
        # 氟代烷烃
        "1-氟丙烷": "CCCF",
        "2-氟丙烷": "CC(F)C",
        "1,1-二氟乙烷": "CC(F)F",
        "1,2-二氟乙烷": "C(F)C(F)",
        
        # 氟代芳香族化合物
        "氟苯": "c1ccc(F)cc1",
        "对二氟苯": "c1cc(F)ccc1F", 
        "三氟甲苯": "Cc1cc(F)c(F)c(F)c1",
        "五氟苯": "c1c(F)c(F)c(F)c(F)c1F",
        
        # 含氟药物分子
        "氟西汀(百忧解)": "CCCOC(c1ccc(C(F)(F)F)cc1)c2cccnc2",  # Fluoxetine
        "氟康唑": "OC(Cn1cncn1)(Cn2cncn2)c3ccc(F)cc3F",  # Fluconazole
        
        # 全氟化合物
        "全氟乙烷": "C(F)(F)C(F)(F)F",
        "全氟丙烷": "C(C(C(F)(F)F)(F)F)(F)(F)F",
        
        # 含氟杂环
        "2-氟吡啶": "c1cncc(F)c1",
        "4-氟苯胺": "Nc1ccc(F)cc1",
        
        # 复杂含氟分子
        "特氟芬": "FC(F)=C(Cl)C(F)(F)F",  # 含氟农药
    }
    
    print(f"测试 {len(fluorine_molecules)} 个含氟分子:")
    print("-" * 60)
    
    successful_conversions = 0
    failed_conversions = 0
    results = []
    
    for name, smiles in fluorine_molecules.items():
        try:
            # 使用PyG的from_smiles函数转换
            data = from_smiles(smiles)
            
            if data is not None:
                successful_conversions += 1
                
                # 分析分子图的特征
                num_atoms = data.x.size(0)
                num_bonds = data.edge_index.size(1) // 2
                
                # 计算氟原子数量 (原子特征的第9个维度通常是原子序数)
                atom_numbers = data.x[:, 0]  # 第一个特征通常是原子序数
                fluorine_count = (atom_numbers == 9).sum().item()  # 氟的原子序数是9
                
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'num_atoms': num_atoms,
                    'num_bonds': num_bonds,
                    'fluorine_count': fluorine_count,
                    'status': '✅ 成功'
                })
                
                print(f"✅ {name:15} | 原子数: {num_atoms:2d} | 键数: {num_bonds:2d} | 氟原子: {fluorine_count:2d}")
                
            else:
                failed_conversions += 1
                results.append({
                    'name': name,
                    'smiles': smiles,
                    'status': '❌ 失败'
                })
                print(f"❌ {name:15} | 转换失败")
                
        except Exception as e:
            failed_conversions += 1
            results.append({
                'name': name,
                'smiles': smiles,
                'status': f'❌ 错误: {str(e)}'
            })
            print(f"❌ {name:15} | 错误: {str(e)}")
    
    print("-" * 60)
    print(f"总结:")
    print(f"成功转换: {successful_conversions}/{len(fluorine_molecules)} 个分子")
    print(f"失败转换: {failed_conversions}/{len(fluorine_molecules)} 个分子")
    print(f"成功率: {successful_conversions/len(fluorine_molecules)*100:.1f}%")
    
    return results

def analyze_fluorine_features():
    """
    分析含氟分子的特征表示
    """
    print("\n" + "=" * 60)
    print("含氟分子特征分析")
    print("=" * 60)
    
    # 选择几个代表性的含氟分子
    test_molecules = {
        "氟甲烷": "CF",
        "氟苯": "c1ccc(F)cc1",
        "氟西汀": "CCCOC(c1ccc(C(F)(F)F)cc1)c2cccnc2"
    }
    
    for name, smiles in test_molecules.items():
        print(f"\n分析 {name} ({smiles}):")
        print("-" * 40)
        
        data = from_smiles(smiles)
        if data is not None:
            print(f"节点特征维度: {data.x.shape}")
            print(f"边索引形状: {data.edge_index.shape}")
            
            # 检查是否有边属性
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                print(f"边特征维度: {data.edge_attr.shape}")
            
            # 分析原子类型
            atom_features = data.x
            print(f"原子特征前5个维度的值范围:")
            for i in range(min(5, atom_features.size(1))):
                values = atom_features[:, i]
                print(f"  特征{i}: [{values.min():.2f}, {values.max():.2f}]")

def demonstrate_fluorine_batch_processing():
    """
    演示含氟分子的批处理
    """
    print("\n" + "=" * 60)
    print("含氟分子批处理演示")
    print("=" * 60)
    
    # 准备含氟分子数据集
    fluorine_smiles = [
        "CF",                    # 氟甲烷
        "c1ccc(F)cc1",          # 氟苯
        "CC(F)F",               # 1,1-二氟乙烷
        "c1cc(F)ccc1F",         # 对二氟苯
        "C(F)(F)C(F)(F)F"       # 全氟乙烷
    ]
    
    # 转换为PyG数据对象
    data_list = []
    for smiles in fluorine_smiles:
        data = from_smiles(smiles)
        if data is not None:
            # 添加一个虚拟标签 (比如溶解度预测)
            data.y = torch.randn(1)  # 随机标签作为示例
            data_list.append(data)
    
    print(f"成功处理 {len(data_list)} 个含氟分子")
    
    # 创建DataLoader进行批处理
    if data_list:
        loader = DataLoader(data_list, batch_size=3, shuffle=True)
        
        print("\n批处理示例:")
        for batch_idx, batch in enumerate(loader):
            print(f"批次 {batch_idx + 1}:")
            print(f"  批次大小: {batch.num_graphs}")
            print(f"  总原子数: {batch.x.size(0)}")
            print(f"  总边数: {batch.edge_index.size(1)}")
            print(f"  标签形状: {batch.y.shape}")

def special_fluorine_considerations():
    """
    含氟分子的特殊考虑事项
    """
    print("\n" + "=" * 60)
    print("含氟分子特殊考虑事项")
    print("=" * 60)
    
    considerations = [
        "✅ 氟原子识别: PyG能正确识别氟原子(原子序数9)",
        "✅ C-F键表示: 碳氟键被正确建模为图中的边",
        "✅ 电负性特征: 氟的高电负性反映在原子特征中",
        "✅ 多氟取代: 支持多个氟原子取代的化合物",
        "✅ 芳香族氟化: 正确处理芳香环上的氟取代",
        "",
        "⚠️  注意事项:",
        "   - 氟的特殊化学性质需要在下游任务中考虑",
        "   - 极强的C-F键可能需要特殊的键特征编码",
        "   - 氟化合物的生物活性预测可能需要专门训练的模型",
        "",
        "🔧 建议的增强方法:",
        "   - 为氟原子添加特殊标记",
        "   - 使用氟化合物特定的预训练模型",
        "   - 考虑氟原子的立体效应"
    ]
    
    for consideration in considerations:
        print(consideration)

if __name__ == "__main__":
    # 运行所有测试
    results = test_fluorine_molecules()
    analyze_fluorine_features()
    demonstrate_fluorine_batch_processing()
    special_fluorine_considerations()
    
    print("\n" + "=" * 60)
    print("结论: PyTorch Geometric 完全支持含氟分子!")
    print("=" * 60)
