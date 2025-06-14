"""
探索 FairChem 的可用模块和类
"""

def explore_fairchem():
    """探索 fairchem 模块的结构"""
    try:
        import fairchem
        print("FairChem 导入成功")
        print(f"FairChem 位置: {fairchem.__file__}")
        
        # 探索 fairchem 的子模块
        import pkgutil
        
        print("\nFairChem 的子模块:")
        for importer, modname, ispkg in pkgutil.iter_modules(fairchem.__path__, fairchem.__name__ + "."):
            print(f"  {modname} (包: {ispkg})")
            
            # 尝试导入并探索一些常见的模块
            if 'dataset' in modname.lower() or 'data' in modname.lower():
                try:
                    module = __import__(modname, fromlist=[''])
                    print(f"    成功导入 {modname}")
                    
                    # 查看模块内容
                    attrs = [attr for attr in dir(module) if not attr.startswith('_')]
                    print(f"    主要属性: {attrs[:10]}")  # 只显示前10个
                    
                    # 查找 LmdbDataset 或类似的类
                    for attr in attrs:
                        if 'lmdb' in attr.lower() or 'dataset' in attr.lower():
                            print(f"    可能相关: {attr}")
                            
                except Exception as e:
                    print(f"    导入 {modname} 失败: {e}")
        
        # 尝试一些可能的导入路径
        possible_imports = [
            "fairchem.core.datasets",
            "fairchem.data.oc.core.dataset", 
            "fairchem.datasets",
            "fairchem.core.data",
            "fairchem.data",
        ]
        
        print("\n尝试可能的导入路径:")
        for import_path in possible_imports:
            try:
                module = __import__(import_path, fromlist=[''])
                print(f"✓ {import_path} 导入成功")
                
                # 查看是否有 LmdbDataset
                if hasattr(module, 'LmdbDataset'):
                    print(f"  找到 LmdbDataset!")
                    return getattr(module, 'LmdbDataset')
                
                # 查看模块内容
                attrs = [attr for attr in dir(module) if 'lmdb' in attr.lower() or 'dataset' in attr.lower()]
                if attrs:
                    print(f"  相关属性: {attrs}")
                    
            except Exception as e:
                print(f"✗ {import_path} 导入失败: {e}")
        
        return None
        
    except Exception as e:
        print(f"探索 FairChem 时出错: {e}")
        return None

def test_lmdb_path_encoding():
    """测试 LMDB 路径编码问题"""
    import lmdb
    import os
    
    file_path = r"D:\code\flu\dataset\train_4M\data0000.aselmdb"
    
    print(f"原始路径: {file_path}")
    print(f"路径存在: {os.path.exists(file_path)}")
    print(f"路径编码: {repr(file_path)}")
    
    # 尝试不同的路径格式
    alternatives = [
        file_path,
        file_path.encode('utf-8').decode('utf-8'),
        os.path.abspath(file_path),
        file_path.replace('\\', '/'),
        f"/{file_path}".replace('\\', '/'),
    ]
    
    for i, alt_path in enumerate(alternatives):
        print(f"\n尝试路径 {i+1}: {alt_path}")
        try:
            env = lmdb.open(alt_path, readonly=True, lock=False, readahead=False, meminit=False)
            print("  ✓ LMDB 打开成功!")
            env.close()
            return alt_path
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    return None

if __name__ == "__main__":
    print("=== FairChem 模块探索 ===\n")
    
    lmdb_dataset_class = explore_fairchem()
    
    print("\n" + "="*50)
    print("=== LMDB 路径编码测试 ===\n")
    
    working_path = test_lmdb_path_encoding()
    
    if working_path:
        print(f"\n找到可用路径: {working_path}")
    else:
        print("\n未找到可用路径")
