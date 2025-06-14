try:
    import fairchem
    print("fairchem 模块成功导入！")
    # 你可以在这里尝试调用 fairchem 的一些基本功能或打印版本号，如果需要的话
    # 例如: print(fairchem.__version__)
except ImportError:
    print("错误：无法导入 fairchem 模块。它可能没有被安装。")
except Exception as e:
    print(f"导入 fairchem 时发生其他错误: {e}")
