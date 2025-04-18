import numpy as np
import sys

if len(sys.argv) < 2:
    print("使用方法: python view_npz.py <file.npz>")
    sys.exit(1)

file_path = sys.argv[1]
data = np.load(file_path)

print(f"文件 '{file_path}' 包含以下数组:")
for name in data.files:
    array = data[name]
    print(f" - {name}: 形状={array.shape}, 类型={array.dtype}")

key = input("\n要查看哪个数组? (输入数组名称): ")
if key in data.files:
    array = data[key]
    print(f"\n数组 '{key}':")
    print(f"形状: {array.shape}")
    print(f"类型: {array.dtype}")
    
    # 大数组显示统计信息而不是全部内容
    if array.size > 100:
        print(f"最小值: {array.min()}")
        print(f"最大值: {array.max()}")
        print(f"均值: {array.mean()}")
        print(f"标准差: {array.std()}")
        print("前几个元素:", array.flatten()[:5])
    else:
        print("内容:", array)
else:
    print(f"数组 '{key}' 不存在!")