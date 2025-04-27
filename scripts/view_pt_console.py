#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速在控制台查看.pt文件内容
直接输出张量数据，不保存到文件
"""

import argparse
import sys
import os
import torch
import numpy as np
from pathlib import Path
from pprint import pprint
from tabulate import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="在控制台查看.pt文件内容")
    parser.add_argument("file", type=str, help=".pt文件路径")
    parser.add_argument("--key", type=str, default=None, 
                        help="要查看的特定键，如'features'或'metadata'")
    parser.add_argument("--max_rows", type=int, default=10, 
                        help="最多显示多少行数据")
    parser.add_argument("--max_cols", type=int, default=10, 
                        help="最多显示多少列数据")
    parser.add_argument("--item", type=int, default=0, 
                        help="对于多个tensor，查看第几个")
    parser.add_argument("--format", choices=['text', 'table', 'numpy'], default='text',
                        help="输出格式: text=文本格式, table=表格格式, numpy=NumPy格式")
    parser.add_argument("--metadata", action="store_true",
                        help="只显示元数据")
    parser.add_argument("--paths", action="store_true", 
                        help="只显示图像路径")
    parser.add_argument("--stats", action="store_true",
                        help="显示统计数据（均值、标准差等）")
    return parser.parse_args()

def format_tensor(tensor, max_rows=10, max_cols=10, format_type='text'):
    """格式化张量显示"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # 获取张量形状
    shape = tensor.shape
    ndim = tensor.ndim
    
    # 准备张量摘要信息
    info = [
        f"形状: {shape}",
        f"维度: {ndim}",
        f"类型: {tensor.dtype}"
    ]
    
    if ndim <= 2:
        # 对于1D或2D张量，直接显示
        if format_type == 'table':
            if ndim == 1:
                # 1D张量转为表格
                data = [[i, val] for i, val in enumerate(tensor[:max_rows])]
                table = tabulate(data, headers=["索引", "值"], tablefmt="grid")
                return "\n".join(info + ["", table])
            else:
                # 2D张量转为表格
                rows = min(tensor.shape[0], max_rows)
                cols = min(tensor.shape[1], max_cols)
                data = tensor[:rows, :cols]
                headers = [f"列{i}" for i in range(cols)]
                table = tabulate(data, headers=headers, tablefmt="grid", floatfmt=".4f")
                return "\n".join(info + ["", table])
        elif format_type == 'numpy':
            # NumPy风格输出
            with np.printoptions(precision=4, threshold=max_rows * max_cols, edgeitems=max_cols):
                return "\n".join(info + ["", str(tensor)])
        else:
            # 默认文本输出
            with np.printoptions(precision=4, threshold=max_rows * max_cols, edgeitems=max_cols):
                return "\n".join(info + ["", str(tensor)])
    else:
        # 高维张量显示部分切片
        if format_type == 'table':
            if ndim == 3:
                # 3D张量，显示第一个切片
                slice_data = tensor[0, :min(tensor.shape[1], max_rows), :min(tensor.shape[2], max_cols)]
                headers = [f"列{i}" for i in range(slice_data.shape[1])]
                table = tabulate(slice_data, headers=headers, tablefmt="grid", floatfmt=".4f")
                return "\n".join(info + ["", f"第0切片:", table])
            elif ndim == 4:
                # 4D张量，显示第一个3D切片的第一个2D切片
                slice_data = tensor[0, 0, :min(tensor.shape[2], max_rows), :min(tensor.shape[3], max_cols)]
                headers = [f"列{i}" for i in range(slice_data.shape[1])]
                table = tabulate(slice_data, headers=headers, tablefmt="grid", floatfmt=".4f")
                return "\n".join(info + ["", f"切片[0,0]:", table])
            else:
                # 更高维，选择显示最里面的两维
                indices = tuple([0] * (ndim - 2))
                slice_data = tensor[indices][:max_rows, :max_cols]
                headers = [f"列{i}" for i in range(slice_data.shape[1])]
                table = tabulate(slice_data, headers=headers, tablefmt="grid", floatfmt=".4f")
                return "\n".join(info + ["", f"切片{indices}:", table])
        else:
            # 文本或NumPy格式，显示部分数据
            slices = []
            if ndim == 3:
                slices.append(tensor[0])
            elif ndim == 4:
                slices.append(tensor[0, 0])
            else:
                indices = tuple([0] * (ndim - 2))
                slices.append(tensor[indices])
            
            with np.printoptions(precision=4, threshold=max_rows * max_cols, edgeitems=max_cols):
                slice_text = "\n".join([f"切片[{i}]:\n{slice}" for i, slice in enumerate(slices)])
                return "\n".join(info + ["", slice_text])

def display_tensor_stats(tensor):
    """显示张量的统计信息"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().float().numpy()
    
    stats = {
        "均值": np.mean(tensor),
        "标准差": np.std(tensor),
        "最小值": np.min(tensor),
        "最大值": np.max(tensor),
        "中位数": np.median(tensor),
        "非零元素比例": np.count_nonzero(tensor) / tensor.size
    }
    
    rows = [[k, f"{v:.6f}"] for k, v in stats.items()]
    return tabulate(rows, headers=["统计量", "值"], tablefmt="grid")

def main():
    args = parse_args()
    
    if not os.path.exists(args.file):
        print(f"错误: 文件 '{args.file}' 不存在!")
        return 1

    try:
        print(f"正在加载 {args.file}...")
        data = torch.load(args.file, map_location='cpu')
        print(f"文件已加载，大小: {os.path.getsize(args.file) / (1024*1024):.2f}MB")
        
        # 检查数据结构
        if isinstance(data, dict):
            print(f"数据类型: 字典，包含键: {list(data.keys())}")
            
            if args.metadata and 'metadata' in data:
                # 只显示元数据
                print("\n元数据:")
                metadata = data['metadata']
                if isinstance(metadata, dict):
                    for k, v in metadata.items():
                        print(f"  {k}: {v}")
                else:
                    print(metadata)
                return 0
                
            if args.paths and 'image_paths' in data:
                # 只显示图像路径
                paths = data['image_paths']
                print(f"\n图像路径 (共{len(paths)}项):")
                for i, path in enumerate(paths[:min(len(paths), args.max_rows)]):
                    print(f"  {i}: {path}")
                if len(paths) > args.max_rows:
                    print(f"  ... (另外还有 {len(paths) - args.max_rows} 项)")
                return 0
            
            # 查看特定键或默认的features
            key = args.key if args.key else ('features' if 'features' in data else list(data.keys())[0])
            
            if key in data:
                print(f"\n查看键: '{key}'")
                value = data[key]
                
                if isinstance(value, torch.Tensor):
                    print(format_tensor(value, args.max_rows, args.max_cols, args.format))
                    if args.stats:
                        print("\n统计信息:")
                        print(display_tensor_stats(value))
                        
                elif isinstance(value, list) and all(isinstance(x, torch.Tensor) for x in value):
                    print(f"张量列表，共 {len(value)} 项")
                    if len(value) > 0:
                        item_idx = min(args.item, len(value) - 1)
                        print(f"\n显示第 {item_idx} 项:")
                        print(format_tensor(value[item_idx], args.max_rows, args.max_cols, args.format))
                        if args.stats:
                            print("\n统计信息:")
                            print(display_tensor_stats(value[item_idx]))
                
                elif isinstance(value, (list, tuple)):
                    print(f"列表/元组，共 {len(value)} 项")
                    for i, item in enumerate(value[:min(len(value), args.max_rows)]):
                        print(f"  {i}: {str(item)[:100]}")
                    if len(value) > args.max_rows:
                        print(f"  ... (另外还有 {len(value) - args.max_rows} 项)")
                
                elif isinstance(value, dict):
                    print("字典内容:")
                    for k, v in list(value.items())[:args.max_rows]:
                        print(f"  {k}: {str(v)[:100]}")
                    if len(value) > args.max_rows:
                        print(f"  ... (另外还有 {len(value) - args.max_rows} 项)")
                
                else:
                    print(f"值类型: {type(value)}")
                    print(str(value)[:1000])
            else:
                print(f"键 '{key}' 不存在")
                
        elif isinstance(data, torch.Tensor):
            print(f"数据是单个张量")
            print(format_tensor(data, args.max_rows, args.max_cols, args.format))
            if args.stats:
                print("\n统计信息:")
                print(display_tensor_stats(data))
        
        else:
            print(f"数据类型: {type(data)}")
            try:
                print(str(data)[:1000])
            except:
                print("无法显示数据内容")
                
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
