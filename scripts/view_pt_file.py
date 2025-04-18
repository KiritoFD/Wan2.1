#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看和分析PyTorch .pt 文件的内容
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import time
import json

def parse_args():
    parser = argparse.ArgumentParser(description="查看和分析PyTorch .pt 文件的内容")
    parser.add_argument("pt_file", type=str, help="要查看的.pt文件路径")
    parser.add_argument("--output", type=str, default=None, help="将分析结果保存到文件")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细信息")
    parser.add_argument("--summarize", "-s", action="store_true", help="仅显示数据摘要")
    parser.add_argument("--tensor_stats", "-t", action="store_true", help="显示张量统计信息(平均值、最大值、最小值等)")
    parser.add_argument("--extract_key", "-k", type=str, default=None, help="提取特定键对应的数据")
    parser.add_argument("--save_extracted", type=str, default=None, help="将提取的数据保存为新的.pt文件")
    return parser.parse_args()

def get_tensor_info(tensor):
    """获取张量的详细信息"""
    info = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "memory_size_mb": tensor.element_size() * tensor.numel() / (1024 * 1024)
    }
    
    # 如果是浮点数类型，计算统计信息
    if tensor.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
        try:
            # 将张量移到CPU，转换为float32计算统计信息
            cpu_tensor = tensor.detach().cpu().float()
            info.update({
                "min": float(cpu_tensor.min()),
                "max": float(cpu_tensor.max()),
                "mean": float(cpu_tensor.mean()),
                "std": float(cpu_tensor.std()),
                "abs_mean": float(cpu_tensor.abs().mean()),
                "has_nan": bool(torch.isnan(cpu_tensor).any()),
                "has_inf": bool(torch.isinf(cpu_tensor).any())
            })
        except Exception as e:
            info["stats_error"] = str(e)
    
    return info

def analyze_nested_dict(data, path="", verbose=False, summarize=False, tensor_stats=False):
    """递归分析嵌套字典结构"""
    results = {}
    
    if isinstance(data, dict):
        # 处理字典
        results["__type__"] = "dict"
        results["__len__"] = len(data)
        
        if not summarize:
            results["__keys__"] = list(data.keys())
        
        if not summarize or verbose:
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                results[key] = analyze_nested_dict(value, current_path, verbose, summarize, tensor_stats)
    
    elif isinstance(data, list):
        # 处理列表
        results["__type__"] = "list"
        results["__len__"] = len(data)
        
        if not summarize or verbose:
            for i, item in enumerate(data[:10]):  # 只显示前10个元素
                results[f"item_{i}"] = analyze_nested_dict(item, f"{path}[{i}]", verbose, summarize, tensor_stats)
            if len(data) > 10:
                results["..."] = f"共 {len(data)} 个元素，仅显示前10个"
    
    elif isinstance(data, torch.Tensor):
        # 处理张量
        results["__type__"] = "tensor"
        results["shape"] = list(data.shape)
        results["dtype"] = str(data.dtype)
        
        if tensor_stats:
            results.update(get_tensor_info(data))
        
        if verbose and not summarize:
            try:
                if data.numel() <= 100:  # 小张量显示所有值
                    results["values"] = data.detach().cpu().tolist()
                else:  # 大张量显示一部分
                    flat_data = data.flatten()
                    results["first_10_values"] = flat_data[:10].detach().cpu().tolist()
            except Exception as e:
                results["values_error"] = str(e)
    
    elif data is None:
        results["__type__"] = "None"
    
    else:
        # 处理其他类型
        results["__type__"] = type(data).__name__
        
        if not summarize or verbose:
            try:
                if hasattr(data, "__len__"):
                    results["__len__"] = len(data)
                
                if isinstance(data, (int, float, bool, str)) or data is None:
                    results["value"] = data
                else:
                    results["repr"] = repr(data)[:100]  # 限制长度
            except Exception as e:
                results["error"] = str(e)
    
    return results

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.pt_file):
        logging.error(f"文件不存在: {args.pt_file}")
        return
    
    # 加载 .pt 文件
    logging.info(f"正在加载 {args.pt_file}...")
    try:
        data = torch.load(args.pt_file, map_location='cpu')
        logging.info(f"文件加载成功")
    except Exception as e:
        logging.error(f"加载文件时出错: {e}")
        return
    
    # 如果指定了提取特定键
    if args.extract_key:
        if isinstance(data, dict) and args.extract_key in data:
            data = data[args.extract_key]
            logging.info(f"已提取键 '{args.extract_key}' 的数据")
            
            # 如果需要保存提取的数据
            if args.save_extracted:
                try:
                    torch.save(data, args.save_extracted)
                    logging.info(f"已将提取的数据保存到: {args.save_extracted}")
                except Exception as e:
                    logging.error(f"保存提取数据时出错: {e}")
        else:
            keys = data.keys() if isinstance(data, dict) else "非字典对象"
            logging.error(f"在数据中找不到键 '{args.extract_key}'. 可用的键: {keys}")
            return
    
    # 分析数据
    logging.info(f"正在分析数据...")
    file_size_mb = os.path.getsize(args.pt_file) / (1024 * 1024)
    analysis = {
        "file_info": {
            "path": args.pt_file,
            "size_mb": file_size_mb,
            "file_type": Path(args.pt_file).suffix
        },
        "content_summary": analyze_nested_dict(data, verbose=args.verbose, 
                                               summarize=args.summarize,
                                               tensor_stats=args.tensor_stats)
    }
    
    # 格式化输出
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        logging.info(f"分析结果已保存到: {args.output}")
    else:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    elapsed_time = time.time() - start_time
    logging.info(f"分析完成，耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
