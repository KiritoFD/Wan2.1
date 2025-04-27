#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查看和分析PyTorch .pt文件的内容
此脚本可以完整展示VAE潜在向量或其他.pt文件的内容和结构
"""

import os
import sys
import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
import time

def parse_args():
    parser = argparse.ArgumentParser(description="查看和分析PyTorch .pt文件内容")
    parser.add_argument("pt_file", type=str, help="要查看的.pt文件路径")
    parser.add_argument("--output", type=str, default=None, 
                        help="输出内容到文本文件，默认为与输入同名但后缀为.txt")
    parser.add_argument("--full", action="store_true", help="显示完整数据内容，而不是摘要")
    parser.add_argument("--max_items", type=int, default=10, 
                        help="每个维度最多显示多少个值，默认10")
    parser.add_argument("--format", choices=['text', 'json', 'both'], default='text',
                        help="输出格式：text=文本形式，json=JSON格式，both=两者都输出")
    parser.add_argument("--stats", action="store_true", 
                        help="计算数值统计信息（均值、标准差、最大值、最小值）")
    parser.add_argument("--histograms", action="store_true",
                        help="生成数据直方图统计")
    parser.add_argument("--image_paths", action="store_true",
                        help="显示图像路径列表")
    args = parser.parse_args()
    
    # 如果未指定输出文件，使用默认命名
    if args.output is None:
        args.output = os.path.splitext(args.pt_file)[0] + ".txt"
    
    return args

def extract_tensor_info(tensor, max_items=10, compute_stats=False, compute_hist=False):
    """提取张量的详细信息"""
    info = {}
    
    # 基本信息
    info["类型"] = str(type(tensor))
    info["设备"] = str(tensor.device) if hasattr(tensor, 'device') else "N/A"
    info["数据类型"] = str(tensor.dtype) if hasattr(tensor, 'dtype') else "N/A" 
    info["形状"] = str(tensor.shape) if hasattr(tensor, 'shape') else "N/A"
    
    # 数据内容摘要
    if hasattr(tensor, 'numel') and tensor.numel() > 0:
        # 生成索引范围，针对每个维度选择开始、中间和结尾部分
        slices = []
        for dim_size in tensor.shape:
            if dim_size <= max_items:
                # 如果维度小于max_items，则显示所有元素
                slices.append(slice(None))
            else:
                # 从头部、中间和尾部选择元素
                head_count = max_items // 3
                tail_count = max_items // 3
                mid_count = max_items - head_count - tail_count
                
                head_indices = list(range(0, head_count))
                mid_start = dim_size // 2 - mid_count // 2
                mid_indices = list(range(mid_start, mid_start + mid_count))
                tail_indices = list(range(dim_size - tail_count, dim_size))
                
                # 组合这些索引
                indices = head_indices + mid_indices + tail_indices
                slices.append(indices)
        
        # 提取样本数据
        try:
            # 生成适合对应维度的索引器
            if len(slices) > 0:
                # 如果是张量
                sample_data = tensor
                for i, s in enumerate(slices):
                    if isinstance(s, list):
                        # 如果是列表索引（非连续区间）
                        # 一次只能在一个维度上使用高级索引，其他维度使用整个范围
                        idx = [slice(None)] * len(tensor.shape)
                        idx[i] = s
                        sample_data = sample_data[tuple(idx)]
                    else:
                        # 如果是常规切片
                        sample_data = sample_data.index_select(i, torch.tensor(list(range(tensor.shape[i]))[s], device='cpu'))
                
                info["样本数据"] = sample_data.cpu().numpy().tolist()
            else:
                info["样本数据"] = tensor.cpu().numpy().tolist() if hasattr(tensor, 'cpu') else str(tensor)
        except Exception as e:
            info["样本数据"] = f"无法提取样本数据: {str(e)}"
            
        # 统计信息
        if compute_stats and hasattr(tensor, 'float') and tensor.numel() > 0:
            try:
                tensor_float = tensor.float().cpu()
                info["统计信息"] = {
                    "均值": tensor_float.mean().item(),
                    "标准差": tensor_float.std().item(),
                    "最大值": tensor_float.max().item(),
                    "最小值": tensor_float.min().item(),
                    "非零元素": torch.count_nonzero(tensor_float).item(),
                    "元素总数": tensor_float.numel()
                }
            except Exception as e:
                info["统计信息"] = f"无法计算统计信息: {str(e)}"
                
        # 直方图
        if compute_hist and hasattr(tensor, 'float') and tensor.numel() > 0:
            try:
                tensor_np = tensor.float().cpu().numpy().flatten()
                hist, bin_edges = np.histogram(tensor_np, bins=10)
                info["直方图"] = {
                    "计数": hist.tolist(),
                    "边界值": bin_edges.tolist()
                }
            except Exception as e:
                info["直方图"] = f"无法生成直方图: {str(e)}"
    else:
        info["数据"] = str(tensor)
    
    return info

def format_tensor_info_text(name, info, indent=0):
    """将张量信息格式化为文本"""
    indent_str = " " * indent
    result = []
    result.append(f"{indent_str}{name}:")
    result.append(f"{indent_str}  类型: {info['类型']}")
    result.append(f"{indent_str}  设备: {info['设备']}")
    result.append(f"{indent_str}  数据类型: {info['数据类型']}")
    result.append(f"{indent_str}  形状: {info['形状']}")
    
    if "样本数据" in info:
        if isinstance(info["样本数据"], str):
            result.append(f"{indent_str}  样本数据: {info['样本数据']}")
        else:
            result.append(f"{indent_str}  样本数据:")
            # 限制嵌套级别和每级显示的项目数
            sample_str = str(info["样本数据"]).replace("], [", "],\n" + " " * (indent+4) + "[")
            # 避免过长的数据输出
            max_len = 1000
            if len(sample_str) > max_len:
                sample_str = sample_str[:max_len] + "..."
            result.append(f"{indent_str}    {sample_str}")
    
    if "统计信息" in info:
        if isinstance(info["统计信息"], dict):
            result.append(f"{indent_str}  统计信息:")
            for k, v in info["统计信息"].items():
                result.append(f"{indent_str}    {k}: {v}")
        else:
            result.append(f"{indent_str}  统计信息: {info['统计信息']}")
            
    if "直方图" in info:
        if isinstance(info["直方图"], dict):
            result.append(f"{indent_str}  数据分布直方图:")
            hist_counts = info["直方图"]["计数"]
            hist_edges = info["直方图"]["边界值"]
            for i in range(len(hist_counts)):
                bin_start = hist_edges[i]
                bin_end = hist_edges[i+1]
                count = hist_counts[i]
                result.append(f"{indent_str}    [{bin_start:.4f}, {bin_end:.4f}): {count} 个元素")
        else:
            result.append(f"{indent_str}  直方图: {info['直方图']}")
    
    return "\n".join(result)

def analyze_pt_file(file_path, max_items=10, compute_stats=False, compute_hist=False):
    """分析.pt文件，返回格式化信息"""
    start_time = time.time()
    
    try:
        # 加载.pt文件
        print(f"正在加载 {file_path}...")
        data = torch.load(file_path, map_location='cpu')
        print(f"文件加载完成，耗时 {time.time() - start_time:.2f} 秒")
        
        # 准备结果
        result = {
            "文件信息": {
                "路径": os.path.abspath(file_path),
                "大小": os.path.getsize(file_path) / (1024 * 1024),  # MB
                "分析时间": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "内容": {}
        }
        
        # 分析主要数据结构
        if isinstance(data, dict):
            print(f"文件内容为字典，包含 {len(data)} 个键")
            result["内容"]["类型"] = "字典"
            result["内容"]["键"] = list(data.keys())
            
            # 分析每个键对应的数据
            for key, value in data.items():
                print(f"正在分析键: {key}...")
                
                if key == "features":
                    # 处理特征数据
                    if isinstance(value, torch.Tensor):
                        # 单个张量
                        result["内容"][key] = extract_tensor_info(value, max_items, compute_stats, compute_hist)
                    elif isinstance(value, list) and all(isinstance(item, torch.Tensor) for item in value):
                        # 张量列表
                        result["内容"][key] = {
                            "类型": "张量列表",
                            "长度": len(value)
                        }
                        # 只采样几个张量
                        for i in range(min(3, len(value))):
                            tensor_key = f"{key}[{i}]"
                            result["内容"][tensor_key] = extract_tensor_info(value[i], max_items, compute_stats, compute_hist)
                    else:
                        # 其他类型
                        result["内容"][key] = {"类型": str(type(value)), "信息": str(value)}
                
                elif key == "image_paths":
                    # 处理图像路径数据
                    if isinstance(value, list):
                        result["内容"][key] = {
                            "类型": "路径列表",
                            "长度": len(value),
                            "样本": value[:min(5, len(value))]
                        }
                    else:
                        result["内容"][key] = {"类型": str(type(value)), "信息": str(value)}
                
                elif key == "metadata":
                    # 处理元数据
                    if isinstance(value, dict):
                        result["内容"][key] = {
                            "类型": "字典",
                            "内容": value
                        }
                    else:
                        result["内容"][key] = {"类型": str(type(value)), "信息": str(value)}
                
                else:
                    # 其他类型的键
                    if isinstance(value, torch.Tensor):
                        result["内容"][key] = extract_tensor_info(value, max_items, compute_stats, compute_hist)
                    elif isinstance(value, (list, dict)):
                        result["内容"][key] = {
                            "类型": str(type(value)),
                            "长度": len(value),
                            "样本": str(value)[:100] + ("..." if len(str(value)) > 100 else "")
                        }
                    else:
                        result["内容"][key] = {"类型": str(type(value)), "信息": str(value)}
        
        elif isinstance(data, torch.Tensor):
            print("文件内容为单个张量")
            result["内容"]["类型"] = "张量"
            result["内容"]["张量信息"] = extract_tensor_info(data, max_items, compute_stats, compute_hist)
        
        elif isinstance(data, list):
            print(f"文件内容为列表，包含 {len(data)} 个元素")
            result["内容"]["类型"] = "列表"
            result["内容"]["长度"] = len(data)
            
            # 分析列表中的前几个元素
            for i in range(min(5, len(data))):
                item = data[i]
                if isinstance(item, torch.Tensor):
                    result["内容"][f"元素[{i}]"] = extract_tensor_info(item, max_items, compute_stats, compute_hist)
                else:
                    result["内容"][f"元素[{i}]"] = {"类型": str(type(item)), "信息": str(item)}
        
        else:
            print(f"文件内容为其他类型: {type(data)}")
            result["内容"]["类型"] = str(type(data))
            result["内容"]["信息"] = str(data)
        
        return result
    
    except Exception as e:
        print(f"分析文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return {"错误": str(e)}

def format_result_text(result):
    """将结果格式化为文本"""
    lines = []
    
    # 文件信息
    lines.append("======== 文件信息 ========")
    file_info = result["文件信息"]
    lines.append(f"路径: {file_info['路径']}")
    lines.append(f"大小: {file_info['大小']:.2f} MB")
    lines.append(f"分析时间: {file_info['分析时间']}")
    lines.append("")
    
    # 内容概述
    lines.append("======== 内容概述 ========")
    content = result["内容"]
    lines.append(f"类型: {content['类型']}")
    
    if content['类型'] == "字典":
        lines.append(f"键: {', '.join(content['键'])}")
        lines.append("")
        
        # 处理每个键的信息
        lines.append("======== 详细内容 ========")
        for key, info in content.items():
            # 跳过类型和键
            if key in ["类型", "键"]:
                continue
                
            if key == "features" and isinstance(info, dict) and "类型" in info and info["类型"] == "张量列表":
                # 对于features张量列表，显示概要
                lines.append(f"features: 张量列表，包含 {info['长度']} 个张量")
            elif key.startswith("features["):
                # 单独处理features中的元素
                lines.append(format_tensor_info_text(key, info, 0))
            elif key == "image_paths" and isinstance(info, dict) and "类型" in info and info["类型"] == "路径列表":
                # 对于image_paths，显示概要和样本
                lines.append(f"image_paths: 路径列表，包含 {info['长度']} 个路径")
                lines.append("  样本路径:")
                for i, path in enumerate(info["样本"]):
                    lines.append(f"    {i}: {path}")
                if info["长度"] > len(info["样本"]):
                    lines.append(f"    ... (共 {info['长度']} 个路径)")
            elif key == "metadata" and isinstance(info, dict) and "类型" in info and info["类型"] == "字典":
                # 处理元数据
                lines.append("metadata: 元数据字典")
                for k, v in info["内容"].items():
                    lines.append(f"  {k}: {v}")
            else:
                # 其他键
                if isinstance(info, dict) and "类型" in info:
                    if "样本数据" in info or "形状" in info:
                        # 这是一个张量信息
                        lines.append(format_tensor_info_text(key, info, 0))
                    else:
                        # 其他类型的信息
                        lines.append(f"{key}: {info}")
                else:
                    lines.append(f"{key}: {info}")
            
            lines.append("")  # 添加空行分隔
    
    elif content['类型'] == "张量":
        # 处理单个张量
        lines.append(format_tensor_info_text("张量", content["张量信息"], 0))
    
    elif content['类型'] == "列表":
        # 处理列表
        lines.append(f"长度: {content['长度']}")
        lines.append("")
        
        # 处理列表中的元素
        for key, info in content.items():
            if key.startswith("元素["):
                lines.append(format_tensor_info_text(key, info, 0))
                lines.append("")
    
    return "\n".join(lines)

def main():
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.pt_file):
        print(f"错误: 文件 '{args.pt_file}' 不存在!")
        return 1
    
    # 分析文件
    result = analyze_pt_file(args.pt_file, args.max_items, args.stats, args.histograms)
    
    # 输出结果
    if args.format == 'text' or args.format == 'both':
        text_result = format_result_text(result)
        
        if args.output:
            text_output = args.output if not args.output.endswith('.json') else args.output.replace('.json', '.txt')
            with open(text_output, "w", encoding="utf-8") as f:
                f.write(text_result)
            print(f"文本分析结果已保存到: {text_output}")
        else:
            print(text_result)
    
    if args.format == 'json' or args.format == 'both':
        json_result = json.dumps(result, indent=2, ensure_ascii=False)
        
        if args.output:
            json_output = args.output if args.output.endswith('.json') else args.output + '.json'
            with open(json_output, "w", encoding="utf-8") as f:
                f.write(json_result)
            print(f"JSON分析结果已保存到: {json_output}")
        else:
            print(json_result)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
