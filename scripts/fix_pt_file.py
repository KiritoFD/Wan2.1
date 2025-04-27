#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查和修复损坏的PT文件
此脚本可以尝试修复损坏的.pt文件，或者提取其中部分数据
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import time
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description="检查和修复损坏的.pt文件")
    parser.add_argument("file", type=str, help="要检查的.pt文件路径")
    parser.add_argument("--output", type=str, default="", help="修复后文件的保存路径")
    parser.add_argument("--method", choices=['full', 'pickle', 'partial', 'repack'], default='full',
                       help="修复方法：full=尝试所有方法, pickle=使用pickle加载, partial=尝试提取部分数据, repack=重新打包数据")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    return parser.parse_args()

def check_pt_file(file_path):
    """检查.pt文件是否完整"""
    try:
        # 检查文件存在
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size < 100:  # 文件太小，可能是空文件
            return False, f"文件过小 ({file_size} 字节)"
        
        # 尝试加载文件
        try:
            data = torch.load(file_path, map_location='cpu')
            # 检查基本内容
            if not isinstance(data, dict):
                return False, f"文件内容格式错误: {type(data)}"
            
            # 检查是否含有features
            if 'features' not in data and 'features_chunk' not in data:
                return False, "文件缺少features或features_chunk"
            
            return True, "文件完整"
        except Exception as e:
            return False, f"加载失败: {str(e)}"
    except Exception as e:
        return False, f"检查文件时出错: {str(e)}"

def try_fix_with_pickle(file_path, output_path):
    """尝试使用pickle直接加载文件"""
    try:
        import pickle
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        # 保存修复后的文件
        torch.save(data, output_path)
        return True, "使用pickle修复成功"
    except Exception as e:
        return False, f"pickle修复失败: {str(e)}"

def try_partial_load(file_path, output_path):
    """尝试加载文件部分内容"""
    try:
        # 尝试使用二进制方式读取文件
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # 尝试从不同位置开始，寻找有效的PyTorch数据
        for offset in [0, 8, 16, 24, 32, 64, 128]:
            try:
                # 创建临时文件
                temp_file = output_path + f".temp{offset}"
                with open(temp_file, 'wb') as f:
                    f.write(file_content[offset:])
                
                # 尝试加载临时文件
                data = torch.load(temp_file, map_location='cpu')
                
                # 如果成功，保存到输出路径
                torch.save(data, output_path)
                os.remove(temp_file)
                return True, f"从偏移量 {offset} 成功加载"
            except:
                # 删除临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return False, "所有偏移量都无法加载"
    except Exception as e:
        return False, f"部分加载失败: {str(e)}"

def repack_data(file_path, output_path):
    """尝试重新打包数据，去除可能导致错误的部分"""
    try:
        # 尝试加载文件
        data = torch.load(file_path, map_location='cpu')
        
        # 提取关键数据
        features = data.get('features', None)
        image_paths = data.get('image_paths', None)
        metadata = data.get('metadata', None)
        
        # 如果features是张量列表，尝试堆叠
        if isinstance(features, list):
            try:
                # 检查所有特征形状是否一致
                if all(isinstance(f, torch.Tensor) and f.shape == features[0].shape for f in features):
                    features = torch.stack(features)
            except:
                # 如果无法堆叠，保留列表形式
                pass
        
        # 创建新的数据结构
        new_data = {}
        if features is not None:
            new_data['features'] = features
        if image_paths is not None:
            new_data['image_paths'] = image_paths
        if metadata is not None:
            new_data['metadata'] = metadata
        
        # 保存重新打包的数据
        torch.save(new_data, output_path)
        return True, "重新打包数据成功"
    except Exception as e:
        return False, f"重新打包失败: {str(e)}"

def main():
    args = parse_args()
    
    # 设置日志
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 检查文件
    print(f"检查文件: {args.file}")
    is_valid, check_msg = check_pt_file(args.file)
    if is_valid:
        print(f"✅ 文件正常: {check_msg}")
        return
    else:
        print(f"❌ 文件损坏: {check_msg}")
    
    # 如果没有指定输出路径，使用默认路径
    if not args.output:
        output_dir = os.path.dirname(args.file)
        filename = os.path.basename(args.file)
        base, ext = os.path.splitext(filename)
        args.output = os.path.join(output_dir, f"{base}_fixed{ext}")
    
    print(f"尝试修复文件，输出路径: {args.output}")
    
    # 根据指定方法尝试修复
    success = False
    
    if args.method == 'full' or args.method == 'pickle':
        print("尝试方法1: 使用pickle直接加载")
        success, msg = try_fix_with_pickle(args.file, args.output)
        if success:
            print(f"✅ 修复成功: {msg}")
            return
        else:
            print(f"❌ 方法1失败: {msg}")
    
    if args.method == 'full' or args.method == 'partial':
        print("尝试方法2: 尝试部分加载文件")
        success, msg = try_partial_load(args.file, args.output)
        if success:
            print(f"✅ 修复成功: {msg}")
            return
        else:
            print(f"❌ 方法2失败: {msg}")
    
    if args.method == 'full' or args.method == 'repack':
        print("尝试方法3: 重新打包数据")
        success, msg = repack_data(args.file, args.output)
        if success:
            print(f"✅ 修复成功: {msg}")
            return
        else:
            print(f"❌ 方法3失败: {msg}")
    
    print("❌ 所有修复方法均失败，无法修复文件")
    
if __name__ == "__main__":
    main()
