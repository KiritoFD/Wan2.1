#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查和修复VAE编码文件
用于验证.pt文件是否完整以及查看其内容
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
    parser = argparse.ArgumentParser(description="检查和修复VAE编码文件")
    parser.add_argument("file", type=str, help="要检查的.pt文件路径")
    parser.add_argument("--repair", type=str, default="", help="如果文件损坏，修复并保存到指定路径")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    return parser.parse_args()

def setup_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def check_pt_file(file_path, verbose=False):
    """检查.pt文件是否有效并返回结构信息"""
    try:
        logging.info(f"检查文件: {file_path}")
        if not os.path.exists(file_path):
            return False, {}, "文件不存在"
        
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        logging.info(f"文件大小: {file_size:.2f} MB")
        
        # 尝试加载文件
        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            return False, {}, f"加载文件失败: {str(e)}"
        
        # 检查内容
        if not isinstance(data, dict):
            return False, {}, f"文件内容格式错误，预期为字典，实际为 {type(data)}"
        
        info = {"keys": list(data.keys())}
        
        # 检查features
        if 'features' in data:
            features = data['features']
            
            # 检查features的类型
            if isinstance(features, torch.Tensor):
                info["features_type"] = "Tensor"
                info["features_shape"] = list(features.shape)
                info["features_dtype"] = str(features.dtype)
                info["features_mean"] = float(features.float().mean())
                info["features_std"] = float(features.float().std())
            elif isinstance(features, list) and len(features) > 0:
                info["features_type"] = "List[Tensor]"
                info["features_length"] = len(features)
                
                # 检查第一个张量
                if isinstance(features[0], torch.Tensor):
                    info["first_tensor_shape"] = list(features[0].shape)
                    info["first_tensor_dtype"] = str(features[0].dtype)
                
                # 检查形状是否一致
                shapes = [tuple(f.shape) for f in features if isinstance(f, torch.Tensor)]
                consistent = len(set(shapes)) == 1 if shapes else False
                info["shapes_consistent"] = consistent
                
                if not consistent and len(shapes) > 0:
                    info["unique_shapes"] = list(set(shapes))
            else:
                info["features_type"] = str(type(features))
        else:
            return False, info, "文件中没有'features'键"
        
        # 检查image_paths
        if 'image_paths' in data:
            paths = data['image_paths']
            if isinstance(paths, list):
                info["num_paths"] = len(paths)
                if paths and verbose:
                    info["first_path"] = paths[0]
            else:
                info["paths_type"] = str(type(paths))
        
        # 检查metadata
        if 'metadata' in data:
            metadata = data['metadata']
            if isinstance(metadata, dict):
                info["metadata_keys"] = list(metadata.keys())
                
                # 检查一些重要的元数据字段
                format_desc = metadata.get('format_description', '')
                if format_desc:
                    info["format_description"] = format_desc
                
                z_dim = metadata.get('z_dim', None)
                if z_dim:
                    info["z_dim"] = z_dim
                
                latent_size = metadata.get('unified_latent_size', None)
                if latent_size:
                    info["unified_latent_size"] = latent_size
            else:
                info["metadata_type"] = str(type(metadata))
        
        return True, info, "文件有效"
    
    except Exception as e:
        logging.error(f"检查文件时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return False, {}, f"检查出错: {str(e)}"

def repair_pt_file(file_path, output_path, verbose=False):
    """尝试修复损坏的.pt文件"""
    try:
        logging.info(f"尝试修复文件: {file_path} -> {output_path}")
        
        # 尝试不同的方式加载文件
        data = None
        try:
            # 尝试使用pickle加载
            logging.info("尝试使用pickle加载...")
            import pickle
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except:
            logging.warning("使用pickle加载失败")
        
        if data is None:
            try:
                # 尝试使用torch.jit.load
                logging.info("尝试使用torch.jit.load加载...")
                data = torch.jit.load(file_path, map_location='cpu')
            except:
                logging.warning("使用torch.jit.load加载失败")
        
        if data is None:
            # 尝试读取部分文件内容
            logging.info("尝试读取文件头部...")
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(1024)
                logging.info(f"文件头部(十六进制): {header.hex()[:100]}...")
            except:
                pass
            
            return False, "所有修复方法均失败，文件可能严重损坏"
        
        # 创建新的保存格式
        if isinstance(data, dict):
            save_data = data
        else:
            # 如果不是字典，创建一个包装字典
            logging.warning(f"数据不是字典，将创建包装结构，原始类型: {type(data)}")
            save_data = {
                'features': data if isinstance(data, (torch.Tensor, list)) else None,
                'metadata': {
                    'repaired': True,
                    'original_type': str(type(data)),
                    'repair_time': time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
        
        # 保存修复后的文件
        torch.save(save_data, output_path)
        logging.info(f"修复文件已保存到: {output_path}")
        
        # 验证修复后的文件
        success, info, msg = check_pt_file(output_path, verbose)
        if success:
            return True, f"文件修复成功: {msg}"
        else:
            return False, f"文件可能已修复，但验证失败: {msg}"
    
    except Exception as e:
        logging.error(f"修复文件时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return False, f"修复出错: {str(e)}"

def print_info(info):
    """打印文件信息"""
    print("\n===== 文件信息 =====")
    for key, value in info.items():
        print(f"{key}: {value}")

def main():
    args = parse_args()
    setup_logging(args.verbose)
    
    # 检查文件
    success, info, message = check_pt_file(args.file, args.verbose)
    
    if success:
        print(f"✅ 文件有效: {message}")
        print_info(info)
    else:
        print(f"❌ 文件无效: {message}")
        print_info(info)
        
        # 如果指定了修复选项
        if args.repair:
            print(f"\n尝试修复文件...")
            repair_success, repair_msg = repair_pt_file(args.file, args.repair, args.verbose)
            
            if repair_success:
                print(f"✅ {repair_msg}")
            else:
                print(f"❌ {repair_msg}")

if __name__ == "__main__":
    main()
