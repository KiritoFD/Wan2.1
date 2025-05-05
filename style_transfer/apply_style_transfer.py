#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换向量处理脚本

使用已训练的风格转换模型对VAE编码向量进行处理，可批量处理文件或目录
"""

import argparse
import logging
import os
import sys
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path

from model import StyleTransferAAE

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="应用风格转换模型处理VAE编码向量")
    parser.add_argument("--input", type=str, required=True, help="输入.pt文件或包含.pt文件的目录")
    parser.add_argument("--model", type=str, required=True, help="风格转换模型路径 (best_model.pth)")
    parser.add_argument("--output", type=str, default=None, 
                      help="输出.pt文件路径(处理单个文件时)或输出目录(处理多个文件时)")
    parser.add_argument("--device", type=str, default="cuda", help="使用设备 (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--recursive", action="store_true", help="递归搜索输入目录")
    parser.add_argument("--style_strength", type=float, default=1.0, help="风格强度 (0.0-1.0)")
    parser.add_argument("--preserve_original", action="store_true", help="在输出中保留原始特征")
    return parser.parse_args()

def find_pt_files(path, recursive=False):
    """查找所有.pt文件"""
    if os.path.isfile(path) and path.endswith('.pt'):
        return [path]
    
    if os.path.isdir(path):
        if recursive:
            return glob(os.path.join(path, '**', '*.pt'), recursive=True)
        else:
            return glob(os.path.join(path, '*.pt'))
    
    return []

def load_features(file_path):
    """加载特征文件"""
    logging.info(f"加载特征文件: {file_path}")
    try:
        data = torch.load(file_path, map_location='cpu')
        
        if 'features' in data:
            return data['features']
        else:
            # 尝试查找其他可能的键
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    logging.info(f"使用键 '{key}' 作为特征")
                    return value
            
            raise ValueError(f"在文件中找不到有效的特征数据")
    except Exception as e:
        logging.error(f"加载文件 {file_path} 时出错: {e}")
        raise

def apply_style_transfer(model, features, batch_size, device, strength=1.0):
    """应用风格转换"""
    logging.info(f"应用风格转换，强度: {strength}...")
    
    # 保存原始形状
    original_shape = features.shape
    needs_squeeze = False
    
    # 添加批次维度（如果需要）
    if len(features.shape) == 3:  # [C, H, W]
        features = features.unsqueeze(0)  # [1, C, H, W]
        needs_squeeze = True
    
    # 处理时间维度
    has_time_dim = False
    if len(features.shape) == 5:  # [B, C, T, H, W]
        has_time_dim = True
        B, C, T, H, W = features.shape
        features = features.transpose(1, 2).reshape(B*T, C, H, W)
    
    # 准备结果存储
    results = []
    
    # 分批处理
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch = features[i:i+batch_size].to(device)
            
            # 应用风格转换，考虑强度
            if strength < 1.0:
                styled = model.transfer_style(batch)
                # 混合原始特征和风格化特征
                styled = batch.cpu() * (1 - strength) + styled * strength
            else:
                styled = model.transfer_style(batch)
                
            results.append(styled.cpu())
    
    # 合并结果
    styled_features = torch.cat(results, dim=0)
    
    # 恢复原始维度
    if has_time_dim:
        styled_features = styled_features.reshape(B, T, C, H, W).transpose(1, 2)  # [B, C, T, H, W]
    
    # 如果输入只有一个样本，移除批次维度
    if needs_squeeze:
        styled_features = styled_features.squeeze(0)
    
    logging.info(f"风格转换完成, 输出形状: {styled_features.shape}")
    return styled_features

def process_file(file_path, model, output_path, batch_size, device, style_strength=1.0, preserve_original=False):
    """处理单个文件"""
    try:
        # 加载特征
        original_features = load_features(file_path)
        
        # 应用风格转换
        styled_features = apply_style_transfer(model, original_features, batch_size, device, style_strength)
        
        # 准备输出数据
        output_data = {
            'features': styled_features,
            'original_path': file_path,
            'model_path': args.model,
            'style_strength': style_strength
        }
        
        # 如果需要，保留原始特征
        if preserve_original:
            output_data['original_features'] = original_features
        
        # 保存结果
        logging.info(f"保存结果到: {output_path}")
        torch.save(output_data, output_path)
        return True
        
    except Exception as e:
        logging.error(f"处理文件 {file_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    logging.info(f"使用设备: {device}")
    
    # 查找所有.pt文件
    pt_files = find_pt_files(args.input, args.recursive)
    if not pt_files:
        logging.error(f"在 {args.input} 中找不到.pt文件")
        return 1
    
    # 确定是处理单个文件还是多个文件
    single_file = len(pt_files) == 1
    
    # 确定输出路径
    if single_file:
        if args.output:
            output_path = args.output
        else:
            # 为单个文件创建默认输出路径
            input_file = pt_files[0]
            file_name = os.path.basename(input_file)
            file_dir = os.path.dirname(input_file)
            output_path = os.path.join(file_dir, f"styled_{file_name}")
    else:
        # 多个文件，需要输出目录
        if args.output:
            output_dir = args.output
        else:
            # 默认在当前目录创建输出文件夹
            output_dir = "styled_output"
        
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"输出目录: {output_dir}")
    
    # 检查模型文件
    if not os.path.exists(args.model):
        logging.error(f"模型文件不存在: {args.model}")
        return 1
    
    try:
        # 初始化模型
        logging.info(f"初始化风格转换模型")
        model = StyleTransferAAE(device=device)
        
        # 加载模型权重
        logging.info(f"加载模型: {args.model}")
        current_epoch = model.load_model(args.model, device=device)
        logging.info(f"模型已加载, 训练轮次: {current_epoch+1}")
        
        # 处理文件
        if single_file:
            # 处理单个文件
            success = process_file(
                pt_files[0], model, output_path, args.batch_size, 
                device, args.style_strength, args.preserve_original
            )
            if success:
                logging.info("处理完成!")
                return 0
            else:
                return 1
        else:
            # 批量处理多个文件
            success_count = 0
            for file_path in tqdm(pt_files, desc="处理文件"):
                file_name = os.path.basename(file_path)
                output_file = os.path.join(output_dir, f"styled_{file_name}")
                
                success = process_file(
                    file_path, model, output_file, args.batch_size,
                    device, args.style_strength, args.preserve_original
                )
                
                if success:
                    success_count += 1
            
            logging.info(f"批处理完成! 成功: {success_count}/{len(pt_files)}")
            return 0 if success_count > 0 else 1
        
    except Exception as e:
        logging.error(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
