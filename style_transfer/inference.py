#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用训练好的风格转换模型对VAE编码特征进行风格转换
"""

import os
import argparse
import logging
import torch
import time
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from style_transfer.model import StyleDataset, StyleTransferAAE

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="使用风格转换模型")
    
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型路径")
    parser.add_argument("--input", type=str, required=True,
                        help="输入VAE编码文件路径(.pt)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出VAE编码文件路径(.pt)")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda",
                        help="推理设备，'cuda'或'cpu'")
    parser.add_argument("--no_squeeze_time", action="store_true",
                        help="不去除时间维度")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        logging.error(f"模型文件不存在: {args.model}")
        return
        
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载输入数据
    try:
        dataset = StyleDataset(args.input, squeeze_time=not args.no_squeeze_time)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        logging.info(f"加载输入数据完成, 共 {len(dataset)} 个样本, 特征形状: {dataset.feature_shape}")
    except Exception as e:
        logging.error(f"加载输入数据时出错: {e}")
        return
    
    # 加载模型
    try:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = StyleTransferAAE(device=device)
        model.load_model(args.model, device=args.device)
        logging.info(f"模型加载完成: {args.model}")
    except Exception as e:
        logging.error(f"加载模型时出错: {e}")
        return
    
    # 执行风格转换
    try:
        logging.info("开始执行风格转换...")
        start_time = time.time()
        
        # 转换所有输入数据
        all_outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="转换进度"):
                batch = batch.to(device)
                output = model.transfer_style(batch)
                all_outputs.append(output)
        
        # 合并所有输出
        transferred_features = torch.cat(all_outputs, dim=0)
        
        # 恢复时间维度（如果原始输入有）
        if not args.no_squeeze_time and dataset.original_shape[2] == 1:
            transferred_features = transferred_features.unsqueeze(2)
            
        elapsed_time = time.time() - start_time
        logging.info(f"风格转换完成, 耗时: {elapsed_time:.2f}秒")
        
        # 准备保存数据
        save_data = {
            'features': transferred_features,
            'image_paths': dataset.get_paths(),
            'metadata': {
                'original_metadata': dataset.get_metadata(),
                'style_transfer': {
                    'model_path': os.path.abspath(args.model),
                    'input_path': os.path.abspath(args.input),
                    'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'feature_shape': list(transferred_features.shape)
                }
            }
        }
        
        # 保存结果
        torch.save(save_data, args.output)
        logging.info(f"转换结果已保存到: {args.output}")
        
    except Exception as e:
        logging.error(f"执行风格转换时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
