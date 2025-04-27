#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练风格转换模型：加载两种不同风格的VAE潜在向量，并训练AAE模型学习风格转换
"""

import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, random_split
import time
from datetime import datetime

# 添加项目根目录到路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from style_transfer.model import StyleDataset, StyleTransferAAE

def setup_logging(log_dir):
    """设置日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练风格转换模型")
    
    parser.add_argument("--style_a", type=str, required=True,
                        help="风格A的VAE编码文件路径(.pt)")
    parser.add_argument("--style_b", type=str, required=True,
                        help="风格B的VAE编码文件路径(.pt)")
    parser.add_argument("--output_dir", type=str, default="models/style_transfer",
                        help="模型输出目录")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="潜在空间维度")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--valid_split", type=float, default=0.1,
                        help="验证集比例")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备，'cuda'或'cpu'")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    model_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(log_dir)
    logging.info(f"开始训练，参数: {args}")
    logging.info(f"日志将保存到: {log_file}")
    
    # 检查文件是否存在
    for file_path in [args.style_a, args.style_b]:
        if not os.path.exists(file_path):
            logging.error(f"文件不存在: {file_path}")
            return
    
    # 加载数据集
    try:
        dataset_a = StyleDataset(args.style_a)
        dataset_b = StyleDataset(args.style_b)
        
        # 检查数据集特征形状
        if dataset_a.feature_shape != dataset_b.feature_shape:
            logging.warning(f"两个数据集的特征形状不一致: {dataset_a.feature_shape} vs {dataset_b.feature_shape}")
        
        # 分割训练集和验证集
        train_size_a = int((1 - args.valid_split) * len(dataset_a))
        valid_size_a = len(dataset_a) - train_size_a
        train_dataset_a, valid_dataset_a = random_split(dataset_a, [train_size_a, valid_size_a])
        
        train_size_b = int((1 - args.valid_split) * len(dataset_b))
        valid_size_b = len(dataset_b) - train_size_b
        train_dataset_b, valid_dataset_b = random_split(dataset_b, [train_size_b, valid_size_b])
        
        # 创建数据加载器
        train_loader_a = DataLoader(train_dataset_a, batch_size=args.batch_size, shuffle=True)
        train_loader_b = DataLoader(train_dataset_b, batch_size=args.batch_size, shuffle=True)
        valid_loader_a = DataLoader(valid_dataset_a, batch_size=args.batch_size)
        valid_loader_b = DataLoader(valid_dataset_b, batch_size=args.batch_size)
        
        logging.info(f"数据集加载完成:")
        logging.info(f"  - 风格A: 训练 {len(train_dataset_a)} 样本, 验证 {len(valid_dataset_a)} 样本")
        logging.info(f"  - 风格B: 训练 {len(train_dataset_b)} 样本, 验证 {len(valid_dataset_b)} 样本")
        
    except Exception as e:
        logging.error(f"加载数据集时出错: {e}")
        return
    
    # 初始化模型
    try:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = StyleTransferAAE(device=device, latent_dim=args.latent_dim)
        
        # 获取输入通道数
        in_channels = dataset_a.feature_shape[0]
        model.build_models(in_channels=in_channels)
        
        # 开始训练
        logging.info("开始训练模型...")
        history = model.train(
            dataloader_a=train_loader_a,
            dataloader_b=train_loader_b,
            num_epochs=args.epochs,
            save_dir=model_dir
        )
        
        # 保存最终模型
        model.save_model(model_dir)
        logging.info(f"模型训练完成，保存到: {model_dir}")
        
    except Exception as e:
        logging.error(f"训练模型时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
