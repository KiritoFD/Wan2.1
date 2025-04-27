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

def inspect_pt_file(file_path):
    """检查.pt文件的有效性和内容"""
    try:
        # 检查文件是否存在和大小
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        if file_size < 0.001:  # 小于1KB
            return False, f"文件可能是空的: {file_path} (大小 {file_size:.2f}MB)"
        
        logging.info(f"检查文件: {file_path} (大小: {file_size:.2f}MB)")
        
        # 尝试简单加载文件
        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            return False, f"无法加载文件: {str(e)}"
        
        # 检查基本结构
        if not isinstance(data, dict):
            return False, f"文件内容不是字典类型: {type(data)}"
        
        # 检查是否包含features键
        if 'features' not in data:
            keys = list(data.keys())
            return False, f"文件中缺少'features'键，包含的键: {keys}"
        
        # 检查features的类型和形状
        features = data['features']
        if isinstance(features, torch.Tensor):
            shape = features.shape
        elif isinstance(features, list) and all(isinstance(x, torch.Tensor) for x in features):
            shape = [x.shape for x in features[:3]]
            shape = f"{shape}... (共{len(features)}项)"
        else:
            return False, f"features不是预期的张量类型: {type(features)}"
        
        # 返回验证成功和基本信息
        return True, f"文件验证通过，形状: {shape}"
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return False, f"检查文件时出错: {str(e)}\n{tb}"

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
    
    # 详细检查输入文件
    logging.info("验证输入文件...")
    for file_path in [args.style_a, args.style_b]:
        success, message = inspect_pt_file(file_path)
        if not success:
            logging.error(message)
            return
        else:
            logging.info(message)
    
    # 加载数据集
    try:
        logging.info("加载风格A数据集...")
        dataset_a = StyleDataset(args.style_a)
        
        logging.info("加载风格B数据集...")
        dataset_b = StyleDataset(args.style_b)
        
        # 检查数据集特征形状
        if dataset_a.feature_shape != dataset_b.feature_shape:
            logging.warning(f"两个数据集的特征形状不一致: {dataset_a.feature_shape} vs {dataset_b.feature_shape}")
            logging.warning("这可能导致训练问题，但尝试继续...")
        
        # 分割训练集和验证集
        train_size_a = int((1 - args.valid_split) * len(dataset_a))
        valid_size_a = len(dataset_a) - train_size_a
        train_dataset_a, valid_dataset_a = random_split(dataset_a, [train_size_a, valid_size_a])
        
        train_size_b = int((1 - args.valid_split) * len(dataset_b))
        valid_size_b = len(dataset_b) - train_size_b
        train_dataset_b, valid_dataset_b = random_split(dataset_b, [train_size_b, valid_size_b])
        
        # 创建数据加载器
        train_loader_a = DataLoader(train_dataset_a, batch_size=args.batch_size, shuffle=True, num_workers=0)
        train_loader_b = DataLoader(train_dataset_b, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader_a = DataLoader(valid_dataset_a, batch_size=args.batch_size, num_workers=0)
        valid_loader_b = DataLoader(valid_dataset_b, batch_size=args.batch_size, num_workers=0)
        
        logging.info(f"数据集加载完成:")
        logging.info(f"  - 风格A: 训练 {len(train_dataset_a)} 样本, 验证 {len(valid_dataset_a)} 样本")
        logging.info(f"  - 风格B: 训练 {len(train_dataset_b)} 样本, 验证 {len(valid_dataset_b)} 样本")
        
    except Exception as e:
        logging.error(f"加载数据集时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
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