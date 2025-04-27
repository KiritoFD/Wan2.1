#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用WGAN-GP训练风格转换模型

这个脚本使用改进的WGAN-GP方法训练风格转换模型，可以获得更好的训练效果
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from model import StyleDataset, StyleTransferAAE

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="使用WGAN-GP训练风格转换模型")
    parser.add_argument("--style_a", type=str, required=True, help="风格A的VAE特征.pt文件")
    parser.add_argument("--style_b", type=str, required=True, help="风格B的VAE特征.pt文件")
    parser.add_argument("--output_dir", type=str, default="models/style_transfer_wgan", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--latent_dim", type=int, default=128, help="潜在空间维度")
    parser.add_argument("--epochs", type=int, default=40, help="训练轮数")
    parser.add_argument("--valid_split", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lambda_gp", type=float, default=10.0, help="梯度惩罚系数")
    parser.add_argument("--n_critic", type=int, default=5, help="判别器每训练n次，生成器训练一次")
    parser.add_argument("--eval_interval", type=int, default=10, help="每多少轮评估一次模型")
    parser.add_argument("--checkpoint", type=str, default=None, help="继续训练的检查点路径")
    parser.add_argument("--best_model", type=str, default=None, help="从最佳模型参数继续训练的路径")
    parser.add_argument("--resume_from_best", action="store_true", help="是否从上一次训练的最佳模型继续训练")
    parser.add_argument("--force_new", action="store_true", help="是否强制从头开始训练新模型")
    parser.add_argument("--ignore_epoch", action="store_true", help="是否忽略保存的训练轮次，从头开始计数")
    return parser.parse_args()

def verify_file(file_path):
    """验证文件是否存在且可以加载"""
    if not os.path.isfile(file_path):
        logging.error(f"文件不存在: {file_path}")
        return False
    file_size = os.path.getsize(file_path) / (1024*1024)
    logging.info(f"检查文件: {file_path} (大小: {file_size:.2f}MB)")
    try:
        data = torch.load(file_path, map_location='cpu')
        if 'features' in data:
            shape = data['features'].shape if isinstance(data['features'], torch.Tensor) else None
            logging.info(f"文件验证通过，形状: {shape}")
            return True
        else:
            logging.error(f"文件不包含'features'字段: {file_path}")
            return False
    except Exception as e:
        logging.error(f"文件加载失败: {e}")
        return False

def main():
    # 解析参数
    args = parse_args()
    # 创建保存日志的目录
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    # 设置文件日志
    log_file = os.path.join(args.output_dir, "logs", f"training_wgan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"开始WGAN-GP训练，参数: {args}")
    logging.info(f"日志将保存到: {log_file}")
    # 验证输入文件
    logging.info("验证输入文件...")
    if not (verify_file(args.style_a) and verify_file(args.style_b)):
        logging.error("文件验证失败，程序退出")
        return
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    # 加载数据集
    logging.info("加载风格A数据集...")
    dataset_a = StyleDataset(args.style_a)
    logging.info("加载风格B数据集...")
    dataset_b = StyleDataset(args.style_b)
    # 划分训练集与验证集
    train_size_a = int(len(dataset_a) * (1 - args.valid_split))
    valid_size_a = len(dataset_a) - train_size_a
    train_size_b = int(len(dataset_b) * (1 - args.valid_split))
    valid_size_b = len(dataset_b) - train_size_b
    train_dataset_a, valid_dataset_a = random_split(
        dataset_a, [train_size_a, valid_size_a],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_dataset_b, valid_dataset_b = random_split(
        dataset_b, [train_size_b, valid_size_b],
        generator=torch.Generator().manual_seed(args.seed)
    )
    # 创建数据加载器
    train_loader_a = DataLoader(
        train_dataset_a, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    train_loader_b = DataLoader(
        train_dataset_b, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    valid_loader_a = DataLoader(
        valid_dataset_a, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    valid_loader_b = DataLoader(
        valid_dataset_b, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    logging.info(f"数据集加载完成:")
    logging.info(f"  - 风格A: 训练 {len(train_dataset_a)} 样本, 验证 {len(valid_dataset_a)} 样本")
    logging.info(f"  - 风格B: 训练 {len(train_dataset_b)} 样本, 验证 {len(valid_dataset_b)} 样本")
    # 初始化模型
    model = StyleTransferAAE(device=args.device, latent_dim=args.latent_dim)
    # 构建模型组件
    in_channels = dataset_a.feature_shape[0]
    model.build_models(in_channels=in_channels)
    # 继续训练逻辑
    loaded_model = False
    start_epoch = 0  # 默认从第0轮开始
    
    # 优先级1：从指定的检查点继续训练
    if args.checkpoint:
        logging.info(f"从检查点继续训练: {args.checkpoint}")
        try:
            # 使用修改后的load_model方法，它会返回当前epoch
            current_epoch = model.load_model(args.checkpoint)
            start_epoch = 0 if args.ignore_epoch else current_epoch + 1
            loaded_model = True
        except Exception as e:
            logging.error(f"加载检查点失败: {e}")
    # 优先级2：从指定的最佳模型继续训练
    elif args.best_model:
        logging.info(f"从最佳模型继续训练: {args.best_model}")
        try:
            # 使用修改后的load_model方法，它会返回当前epoch
            current_epoch = model.load_model(args.best_model)
            start_epoch = 0 if args.ignore_epoch else current_epoch + 1
            loaded_model = True
        except Exception as e:
            logging.error(f"加载最佳模型失败: {e}")
    # 优先级3：如果没有强制从头开始训练，则自动从上一次的最佳模型继续训练
    elif not args.force_new:
        best_model_path = os.path.join(args.output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            logging.info(f"从上一次的最佳模型继续训练: {best_model_path}")
            try:
                # 使用修改后的load_model方法，它会返回当前epoch
                current_epoch = model.load_model(best_model_path)
                start_epoch = 0 if args.ignore_epoch else current_epoch + 1
                loaded_model = True
            except Exception as e:
                logging.error(f"加载最佳模型失败: {e}")
    
    # 如果没有加载任何模型，则从头开始训练
    if not loaded_model:
        logging.info("从头开始训练新模型")
    else:
        logging.info(f"已加载模型，将从第 {start_epoch+1} 轮开始训练")
    start_time = time.time()
    # 调整实际训练轮数，考虑已经训练过的轮数
    actual_epochs = args.epochs - start_epoch
    if start_epoch > 0 and not args.ignore_epoch:
        logging.info(f"已训练 {start_epoch} 轮，将继续训练 {actual_epochs} 轮")
    
    # 使用优化后的稳定训练方法
    logging.info("使用优化的稳定训练方法...")
    start_time = time.time()
    history = model.train_stable(
        train_loader_a,
        train_loader_b,
        epochs=actual_epochs,
        save_dir=args.output_dir,
        lambda_gp=args.lambda_gp,
        clip_value=1.0,  # 添加梯度裁剪值
        lr_decay=0.995,  # 添加学习率衰减参数
        start_epoch=start_epoch  # 添加起始轮次参数
    )
    eval_dir = os.path.join(args.output_dir, "final_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 训练完成
    elapsed_time = time.time() - start_time
    logging.info(f"稳定训练完成，总耗时: {elapsed_time:.2f}秒")
    # 执行最终评估
    logging.info("执行最终模型评估...")
    # 使用训练期间保存的最佳模型进行评估
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        logging.info(f"加载最佳模型参数: {best_model_path}")
        model.load_model(best_model_path)  # 加载最佳模型
        logging.info("已加载最佳模型参数进行评估")
        logging.info(f"样例已保存到: {os.path.join(eval_dir, 'sample_transfers.pt')}")
        
    # 简单的评估方法，替代不存在的evaluate方法
    try:
        # 从验证集抽取少量样本进行风格转换，保存结果
        logging.info("生成风格转换样例...")
        with torch.no_grad():
            # 从验证集获取一些样本
            sample_batch_a = next(iter(valid_loader_a)).to(args.device)
            # 执行风格转换
            transferred = model.transfer_style(sample_batch_a)
            # 保存样本数据
            sample_data = {
                'original': sample_batch_a.cpu(),
                'transferred': transferred
            }
            torch.save(sample_data, os.path.join(eval_dir, "sample_transfers.pt"))
            logging.info(f"样例已保存到: {os.path.join(eval_dir, 'sample_transfers.pt')}")
    except Exception as e:
        logging.error(f"生成样例时出错: {e}")
    # 计算并记录最终训练指标
    metrics = {
        'final_gen_loss': history['gen_loss'][-1] if history['gen_loss'] else float('nan'),
        'final_recon_loss': history['recon_loss'][-1] if history['recon_loss'] else float('nan'),
        'final_disc_loss': history['disc_loss'][-1] if history['disc_loss'] else float('nan'),
    }
    logging.info(f"最终训练指标:")
    for name, value in metrics.items():
        logging.info(f"  - {name}: {value:.4f}")

if __name__ == "__main__":
    main()