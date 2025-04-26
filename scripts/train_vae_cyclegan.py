#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE特征向量的CycleGAN训练入口点
用于在两个领域的VAE编码向量之间学习映射关系

使用示例:
    python train_vae_cyclegan.py --domain_a path/to/domain_a.pt --domain_b path/to/domain_b.pt
"""

import os
import sys
import argparse
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 自定义尺寸匹配的MSE损失函数
class SizeMatchingMSELoss(nn.Module):
    def __init__(self):
        super(SizeMatchingMSELoss, self).__init__()
        
    def forward(self, input, target):
        # 确保输入和目标具有相同的尺寸
        if input.size() != target.size():
            # 将输入调整为与目标相同的尺寸
            input = F.interpolate(input, size=target.size()[2:], mode='bilinear', align_corners=False)
            # 确保通道数匹配
            if input.size(1) != target.size(1):
                # 扩展通道维度
                input = input.repeat(1, target.size(1)//input.size(1), 1, 1)
        return F.mse_loss(input, target)

# 导入VAECycleGAN模块和补丁
from scripts.vae_cyclegan import Config, VAEFeatureDataset, ShapeAwareDataLoader, VAECycleGANTrainer
from scripts.vae_cyclegan_patch import patch_vae_cyclegan, safe_mse_loss

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="VAE特征向量的CycleGAN训练")
    
    # 数据参数
    parser.add_argument("--domain_a", type=str, required=True, help="域A特征文件路径")
    parser.add_argument("--domain_b", type=str, required=True, help="域B特征文件路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--max_samples", type=int, default=None, help="每个域使用的最大样本数，None表示全部使用")
    parser.add_argument("--shuffle", action='store_true', help="是否打乱数据集")
    parser.add_argument("--drop_last", action='store_true', help="丢弃最后不完整的批次")
    
    # 模型参数
    parser.add_argument("--input_channels", type=int, default=16, help="输入通道数")
    parser.add_argument("--output_channels", type=int, default=16, help="输出通道数")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="残差块数量")
    parser.add_argument("--base_filters", type=int, default=64, help="基础滤波器数量")    
    parser.add_argument("--discriminator_layers", type=int, default=3, help="判别器层数")
    
    # 训练参数
    parser.add_argument("--n_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--decay_epoch", type=int, default=100, help="学习率开始衰减的轮数")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam优化器beta1参数")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器beta2参数")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="循环一致性损失权重")
    parser.add_argument("--lambda_identity", type=float, default=5.0, help="身份损失权重")        
    parser.add_argument("--buffer_size", type=int, default=50, help="经验回放缓冲区大小")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--output_dir", type=str, default="./output/vae_cyclegan", help="输出目录")
    parser.add_argument("--save_freq", type=int, default=10, help="模型保存频率（轮数）")
    parser.add_argument("--sample_freq", type=int, default=5, help="样本生成频率（轮数）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的模型路径或轮数")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="训练或测试模式")
    parser.add_argument("--config_path", type=str, default=None, help="配置文件路径，优先级高于命令行参数")
    
    # 测试参数
    parser.add_argument("--test_input", type=str, default=None, help="测试模式下的输入文件")
    parser.add_argument("--test_output", type=str, default=None, help="测试模式下的输出文件")
    parser.add_argument("--test_direction", type=str, default="A2B", choices=["A2B", "B2A"], help="测试模式下的转换方向")
    parser.add_argument("--test_batch_size", type=int, default=4, help="测试模式下的批处理大小")
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config.from_args(args)
    
    # 如果配置文件存在，则覆盖命令行参数
    if args.config_path and os.path.exists(args.config_path):
        config = Config.from_file(args.config_path)
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 配置日志
    log_file = os.path.join(config.output_dir, "vae_cyclegan.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 保存配置
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    
    # 打印配置信息
    logging.info(f"配置参数: {config}")

    # 加载数据集
    dataset_A = VAEFeatureDataset(config.domain_a_path, config.max_samples)
    dataset_B = VAEFeatureDataset(config.domain_b_path, config.max_samples)
    
    # 创建形状感知的数据加载器
    dataloader_A = ShapeAwareDataLoader(
        dataset_A, 
        batch_size=config.batch_size,
        shuffle=config.shuffle, 
        num_workers=config.num_workers,
        drop_last=config.drop_last
    )
    
    dataloader_B = ShapeAwareDataLoader(
        dataset_B, 
        batch_size=config.batch_size,
        shuffle=config.shuffle, 
        num_workers=config.num_workers,
        drop_last=config.drop_last
    )
    
    logging.info(f"数据集A: {len(dataset_A)}个样本，数据集B: {len(dataset_B)}个样本")
    
    # 初始化训练器，使用正确设计的固定尺寸网络
    from scripts.fixed_size_models import FixedSizeVAECycleGANTrainer
    
    # 减少日志输出频率，防止进度条异常
    logging_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)  # 临时提高日志级别
    
    # 启用细粒度容错模式，遇到形状不匹配时只跳过单个样本而不是整个批次
    trainer = FixedSizeVAECycleGANTrainer(config, fault_tolerant=True, skip_individual=True)
    logging.getLogger().setLevel(logging_level)  # 恢复原始日志级别
    
    logging.info("使用固定尺寸网络模型（已启用细粒度容错模式）")
    
    # 加载预训练模型（如果指定）
    if config.resume_path:
        try:
            if config.resume_path.isdigit():
                trainer.load_models(epoch=int(config.resume_path))
            else:
                trainer.load_models(path=config.resume_path)
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            return
    
    # 训练或测试
    if args.mode == "train":
        # 训练模型
        logging.info("开始训练...")
        losses = trainer.train(dataloader_A, dataloader_B)
        logging.info("训练完成")
        
        # 保存最终模型
        trainer.save_models(epoch="final")
    else:
        # 测试模式
        logging.info("开始测试...")
        if args.test_input and args.test_output:
            # 转换整个文件
            trainer.translate_file(
                args.test_input, 
                args.test_output, 
                direction=args.test_direction, 
                batch_size=args.test_batch_size
            )
        else:
            # 生成一些示例
            trainer.save_samples("test", dataloader_A, dataloader_B, num_samples=10)
        
        logging.info("测试完成")


if __name__ == "__main__":
    main()
