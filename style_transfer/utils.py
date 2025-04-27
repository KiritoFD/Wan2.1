#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型的工具函数
"""

import torch
import torch.nn.functional as F
import logging


def calc_style_statistics(features):
    """计算风格特征的统计特性(均值和方差)"""
    mean = features.mean(0, keepdim=True)
    std = features.std(0, keepdim=True)
    return {'mean': mean, 'std': std}


def style_distance(style_a, style_b):
    """计算两个风格特征的距离"""
    mean_loss = F.mse_loss(style_a['mean'], style_b['mean'])
    std_loss = F.mse_loss(style_a['std'], style_b['std'])
    return mean_loss + std_loss


def check_flash_attn_available():
    """检查是否可以使用FlashAttention"""
    try:
        from flash_attn import __version__
        return True, __version__
    except ImportError:
        return False, None


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_features(features, eps=1e-5):
    """归一化特征，适用于可视化"""
    min_val = features.min()
    max_val = features.max()
    if max_val - min_val > eps:
        normalized = (features - min_val) / (max_val - min_val)
    else:
        normalized = torch.zeros_like(features)
    return normalized


def setup_logger(log_file=None, level=logging.INFO):
    """
    配置日志器
    
    Args:
        log_file: 日志文件路径，None表示仅控制台输出
        level: 日志级别
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除已有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器(如果提供了文件路径)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    计算WGAN-GP中的梯度惩罚
    
    Args:
        discriminator: 判别器模型
        real_samples: 真实样本
        fake_samples: 生成的样本
        device: 计算设备
    
    Returns:
        gradient_penalty: 梯度惩罚项
    """
    # 在真实样本和生成样本之间随机插值
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # 计算判别器对插值样本的输出
    d_interpolates = discriminator(interpolates)
    
    # 创建填充的梯度输出
    fake = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 计算梯度惩罚
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty
