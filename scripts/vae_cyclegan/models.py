#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CycleGAN模型组件"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init_normal(m):
    """初始化网络权重为正态分布"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    """灵活的残差块，适应不同输入尺寸"""
    def __init__(self, in_features, dropout=0.0):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class AdaptiveGenerator(nn.Module):
    """自适应生成器，可处理不同尺寸的特征向量"""
    
    def __init__(self, input_channels=16, output_channels=16, n_residual_blocks=9, 
                 base_filters=64, dropout=0.0):
        super(AdaptiveGenerator, self).__init__()
        
        # 初始卷积块
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, 7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]

        # 下采样 - 使用计算填充来保持空间维度比例
        curr_dim = base_filters
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim *= 2

        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(curr_dim, dropout=dropout)]

        # 上采样 - 使用更精确的上采样方法
        for _ in range(2):
            model += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.ReflectionPad2d(1),  # 增加填充以保持边缘
                nn.Conv2d(curr_dim, curr_dim // 2, 3, stride=1, padding=0),  # 使用填充=0，因为已经添加了反射填充
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(inplace=True),
            ]
            curr_dim //= 2

        # 输出层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_filters, output_channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        # 记录输入尺寸，以便最终调整输出大小
        input_height, input_width = x.shape[-2], x.shape[-1]
        
        # 运行模型
        out = self.model(x)
        
        # 如果输出形状与输入不一致，调整回原始尺寸
        if out.shape[-2] != input_height or out.shape[-1] != input_width:
            out = F.interpolate(out, size=(input_height, input_width), 
                               mode='bilinear', align_corners=False)
        
        return out


class Discriminator(nn.Module):
    """PatchGAN判别器，用于判断真假特征"""
    
    def __init__(self, input_channels=16, base_filters=64, n_layers=3):
        super(Discriminator, self).__init__()

        # 构建一系列的判别层
        model = [
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # 增加通道数，减小特征图尺寸
        curr_dim = base_filters
        for i in range(1, n_layers):
            next_dim = min(curr_dim * 2, 512)
            model += [
                nn.Conv2d(curr_dim, next_dim, 4, stride=2, padding=1),
                nn.InstanceNorm2d(next_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            curr_dim = next_dim

        # 输出层
        next_dim = min(curr_dim * 2, 512)
        model += [
            nn.Conv2d(curr_dim, next_dim, 4, stride=1, padding=1),
            nn.InstanceNorm2d(next_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(next_dim, 1, 4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
