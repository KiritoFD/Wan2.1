#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型中使用的网络架构模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .utils import check_flash_attn_available


class ResidualDownBlock(nn.Module):
    """
    残差下采样块
    
    输入形状: [N, in_channels, H, W]
    输出形状: [N, out_channels, H/2, W/2]
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualDownBlock, self).__init__()
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)

        # 残差分支
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        residual = x
        
        # 主分支
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        out = F.leaky_relu(self.norm2(self.conv2(out)), 0.2)
        
        # 残差连接
        out = out + self.shortcut(residual)
        
        return out


class ResidualBlock(nn.Module):
    """
    残差块
    
    输入形状: [N, channels, H, W]
    输出形状: [N, channels, H, W]
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        out = out + residual
        out = F.relu(out)
        
        return out


class ResidualUpBlock(nn.Module):
    """
    残差上采样块
    
    输入形状: [N, in_channels, H, W]
    输出形状: [N, out_channels, 2H, 2W]
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualUpBlock, self).__init__()
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)

        # 残差分支
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        residual = x
        
        # 主分支
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        
        # 残差连接
        out = out + self.shortcut(residual)
        
        return out


class SelfAttention(nn.Module):
    def __init__(self, channels, use_flash_attn=True):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.use_flash_attn = use_flash_attn
        
        # 检测是否可以使用FlashAttention
        self.flash_attn_available = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_available = True
        except ImportError:
            if use_flash_attn:
                logging.warning("FlashAttention未安装，使用标准注意力机制。可通过pip install flash-attn安装")
    
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 线性映射
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C/8
        k = self.key(x).view(batch_size, -1, height * width)  # B x C/8 x HW
        v = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        
        # 使用FlashAttention或标准注意力机制
        if self.use_flash_attn and self.flash_attn_available:
            try:
                from flash_attn import flash_attn_func
                
                # 使用标准注意力，因为Flash Attention有维度问题
                energy = torch.bmm(q, k)  # B x HW x HW
                attention = F.softmax(energy, dim=2)
                attn_output = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
            except Exception as e:
                # 回退到标准注意力
                logging.warning(f"FlashAttention错误，回退到标准注意力: {e}")
                
                # 标准注意力计算
                energy = torch.bmm(q, k)  # B x HW x HW
                attention = F.softmax(energy, dim=2)
                attn_output = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
        else:
            # 标准注意力计算
            energy = torch.bmm(q, k)  # B x HW x HW
            attention = F.softmax(energy, dim=2)
            attn_output = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
        
        # 重塑回原始维度
        out = attn_output.view(batch_size, C, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class CouplingLayer(nn.Module):
    """实现简单的仿射耦合层"""
    def __init__(self, latent_dim):
        super(CouplingLayer, self).__init__()
        self.split_size = latent_dim // 2
        
        # 变换网络
        self.net = nn.Sequential(
            nn.Linear(self.split_size, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, self.split_size * 2)
        )
    
    def forward(self, x):
        x1, x2 = torch.split(x, [self.split_size, self.split_size], dim=1)
        h = self.net(x1)
        shift, scale = torch.split(h, [self.split_size, self.split_size], dim=1)
        scale = torch.sigmoid(scale + 2) + 1e-5  # 确保缩放因子始终为正
        
        y1 = x1
        y2 = x2 * scale + shift
        
        return torch.cat([y1, y2], dim=1)


class EnergyDiscriminator(nn.Module):
    """
    基于能量的GAN判别器
    """
    def __init__(self, latent_dim=256, use_spectral_norm=True):
        super(EnergyDiscriminator, self).__init__()
        
        def get_norm_layer(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        
        # 更深更稳定的判别器
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            get_norm_layer(nn.Linear(latent_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            
            nn.Dropout(0.2),
            get_norm_layer(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            
            nn.Dropout(0.1),
            get_norm_layer(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            
            get_norm_layer(nn.Linear(256, 1)),
        )
        
        # 添加自注意力机制
        self.self_attn = nn.MultiheadAttention(512, 4, batch_first=True)
        
    def forward(self, x):
        # 初始映射
        x = self.model[0:4](x)  # 到第一个LayerNorm
        
        # 应用自注意力
        x_reshaped = x.unsqueeze(1)  # [B, 1, 512]
        attn_out, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        x = x + attn_out.squeeze(1)  # 残差连接
        
        # 继续处理
        x = self.model[4:](x)
        
        return x
