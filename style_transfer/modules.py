#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型中的基本模块实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from .networks import ResidualDownBlock, ResidualBlock, ResidualUpBlock, SelfAttention
from .utils import check_flash_attn_available


class Encoder(nn.Module):
    """
    编码器：将输入的VAE潜在向量映射到新的潜在空间
    
    输入形状: [N, 16, 32, 32] (已去除时间维度)
    输出形状: [N, latent_dim]
    """
    def __init__(self, in_channels=16, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 降采样: 32x32 -> 16x16
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 降采样: 16x16 -> 8x8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # 自适应平均池化，保证输出大小固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层，输出潜在向量
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # 输入形状: [N, 16, 32, 32]
        x = self.conv1(x)          # -> [N, 64, 16, 16]
        x = self.conv2(x)          # -> [N, 256, 8, 8]
        x = self.adaptive_pool(x)  # -> [N, 256, 1, 1]
        x = torch.flatten(x, 1)    # -> [N, 256]
        x = self.fc(x)             # -> [N, latent_dim]
        return x


class Decoder(nn.Module):
    """
    解码器：将潜在空间映射回VAE潜在向量
    
    输入形状: [N, latent_dim]
    输出形状: [N, 16, 32, 32]
    """
    def __init__(self, out_channels=16, latent_dim=128):
        super(Decoder, self).__init__()
        
        # 全连接层，从潜在向量生成特征图
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 上采样: 8x8 -> 16x16
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 上采样: 16x16 -> 32x32
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            # 不使用激活函数，因为VAE输出范围不确定
        )

    def forward(self, x):
        # 输入形状: [N, latent_dim]
        x = self.fc(x)                            # -> [N, 256*8*8]
        x = x.view(-1, 256, 8, 8)                 # -> [N, 256, 8, 8]
        x = self.conv1(x)                         # -> [N, 128, 16, 16]
        x = self.conv2(x)                         # -> [N, out_channels, 32, 32]
        return x


class Discriminator(nn.Module):
    """
    判别器：区分潜在向量的来源（风格A或风格B）
    
    输入形状: [N, latent_dim]
    输出形状: [N, 1]
    """
    def __init__(self, latent_dim=128):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class StyleMapper(nn.Module):
    """
    风格映射网络 - MLP形式
    
    输入形状: [N, latent_dim]
    输出形状: [N, latent_dim]
    """
    def __init__(self, latent_dim=256, depth=4, width_factor=2):
        super(StyleMapper, self).__init__()
        
        hidden_dim = latent_dim * width_factor
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 中间层
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)


class EnhancedEncoder(nn.Module):
    def __init__(self, in_channels=16, latent_dim=256, use_attention=True, use_flash_attn=True):
        super(EnhancedEncoder, self).__init__()
        
        # 激活函数
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.norm_in = nn.InstanceNorm2d(64)
        
        # 下采样块
        self.down1 = ResidualDownBlock(64, 128)  # 32x32 -> 16x16
        self.down2 = ResidualDownBlock(128, 256)  # 16x16 -> 8x8
        self.down3 = ResidualDownBlock(256, 512)  # 8x8 -> 4x4
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(512, use_flash_attn=use_flash_attn)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 映射到潜在空间
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        # 初始处理
        x = self.act(self.norm_in(self.conv_in(x)))
        
        # 下采样
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # 注意力
        if self.use_attention:
            x = self.attention(x)
        
        # 全局池化和映射
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        
        return x


class EnhancedDecoder(nn.Module):
    """
    解码器：将潜在空间映射回VAE潜在向量
    
    输入形状: [N, latent_dim]
    输出形状: [N, 16, 32, 32]
    """
    def __init__(self, out_channels=16, latent_dim=256, use_residual=True):
        super(EnhancedDecoder, self).__init__()
        
        # 潜在空间 -> 4x4 特征图
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.norm_fc = nn.BatchNorm2d(512)
        
        # 上采样块
        self.up1 = ResidualUpBlock(512, 256)  # 4x4 -> 8x8
        self.up2 = ResidualUpBlock(256, 128)  # 8x8 -> 16x16
        self.up3 = ResidualUpBlock(128, 64)   # 16x16 -> 32x32
        
        # Residual连接 - 修改为与上采样块的输出通道数匹配
        self.use_residual = use_residual
        if use_residual:
            self.res_512 = ResidualBlock(512)
            self.res_256 = ResidualBlock(256)
            self.res_128 = ResidualBlock(128)
            self.res_64 = ResidualBlock(64)
        
        # 输出层
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 从潜在空间重建特征图
        x = self.fc(x).view(-1, 512, 4, 4)
        x = F.relu(self.norm_fc(x))
        
        # 残差层应用于对应通道数的特征图
        if self.use_residual:
            x = self.res_512(x)
        
        # 上采样512->256
        x = self.up1(x)
        if self.use_residual:
            x = self.res_256(x)
        
        # 上采样256->128
        x = self.up2(x)
        if self.use_residual:
            x = self.res_128(x)
        
        # 上采样128->64
        x = self.up3(x)
        if self.use_residual:
            x = self.res_64(x)
            
        # 输出层 64->out_channels
        x = self.conv_out(x)
        
        return x


class EnhancedDiscriminator(nn.Module):
    """
    判别器：区分潜在向量的来源（风格A或风格B）
    
    输入形状: [N, latent_dim]
    输出形状: [N, 1]
    """
    def __init__(self, latent_dim=256, use_spectral_norm=True):
        super(EnhancedDiscriminator, self).__init__()
        
        def get_norm_layer(dim):
            if use_spectral_norm:
                return nn.utils.spectral_norm(nn.Linear(dim, dim))
            else:
                return nn.Identity()
        
        # 更深的判别器网络
        self.model = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 512),
            get_norm_layer(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            get_norm_layer(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            get_norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 输出原始logits，不用sigmoid
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


class ContentExtractionModule(nn.Module):
    def __init__(self, in_channels=16, latent_dim=128):
        super(ContentExtractionModule, self).__init__()
        
        # 收缩通道，捕获结构特征
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )
    
    def forward(self, x):
        return self.content_encoder(x)


class CouplingLayer(nn.Module):
    """
    Coupling layer for normalizing flow
    """
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim//2, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim//2)
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1 = x1
        y2 = x2 * torch.exp(self.net(x1)) + self.net(x1)
        return torch.cat([y1, y2], dim=-1)


class NormalizingFlowStyleMapper(nn.Module):
    """
    使用归一化流的风格映射器 - 可逆且强大的表达能力
    """
    def __init__(self, latent_dim=256, flow_steps=4):
        super(NormalizingFlowStyleMapper, self).__init__()
        
        # 基础映射网络
        self.base_mapper = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # 归一化流步骤
        self.flow_steps = nn.ModuleList([
            CouplingLayer(latent_dim) for _ in range(flow_steps)
        ])
    
    def forward(self, x):
        # 基础映射
        h = self.base_mapper(x)
        
        # 应用归一化流
        for flow in self.flow_steps:
            h = flow(h)
            
        return h
