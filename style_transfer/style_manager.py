#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型管理器
负责加载和应用风格转换模型
"""

import os
import logging
import torch
from .model import StyleTransferAAE

class StyleManager:
    """风格转换模型管理器"""
    
    def __init__(self, model_path, device='cuda'):
        """
        初始化风格转换模型管理器
        
        Args:
            model_path: 风格转换模型路径
            device: 运行设备
        """
        self.device = device
        self.model_path = model_path
        
        # 确保模型文件存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"风格转换模型文件不存在: {model_path}")
        
        # 加载模型
        self.model = StyleTransferAAE(device=device)
        in_channels = 16  # VAE的默认输出通道
        self.model.build_models(in_channels=in_channels)
        self.model.load_model(model_path)
        
        logging.info(f"已加载风格转换模型: {model_path}")
    
    def transfer_features(self, features):
        """
        对VAE编码的特征应用风格转换
        
        Args:
            features: VAE编码的特征 [B, C, H, W]
            
        Returns:
            转换后的特征 [B, C, H, W]
        """
        # 确保模型处于评估模式
        original_shape = features.shape
        
        # 处理输入特征
        if len(features.shape) == 3:  # [C, H, W]
            features = features.unsqueeze(0)  # 添加批次维度
        
        # 记录输入形状
        logging.info(f"风格转换输入形状: {features.shape}")
        
        # 执行风格转换
        with torch.no_grad():
            transferred = self.model.transfer_style(features.to(self.device))
        
        # 恢复原始形状
        if len(original_shape) == 3:  # [C, H, W]
            transferred = transferred.squeeze(0)
        
        logging.info(f"风格转换完成，输出形状: {transferred.shape}")
        return transferred
    
    def __call__(self, features):
        """方便调用的别名"""
        return self.transfer_features(features)
