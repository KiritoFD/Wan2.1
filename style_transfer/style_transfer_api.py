#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换API模块

提供简单接口调用训练好的风格转换模型进行VAE向量的风格转换
"""

import os
import logging
import torch
from typing import Union, Dict, List, Tuple, Optional
import numpy as np

from .model import StyleTransferAAE

logger = logging.getLogger(__name__)

class StyleTransferAPI:
    """
    风格转换API类，提供模型加载和向量处理功能
    """
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化风格转换API
        
        参数:
            model_path (str): 训练好的模型路径
            device (str, optional): 运行设备, 'cuda'或'cpu', 默认为自动选择
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"初始化风格转换API, 设备: {self.device}")
        
        # 加载模型
        self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> None:
        """加载预训练模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            self.model = StyleTransferAAE(device=self.device)
            current_epoch = self.model.load_model(model_path, device=self.device)
            logger.info(f"已加载模型: {model_path}, 训练轮次: {current_epoch+1}")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise RuntimeError(f"模型加载失败: {e}")
        
    def transfer_style(self, 
                      features: Union[torch.Tensor, np.ndarray], 
                      strength: float = 1.0,
                      batch_size: int = 16) -> torch.Tensor:
        """
        对输入特征进行风格转换
        
        参数:
            features: 输入特征张量，支持多种格式:
                      - 单一样本: [C, H, W]
                      - 批次样本: [B, C, H, W]
                      - 时序批次样本: [B, C, T, H, W]
            strength: 风格强度，范围[0,1], 1表示完全转换，0表示不变
            batch_size: 处理批次大小，可根据显存调整
            
        返回:
            转换后的特征张量，与输入形状相同
        """
        # 转换为torch张量
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        # 确保模型在评估模式
        self.model.encoder.eval()
        self.model.mapper.eval()
        self.model.decoder.eval()
        
        # 记录原始形状
        original_shape = features.shape
        needs_squeeze = False
        
        # 标准化处理
        if len(original_shape) == 3:  # [C, H, W]
            features = features.unsqueeze(0)
            needs_squeeze = True
        
        # 处理时间维度
        has_time_dim = False
        if len(features.shape) == 5:  # [B, C, T, H, W]
            has_time_dim = True
            B, C, T, H, W = features.shape
            # 重新排列为2D批次
            features = features.transpose(1, 2).reshape(B*T, C, H, W)
        
        # 分批处理
        results = []
        total_batches = (features.size(0) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in range(0, features.size(0), batch_size):
                batch = features[i:i+batch_size]
                
                # 应用风格转换，考虑强度混合
                if strength < 1.0:
                    styled = self.model.transfer_style(batch)
                    styled = batch.to(styled.device) * (1 - strength) + styled * strength
                else:
                    styled = self.model.transfer_style(batch)
                
                results.append(styled.cpu())
                
                logger.debug(f"已处理批次 {len(results)}/{total_batches}")
        
        # 合并结果
        styled_features = torch.cat(results, dim=0)
        
        # 恢复原始形状
        if has_time_dim:
            styled_features = styled_features.reshape(B, T, C, H, W).transpose(1, 2)  # [B, C, T, H, W]
        
        # 移除批次维度(如果需要)
        if needs_squeeze:
            styled_features = styled_features.squeeze(0)
        
        return styled_features
    
    def process_file(self, 
                    input_path: str, 
                    output_path: Optional[str] = None,
                    strength: float = 1.0,
                    batch_size: int = 16,
                    preserve_original: bool = False) -> str:
        """
        处理.pt文件中的特征
        
        参数:
            input_path: 输入.pt文件路径
            output_path: 输出.pt文件路径，如未指定则在输入路径基础上添加'_styled'
            strength: 风格强度，范围[0,1]
            batch_size: 处理批次大小
            preserve_original: 是否在输出中保留原始特征
            
        返回:
            输出文件路径
        """
        # 检查文件
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
        if not input_path.endswith('.pt'):
            raise ValueError("输入文件必须是.pt格式")
        
        # 确定输出路径
        if not output_path:
            file_name = os.path.basename(input_path)
            file_dir = os.path.dirname(input_path)
            file_base, file_ext = os.path.splitext(file_name)
            output_path = os.path.join(file_dir, f"{file_base}_styled{file_ext}")
        
        logger.info(f"处理文件: {input_path}")
        
        try:
            # 加载文件
            data = torch.load(input_path, map_location='cpu')
            
            # 提取特征
            if 'features' in data:
                features = data['features']
            else:
                # 尝试找到特征张量
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        features = value
                        logger.info(f"使用键 '{key}' 作为特征")
                        break
                else:
                    raise ValueError(f"在文件中找不到特征张量")
            
            # 转换风格
            styled_features = self.transfer_style(
                features, 
                strength=strength, 
                batch_size=batch_size
            )
            
            # 准备输出数据
            output_data = {
                'features': styled_features,
                'original_path': input_path,
                'style_strength': strength
            }
            
            # 如果需要保留原始特征
            if preserve_original:
                output_data['original_features'] = features
            
            # 保存结果
            logger.info(f"保存结果到: {output_path}")
            torch.save(output_data, output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"处理文件失败: {e}")
            raise
    
    def __call__(self, 
                features: Union[torch.Tensor, np.ndarray], 
                strength: float = 1.0,
                batch_size: int = 16) -> torch.Tensor:
        """
        让类实例可直接调用，等同于transfer_style方法
        
        示例:
            api = StyleTransferAPI("model.pth")
            styled_features = api(input_features)
        """
        return self.transfer_style(features, strength, batch_size)


# 简易使用示例
def create_style_transfer(model_path: str, device: str = None) -> StyleTransferAPI:
    """
    创建风格转换API实例的便捷函数
    
    参数:
        model_path: 预训练模型路径
        device: 运行设备
        
    返回:
        StyleTransferAPI实例
    """
    return StyleTransferAPI(model_path, device)
