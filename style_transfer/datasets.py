#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型数据集
"""

import os
import torch
import logging
from torch.utils.data import Dataset


class StyleDataset(Dataset):
    """
    风格数据集：加载.pt文件中的VAE编码特征
    
    Args:
        data_path (str): .pt文件路径
        squeeze_time (bool): 是否去除时间维度(通常为1)
    """
    def __init__(self, data_path, squeeze_time=True):
        self.path = data_path
        
        try:
            # 尝试加载文件的不同方式
            try:
                # 主要加载方式
                logging.info(f"尝试加载数据: {data_path}")
                data = torch.load(data_path)
                logging.info(f"数据加载成功，文件大小: {os.path.getsize(data_path) / (1024*1024):.2f}MB")
            except Exception as e:
                # 如果主要方式失败，尝试备用加载方式
                logging.warning(f"常规加载失败，尝试使用备用方式: {str(e)}")
                
                # 使用map_location确保加载到CPU，避免CUDA内存问题
                data = torch.load(data_path, map_location='cpu')
            
            # 处理features，根据不同的格式进行适配
            if 'features' in data:
                features = data['features']
                
                # 如果features是单个张量
                if isinstance(features, torch.Tensor):
                    if features.dim() == 4:  # [16, 1, 32, 32]
                        # 添加批次维度
                        self.features = features.unsqueeze(0)
                    else:
                        # 假设已有批次维度 [N, 16, 1, 32, 32]
                        self.features = features
                # 如果features是张量列表
                elif isinstance(features, list):
                    # 堆叠为单个张量
                    logging.info(f"发现特征列表，包含 {len(features)} 个张量")
                    try:
                        self.features = torch.stack(features)
                    except Exception as stack_err:
                        # 如果无法直接堆叠，检查维度不一致问题
                        logging.warning(f"无法直接堆叠特征: {stack_err}")
                        
                        # 转换为统一大小
                        if len(features) > 0:
                            sample = features[0]
                            expected_shape = list(sample.shape)
                            consistent_features = []
                            
                            for i, feat in enumerate(features):
                                if feat.shape == sample.shape:
                                    consistent_features.append(feat)
                                else:
                                    logging.warning(f"跳过形状不一致的特征 {i}: {feat.shape} != {expected_shape}")
                            
                            if consistent_features:
                                self.features = torch.stack(consistent_features)
                                logging.info(f"成功堆叠 {len(consistent_features)}/{len(features)} 个形状一致的特征")
                            else:
                                raise ValueError(f"无法找到形状一致的特征进行堆叠")
                        else:
                            raise ValueError("特征列表为空")
                else:
                    raise ValueError(f"不支持的features类型: {type(features)}")
            else:
                # 尝试检查文件包含的内容
                keys = list(data.keys()) if isinstance(data, dict) else []
                raise ValueError(f"数据中找不到'features'键，可用键: {keys}")
            
            # 记录原始形状
            self.original_shape = self.features.shape
            logging.info(f"加载的特征形状: {self.original_shape}")
            
            # 可选去除时间维度（通常为1）
            if squeeze_time and self.features.shape[2] == 1:
                self.features = self.features.squeeze(2)
                logging.info(f"去除时间维度后的形状: {self.features.shape}")
            
            # 获取图像路径和元数据，如果有
            self.image_paths = data.get('image_paths', None)
            self.metadata = data.get('metadata', None)
            
            logging.info(f"数据集加载完成: {data_path}")
            logging.info(f"  - 特征形状: {self.features.shape}")
            logging.info(f"  - 样本数量: {len(self.features)}")
            
        except Exception as e:
            logging.error(f"加载数据集 {data_path} 时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
    
    @property
    def feature_shape(self):
        return self.features.shape[1:]
    
    def get_metadata(self):
        return self.metadata
    
    def get_paths(self):
        return self.image_paths
