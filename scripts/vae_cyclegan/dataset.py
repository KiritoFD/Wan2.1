#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据集和数据加载器模块"""

import logging
import random
import torch
from torch.utils.data import Dataset, DataLoader


class VAEFeatureDataset(Dataset):
    """加载.pt文件中的VAE特征向量数据集"""
    
    def __init__(self, feature_file, max_samples=None, transform=None):
        """
        参数:
            feature_file (str): 特征文件的路径 (.pt 格式)
            max_samples (int, 可选): 最大样本数量，None表示加载全部
            transform (callable, 可选): 应用于特征的变换
        """
        self.transform = transform
        
        # 加载数据
        data = torch.load(feature_file)
        self.features = data['features']
        self.image_paths = data['image_paths']
        self.metadata = data.get('metadata', {})
        
        # 限制样本数量
        if max_samples is not None and isinstance(self.features, list):
            self.features = self.features[:max_samples]
            self.image_paths = self.image_paths[:max_samples]
        elif max_samples is not None:
            self.features = self.features[:max_samples]
            self.image_paths = self.image_paths[:max_samples]
            
        # 打印数据集统计信息
        logging.info(f"已加载数据集: {feature_file}")
        
        # 获取形状分布统计
        if isinstance(self.features, list):
            shapes = {}
            for feat in self.features:
                shape_key = 'x'.join([str(s) for s in feat.shape])
                shapes[shape_key] = shapes.get(shape_key, 0) + 1
            
            logging.info(f"特征数量: {len(self.features)}")
            logging.info(f"特征形状分布: {shapes}")
        else:
            logging.info(f"特征数量: {self.features.shape[0]}")
            logging.info(f"特征形状: {self.features.shape[1:]}")

    def __len__(self):
        if isinstance(self.features, list):
            return len(self.features)
        return self.features.shape[0]

    def __getitem__(self, idx):
        # 获取特征向量
        if isinstance(self.features, list):
            feature = self.features[idx]
        else:
            feature = self.features[idx]
            
        # 应用变换
        if self.transform:
            feature = self.transform(feature)
            
        return {
            "feature": feature, 
            "path": self.image_paths[idx], 
            "shape": feature.shape,
            "idx": idx
        }


class ShapeAwareDataLoader:
    """按形状分组的数据加载器，确保每个批次中的样本具有相同的空间维度"""
    
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers
        
        # 按形状对样本分组
        self.shape_groups = {}
        for idx in range(len(dataset)):
            sample = dataset[idx]
            shape_key = '_'.join(map(str, sample["feature"].shape[-2:]))  # 使用高度和宽度作为键
            if shape_key not in self.shape_groups:
                self.shape_groups[shape_key] = []
            self.shape_groups[shape_key].append(idx)
        
        # 创建每个形状组的数据加载器
        self.dataloaders = {}
        for shape_key, indices in self.shape_groups.items():
            # 创建子集
            subset = torch.utils.data.Subset(dataset, indices)
            # 创建数据加载器
            self.dataloaders[shape_key] = DataLoader(
                subset, 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                drop_last=drop_last
            )
        
        # 计算迭代的长度
        self.length = sum([len(dl) for dl in self.dataloaders.values()])
        
        # 获取所有形状组的迭代器
        self.iterators = {}
        self.current_shape_keys = list(self.shape_groups.keys())
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        # 重置迭代器
        self.iterators = {
            shape_key: iter(self.dataloaders[shape_key]) 
            for shape_key in self.shape_groups
        }
        self.current_shape_keys = list(self.shape_groups.keys())
        if self.shuffle:
            random.shuffle(self.current_shape_keys)
        
        self.remaining_batches = self.length
        return self
    
    def __next__(self):
        if self.remaining_batches <= 0:
            raise StopIteration
        
        # 轮流从不同形状组中获取批次
        for i in range(len(self.current_shape_keys)):
            shape_key = self.current_shape_keys[i]
            try:
                batch = next(self.iterators[shape_key])
                self.remaining_batches -= 1
                # 轮换形状键的顺序
                self.current_shape_keys = self.current_shape_keys[1:] + [self.current_shape_keys[0]]
                return batch
            except StopIteration:
                # 如果这个形状组已经迭代完，从列表中移除
                self.current_shape_keys.pop(i)
                if not self.current_shape_keys:  # 所有形状组都迭代完了
                    raise StopIteration
                return next(self)
        
        # 应该不会走到这里，但为了安全起见
        raise StopIteration


class VAEFeatureNormalizer(torch.nn.Module):
    """VAE特征归一化器，用于标准化不同形状的特征"""
    
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        """特征归一化"""
        # 计算每个特征的均值和标准差
        if x.dim() == 4:  # [C,T,H,W]
            # 在空间维度上计算
            mean = x.mean(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        else:
            mean = x.mean()
            std = x.std() + 1e-6
        
        # 标准化
        x = (x - mean) / std
        
        # 应用目标分布参数
        x = x * self.std + self.mean
        
        return x


class ReplayBuffer:
    """经验回放缓冲区，用于训练稳定性"""
    
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result)
