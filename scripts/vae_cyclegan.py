#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE特征向量的CycleGAN实现
用于在两个领域的VAE编码向量之间学习映射关系

特性:
- 支持处理可变形状的VAE特征向量 (16x1xHxW)
- 适应不同形状与分辨率的特征向量映射
- 灵活的配置参数与命令行接口
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import random
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wan.modules.vae import WanVAE

# 全局默认配置
DEFAULT_CONFIG = {
    # 数据参数
    "domain_a_path": "C:/Users/xy/Downloads/Train/Train/Train/shadow/shadow.pt",
    "domain_b_path": "C:/Users/xy/Downloads/Train/Train/Train/shadow_free/no_shadow.pt",
    "batch_size": 1,
    "num_workers": 4,
    "max_samples": None,
    "shuffle": True,
    "drop_last": True,
    
    # 模型参数
    "input_channels": 16,
    "output_channels": 16,
    "n_residual_blocks": 9,
    "base_filters": 64,
    "discriminator_layers": 3,
    
    # 训练参数
    "n_epochs": 200,
    "decay_epoch": 100,
    "lr": 0.0002,
    "b1": 0.5,
    "b2": 0.999,
    "lambda_cycle": 10.0,
    "lambda_identity": 5.0,
    "buffer_size": 50,
    
    # 其他参数
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "output_dir": "./output/vae_cyclegan",
    "save_freq": 10,
    "sample_freq": 5,
    "seed": 42,
    "model_name": f"vae_cyclegan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "resume_path": None,
    
    # 测试参数
    "test_input": None,
    "test_output": None,
    "test_direction": "A2B",
    "test_batch_size": 4,
}

class Config:
    """配置管理类"""
    
    def __init__(self, **kwargs):
        """初始化配置"""
        # 设置默认值
        for key, value in DEFAULT_CONFIG.items():
            setattr(self, key, value)
        
        # 更新传入的参数
        for key, value in kwargs.items():
            if key in DEFAULT_CONFIG or not hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置"""
        config_dict = vars(args)
        # 特殊处理domain路径
        if hasattr(args, "domain_a"):
            config_dict["domain_a_path"] = args.domain_a
        if hasattr(args, "domain_b"):
            config_dict["domain_b_path"] = args.domain_b
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, filepath):
        """从JSON文件加载配置"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save(self, filepath):
        """将配置保存到JSON文件"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def __str__(self):
        """友好的字符串表示"""
        return json.dumps(self.__dict__, indent=4)

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


# 特征预处理与归一化
class VAEFeatureNormalizer(nn.Module):
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


# 灵活的残差块，适应不同输入尺寸
class ResidualBlock(nn.Module):
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


class VAECycleGAN:
    """VAE特征向量的CycleGAN实现"""
    
    def __init__(self, config, criterion=None):
        """
        初始化VAE CycleGAN模型
        
        参数:
            config: 配置对象，包含模型参数
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化网络
        self.netG_A2B = AdaptiveGenerator(
            input_channels=config.input_channels, 
            output_channels=config.output_channels,
            n_residual_blocks=config.n_residual_blocks,
            base_filters=config.base_filters
        )
        
        self.netG_B2A = AdaptiveGenerator(
            input_channels=config.output_channels, 
            output_channels=config.input_channels,
            n_residual_blocks=config.n_residual_blocks,
            base_filters=config.base_filters
        )
        
        self.netD_A = Discriminator(
            input_channels=config.input_channels,
            base_filters=config.base_filters,
            n_layers=config.discriminator_layers
        )
        
        self.netD_B = Discriminator(
            input_channels=config.output_channels,
            base_filters=config.base_filters,
            n_layers=config.discriminator_layers
        )
        
        # 移动模型到指定设备
        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)
        
        # 应用权重初始化
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        
        # 定义损失函数
        self.criterion_GAN = nn.MSELoss()         # 对抗损失
        # 使用自定义损失函数或默认MSE损失
        self.criterion_cycle = criterion if criterion is not None else nn.MSELoss()
        self.criterion_identity = criterion if criterion is not None else nn.MSELoss()
        
        # 定义优化器
        self.optimizer_G = optim.Adam(
            list(self.netG_A2B.parameters()) + list(self.netG_B2A.parameters()),
            lr=config.lr, betas=(config.b1, config.b2)
        )
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=config.lr, betas=(config.b1, config.b2))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=config.lr, betas=(config.b1, config.b2))
        
        # 定义学习率调度器
        self.lr_scheduler_G = lr_scheduler.LambdaLR(
            self.optimizer_G, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / float(config.n_epochs - config.decay_epoch)
        )
        self.lr_scheduler_D_A = lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / float(config.n_epochs - config.decay_epoch)
        )
        self.lr_scheduler_D_B = lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / float(config.n_epochs - config.decay_epoch)
        )
        
        # 创建经验回放缓冲区
        self.fake_A_buffer = ReplayBuffer(config.buffer_size)
        self.fake_B_buffer = ReplayBuffer(config.buffer_size)
        
        # 创建日志记录器
        self.history = {
            'train_losses': {'G': [], 'D_A': [], 'D_B': [], 'cycle': [], 'identity': []},
            'lr': {'G': [], 'D_A': [], 'D_B': []}
        }
        
        logging.info(f"模型已初始化，设备: {self.device}")

    def _resize_tensor(self, tensor, size):
        """调整张量的空间维度尺寸到指定大小，增强处理能力
        
        参数:
            tensor: 输入张量，形状为[B,C,H,W]
            size: 目标尺寸 (H, W)或(H_target, W_target)
            
        返回:
            调整后的张量
        """
        # 确认大小参数
        if isinstance(size, tuple) and len(size) == 2:
            target_h, target_w = size
        else:
            target_h, target_w = size, size
            
        current_h, current_w = tensor.shape[-2], tensor.shape[-1]
        
        # 如果已经匹配，直接返回
        if current_h == target_h and current_w == target_w:
            return tensor
        
        # 使用双线性插值调整尺寸
        try:
            # 保存输入形状
            original_shape = tensor.shape
            b = original_shape[0] if len(original_shape) > 3 else 1
            c = original_shape[1] if len(original_shape) > 3 else original_shape[0]
            
            # 调整尺寸
            if len(original_shape) > 3:  # 批量张量 [B,C,H,W]
                resized = F.interpolate(
                    tensor, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:  # 单个张量 [C,H,W]
                resized = F.interpolate(
                    tensor.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
            return resized
        except Exception as e:
            logging.error(f"调整张量尺寸时出错: {e}, 输入形状: {tensor.shape}, 目标尺寸: {(target_h, target_w)}")
            # 返回原始张量
            return tensor

    def _update_epoch_history(self, epoch_losses, batch_count):
        """计算平均损失并更新训练历史"""
        # 计算平均损失
        for k in epoch_losses.keys():
            epoch_losses[k] /= max(1, batch_count)
            self.history['train_losses'][k].append(epoch_losses[k])
        return epoch_losses
    
    def _update_learning_rates(self):
        """更新所有优化器的学习率并记录当前值"""
        # 更新学习率
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()
        
        # 记录当前学习率
        lr_g = self.optimizer_G.param_groups[0]['lr']
        lr_d_a = self.optimizer_D_A.param_groups[0]['lr']
        lr_d_b = self.optimizer_D_B.param_groups[0]['lr']
        self.history['lr']['G'].append(lr_g)
        self.history['lr']['D_A'].append(lr_d_a)
        self.history['lr']['D_B'].append(lr_d_b)
        
        return {"G": lr_g, "D_A": lr_d_a, "D_B": lr_d_b}
        
    def _check_shape_compatibility(self, real_A, real_B):
        """检查两个域的形状兼容性并返回是否匹配"""
        A_shape = real_A.shape[-2:]
        B_shape = real_B.shape[-2:]
        
        # 检查高度和宽度比例，如果差异太大则不兼容
        h_ratio = A_shape[0] / B_shape[0]
        w_ratio = A_shape[1] / B_shape[1]
        
        # 如果比例相差太大（超过2倍），则认为不兼容
        if h_ratio > 2 or h_ratio < 0.5 or w_ratio > 2 or w_ratio < 0.5:
            return False
        
        # 即使比例接近，如果实际尺寸差距太大，可能仍会导致判别器输出尺寸不匹配
        if abs(A_shape[0] - B_shape[0]) > 32 or abs(A_shape[1] - B_shape[1]) > 32:
            return False
        
        return True

    def train(self, dataloader_A, dataloader_B):
        """训练CycleGAN模型"""
        # 确保输出目录存在
        os.makedirs(os.path.join(self.config.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)
        
        # 保存配置
        if isinstance(self.config, Config):
            self.config.save(os.path.join(self.config.output_dir, "config.json"))
        
        logging.info(f"开始训练，共{self.config.n_epochs}个epoch")
        
        # 主训练循环
        for epoch in range(1, self.config.n_epochs + 1):
            start_time = time.time()
            epoch_losses = {'G': 0, 'D_A': 0, 'D_B': 0, 'cycle': 0, 'identity': 0}
            
            # 创建进度条
            with tqdm(total=min(len(dataloader_A), len(dataloader_B)), 
                     desc=f"Epoch {epoch}/{self.config.n_epochs}") as pbar:
                # 同时遍历两个数据加载器
                for batch_idx, (batch_A, batch_B) in enumerate(zip(dataloader_A, dataloader_B)):
                    real_A = batch_A["feature"].to(self.device)
                    real_B = batch_B["feature"].to(self.device)
                    
                    # 新增：检查数据批次的形状兼容性
                    if not self._check_shape_compatibility(real_A, real_B):
                        logging.warning(f"跳过形状不兼容的批次: A={real_A.shape}, B={real_B.shape}")
                        continue
                    
                    # 处理VAE特征的特殊维度：保持时间维度为1
                    # VAE特征形状: [B, C, T, H, W] 或 [C, T, H, W]
                    if real_A.dim() == 5:  # 批处理多帧情况
                        real_A = real_A[:, :, 0]  # 只使用第一帧
                    elif real_A.dim() == 4 and real_A.size(1) > 1:  # 单样本多帧情况
                        real_A = real_A[:, 0].unsqueeze(1)  # 只使用第一帧，保持时间维度
                    
                    if real_B.dim() == 5:
                        real_B = real_B[:, :, 0]  # 只使用第一帧
                    elif real_B.dim() == 4 and real_B.size(1) > 1:
                        real_B = real_B[:, 0].unsqueeze(1)  # 只使用第一帧，保持时间维度
                    
                    # 确保形状是[B,C,H,W]用于2D卷积
                    if real_A.dim() == 4 and real_A.size(1) == 1:  # 如果是[B,1,H,W]
                        real_A = real_A.squeeze(1)  # 变成[B,H,W]
                        if real_A.dim() == 3:  # 如果只有3维，添加通道维度
                            real_A = real_A.unsqueeze(1)  # 变成[B,1,H,W]
                    
                    if real_B.dim() == 4 and real_B.size(1) == 1:
                        real_B = real_B.squeeze(1)
                        if real_B.dim() == 3:
                            real_B = real_B.unsqueeze(1)
                    
                    batch_size = real_A.size(0)
                    
                    # 检查形状，记录原始形状以便后续恢复
                    A_shape = real_A.shape
                    B_shape = real_B.shape
                    
                    # 如果高度或宽度不匹配，记录并跳过该批次
                    if A_shape[-2:] != B_shape[-2:]:
                        logging.warning(f"跳过形状不匹配的批次: A={A_shape}, B={B_shape}")
                        continue
                    
                    # 计算判别器输出大小 - 自适应判别器输出大小
                    try:
                        valid_shape_A = self.netD_A(real_A).shape
                        valid_shape_B = self.netD_B(real_B).shape
                    except Exception as e:
                        logging.error(f"判别器前向传播错误: {e}, A形状: {real_A.shape}, B形状: {real_B.shape}")
                        continue
                    
                    # 创建真假标签
                    valid_A = torch.ones(valid_shape_A, device=self.device)
                    fake_A = torch.zeros(valid_shape_A, device=self.device)
                    valid_B = torch.ones(valid_shape_B, device=self.device)
                    fake_B = torch.zeros(valid_shape_B, device=self.device)
                    
                    # 训练生成器
                    #----------------------
                    self.optimizer_G.zero_grad()
                    # 身份损失 (如果lambda_identity > 0)
                    if self.config.lambda_identity > 0:
                        try:
                            # G_A2B(B) 应接近 B
                            identity_B = self.netG_A2B(real_B)
                            # 检查输出形状是否匹配
                            if identity_B.shape[-2:] != real_B.shape[-2:]:
                                identity_B = F.interpolate(identity_B, size=real_B.shape[-2:], mode='bilinear', align_corners=False)
                            loss_identity_B = self.criterion_identity(identity_B, real_B) * self.config.lambda_identity
                            
                            # G_B2A(A) 应接近 A
                            identity_A = self.netG_B2A(real_A)
                            # 检查输出形状是否匹配
                            if identity_A.shape[-2:] != real_A.shape[-2:]:
                                identity_A = F.interpolate(identity_A, size=real_A.shape[-2:], mode='bilinear', align_corners=False)
                            loss_identity_A = self.criterion_identity(identity_A, real_A) * self.config.lambda_identity
                            
                            loss_identity = loss_identity_A + loss_identity_B
                        except RuntimeError as e:
                            logging.warning(f"计算身份损失时出错: {e}, 跳过身份损失计算")
                            loss_identity = torch.tensor(0.0, device=self.device)
                    else:
                        loss_identity = torch.tensor(0.0, device=self.device)
                    
                    # GAN 损失
                    try:
                        # 生成假样本
                        fake_B = self.netG_A2B(real_A)
                        # 确保形状匹配
                        if fake_B.shape[-2:] != real_B.shape[-2:]:
                            fake_B = F.interpolate(fake_B, size=real_B.shape[-2:], mode='bilinear', align_corners=False)
                        loss_GAN_A2B = self.criterion_GAN(self.netD_B(fake_B), valid_B)
                        
                        fake_A = self.netG_B2A(real_B)
                        # 确保形状匹配
                        if fake_A.shape[-2:] != real_A.shape[-2:]:
                            fake_A = F.interpolate(fake_A, size=real_A.shape[-2:], mode='bilinear', align_corners=False)
                        loss_GAN_B2A = self.criterion_GAN(self.netD_A(fake_A), valid_A)
                        
                        loss_GAN = loss_GAN_A2B + loss_GAN_B2A
                    except Exception as e:
                        logging.error(f"计算GAN损失时出错: {e}")
                        loss_GAN = torch.tensor(0.0, device=self.device)
                    
                    # 循环一致性损失
                    try:
                        # A -> B -> A 应接近原始 A
                        recovered_A = self.netG_B2A(fake_B)
                        if recovered_A.shape[-2:] != real_A.shape[-2:]:
                            recovered_A = F.interpolate(recovered_A, size=real_A.shape[-2:], mode='bilinear', align_corners=False)
                        loss_cycle_A = self.criterion_cycle(recovered_A, real_A) * self.config.lambda_cycle
                        
                        # B -> A -> B 应接近原始 B
                        recovered_B = self.netG_A2B(fake_A)
                        if recovered_B.shape[-2:] != real_B.shape[-2:]:
                            recovered_B = F.interpolate(recovered_B, size=real_B.shape[-2:], mode='bilinear', align_corners=False)
                        loss_cycle_B = self.criterion_cycle(recovered_B, real_B) * self.config.lambda_cycle
                        
                        loss_cycle = loss_cycle_A + loss_cycle_B
                    except RuntimeError as e:
                        logging.warning(f"计算循环一致性损失时出错: {e}, 跳过循环一致性损失计算")
                        loss_cycle = torch.tensor(0.0, device=self.device)
                    
                    # 总生成器损失
                    loss_G = loss_GAN + loss_cycle + loss_identity
                    
                    # 反向传播和优化
                    loss_G.backward()
                    self.optimizer_G.step()
                    
                    #----------------------
                    # 训练判别器 A
                    #----------------------
                    self.optimizer_D_A.zero_grad()
                    # 真实样本
                    loss_real = self.criterion_GAN(self.netD_A(real_A), valid_A)
                    # 假样本 (从缓冲区获取)
                    fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A.detach())
                    # 获取判别器输出形状，确保标签匹配
                    fake_A_output = self.netD_A(fake_A_buffer)
                    # 创建与输出匹配形状的标签
                    fake_A_target = torch.zeros_like(fake_A_output, device=self.device)
                    loss_fake = self.criterion_GAN(fake_A_output, fake_A_target)
                    # 总判别器损失
                    loss_D_A = (loss_real + loss_fake) * 0.5
                    
                    # 反向传播和优化
                    loss_D_A.backward()
                    self.optimizer_D_A.step()
                    
                    #----------------------
                    # 训练判别器 B
                    #----------------------
                    self.optimizer_D_B.zero_grad()
                    # 真实样本
                    loss_real = self.criterion_GAN(self.netD_B(real_B), valid_B)
                    # 假样本 (从缓冲区获取)
                    fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B.detach())
                    # 获取判别器输出形状，确保标签匹配
                    fake_B_output = self.netD_B(fake_B_buffer)
                    # 创建与输出匹配形状的标签
                    fake_B_target = torch.zeros_like(fake_B_output, device=self.device)
                    loss_fake = self.criterion_GAN(fake_B_output, fake_B_target)
                    # 总判别器损失
                    loss_D_B = (loss_real + loss_fake) * 0.5
                    
                    # 反向传播和优化
                    loss_D_B.backward()
                    self.optimizer_D_B.step()
                    
                    #----------------------
                    # 记录损失
                    #----------------------
                    epoch_losses['G'] += loss_G.item()
                    epoch_losses['D_A'] += loss_D_A.item()
                    epoch_losses['D_B'] += loss_D_B.item()
                    epoch_losses['cycle'] += loss_cycle.item()
                    epoch_losses['identity'] += loss_identity.item()            
                    
                    # 更新进度条
                    description = f"Epoch {epoch}/{self.config.n_epochs} "
                    description += f"[G: {loss_G.item():.3f}, D_A: {loss_D_A.item():.3f}, D_B: {loss_D_B.item():.3f}]"
                    pbar.set_description(description)
                    pbar.update(1)                    
                
            # 计算平均损失并更新历史
            batch_count = min(len(dataloader_A), len(dataloader_B))
            epoch_losses = self._update_epoch_history(epoch_losses, batch_count)
            
            # 更新学习率并获取当前值
            learning_rates = self._update_learning_rates()
            
            # 计算训练时间
            epoch_time = time.time() - start_time
            
            # 打印统计信息
            logging.info(f"[Epoch {epoch}/{self.config.n_epochs}] "
                         f"Loss_G: {epoch_losses['G']:.4f}, Loss_D_A: {epoch_losses['D_A']:.4f}, "
                         f"Loss_D_B: {epoch_losses['D_B']:.4f}, Loss_cycle: {epoch_losses['cycle']:.4f}, "
                         f"Loss_identity: {epoch_losses['identity']:.4f}, "
                         f"LR: G={learning_rates['G']:.6f}, D_A={learning_rates['D_A']:.6f}, D_B={learning_rates['D_B']:.6f}, "
                         f"Time: {epoch_time:.2f}s")
            
            # 保存模型
            if epoch % self.config.save_freq == 0 or epoch == self.config.n_epochs:
                self.save_models(epoch)
                self.plot_losses(epoch)
                logging.info(f"模型已保存 (Epoch {epoch})")
            
            # 保存一些示例结果
            if epoch % self.config.sample_freq == 0:
                self.save_samples(epoch, dataloader_A, dataloader_B)
        
        # 保存最终模型和历史记录
        self.save_models("final")
        self.save_history()
        self.plot_losses("final")
        logging.info("训练完成")
    
    def _prepare_features(self, features):
        """准备VAE特征以适应网络处理
        
        参数:
            features: 批量特征张量，形状可能是[B,C,T,H,W]或[B,C,H,W]
        
        返回:
            准备好的特征，形状为[B,C,H,W]
        """
        # 处理5D张量（有时间维度）
        if features.dim() == 5:  # [B,C,T,H,W]
            # 只取第一帧
            features = features[:, :, 0]
        
        # 处理4D张量（可能有单一时间维度）
        if features.dim() == 4 and features.size(1) == 1:  # [B,1,H,W]
            # 检查第二维是否为时间维度
            if features.size(2) > 1 and features.size(3) > 1:
                # 可能是[B,C,H,W]，保持原样
                pass
            else:
                # 可能是[B,T,H,W]，转换为[B,1,H,W]
                features = features.permute(0, 2, 3, 1)
        
        return features
    
    def translate_domain(self, features, direction="A2B"):
        """ 将特征从一个域转换到另一个域
        
        参数:
            features: 要转换的特征列表或张量
            direction: 转换方向 "A2B" 或 "B2A"
            
        返回:
            转换后的特征
        """
        # 设置为评估模式
        self.netG_A2B.eval()
        self.netG_B2A.eval()
        
        results = []
        
        with torch.no_grad():
            if isinstance(features, list):
                for feat in features:
                    # 准备特征
                    feat = self._prepare_features(feat.unsqueeze(0) if feat.dim() < 4 else feat)
                    # 选择转换方向
                    if direction == "A2B":
                        translated = self.netG_A2B(feat.to(self.device)).cpu()
                    else:
                        translated = self.netG_B2A(feat.to(self.device)).cpu()
                    
                    results.append(translated.squeeze(0) if translated.size(0) == 1 else translated)
            else:
                # 准备特征
                features = self._prepare_features(features.unsqueeze(0) if features.dim() < 4 else features)
                # 选择转换方向
                if direction == "A2B":
                    results = self.netG_A2B(features.to(self.device)).cpu()
                else:
                    results = self.netG_B2A(features.to(self.device)).cpu()
        
        # 恢复训练模式
        self.netG_A2B.train()
        self.netG_B2A.train()
        
        return results
    
    def translate_file(self, input_file, output_file, direction="A2B", batch_size=4):
        """
        转换整个文件中的特征
        
        参数:
            input_file: 输入特征文件路径
            output_file: 输出特征文件路径
            direction: 转换方向 "A2B" 或 "B2A"
            batch_size: 批处理大小
        """
        # 加载输入数据
        data = torch.load(input_file)
        features = data['features']
        image_paths = data['image_paths']
        metadata = data.get('metadata', {})
        
        # 创建数据加载器
        dataset = VAEFeatureDataset(input_file)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 设置为评估模式
        self.netG_A2B.eval()
        self.netG_B2A.eval()
        
        # 进行转换
        translated_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"转换 {direction}"):
                features_batch = self._prepare_features(batch["feature"]).to(self.device)
                if direction == "A2B":
                    translated_batch = self.netG_A2B(features_batch).cpu()
                else:
                    translated_batch = self.netG_B2A(features_batch).cpu()
                
                # 添加到结果列表
                if isinstance(translated_batch, list):
                    translated_features.extend(translated_batch)
                else:
                    for i in range(translated_batch.size(0)):
                        translated_features.append(translated_batch[i])
        
        # 恢复训练模式
        self.netG_A2B.train()
        self.netG_B2A.train()
        
        # 更新元数据
        metadata.update({
            "translated_direction": direction,
            "original_file": os.path.basename(input_file),
            "translation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": self.config.__dict__ if hasattr(self.config, '__dict__') else None
        })
        
        # 保存结果
        torch.save({
            'features': translated_features if len(translated_features) > 1 else translated_features[0],
            'image_paths': image_paths,
            'metadata': metadata
        }, output_file)
        
        logging.info(f"已将 {len(translated_features)} 个特征从 {input_file} 转换并保存到 {output_file}")


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
    
    # 初始化模型
    model = VAECycleGAN(config)
    
    # 加载预训练模型（如果指定）
    if config.resume_path:
        try:
            if config.resume_path.isdigit():
                model.load_models(epoch=int(config.resume_path))
            else:
                model.load_models(path=config.resume_path)
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            return
    
    # 训练或测试
    if args.mode == "train":
        # 训练模型
        logging.info("开始训练...")
        losses = model.train(dataloader_A, dataloader_B)
    else:
        logging.info("开始测试...")
        # 测试模式
        if args.test_input and args.test_output:
            model.translate_file(
                # 转换整个文件
                args.test_input,
                args.test_output,
                batch_size=args.test_batch_size,
                direction=args.test_direction,
            )
        else:
            logging.info("测试输入或输出文件未提供，跳过测试步骤。")


if __name__ == "__main__":
    main()
