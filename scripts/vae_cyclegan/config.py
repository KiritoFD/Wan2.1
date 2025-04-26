#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""配置管理模块"""

import json
from datetime import datetime
import torch

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
