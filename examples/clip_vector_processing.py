#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
示例：如何使用CLIPVectorEncoder处理CLIP向量
"""

import os
import sys
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.encode_clip_vectors import CLIPVectorEncoder
from wan.modules.vae import WanVAE

# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_sample_clip_vector(shape=(3, 8, 32, 32)):
    """生成示例CLIP向量用于测试"""
    return torch.randn(shape)

def main():
    # 检查VAE预训练模型是否存在
    vae_path = "cache/vae_step_411000.pth"
    if not os.path.exists(vae_path):
        logging.warning(f"找不到VAE预训练模型: {vae_path}")
        logging.warning("请下载模型或修改路径指向有效的模型文件")
        vae_path = input("请输入VAE模型路径 (按回车使用默认路径): ") or vae_path
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"使用设备: {device}")
    
    # 示例1: 生成样本CLIP向量并编码
    logging.info("示例1: 处理单个CLIP向量")
    clip_vector = generate_sample_clip_vector()
    logging.info(f"生成的CLIP向量形状: {clip_vector.shape}")
    
    # 初始化编码器
    encoder = CLIPVectorEncoder(vae_pth=vae_path, device=device)
    
    # 编码向量
    encoded_vector = encoder.encode(clip_vector)
    logging.info(f"编码后的向量形状: {encoded_vector.shape}")
    
    # 示例2: 批处理多个CLIP向量
    logging.info("\n示例2: 批处理多个CLIP向量")
    clip_vectors = [generate_sample_clip_vector() for _ in range(3)]
    encoded_vectors = encoder.encode(clip_vectors)
    logging.info(f"编码了 {len(encoded_vectors)} 个向量")
    
    # 示例3: 与完整VAE对比
    logging.info("\n示例3: 与完整VAE编码器对比")
    try:
        # 加载完整VAE
        full_vae = WanVAE(z_dim=16, vae_pth=vae_path, device=device)
        
        # 使用完整VAE编码
        sample_vector = clip_vector.unsqueeze(0).to(device)  # 添加批次维度
        vae_encoded = full_vae.encode([sample_vector.to(device)])[0]
        
        # 使用CLIPVectorEncoder编码
        clip_encoded = encoder.encode(sample_vector)
        
        # 比较结果
        diff = torch.mean(torch.abs(vae_encoded - clip_encoded)).item()
        logging.info(f"VAE和CLIPVectorEncoder结果平均差异: {diff}")
        
        if diff < 1e-5:
            logging.info("两种方法的编码结果基本一致，证明CLIPVectorEncoder正确实现")
        else:
            logging.warning("两种方法的编码结果存在差异，请检查实现")
    except Exception as e:
        logging.error(f"加载完整VAE时出错: {e}")
    
    # 示例4: 保存和加载编码结果
    logging.info("\n示例4: 保存和加载编码结果")
    output_path = "encoded_clip_vector.pt"
    
    # 保存编码结果
    torch.save(encoded_vector, output_path)
    logging.info(f"编码结果已保存到: {output_path}")
    
    # 加载编码结果
    loaded_vector = torch.load(output_path)
    logging.info(f"加载的编码向量形状: {loaded_vector.shape}")
    
    # 清理
    if os.path.exists(output_path):
        os.remove(output_path)
        logging.info(f"临时文件已删除: {output_path}")

if __name__ == "__main__":
    main()
