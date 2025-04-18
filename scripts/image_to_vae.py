#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一脚本：从图像提取CLIP特征，然后通过VAE进行编码
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
import time
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wan.modules.vae import Encoder3d, WanVAE, count_conv3d
from scripts.encode_clip_vectors import CLIPVectorEncoder, reshape_clip_vector

def parse_args():
    parser = argparse.ArgumentParser(description="图像到VAE编码的统一处理流程")
    parser.add_argument("--image_path", type=str, required=True,
                        help="输入图像文件路径或包含图像的目录")
    parser.add_argument("--vae_path", type=str, default="Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                        help="VAE预训练模型路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--z_dim", type=int, default=16,
                        help="潜在空间维度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--input_dim", type=str, default="3,1,32,32",
                        help="指定输入维度，格式为'C,T,H,W'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="处理批次大小")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14", 
                        help="CLIP模型名称")
    parser.add_argument("--image_size", type=int, default=224,
                        help="调整图像尺寸")
    parser.add_argument("--save_clip_features", action="store_true",
                        help="是否保存中间CLIP特征")
    parser.add_argument("--clip_batch_size", type=int, default=32,
                        help="CLIP批处理大小")
    return parser.parse_args()

def load_clip_model(model_name, device):
    """
    加载CLIP模型
    
    参数:
        model_name: CLIP模型名称
        device: 计算设备
        
    返回:
        model: CLIP模型
        preprocess: 预处理函数
    """
    try:
        import clip
    except ImportError:
        raise ImportError("请安装CLIP: pip install git+https://github.com/openai/CLIP.git")
    
    model, preprocess = clip.load(model_name, device=device)
    model.eval()  # 设置为评估模式
    return model, preprocess

def process_images_with_clip(model, preprocess, image_paths, device, batch_size=32):
    """
    使用CLIP处理图像并提取特征
    
    参数:
        model: CLIP模型
        preprocess: 预处理函数
        image_paths: 图像路径列表
        device: 计算设备
        batch_size: 批处理大小
        
    返回:
        features: 提取的CLIP特征
    """
    all_features = []
    
    # 分批处理图像
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        
        # 加载并预处理图像
        for img_path in batch_paths:
            try:
                image = preprocess(Image.open(img_path).convert("RGB"))
                images.append(image)
            except Exception as e:
                logging.error(f"处理图像 {img_path} 时出错: {e}")
                continue
        
        if not images:
            continue
            
        # 将图像堆叠为批次
        batch = torch.stack(images).to(device)
        
        # 提取特征
        with torch.no_grad():
            features = model.encode_image(batch)
            # 归一化特征
            features = F.normalize(features, dim=-1)
            all_features.append(features.cpu())
    
    # 合并所有批次的特征
    if all_features:
        return torch.cat(all_features, dim=0)
    else:
        return None

def gather_image_paths(image_path):
    """
    收集所有图像文件路径
    
    参数:
        image_path: 图像文件路径或目录
        
    返回:
        image_paths: 图像路径列表
    """
    image_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    if os.path.isdir(image_path):
        # 如果是目录，收集所有图像文件
        for root, _, files in os.walk(image_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
    elif os.path.isfile(image_path):
        # 如果是文件，检查是否为图像文件
        if any(image_path.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(image_path)
    
    return sorted(image_paths)

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入路径
    if not os.path.exists(args.image_path):
        logging.error(f"输入路径不存在: {args.image_path}")
        return
    
    # 检查VAE模型文件
    if not os.path.exists(args.vae_path):
        logging.error(f"VAE模型文件不存在: {args.vae_path}")
        return
    
    # 收集图像文件路径
    image_paths = gather_image_paths(args.image_path)
    if not image_paths:
        logging.error(f"没有找到有效的图像文件")
        return
    
    logging.info(f"找到 {len(image_paths)} 个图像文件")
    
    # 加载CLIP模型
    logging.info(f"加载CLIP模型: {args.clip_model}")
    try:
        clip_model, preprocess = load_clip_model(args.clip_model, args.device)
    except Exception as e:
        logging.error(f"加载CLIP模型时出错: {e}")
        return
    
    # 使用CLIP处理图像
    logging.info("使用CLIP提取图像特征...")
    clip_features = process_images_with_clip(clip_model, preprocess, image_paths, args.device, args.clip_batch_size)
    
    if clip_features is None:
        logging.error("无法提取CLIP特征")
        return
        
    logging.info(f"CLIP特征形状: {clip_features.shape}")
    
    # 保存CLIP特征（如果需要）
    if args.save_clip_features:
        clip_features_path = args.output.replace(".pt", "_clip_features.pt") if args.output else "clip_features.pt"
        torch.save(clip_features, clip_features_path)
        logging.info(f"CLIP特征已保存到: {clip_features_path}")
    
    # 重塑CLIP特征以适应VAE编码器
    try:
        reshaped_features = reshape_clip_vector(clip_features, args.input_dim, args.batch_size)
        logging.info(f"重塑后的特征形状: {reshaped_features.shape}")
    except Exception as e:
        logging.error(f"重塑特征时出错: {e}")
        return
    
    # 初始化VAE编码器
    try:
        encoder = CLIPVectorEncoder(
            vae_pth=args.vae_path,
            z_dim=args.z_dim,
            device=args.device
        )
    except Exception as e:
        logging.error(f"初始化VAE编码器时出错: {e}")
        return
    
    # 使用VAE编码特征
    logging.info("使用VAE编码特征...")
    try:
        encoded_features = encoder.encode(reshaped_features)
        logging.info(f"编码后的特征形状: {encoded_features.shape}")
    except Exception as e:
        logging.error(f"编码特征时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 确定输出文件路径
    if args.output is None:
        if os.path.isdir(args.image_path):
            output_dir = args.image_path
            output_filename = "vae_encoded_features.pt"
        else:
            output_dir = os.path.dirname(args.image_path)
            output_filename = f"{os.path.basename(args.image_path).split('.')[0]}_vae_encoded.pt"
        args.output = os.path.join(output_dir, output_filename)
    
    # 保存编码后的特征
    try:
        torch.save({
            'features': encoded_features,
            'image_paths': image_paths,
            'clip_model': args.clip_model,
            'vae_path': args.vae_path,
            'z_dim': args.z_dim,
            'input_dim': args.input_dim
        }, args.output)
        logging.info(f"VAE编码特征已保存到: {args.output}")
    except Exception as e:
        logging.error(f"保存编码特征时出错: {e}")
        return
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
