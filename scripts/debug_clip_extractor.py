#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试脚本: 专门用于测试和调试CLIP特征提取功能
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
import warnings
from tqdm import tqdm
import traceback
import gc

# 忽略PIL警告
warnings.filterwarnings("ignore", category=UserWarning, module='PIL.TiffImagePlugin')

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wan.modules.clip import CLIPModel

def parse_args():
    parser = argparse.ArgumentParser(description="调试CLIP特征提取")
    parser.add_argument("image_path", type=str, help="输入图像文件路径或包含图像的目录")
    parser.add_argument("--clip_path", type=str, default="Wan2.1-T2V-14B/clip_vit_h_14.pth",
                        help="CLIP预训练模型路径")
    parser.add_argument("--tokenizer_path", type=str, default="Wan2.1-T2V-14B/xlm_roberta_tokenizer",
                        help="CLIP分词器路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小，建议从1开始调试")
    parser.add_argument("--max_images", type=int, default=10, help="测试的最大图像数量")
    parser.add_argument("--input_size", type=str, default=None, help="调整输入图像大小，格式为'H,W'")
    
    return parser.parse_args()

def gather_image_paths(image_path, max_images=None):
    """收集所有图像文件路径"""
    image_paths = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    if os.path.isdir(image_path):
        # 如果是目录，收集所有图像文件
        for root, _, files in os.walk(image_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
                    if max_images and len(image_paths) >= max_images:
                        break
            if max_images and len(image_paths) >= max_images:
                break
    elif os.path.isfile(image_path):
        # 如果是文件，检查是否为图像文件
        if any(image_path.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(image_path)
    
    return sorted(image_paths)[:max_images] if max_images else sorted(image_paths)

def debug_clip_extraction(args):
    """调试CLIP特征提取"""
    # 配置日志
    logging.basicConfig(level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 收集图像路径
    logging.info(f"收集图像路径从: {args.image_path}")
    image_paths = gather_image_paths(args.image_path, args.max_images)
    
    if not image_paths:
        logging.error("未找到任何图像文件")
        return
    
    logging.info(f"找到 {len(image_paths)} 个图像")
    
    # 解析输入尺寸
    input_size = None
    if args.input_size:
        try:
            h, w = map(int, args.input_size.split(','))
            input_size = (w, h)  # PIL使用(width, height)格式
            logging.info(f"将调整图像尺寸为: {input_size}")
        except:
            logging.warning(f"无效的输入尺寸格式: {args.input_size}, 将使用原始尺寸")
    
    # 加载CLIP模型
    try:
        logging.info(f"尝试加载CLIP模型: {args.clip_path}")
        logging.info(f"目标设备: {args.device}")
        
        # 创建Python对象的可视化详细日志函数
        def describe_object(obj, name="对象", max_depth=2, current_depth=0):
            indent = "  " * current_depth
            if current_depth >= max_depth:
                return f"{indent}{name}: [最大递归深度]"
            
            output = []
            output.append(f"{indent}{name} ({type(obj).__name__}):")
            
            # 如果是torch.nn.Module，打印相关信息
            if isinstance(obj, torch.nn.Module):
                try:
                    output.append(f"{indent}  - 参数数量: {sum(p.numel() for p in obj.parameters())}")
                    output.append(f"{indent}  - 训练模式: {obj.training}")
                    output.append(f"{indent}  - 设备: {next(obj.parameters()).device}")
                except Exception as e:
                    output.append(f"{indent}  - 无法获取模型信息: {str(e)}")
            
            # 如果是字典或类似对象，递归打印属性
            if hasattr(obj, "__dict__") and current_depth < max_depth:
                for k, v in obj.__dict__.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        output.append(f"{indent}  {k}: {v}")
                    else:
                        output.append(describe_object(v, k, max_depth, current_depth + 1))
            
            return "\n".join(output)
        
        # 加载CLIP模型
        clip_model = CLIPModel(
            dtype=torch.float16,
            device=args.device,
            checkpoint_path=args.clip_path,
            tokenizer_path=args.tokenizer_path
        )
        
        # 打印CLIP模型详细信息
        logging.info(f"CLIP模型成功加载，类型: {type(clip_model).__name__}")
        logging.info(describe_object(clip_model, "CLIP模型", max_depth=1))
        
        # 逐个处理图像以调试
        logging.info("开始逐个测试图像...")
        
        for i, img_path in enumerate(image_paths):
            logging.info(f"\n--- 测试图像 {i+1}/{len(image_paths)}: {img_path} ---")
            
            try:
                # 1. 加载图像
                logging.info(f"尝试加载图像: {img_path}")
                img = Image.open(img_path).convert("RGB")
                logging.info(f"成功加载图像，尺寸: {img.size}, 格式: {img.format}")
                
                # 2. 调整尺寸
                if input_size:
                    img = img.resize(input_size, Image.Resampling.LANCZOS)
                    logging.info(f"已调整图像尺寸为: {img.size}")
                
                # 3. 转换为张量
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2 - 1
                logging.info(f"图像张量形状: {img_tensor.shape}")
                
                # 检查张量值范围
                logging.info(f"张量值范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                has_nan = torch.isnan(img_tensor).any()
                has_inf = torch.isinf(img_tensor).any()
                logging.info(f"张量包含NaN: {has_nan}, 张量包含Inf: {has_inf}")
                
                # 4. 添加时间维度
                img_tensor = img_tensor.unsqueeze(1)
                logging.info(f"添加时间维度后张量形状: {img_tensor.shape}")
                
                # 5. 移动到设备
                img_tensor = img_tensor.to(args.device)
                logging.info(f"已将张量移动到设备: {args.device}")
                
                # 6. 进行CLIP特征提取
                with torch.no_grad():
                    # 时间性能测量
                    start_time = time.time()
                    features = clip_model.visual([img_tensor])
                    elapsed = time.time() - start_time
                    
                    logging.info(f"CLIP特征提取成功! 耗时: {elapsed:.3f}秒")
                    logging.info(f"特征形状: {features.shape}")
                    logging.info(f"特征值范围: [{features.min().item():.3f}, {features.max().item():.3f}]")
                    
                    # 移回CPU以释放GPU内存
                    features = features.cpu()
                    del img_tensor
                    torch.cuda.empty_cache()
                    
                # 成功完成一张图像的测试
                logging.info(f"图像 {img_path} 测试成功!\n")
                
            except Exception as e:
                logging.error(f"处理图像 {img_path} 失败: {str(e)}")
                logging.error(f"异常堆栈: {traceback.format_exc()}")
        
        # 测试批量处理
        if len(image_paths) >= args.batch_size and args.batch_size > 1:
            logging.info(f"\n--- 测试批量处理 (批次大小 {args.batch_size}) ---")
            
            try:
                batch_images = []
                batch_paths = image_paths[:args.batch_size]
                
                # 预处理所有图像
                for img_path in batch_paths:
                    img = Image.open(img_path).convert("RGB")
                    if input_size:
                        img = img.resize(input_size, Image.Resampling.LANCZOS)
                        
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor * 2 - 1
                    img_tensor = img_tensor.unsqueeze(1)  # 添加时间维度
                    batch_images.append(img_tensor)
                
                # 批量处理测试
                start_time = time.time()
                with torch.no_grad():
                    videos = [img.to(args.device) for img in batch_images]
                    features = clip_model.visual(videos)
                elapsed = time.time() - start_time
                
                logging.info(f"批量处理成功! 耗时: {elapsed:.3f}秒")
                logging.info(f"批量特征形状: {features.shape}")
                
                del videos, features, batch_images
                torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"批量处理测试失败: {str(e)}")
                logging.error(traceback.format_exc())
    
    except Exception as e:
        logging.error(f"加载CLIP模型失败: {str(e)}")
        logging.error(traceback.format_exc())
    
    logging.info("调试结束")

if __name__ == "__main__":
    args = parse_args()
    debug_clip_extraction(args)
