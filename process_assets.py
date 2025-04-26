#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量处理图片目录，使用encode_image.py将它们转换为VAE编码格式
"""

import os
import argparse
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
import glob

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# 支持的图片扩展名
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def parse_args():
    parser = argparse.ArgumentParser(description="批量处理图片目录")
    parser.add_argument("--input_dir", type=str, default=".", help="输入图片目录，默认为当前目录")
    parser.add_argument("--output_dir", type=str, default="vae_encoded_output", help="输出目录")
    parser.add_argument("--recursive", action="store_true", help="是否递归处理子目录")
    parser.add_argument("--max_workers", type=int, default=32, help="最大并发工作线程数")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--max_area", type=int, default=720*1280, help="最大像素面积")
    parser.add_argument("--frame_num", type=int, default=81, help="视频帧数")
    
    # 模型路径参数
    parser.add_argument("--vae_path", type=str, default="Wan2.1-I2V-14B/Wan2.1_VAE.pth", help="VAE预训练模型路径")
    parser.add_argument("--clip_path", type=str, default="Wan2.1-I2V-14B/clip_vit_h_14.pth", help="CLIP预训练模型路径")
    parser.add_argument("--tokenizer_path", type=str, default="Wan2.1-I2V-14B/clip_tokenizer", help="CLIP分词器路径")
    parser.add_argument("--z_dim", type=int, default=16, help="潜在空间维度")
    parser.add_argument("--input_dim", type=str, default="5,1,16,16", help="指定输入维度，格式为'C,T,H,W'")
    parser.add_argument("--feature_mode", type=str, default="auto", 
                        choices=["auto", "pad", "project", "reshape", "pca"],
                        help="CLIP特征调整方法")
    
    # 直接处理文件的选项
    parser.add_argument("--file_pattern", type=str, default=None, help="图片文件模式，例如'*.jpg'")
    
    args = parser.parse_args()
    return args

def get_all_images(input_dir, recursive=False, file_pattern=None):
    """获取目录中所有图片文件的路径"""
    images = []
    
    # 如果提供了文件模式，直接使用glob
    if file_pattern:
        if os.path.isdir(input_dir):
            pattern = os.path.join(input_dir, file_pattern)
            images = glob.glob(pattern, recursive=recursive)
            logging.info(f"使用模式 '{pattern}' 找到 {len(images)} 个文件")
        else:
            logging.error(f"目录不存在: {input_dir}")
        return sorted(images)
    
    # 常规目录处理
    input_path = Path(input_dir)
    if not input_path.exists():
        logging.error(f"目录不存在: {input_dir}")
        return images
        
    # 如果是递归模式，使用glob查找所有图片
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            images.extend([str(p) for p in input_path.glob(f"**/*{ext}")])
            images.extend([str(p) for p in input_path.glob(f"**/*{ext.upper()}")])
    else:
        # 非递归模式，只查找当前目录
        for ext in IMAGE_EXTENSIONS:
            images.extend([str(p) for p in input_path.glob(f"*{ext}")])
            images.extend([str(p) for p in input_path.glob(f"*{ext.upper()}")])
    
    return sorted(images)

def process_image(args_tuple):
    """处理单个图片"""
    image_path, output_dir, cmd_args = args_tuple
    
    # 确定输出文件名
    image_rel_path = os.path.relpath(image_path, os.path.abspath('.'))
    output_name = image_rel_path.replace('/', '_').replace('\\', '_')
    output_name = os.path.splitext(output_name)[0] + '.pt'
    output_path = os.path.join(output_dir, output_name)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 如果输出文件已存在，则跳过
    if os.path.exists(output_path):
        return f"跳过已处理文件: {image_path}"
    
    # 构建命令
    cmd = [
        "python", "encode_image.py",
        "--image", image_path,
        "--output", output_path,
        "--device", cmd_args.device,
        "--max_area", str(cmd_args.max_area),
        "--frame_num", str(cmd_args.frame_num),
        "--vae_path", cmd_args.vae_path,
        "--clip_path", cmd_args.clip_path,
        "--tokenizer_path", cmd_args.tokenizer_path,
        "--z_dim", str(cmd_args.z_dim),
        "--input_dim", cmd_args.input_dim,
        "--feature_mode", cmd_args.feature_mode
    ]
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"成功处理: {image_path} -> {output_path}"
    except subprocess.CalledProcessError as e:
        return f"处理失败: {image_path}, 错误: {e.stderr}"

def main():
    args = parse_args()
    
    # 检查encode_image.py是否存在
    if not os.path.exists("encode_image.py"):
        logging.error("encode_image.py不存在，请确保脚本在正确的目录中运行")
        return
    
    # 获取所有图片
    logging.info(f"开始搜索图片，{'递归' if args.recursive else '不递归'}查找子目录...")
    images = get_all_images(args.input_dir, args.recursive, args.file_pattern)
    logging.info(f"找到 {len(images)} 张图片")
    
    if not images:
        logging.warning(f"在 {args.input_dir} 没有找到图片")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备处理参数
    process_args = [
        (img, args.output_dir, args) for img in images
    ]
    
    # 并行处理图片
    logging.info(f"开始处理图片，使用 {args.max_workers} 个工作线程，设备: {args.device}")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(tqdm(
            executor.map(process_image, process_args),
            total=len(images),
            desc="处理图片"
        ))
    
    # 统计结果
    success_count = sum(1 for r in results if r.startswith("成功处理"))
    skip_count = sum(1 for r in results if r.startswith("跳过已处理"))
    fail_count = sum(1 for r in results if r.startswith("处理失败"))
    
    logging.info(f"处理完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {fail_count}")
    
    # 打印失败的文件
    if fail_count > 0:
        logging.warning("以下文件处理失败:")
        for r in results:
            if r.startswith("处理失败"):
                logging.warning(r)

if __name__ == "__main__":
    main()