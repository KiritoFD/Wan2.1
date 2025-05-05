#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量风格转换训练脚本

自动对pt文件夹内的所有.pt文件两两配对进行训练，并以两个文件名字命名保存结果
"""
import torch
import argparse
import itertools
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="批量训练风格转换模型")
    parser.add_argument("--pt_folder", type=str, required=True, help="包含所有.pt特征文件的文件夹路径")
    parser.add_argument("--output_base", type=str, default="models/batch_styles", help="输出目录的基础路径")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="每对样式的训练轮数")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在训练结果的样式对")
    parser.add_argument("--skip_same_prefix", action="store_true", help="跳过具有相同前缀的文件对")
    parser.add_argument("--prefix_delimiter", type=str, default="_", help="提取前缀时使用的分隔符")
    parser.add_argument("--mixed_precision", action="store_true", help="使用混合精度训练")
    return parser.parse_args()

def get_file_prefix(filename, delimiter="_"):
    """从文件名提取前缀，用于对样式进行分组"""
    base_name = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext.split(delimiter)[0]

def main():
    args = parse_args()
    
    # 创建主输出目录
    os.makedirs(args.output_base, exist_ok=True)
    
    # 设置日志文件
    log_file = os.path.join(args.output_base, f"batch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"开始批量训练，参数: {args}")
    logging.info(f"日志将保存到: {log_file}")
    
    # 检查文件夹是否存在
    pt_folder = Path(args.pt_folder)
    if not pt_folder.exists() or not pt_folder.is_dir():
        logging.error(f"文件夹不存在或不是一个目录: {args.pt_folder}")
        return
    
    # 获取所有.pt文件
    pt_files = list(pt_folder.glob("*.pt"))
    if not pt_files:
        logging.error(f"在 {args.pt_folder} 中未找到.pt文件")
        return
    
    logging.info(f"找到 {len(pt_files)} 个.pt文件")
    
    # 生成所有可能的文件对组合
    file_pairs = list(itertools.combinations(pt_files, 2))
    logging.info(f"生成了 {len(file_pairs)} 对文件组合")
    
    # 如果需要，跳过具有相同前缀的文件对
    if args.skip_same_prefix:
        filtered_pairs = []
        for pair in file_pairs:
            prefix_a = get_file_prefix(pair[0], args.prefix_delimiter)
            prefix_b = get_file_prefix(pair[1], args.prefix_delimiter)
            if prefix_a != prefix_b:
                filtered_pairs.append(pair)
        
        logging.info(f"跳过相同前缀后剩余 {len(filtered_pairs)} 对文件组合")
        file_pairs = filtered_pairs
    
    # 获取当前脚本目录，确保我们能找到训练脚本
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(current_dir, "train_wgan_gp.py")
    
    # 验证训练脚本是否存在
    if not os.path.exists(train_script_path):
        logging.error(f"训练脚本不存在: {train_script_path}")
        return
    
    logging.info(f"使用训练脚本: {train_script_path}")
    
    # 设置最小样本量要求
    MIN_SAMPLES = 20  # 最小样本数，避免除零错误
    
    # 预先检查每个文件的样本量
    file_sample_counts = {}
    for file_path in pt_files:
        try:
            data = torch.load(file_path, map_location='cpu')
            if 'features' in data and isinstance(data['features'], torch.Tensor):
                file_sample_counts[file_path] = data['features'].size(0)
            else:
                logging.warning(f"文件不包含有效特征: {file_path}")
                file_sample_counts[file_path] = 0
        except Exception as e:
            logging.error(f"无法加载文件 {file_path}: {e}")
            file_sample_counts[file_path] = 0
    
    # 对每一对文件执行训练
    total_pairs = len(file_pairs)
    skipped_count = 0
    processed_count = 0
    
    for idx, (file_a, file_b) in enumerate(file_pairs):
        # 从文件路径中获取文件名（不含扩展名）
        name_a = os.path.splitext(file_a.name)[0]
        name_b = os.path.splitext(file_b.name)[0]
        
        # 检查样本量
        samples_a = file_sample_counts.get(file_a, 0)
        samples_b = file_sample_counts.get(file_b, 0)
        
        if samples_a < MIN_SAMPLES or samples_b < MIN_SAMPLES:
            logging.warning(f"[{idx+1}/{total_pairs}] 跳过样本量不足的文件对: {name_a}({samples_a}) -> {name_b}({samples_b}), 最小要求: {MIN_SAMPLES}")
            skipped_count += 1
            continue
        
        # 创建目录名时替换文件名中的空格为下划线
        safe_name_a = name_a.replace(" ", "_")
        safe_name_b = name_b.replace(" ", "_")
        output_dir = os.path.join(args.output_base, f"{safe_name_a}_to_{safe_name_b}")
        
        # 如果已存在且设置了跳过，则跳过此对
        if args.skip_existing and os.path.exists(output_dir) and os.path.exists(os.path.join(output_dir, "best_model.pth")):
            logging.info(f"[{idx+1}/{total_pairs}] 跳过已存在的结果: {name_a} -> {name_b}")
            skipped_count += 1
            continue
        
        logging.info(f"[{idx+1}/{total_pairs}] 开始训练: {name_a}({samples_a}) -> {name_b}({samples_b})")
        logging.info(f"输出目录: {output_dir}")
        processed_count += 1
        
        # 构建train_wgan_gp.py的命令参数，使用绝对路径
        cmd = [
            sys.executable,  # 使用当前Python解释器
            train_script_path,  # 使用完整路径
            f"--style_a={file_a}",
            f"--style_b={file_b}",
            f"--output_dir={output_dir}",
            f"--batch_size={min(args.batch_size, samples_a, samples_b)}",  # 确保批次大小不超过样本量
            f"--epochs={args.epochs}",
            f"--device={args.device}",
            f"--seed={args.seed}"
        ]
        
        # 添加混合精度参数（如果启用）
        if args.mixed_precision:
            cmd.append("--mixed_precision")
        
        # 执行训练命令
        try:
            logging.info(f"执行命令: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                errors='replace'  # 添加错误处理，防止解码错误
            )
            
            # 实时输出训练进程的日志，增加错误处理
            try:
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        logging.info(f"[{name_a}->{name_b}] {line}")
            except Exception as e:
                logging.error(f"读取进程输出时出错: {e}")
            
            # 等待进程完成
            return_code = process.wait()
            
            if return_code == 0:
                logging.info(f"[{idx+1}/{total_pairs}] 训练完成: {name_a} -> {name_b}")
            else:
                logging.error(f"[{idx+1}/{total_pairs}] 训练失败: {name_a} -> {name_b}, 返回码: {return_code}")
        
        except Exception as e:
            logging.error(f"[{idx+1}/{total_pairs}] 执行训练时出错: {e}")
    
    logging.info(f"批量训练完成! 总任务: {total_pairs}, 已处理: {processed_count}, 跳过: {skipped_count}")

if __name__ == "__main__":
    main()
