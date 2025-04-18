#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE潜在空间表示的实用工具
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wan.modules.vae import WanVAE
from scripts.encode_clip_vectors import CLIPVectorEncoder

class VAELatentUtils:
    """
    VAE潜在空间表示的实用工具类
    
    提供用于可视化、分析和操作VAE潜在空间表示的实用功能。
    """
    
    @staticmethod
    def visualize_latent_space(latent_vectors, output_path=None, pca_components=2):
        """
        可视化潜在空间表示
        
        通过PCA降维，将高维潜在空间表示可视化为2D或3D散点图
        
        参数:
            latent_vectors: 潜在空间表示列表或批次 [B,C,T,H,W] 或 [C,T,H,W]
            output_path: 输出图像路径
            pca_components: PCA降维后的组件数(2或3)
        """
        # 检查输入
        if isinstance(latent_vectors, list):
            # 转换为批次格式
            latent_vectors = torch.stack(latent_vectors)
        
        # 确保是张量
        if not isinstance(latent_vectors, torch.Tensor):
            latent_vectors = torch.tensor(latent_vectors)
        
        # 移至CPU
        latent_vectors = latent_vectors.detach().cpu()
        
        # 获取维度信息
        if len(latent_vectors.shape) == 5:  # [B,C,T,H,W]
            b, c, t, h, w = latent_vectors.shape
            # 重塑为 [B*T, C*H*W]
            features = latent_vectors.permute(0, 2, 1, 3, 4).reshape(b * t, c * h * w)
        elif len(latent_vectors.shape) == 4:  # [C,T,H,W]
            c, t, h, w = latent_vectors.shape
            # 重塑为 [T, C*H*W]
            features = latent_vectors.permute(1, 0, 2, 3).reshape(t, c * h * w)
        else:
            raise ValueError(f"不支持的潜在向量形状: {latent_vectors.shape}")
        
        # 转换为numpy数组
        features = features.numpy()
        
        # 应用PCA降维
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        reduced_features = pca.fit_transform(features)
        
        # 绘制图像
        plt.figure(figsize=(10, 8))
        if pca_components == 2:
            plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7)
            plt.xlabel('主成分1')
            plt.ylabel('主成分2')
        elif pca_components == 3:
            ax = plt.figure().add_subplot(projection='3d')
            ax.scatter(
                reduced_features[:, 0],
                reduced_features[:, 1],
                reduced_features[:, 2],
                alpha=0.7
            )
            ax.set_xlabel('主成分1')
            ax.set_ylabel('主成分2')
            ax.set_zlabel('主成分3')
        
        plt.title('VAE潜在空间可视化')
        plt.grid(True)
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"潜在空间可视化已保存到: {output_path}")
        else:
            plt.show()
    
    @staticmethod
    def interpolate_latents(start_latent, end_latent, steps=10, method='linear'):
        """
        在两个潜在空间表示之间进行插值
        
        参数:
            start_latent: 起始潜在表示
            end_latent: 结束潜在表示
            steps: 插值步数
            method: 插值方法 ('linear', 'spherical')
            
        返回:
            插值后的潜在表示列表
        """
        # 确保是张量
        if not isinstance(start_latent, torch.Tensor):
            start_latent = torch.tensor(start_latent)
        if not isinstance(end_latent, torch.Tensor):
            end_latent = torch.tensor(end_latent)
            
        # 确保形状一致
        assert start_latent.shape == end_latent.shape, "起始和结束潜在表示形状不一致"
        
        # 生成插值权重
        weights = torch.linspace(0, 1, steps, device=start_latent.device)
        
        results = []
        if method == 'linear':
            # 线性插值
            for w in weights:
                interpolated = start_latent * (1 - w) + end_latent * w
                results.append(interpolated)
        elif method == 'spherical':
            # 球面插值 (保持范数)
            start_norm = torch.norm(start_latent)
            end_norm = torch.norm(end_latent)
            
            # 归一化
            start_normalized = start_latent / start_norm
            end_normalized = end_latent / end_norm
            
            # 计算角度
            dot_product = torch.sum(start_normalized * end_normalized)
            angle = torch.acos(torch.clamp(dot_product, -1.0, 1.0))
            
            for w in weights:
                # 球面插值公式
                interp = (torch.sin((1 - w) * angle) * start_normalized + 
                          torch.sin(w * angle) * end_normalized) / torch.sin(angle)
                
                # 重新应用范数
                norm_interp = start_norm * (1 - w) + end_norm * w
                interpolated = interp * norm_interp
                results.append(interpolated)
        else:
            raise ValueError(f"不支持的插值方法: {method}")
            
        return results
    
    @staticmethod
    def compare_encoded_vectors(vae_latent, clip_encoded_latent, output_path=None):
        """
        比较VAE编码和通过CLIP编码器编码的向量
        
        参数:
            vae_latent: 使用VAE直接编码的潜在表示
            clip_encoded_latent: 使用CLIPVectorEncoder编码的潜在表示
            output_path: 输出图像路径
        """
        # 确保是张量并移至CPU
        vae_latent = vae_latent.detach().cpu()
        clip_encoded_latent = clip_encoded_latent.detach().cpu()
        
        # 确保维度匹配
        if vae_latent.shape != clip_encoded_latent.shape:
            logging.warning(f"潜在表示形状不匹配: VAE={vae_latent.shape}, CLIP编码={clip_encoded_latent.shape}")
            return
            
        # 计算差异统计
        diff = vae_latent - clip_encoded_latent
        abs_diff = torch.abs(diff)
        
        mean_diff = torch.mean(abs_diff).item()
        max_diff = torch.max(abs_diff).item()
        std_diff = torch.std(diff).item()
        
        # 可视化差异
        plt.figure(figsize=(15, 10))
        
        # 转换为可绘制格式
        if len(vae_latent.shape) == 5:  # [B,C,T,H,W]
            b, c, t, h, w = vae_latent.shape
            vae_plot = vae_latent[0].mean(dim=(1, 2)).flatten().numpy()
            clip_plot = clip_encoded_latent[0].mean(dim=(1, 2)).flatten().numpy()
            diff_plot = diff[0].mean(dim=(1, 2)).flatten().numpy()
        elif len(vae_latent.shape) == 4:  # [C,T,H,W]
            vae_plot = vae_latent.mean(dim=(1, 2)).flatten().numpy()
            clip_plot = clip_encoded_latent.mean(dim=(1, 2)).flatten().numpy()
            diff_plot = diff.mean(dim=(1, 2)).flatten().numpy()
        
        # 绘制比较图
        plt.subplot(3, 1, 1)
        plt.bar(range(len(vae_plot)), vae_plot)
        plt.title('VAE编码潜在表示')
        
        plt.subplot(3, 1, 2)
        plt.bar(range(len(clip_plot)), clip_plot)
        plt.title('CLIP编码潜在表示')
        
        plt.subplot(3, 1, 3)
        plt.bar(range(len(diff_plot)), diff_plot)
        plt.title(f'差异 (平均={mean_diff:.4f}, 最大={max_diff:.4f}, 标准差={std_diff:.4f})')
        
        plt.tight_layout()
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"比较结果已保存到: {output_path}")
        else:
            plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="VAE潜在空间表示工具")
    parser.add_argument("--operation", type=str, required=True, 
                        choices=["visualize", "interpolate", "compare"],
                        help="要执行的操作")
    parser.add_argument("--input", type=str, required=True,
                        help="输入文件路径(.npy或.pt格式)")
    parser.add_argument("--input2", type=str, default=None,
                        help="插值或比较操作的第二个输入文件")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--steps", type=int, default=10,
                        help="插值步数")
    parser.add_argument("--method", type=str, default="linear",
                        choices=["linear", "spherical"],
                        help="插值方法")
    return parser.parse_args()

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logging.error(f"输入文件不存在: {args.input}")
        return
    
    # 加载输入文件
    input_ext = os.path.splitext(args.input)[1].lower()
    try:
        if input_ext == '.npy':
            input_data = np.load(args.input)
            input_data = torch.from_numpy(input_data)
        elif input_ext == '.pt':
            input_data = torch.load(args.input)
        else:
            logging.error(f"不支持的文件格式: {input_ext}，请提供.npy或.pt格式")
            return
    except Exception as e:
        logging.error(f"加载输入文件时出错: {e}")
        return
    
    # 执行选定的操作
    if args.operation == "visualize":
        VAELatentUtils.visualize_latent_space(input_data, args.output)
    
    elif args.operation == "interpolate":
        # 检查第二个输入文件
        if not args.input2:
            logging.error("插值操作需要提供第二个输入文件")
            return
            
        if not os.path.exists(args.input2):
            logging.error(f"第二个输入文件不存在: {args.input2}")
            return
        
        # 加载第二个输入文件
        input2_ext = os.path.splitext(args.input2)[1].lower()
        try:
            if input2_ext == '.npy':
                input2_data = np.load(args.input2)
                input2_data = torch.from_numpy(input2_data)
            elif input2_ext == '.pt':
                input2_data = torch.load(args.input2)
            else:
                logging.error(f"不支持的文件格式: {input2_ext}，请提供.npy或.pt格式")
                return
        except Exception as e:
            logging.error(f"加载第二个输入文件时出错: {e}")
            return
            
        # 执行插值
        interpolated = VAELatentUtils.interpolate_latents(
            input_data, 
            input2_data, 
            steps=args.steps, 
            method=args.method
        )
        
        # 保存结果
        if args.output:
            torch.save(interpolated, args.output)
            logging.info(f"插值结果已保存到: {args.output}")
    
    elif args.operation == "compare":
        # 检查第二个输入文件
        if not args.input2:
            logging.error("比较操作需要提供第二个输入文件")
            return
            
        if not os.path.exists(args.input2):
            logging.error(f"第二个输入文件不存在: {args.input2}")
            return
        
        # 加载第二个输入文件
        input2_ext = os.path.splitext(args.input2)[1].lower()
        try:
            if input2_ext == '.npy':
                input2_data = np.load(args.input2)
                input2_data = torch.from_numpy(input2_data)
            elif input2_ext == '.pt':
                input2_data = torch.load(args.input2)
            else:
                logging.error(f"不支持的文件格式: {input2_ext}，请提供.npy或.pt格式")
                return
        except Exception as e:
            logging.error(f"加载第二个输入文件时出错: {e}")
            return
            
        # 执行比较
        VAELatentUtils.compare_encoded_vectors(
            input_data,
            input2_data,
            args.output
        )

if __name__ == "__main__":
    main()
