#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用WanVAE编码器处理CLIP提取的向量
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wan.modules.vae import Encoder3d, WanVAE, count_conv3d

class CLIPVectorEncoder:
    """
    只使用WanVAE的编码器部分处理CLIP向量的封装类
    
    该类提取并使用WanVAE的编码器组件，专门用于处理CLIP提取的向量，
    避免了加载和使用完整VAE模型的开销。
    """
    
    def __init__(
        self,
        vae_pth="Wan2.1-T2V-14B/Wan2.1_VAE.pth",
        z_dim=16,
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化CLIP向量编码器
        
        参数:
            vae_pth: VAE预训练模型路径
            z_dim: 潜在空间维度
            dtype: 计算精度类型
            device: 计算设备
        """
        self.dtype = dtype
        self.device = device
        self.z_dim = z_dim
        
        # 预定义的潜在空间归一化参数（与WanVAE中相同）
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        # 如果z_dim小于预定义参数长度，截取对应维度
        if z_dim < len(mean):
            mean = mean[:z_dim]
            std = std[:z_dim]
        # 转换为张量并移至指定设备
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]
        
        # 只加载编码器部分
        logging.info(f"加载VAE模型: {vae_pth}")
        self._load_encoder(vae_pth)
        logging.info(f"编码器已加载到设备: {device}")
    
    def _load_encoder(self, vae_pth):
        """
        只加载编码器部分，节省内存
        
        参数:
            vae_pth: 预训练VAE模型路径
        """
        # 配置编码器参数
        encoder_cfg = {
            "dim": 96,
            "z_dim": self.z_dim * 2,  # 编码器输出通道是z_dim*2，包含均值和方差
            "dim_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_scales": [],
            "temperal_downsample": [False, True, True],
            "dropout": 0.0
        }
        
        # 初始化编码器
        self.encoder = Encoder3d(**encoder_cfg).to(self.device).eval().requires_grad_(False)
        self.conv1 = torch.nn.Conv3d(self.z_dim * 2, self.z_dim * 2, 1).to(self.device)
        
        # 加载预训练权重
        full_state_dict = torch.load(vae_pth, map_location=self.device)
        
        # 筛选只属于编码器的权重
        encoder_state_dict = {}
        conv1_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith('encoder.'):
                encoder_state_dict[key.replace('encoder.', '')] = value
            elif key.startswith('conv1.'):
                conv1_state_dict[key.replace('conv1.', '')] = value
        
        # 加载权重到模型
        self.encoder.load_state_dict(encoder_state_dict)
        self.conv1.load_state_dict(conv1_state_dict)
        
        # 初始化特征缓存
        self.clear_cache()
    
    def clear_cache(self):
        """清除编码器特征缓存"""
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num
    
    def encode(self, clip_vectors, normalize=True):
        """
        编码CLIP向量到VAE潜在空间
        
        参数:
            clip_vectors: CLIP提取的向量，形状为[B,C,T,H,W]或列表
            normalize: 是否应用归一化
            
        返回:
            编码后的潜在表示
        """
        # 确保输入格式正确
        if isinstance(clip_vectors, list):
            # 如果输入是列表，处理每个元素
            results = []
            for vec in clip_vectors:
                if len(vec.shape) == 4:  # [C,T,H,W]
                    vec = vec.unsqueeze(0)  # 添加批次维度
                results.append(self._encode_single(vec, normalize).squeeze(0))
            return results
        else:
            # 如果输入已经是批量格式 [B,C,T,H,W]
            return self._encode_single(clip_vectors, normalize)
    
    def _encode_single(self, x, normalize=True):
        """
        处理单个输入批次
        
        参数:
            x: 输入张量 [B,C,T,H,W]
            normalize: 是否应用归一化
            
        返回:
            编码后的潜在表示
        """
        self.clear_cache()
        with torch.no_grad():
            # 使用新的 torch.amp.autocast 方式替代 torch.cuda.amp.autocast
            with torch.amp.autocast(device_type=self.device.split(":")[0], dtype=self.dtype):
                # 转换到目标设备和数据类型
                x = x.to(self.device, self.dtype)
                
                # 获取时间维度长度
                t = x.shape[2]
                iter_ = 1 + (t - 1) // 4
                
                # 分块处理视频帧
                for i in range(iter_):
                    self._enc_conv_idx = [0]
                    if i == 0:
                        # 首帧单独处理
                        out = self.encoder(
                            x[:, :, :1, :, :],
                            feat_cache=self._enc_feat_map,
                            feat_idx=self._enc_conv_idx
                        )
                    else:
                        # 后续帧分组处理，每组4帧
                        out_ = self.encoder(
                            x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                            feat_cache=self._enc_feat_map,
                            feat_idx=self._enc_conv_idx
                        )
                        out = torch.cat([out, out_], 2)
                
                # 分离均值和对数方差
                mu, log_var = self.conv1(out).chunk(2, dim=1)
                
                # 执行归一化操作
                if normalize:
                    if isinstance(self.scale[0], torch.Tensor):
                        mu = (mu - self.scale[0].view(1, self.z_dim, 1, 1, 1)) * self.scale[1].view(
                            1, self.z_dim, 1, 1, 1)
                    else:
                        mu = (mu - self.scale[0]) * self.scale[1]
                
                # 清理缓存
                self.clear_cache()
                
                return mu.float()

def parse_args():
    parser = argparse.ArgumentParser(description="使用WanVAE编码器处理CLIP向量")
    parser.add_argument("clip_vector", type=str, nargs="?", default=None,
                        help="CLIP向量文件路径(位置参数)")
    parser.add_argument("--vae_path", type=str, default="Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                        help="VAE预训练模型路径")
    parser.add_argument("--clip_vector", type=str, dest="clip_vector_arg", default=None,
                        help="CLIP向量文件路径(.npy, .pt 或 .npz格式)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--z_dim", type=int, default=16,
                        help="潜在空间维度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--reshape", action="store_true", default=True,
                        help="是否自动重塑CLIP向量形状以适应VAE编码器")
    parser.add_argument("--input_dim", type=str, default="3,1,32,32",
                        help="指定输入维度，例如 '3,1,32,32'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="处理批次大小")
    parser.add_argument("--no_reshape", dest="reshape", action="store_false",
                        help="禁用自动重塑CLIP向量")
    return parser.parse_args()

def load_clip_vector(file_path):
    """加载CLIP向量文件，支持.npy, .pt, .npz格式"""
    vector_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if vector_ext == '.npy':
            clip_vector = np.load(file_path)
            clip_vector = torch.from_numpy(clip_vector)
        elif vector_ext == '.pt':
            clip_vector = torch.load(file_path)
        elif vector_ext == '.npz':
            # 对于npz文件，尝试加载第一个数组
            loaded_data = np.load(file_path)
            if isinstance(loaded_data, np.lib.npyio.NpzFile):
                first_key = list(loaded_data.keys())[0]
                logging.info(f"从NPZ文件中加载键为 '{first_key}' 的数组")
                clip_vector = loaded_data[first_key]
            else:
                clip_vector = loaded_data
            clip_vector = torch.from_numpy(clip_vector)
        else:
            raise ValueError(f"不支持的文件格式: {vector_ext}，请提供.npy、.pt或.npz格式")
        
        return clip_vector
    
    except Exception as e:
        raise RuntimeError(f"加载CLIP向量时出错: {e}")

def reshape_clip_vector(clip_vector, input_dim=None, batch_size=1):
    """
    将CLIP向量重塑为VAE编码器期望的形状
    
    参数:
        clip_vector: CLIP向量
        input_dim: 目标维度字符串，格式为"C,T,H,W"
        batch_size: 批次大小
    
    返回:
        重塑后的向量
    """
    # 处理3D向量，例如 [N, L, C] 形状 (CLIP输出的序列特征)
    if len(clip_vector.shape) == 3:
        # 将序列维度平均池化，得到 [N, C] 形状
        logging.info(f"检测到3D CLIP向量形状 {clip_vector.shape}，进行平均池化")
        clip_vector = clip_vector.mean(dim=1)
        # 然后按2D向量处理
    
    # 处理2D向量，例如 [N, C] 形状
    if len(clip_vector.shape) == 2:
        n, c = clip_vector.shape
        
        if input_dim is not None:
            # 使用用户指定的维度
            dims = list(map(int, input_dim.split(',')))
            if len(dims) != 4:
                raise ValueError("输入维度必须是4个值，格式为'C,T,H,W'")
            c_out, t, h, w = dims
            
            # 验证总特征数是否匹配
            total_elements_needed = c_out * t * h * w
            
            # 如果总特征数不匹配，但每个样本的特征数是一致的
            if c * n == total_elements_needed:
                logging.info(f"将整个批次视为单个样本，重塑为 [1, {c_out}, {t}, {h}, {w}]")
                return clip_vector.reshape(1, c_out, t, h, w)
            
            # 如果指定的维度与每个样本的特征数不匹配
            if c != total_elements_needed:
                logging.warning(f"特征数量不匹配: {c} vs {total_elements_needed}，将尝试调整")
                
                # 计算每个样本所需的特征数
                features_per_sample = c
                
                # 尝试自动确定合理的维度，保持通道数为3(RGB)
                if c_out == 3:
                    # 尝试为t=1找到合适的h和w
                    spatial_dim = int(np.sqrt(features_per_sample / c_out))
                    if spatial_dim * spatial_dim * c_out == features_per_sample:
                        h = w = spatial_dim
                        t = 1
                        logging.info(f"自动调整为 [{c_out}, {t}, {h}, {w}]")
                    else:
                        # 尝试不同的配置
                        for test_dim in range(spatial_dim, 1, -1):
                            if features_per_sample % (c_out * test_dim * test_dim) == 0:
                                h = w = test_dim
                                t = features_per_sample // (c_out * h * w)
                                logging.info(f"自动调整为 [{c_out}, {t}, {h}, {w}]")
                                break
                else:
                    # 如果要求的通道数不是3，尝试调整空间维度
                    spatial_dim = int(np.sqrt(features_per_sample / c_out))
                    h = w = max(1, spatial_dim)
                    t = max(1, features_per_sample // (c_out * h * w))
                    logging.info(f"自动调整为 [{c_out}, {t}, {h}, {w}]")
        else:
            # 自动确定合理的维度
            # 默认为3通道(RGB)
            c_out = 3
            
            # 尝试确定合适的空间尺寸，使时间维度为1
            spatial_size = int(np.sqrt(c / c_out))
            if spatial_size * spatial_size * c_out == c:
                h = w = spatial_size
                t = 1
            else:
                # 尝试找到能整除的空间尺寸
                for test_size in range(spatial_size, 1, -1):
                    if c % (c_out * test_size * test_size) == 0:
                        h = w = test_size
                        t = c // (c_out * h * w)
                        break
                else:
                    # 如果找不到合适的维度，尝试调整通道数
                    for test_channels in [1, 4, 8, 16]:
                        spatial_size = int(np.sqrt(c / test_channels))
                        for test_size in range(spatial_size, 1, -1):
                            if c % (test_channels * test_size * test_size) == 0:
                                c_out = test_channels
                                h = w = test_size
                                t = c // (c_out * h * w)
                                break
                        else:
                            continue
                        break
                    else:
                        # 如果仍然找不到，使用默认值并进行填充或截断
                        c_out = 3
                        h = w = max(1, spatial_size)
                        t = 1
        
        # 确保有合理的形状
        total_elements_needed = c_out * t * h * w
        if total_elements_needed != c:
            logging.warning(f"特征数不匹配 ({c} vs {total_elements_needed})，将进行调整")
            # 填充或截断到目标大小
            if total_elements_needed > c:
                padding_needed = total_elements_needed - c
                logging.info(f"填充 {padding_needed} 个零以匹配目标形状")
                padded = torch.zeros((n, total_elements_needed), dtype=clip_vector.dtype)
                padded[:, :c] = clip_vector
                clip_vector = padded
            else:
                logging.info(f"截断到 {total_elements_needed} 个特征以匹配目标形状")
                clip_vector = clip_vector[:, :total_elements_needed]
            
        # 重塑为5D张量 [B, C, T, H, W]
        try:
            reshaped = clip_vector.reshape(n, c_out, t, h, w)
            logging.info(f"成功重塑为 {reshaped.shape}")
            return reshaped
        except RuntimeError as e:
            raise ValueError(f"重塑失败: {e}，尝试其他维度，比如 --input_dim '3,1,32,32'")
    
    # 处理已经是4D的向量 [C, T, H, W]
    elif len(clip_vector.shape) == 4:
        # 已经是符合要求的形状，添加批次维度
        return clip_vector.unsqueeze(0)
    
    # 处理已经是5D的向量 [B, C, T, H, W]
    elif len(clip_vector.shape) == 5:
        # 已经是符合要求的形状
        return clip_vector
    
    else:
        raise ValueError(f"不支持的CLIP向量形状: {clip_vector.shape}")

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析命令行参数
    args = parse_args()
    
    # 确定CLIP向量文件路径（优先使用位置参数）
    clip_vector_path = args.clip_vector if args.clip_vector else args.clip_vector_arg
    
    if not clip_vector_path:
        logging.error("必须提供CLIP向量文件路径")
        return
    
    # 检查VAE文件是否存在
    if not os.path.exists(args.vae_path):
        logging.error(f"VAE模型文件不存在: {args.vae_path}")
        return
    
    # 检查CLIP向量文件是否存在
    if not os.path.exists(clip_vector_path):
        logging.error(f"CLIP向量文件不存在: {clip_vector_path}")
        return
    
    # 加载CLIP向量
    try:
        clip_vector = load_clip_vector(clip_vector_path)
    except Exception as e:
        logging.error(str(e))
        return
    
    logging.info(f"原始CLIP向量形状: {clip_vector.shape}")
    
    # 重塑CLIP向量为VAE编码器期望的形状
    if args.reshape:
        try:
            clip_vector = reshape_clip_vector(clip_vector, args.input_dim, args.batch_size)
            logging.info(f"重塑后的CLIP向量形状: {clip_vector.shape}")
        except Exception as e:
            logging.error(f"重塑CLIP向量时出错: {e}")
            return
    
    # 确定输出文件路径
    if args.output is None:
        output_dir = os.path.dirname(clip_vector_path)
        filename = os.path.basename(clip_vector_path).split('.')[0]
        args.output = os.path.join(output_dir, f"{filename}_encoded.pt")
    
    # 初始化编码器
    try:
        encoder = CLIPVectorEncoder(
            vae_pth=args.vae_path,
            z_dim=args.z_dim,
            device=args.device
        )
    except Exception as e:
        logging.error(f"初始化编码器时出错: {e}")
        return
    
    # 编码向量
    try:
        encoded_vector = encoder.encode(clip_vector)
        logging.info(f"编码后的向量形状: {encoded_vector.shape}")
    except Exception as e:
        logging.error(f"编码向量时出错: {e}")
        logging.error(f"错误详情: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 保存编码后的向量
    try:
        torch.save(encoded_vector, args.output)
        logging.info(f"编码后的向量已保存到: {args.output}")
    except Exception as e:
        logging.error(f"保存编码后的向量时出错: {e}")
        return

if __name__ == "__main__":
    main()
