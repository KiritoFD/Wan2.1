#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一脚本：从图像提取CLIP特征，然后通过VAE进行编码
直接使用wan.modules.clip.py和wan.modules.vae.py中的组件
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
import torch.nn.functional as F

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wan.modules.vae import WanVAE, Encoder3d, count_conv3d
from wan.modules.clip import CLIPModel
from scripts.encode_clip_vectors import reshape_clip_vector

def parse_args():
    parser = argparse.ArgumentParser(description="图像到VAE编码的统一处理流程")
    parser.add_argument("image_path", type=str, nargs="?", default=None,
                        help="输入图像文件路径或包含图像的目录")
    parser.add_argument("--image_path", dest="image_path_arg", type=str, default=None,
                        help="输入图像文件路径或包含图像的目录")
    parser.add_argument("--vae_path", type=str, default="Wan2.1-T2V-14B/Wan2.1_VAE.pth",
                        help="VAE预训练模型路径")
    parser.add_argument("--clip_path", type=str, default="Wan2.1-T2V-14B/clip_vit_h_14.pth",
                        help="CLIP预训练模型路径")
    parser.add_argument("--tokenizer_path", type=str, default="Wan2.1-T2V-14B/xlm_roberta_tokenizer",
                        help="CLIP分词器路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--z_dim", type=int, default=16,
                        help="潜在空间维度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--input_dim", type=str, default=None,
                        help="指定输入维度，格式为'C,T,H,W'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="处理批次大小")
    parser.add_argument("--clip_batch_size", type=int, default=32,
                        help="CLIP批处理大小")
    parser.add_argument("--save_clip_features", action="store_true",
                        help="是否保存中间CLIP特征")
    parser.add_argument("--feature_mode", type=str, default="auto", 
                        choices=["auto", "pad", "project", "reshape", "pca"],
                        help="CLIP特征调整方法: auto(自动选择), pad(零填充), project(线性投影), reshape(直接重塑), pca(主成分分析)")
    parser.add_argument("--feature_dim", type=int, default=None,
                        help="调整后的特征维度，仅用于project和pca模式")
    args = parser.parse_args()
    
    # 优先使用位置参数中的image_path
    if args.image_path is None and args.image_path_arg is None:
        parser.error("必须提供图像路径")
    elif args.image_path is None:
        args.image_path = args.image_path_arg

    # 如果未指定input_dim，设置默认值为原始CLIP输出维度的合适重组
    if args.input_dim is None:
        if args.feature_mode == "reshape":
            args.input_dim = "5,1,16,16"  # 适合1280维CLIP输出
        else:
            args.input_dim = "3,1,32,32"  # 标准配置，确保通道数为3
    else:
        # 确保input_dim中的通道数为3，防止与VAE不兼容
        dims = list(map(int, args.input_dim.split(',')))
        if dims[0] != 3:
            logging.warning(f"将通道数从 {dims[0]} 修改为 3 以兼容VAE")
            dims[0] = 3
            args.input_dim = ','.join(map(str, dims))

    return args

def gather_image_paths(image_path):
    """收集所有图像文件路径"""
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

def process_images_with_clip(clip_model, image_paths, device, batch_size=32):
    """使用CLIP模型处理图像并提取特征"""
    all_features = []
    
    # 分批处理图像
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        
        # 加载图像
        for img_path in batch_paths:
            try:
                # 加载图像
                img = Image.open(img_path).convert("RGB")
                # 转换为张量
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                # 归一化到[-1, 1]
                img_tensor = img_tensor * 2 - 1
                images.append(img_tensor)
            except Exception as e:
                logging.error(f"处理图像 {img_path} 时出错: {e}")
                continue
        
        if not images:
            continue
        
        # 准备CLIP输入
        videos = [img.unsqueeze(1).to(device) for img in images]  # 添加时间维度
        
        # 提取CLIP特征
        with torch.no_grad():
            features = clip_model.visual(videos)
            # 处理CLIP特征以便兼容后续处理
            # 如果特征是三维的 [batch_size, seq_len, embedding_dim]，转换为二维 [batch_size, embedding_dim]
            if len(features.shape) == 3:
                # 取每个序列的均值作为全局特征
                features = features.mean(dim=1)
            all_features.append(features.cpu())
    
    # 合并所有批次的特征
    if all_features:
        return torch.cat(all_features, dim=0)
    else:
        return None

def prepare_clip_features(features, mode="project", target_dim=None, input_shape=None):
    """
    处理CLIP特征以适应VAE编码器输入
    
    参数:
        features: CLIP输出特征
        mode: 处理模式
        target_dim: 目标特征维度(用于project和pca模式)
        input_shape: 目标形状，格式为(C,T,H,W)
        
    返回:
        处理后的特征
    """
    # 确保特征是2D
    if len(features.shape) == 3:
        logging.info(f"处理3D CLIP特征，原始形状: {features.shape}")
        features = features.mean(dim=1)
        logging.info(f"平均池化后形状: {features.shape}")
    
    # 获取批次大小和特征维度
    batch_size, feat_dim = features.shape
    
    # 解析目标输入形状，强制通道数为3以匹配VAE要求
    if input_shape:
        c, t, h, w = input_shape
        # 强制通道数为3
        c = 3
        target_elements = c * t * h * w
    else:
        # 为常见的CLIP输出维度设置优化的形状配置
        dim_to_shape = {
            768: (3, 1, 16, 16),    # 3*1*16*16=768
            1024: (3, 1, 18, 19),   # 3*1*18*19≈1024
            1280: (3, 1, 20, 21),   # 3*1*20*21≈1280
            2048: (3, 1, 26, 26),   # 3*1*26*26≈2028
            2560: (3, 1, 29, 29)    # 3*1*29*29≈2523
        }
        
        if feat_dim in dim_to_shape:
            c, t, h, w = dim_to_shape[feat_dim]
            logging.info(f"为CLIP维度 {feat_dim} 使用优化形状: [{c}, {t}, {h}, {w}]")
        else:
            # 尝试找到能够整除特征维度的配置
            found = False
            # 尝试不同的通道数
            for c_try in [3, 4, 5, 8, 10]:
                t_try = 1  # 固定时间维度为1
                size = int(np.sqrt(feat_dim / c_try))
                if c_try * t_try * size * size == feat_dim:
                    c, t, h, w = c_try, t_try, size, size
                    found = True
                    logging.info(f"为CLIP维度 {feat_dim} 找到精确匹配: [{c}, {t}, {h}, {w}]")
                    break
            
            if not found:
                # 默认配置
                c = 3  # 默认3通道
                t = 1  # 默认1帧
                h = w = int(np.sqrt(feat_dim / c))
                logging.info(f"无法找到精确匹配，使用近似形状: [{c}, {t}, {h}, {w}]")
        
        target_elements = c * t * h * w
    
    logging.info(f"原始CLIP特征维度: {feat_dim}, 目标形状元素数: {target_elements}")
    
    # 根据模式处理特征
    if mode == "auto":
        # 自动选择处理模式
        if feat_dim == target_elements:
            logging.info(f"自动模式: 特征维度{feat_dim}与目标元素数相等，选择reshape模式")
            mode = "reshape"
        elif feat_dim in [768, 1024, 1280, 2048, 2560]:
            logging.info(f"自动模式: 特征维度{feat_dim}是常见值，选择reshape模式")
            mode = "reshape"
        else:
            logging.info(f"自动模式: 特征维度{feat_dim}不适合直接重塑，选择project模式")
            mode = "project"
    
    if mode == "reshape":
        # 尝试直接重塑特征为合适的形状
        try:
            processed = features.reshape(batch_size, c, t, h, w)
            logging.info(f"成功将特征直接重塑为 {processed.shape}")
            return processed
        except Exception as e:
            logging.warning(f"无法直接重塑特征: {e}")
            logging.info("切换到线性投影模式")
            mode = "project"
    
    if mode == "project":
        # 线性投影到目标维度
        if target_dim is None:
            target_dim = target_elements
        
        logging.info(f"线性投影特征从 {feat_dim} 到 {target_elements}")
        # 创建投影层
        projection = torch.nn.Linear(feat_dim, target_elements, dtype=features.dtype).to(features.device)
        # 确保权重与输入特征具有相同的数据类型
        projection.weight.data = projection.weight.data.to(features.dtype)
        if projection.bias is not None:
            projection.bias.data = projection.bias.data.to(features.dtype)
            
        with torch.no_grad():
            # 使用正交初始化，保持特征的分布特性
            # 临时转换为float32进行正交初始化，然后转回原始类型
            weight_dtype = projection.weight.dtype
            projection.weight.data = projection.weight.data.to(torch.float32)
            torch.nn.init.orthogonal_(projection.weight)
            projection.weight.data = projection.weight.data.to(weight_dtype)
            processed = projection(features)
    
    elif mode == "pca":
        # 使用PCA降维
        if target_dim is None:
            target_dim = min(feat_dim, target_elements)
            
        logging.info(f"使用PCA将特征从 {feat_dim} 降维到 {target_dim}")
        from sklearn.decomposition import PCA
        # 确保PCA结果与原始特征有相同的数据类型
        # 将特征移至CPU用于PCA处理
        cpu_features = features.cpu().numpy()
        pca = PCA(n_components=target_dim)
        processed_np = pca.fit_transform(cpu_features)
        processed = torch.from_numpy(processed_np).to(device=features.device, dtype=features.dtype)
        
        # 如果PCA降维后维度不等于目标维度，需要进行调整
        if target_dim != target_elements:
            logging.info(f"通过线性投影调整维度从 {target_dim} 到 {target_elements}")
            projection = torch.nn.Linear(target_dim, target_elements).to(features.device)
            projection.weight.data = projection.weight.data.to(features.dtype)
            if projection.bias is not None:
                projection.bias.data = projection.bias.data.to(features.dtype)
            with torch.no_grad():
                torch.nn.init.orthogonal_(projection.weight)
                processed = projection(processed)
    
    elif mode == "pad":
        # 零填充或截断
        if feat_dim < target_elements:
            logging.info(f"使用零填充将特征从 {feat_dim} 到 {target_elements}")
            padded = torch.zeros((batch_size, target_elements), dtype=features.dtype, device=features.device)
            padded[:, :feat_dim] = features
            processed = padded
        else:
            logging.info(f"截断特征从 {feat_dim} 到 {target_elements}")
            processed = features[:, :target_elements]
    
    else:
        raise ValueError(f"不支持的特征处理模式: {mode}")
    
    # 重塑为5D张量 [B, C, T, H, W]，确保通道数为3
    processed = processed.reshape(batch_size, c, t, h, w)
    logging.info(f"最终特征形状: {processed.shape}")
    return processed

def fix_amp_warnings():
    """修复torch.cuda.amp.autocast弃用警告
    通过monkey patching将旧的API重定向到新的API
    """
    old_autocast = torch.cuda.amp.autocast
    def new_autocast(*args, **kwargs):
        if 'enabled' in kwargs:
            enabled = kwargs.pop('enabled')
            if not enabled:
                return torch.autocast('cuda', enabled=False, *args, **kwargs)
        return torch.autocast('cuda', *args, **kwargs)
    torch.cuda.amp.autocast = new_autocast

def main():
    # 修复torch.cuda.amp.autocast弃用警告
    fix_amp_warnings()
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()
    # 解析命令行参数
    args = parse_args()
    # 检查输入路径
    if not os.path.exists(args.image_path):
        logging.error(f"输入路径不存在: {args.image_path}")
        return
    # 检查模型文件
    for path, name in [(args.vae_path, "VAE"), (args.clip_path, "CLIP")]:
        if not os.path.exists(path):
            logging.error(f"{name}模型文件不存在: {path}")
            return
    
    # 收集图像文件路径
    image_paths = gather_image_paths(args.image_path)
    if not image_paths:
        logging.error(f"没有找到有效的图像文件")
        return
    logging.info(f"找到 {len(image_paths)} 个图像文件")
    
    # 加载CLIP模型
    logging.info("加载CLIP模型...")
    try:
        clip_model = CLIPModel(
            dtype=torch.float16,
            device=args.device,
            checkpoint_path=args.clip_path,
            tokenizer_path=args.tokenizer_path
        )
    except Exception as e:
        logging.error(f"加载CLIP模型时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    # 使用CLIP处理图像
    logging.info("使用CLIP提取图像特征...")
    clip_features = process_images_with_clip(
        clip_model, image_paths, args.device, args.clip_batch_size
    )
    if clip_features is None:
        logging.error("无法提取CLIP特征")
        return
    logging.info(f"CLIP特征形状: {clip_features.shape}")
    
    # 保存CLIP特征（如果需要）
    if args.save_clip_features:
        clip_features_path = args.output.replace(".pt", "_clip_features.pt") if args.output else os.path.join(
            os.path.dirname(args.image_path) if os.path.isfile(args.image_path) else args.image_path,
            "clip_features.pt"
        )
        torch.save(clip_features, clip_features_path)
        logging.info(f"CLIP特征已保存到: {clip_features_path}")
    
    # 解析输入维度
    input_dim = tuple(map(int, args.input_dim.split(',')))
    if len(input_dim) != 4:
        logging.error(f"输入维度格式不正确: {args.input_dim}，应为'C,T,H,W'")
        return
    
    # 确保通道数为3
    if input_dim[0] != 3:
        input_dim = (3, input_dim[1], input_dim[2], input_dim[3])
        logging.info(f"强制将通道数设置为3，新的输入维度: {input_dim}")
    
    # 处理CLIP特征以适应VAE编码器
    try:
        # 判断是否为auto模式，并自动选择最合适的处理方式
        if args.feature_mode == "auto":
            # 获取特征维度和目标维度
            feat_dim = clip_features.shape[1]
            input_dim_elements = np.prod(input_dim)
            
            # 根据维度匹配情况自动选择处理模式
            if feat_dim == input_dim_elements:
                # 维度完全匹配，使用reshape
                logging.info(f"自动选择reshape模式: 特征维度 {feat_dim} 完全匹配目标维度 {input_dim_elements}")
                args.feature_mode = "reshape"
            elif any(feat_dim == dim for dim in [768, 1024, 1280, 2048, 2560]):
                # 常见的CLIP输出维度，使用对应的优化配置
                logging.info(f"自动选择优化配置: 检测到常见CLIP维度 {feat_dim}")
                # 更新input_dim为更合适的值
                if feat_dim == 1280:
                    args.input_dim = "5,1,16,16"  # 5*1*16*16=1280
                    input_dim = (5, 1, 16, 16)
                    logging.info(f"更新input_dim为 {args.input_dim}")
            # 如果维度不匹配且不是常见维度，使用project模式
            if args.feature_mode == "auto":
                logging.info(f"自动选择project模式: 无精确匹配配置")
                args.feature_mode = "project"
        # 使用新的特征处理方法
        processed_features = prepare_clip_features(
            clip_features,
            mode=args.feature_mode,
            target_dim=args.feature_dim,
            input_shape=input_dim
        )
        logging.info(f"处理后的特征形状: {processed_features.shape}")
    except Exception as e:
        logging.error(f"处理特征时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 加载VAE模型
    logging.info("加载VAE模型...")
    try:
        # 创建自定义scale参数以匹配z_dim
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        # 如果z_dim与默认值不一致，进行截断或填充
        if args.z_dim < len(mean):
            mean = mean[:args.z_dim]
            std = std[:args.z_dim]
        elif args.z_dim > len(mean):
            # 填充额外的维度
            mean = mean + [0.0] * (args.z_dim - len(mean))
            std = std + [1.0] * (args.z_dim - len(std))
        
        vae = WanVAE(
            z_dim=args.z_dim,
            vae_pth=args.vae_path,
            dtype=torch.float16,
            device=args.device
        )
    except Exception as e:
        logging.error(f"加载VAE模型时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 使用VAE编码特征
    logging.info("使用VAE编码特征...")
    try:
        # 准备列表形式的输入
        if processed_features.shape[0] == 1:
            # 单个样本
            inputs = [processed_features.squeeze(0).to(args.device)]
        else:
            # 多个样本
            inputs = [processed_features[i].to(args.device) for i in range(processed_features.shape[0])]
        
        # 使用VAE编码
        encoded_features = vae.encode(inputs)
        # 将结果转换为张量
        if len(encoded_features) == 1:
            encoded_features = encoded_features[0]
        else:
            encoded_features = torch.stack(encoded_features)
        
        logging.info(f"编码后的特征形状: {encoded_features.shape if isinstance(encoded_features, torch.Tensor) else [f.shape for f in encoded_features]}")
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
            'features': encoded_features.cpu() if isinstance(encoded_features, torch.Tensor) else [f.cpu() for f in encoded_features],
            'image_paths': image_paths,
            'vae_path': args.vae_path,
            'z_dim': args.z_dim,
            'input_dim': args.input_dim
        }, args.output)
        logging.info(f"VAE编码特征已保存到: {args.output}")
    except Exception as e:
        logging.error(f"保存编码特征时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main()
