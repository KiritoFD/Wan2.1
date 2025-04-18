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
    parser.add_argument("--feature_mode", type=str, default="auto", "auto"],
                        choices=["auto", "pad", "project", "reshape", "pca"],auto(自动选择)")
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
            args.input_dim = "3,1,32,32"  # 标准配置

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
    
    # 解析目标输入形状
    if input_shape:
        c, t, h, w = input_shape
        target_elements = c * t * h * w
    else:
        # 默认目标形状
        c, t = 3, 180)直接计算合适的空间分辨率
        # 计算h和w，使其接近平方形
        elements_per_channel = feat_dim // c0]:
        h = w = int(np.sqrt(elements_per_channel))
        target_elements = c * t * h * w            768: (3, 1, 16, 16),
    
    logging.info(f"原始CLIP特征维度: {feat_dim}, 目标形状元素数: {target_elements}")            1280: (5, 1, 16, 16),
      2048: (8, 1, 16, 16),
    # 根据模式处理特征 (10, 1, 16, 16)
    if mode == "pad":
        if feat_dim < target_elements:at_dim in dim_to_shape:
            # 零填充
            logging.info(f"使用零填充将特征从 {feat_dim} 扩展到 {target_elements}")
            padded = torch.zeros((batch_size, target_elements), dtype=features.dtype, device=features.device)
            padded[:, :feat_dim] = features [3, 4, 5, 8, 10, 16]:
            processed = padded       size = int(np.sqrt(feat_dim / channel))
        else:    if channel * size * size == feat_dim:
            # 截断
            logging.info(f"截断特征从 {feat_dim} 到 {target_elements}")dim} 找到精确匹配: [{c}, {t}, {h}, {w}]")
            processed = features[:, :target_elements]            break
            
    elif mode == "project": c = 3
        # 线性投影到目标维度(np.sqrt(feat_dim / c))
        if target_dim is None:
            target_dim = target_elements            logging.info(f"无法找到精确匹配，使用近似形状: [{c}, {t}, {h}, {w}]")
        
        logging.info(f"线性投影特征从 {feat_dim} 到 {target_dim}")
        projection = torch.nn.Linear(feat_dim, target_dim).to(features.device) int(np.sqrt(feat_dim)) + 1):
        with torch.no_grad():at_dim % i == 0:
            # 使用正交初始化
            torch.nn.init.orthogonal_(projection.weight)
            processed = projection(features)if divisors:
            n sorted(divisors, reverse=True):
    elif mode == "pca":   if d % 3 == 0:
        # 使用PCA降维= d // 3
        if target_dim is None:
            target_dim = min(feat_dim, target_elements)            break
            
        logging.info(f"使用PCA将特征从 {feat_dim} 降维到 {target_dim}")
        from sklearn.decomposition import PCA            c = feat_dim // (h * w)
        
        # 将特征移至CPU用于PCA处理
        cpu_features = features.cpu().numpy()
        pca = PCA(n_components=target_dim)
        processed_np = pca.fit_transform(cpu_features)
        processed = torch.from_numpy(processed_np).to(features.device)
        eat_dim}, 目标形状元素数: {target_elements}")
        # 如果目标维度与PCA后维度不同，进行填充或截断
        if target_dim < target_elements:
            padded = torch.zeros((batch_size, target_elements), dtype=processed.dtype, device=processed.device)
            padded[:, :target_dim] = processed65, feat_dim + 1)):
            processed = padded
            logging.info(f"PCA降维后填充到目标维度 {target_elements}")    factors.append(i)
            
    elif mode == "reshape":ne
        # 尝试直接重塑特征为合适的形状'inf')
        # 计算使特征能够完美重塑的h和w
        divisors = []
        for i in range(1, int(np.sqrt(feat_dim)) + 1):
            if feat_dim % i == 0:rs:
                divisors.append(i)            if feat_dim % (c_try * t_try * h_try) == 0:
            w_try = feat_dim // (c_try * t_try * h_try)
        if divisors:atio_diff = abs(h_try / w_try - 1.0)
            # 找到最接近平方根的除数
            h = max([d for d in divisors if d <= np.sqrt(feat_dim)])in_diff = ratio_diff
            w = feat_dim // h                best_config = (c_try, t_try, h_try, w_try)
            
            # 直接重塑，不进行填充
            logging.info(f"将特征直接重塑为 [B, 1, 1, {h}, {w}]")
            processed = features.reshape(batch_size, 1, 1, h, w)logging.info(f"找到最佳形状配置: [{c}, {t}, {h}, {w}], 总元素数: {c*t*h*w}")
            s.reshape(batch_size, c, t, h, w)
            # 如果需要3通道，复制通道
            if c > 1:
                processed = processed.repeat(1, c, 1, 1, 1)
                logging.info(f"复制特征到 {c} 个通道")
            
            return processedrget_dim is None:
        else:
            logging.warning(f"无法找到合适的形状重塑 {feat_dim}，切换到填充模式")
            mode = "pad"
            padded = torch.zeros((batch_size, target_elements), dtype=features.dtype, device=features.device)
            padded[:, :feat_dim] = featuresLinear(feat_dim, target_elements).to(features.device)
            processed = paddedith torch.no_grad():
    else:n.weight)
        raise ValueError(f"不支持的特征处理模式: {mode}")        processed = projection(features)
    
    # 重塑为5D张量
    processed = processed.reshape(batch_size, c, t, h, w)
    logging.info(f"最终特征形状: {processed.shape}")        target_dim = min(feat_dim, target_elements)
    
    return processed        logging.info(f"使用PCA将特征从 {feat_dim} 降维到 {target_dim}")
composition import PCA
def fix_amp_warnings(): 
    """pu().numpy()
    修复torch.cuda.amp.autocast弃用警告t_dim)
    通过monkey patching将旧的API重定向到新的API processed_np = pca.fit_transform(cpu_features)
    """ssed_np).to(features.device)
    old_autocast = torch.cuda.amp.autocast    
    ts:
    def new_autocast(*args, **kwargs):nn.Linear(target_dim, target_elements).to(features.device)
        if 'enabled' in kwargs:
            enabled = kwargs.pop('enabled')it.orthogonal_(projection.weight)
            if not enabled:
                return torch.autocast('cuda', enabled=False, *args, **kwargs)
        return torch.autocast('cuda', *args, **kwargs)elif mode == "pad":
    
    torch.cuda.amp.autocast = new_autocast            logging.info(f"零填充特征从 {feat_dim} 到 {target_elements}")
 padded = torch.zeros((batch_size, target_elements), dtype=features.dtype, device=features.device)
def main():features
    # 修复torch.cuda.amp.autocast弃用警告= padded
    fix_amp_warnings()    else:
      logging.info(f"截断特征从 {feat_dim} 到 {target_elements}")
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    start_time = time.time()    raise ValueError(f"不支持的特征处理模式: {mode}")
    
    # 解析命令行参数ed.reshape(batch_size, c, t, h, w)
    args = parse_args()logging.info(f"最终特征形状: {processed.shape}")
    
    # 检查输入路径
    if not os.path.exists(args.image_path):
        logging.error(f"输入路径不存在: {args.image_path}")rnings():
        return"""
    cuda.amp.autocast弃用警告
    # 检查模型文件
    for path, name in [(args.vae_path, "VAE"), (args.clip_path, "CLIP")]:
        if not os.path.exists(path):
            logging.error(f"{name}模型文件不存在: {path}")
            returndef new_autocast(*args, **kwargs):
    abled' in kwargs:
    # 收集图像文件路径
    image_paths = gather_image_paths(args.image_path)led:
    if not image_paths:('cuda', enabled=False, *args, **kwargs)
        logging.error(f"没有找到有效的图像文件") torch.autocast('cuda', *args, **kwargs)
        return
    
    logging.info(f"找到 {len(image_paths)} 个图像文件")
    
    # 加载CLIP模型弃用警告
    logging.info("加载CLIP模型...")amp_warnings()
    try:
        clip_model = CLIPModel(
            dtype=torch.float16,ogging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            device=args.device,
            checkpoint_path=args.clip_path,
            tokenizer_path=args.tokenizer_path令行参数
        )
    except Exception as e:
        logging.error(f"加载CLIP模型时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())g.error(f"输入路径不存在: {args.image_path}")
        return    return
    
    # 使用CLIP处理图像
    logging.info("使用CLIP提取图像特征..."), (args.clip_path, "CLIP")]:
    clip_features = process_images_with_clip(
        clip_model, image_paths, args.device, args.clip_batch_size       logging.error(f"{name}模型文件不存在: {path}")
    )        return
    
    if clip_features is None:
        logging.error("无法提取CLIP特征")s = gather_image_paths(args.image_path)
        returnot image_paths:
        
    logging.info(f"CLIP特征形状: {clip_features.shape}")    return
    
    # 保存CLIP特征（如果需要）e_paths)} 个图像文件")
    if args.save_clip_features:
        clip_features_path = args.output.replace(".pt", "_clip_features.pt") if args.output else os.path.join(
            os.path.dirname(args.image_path) if os.path.isfile(args.image_path) else args.image_path,)
            "clip_features.pt"
        )
        torch.save(clip_features, clip_features_path)
        logging.info(f"CLIP特征已保存到: {clip_features_path}")        device=args.device,
    checkpoint_path=args.clip_path,
    # 解析输入维度
    input_dim = tuple(map(int, args.input_dim.split(',')))
    if len(input_dim) != 4:
        logging.error(f"输入维度格式不正确: {args.input_dim}，应为'C,T,H,W'")g.error(f"加载CLIP模型时出错: {e}")
        return    import traceback
    raceback.format_exc())
    # 处理CLIP特征以适应VAE编码器return
    try:
        # 使用新的特征处理方法
        processed_features = prepare_clip_features(像特征...")
            clip_features, with_clip(
            mode=args.feature_mode,vice, args.clip_batch_size
            target_dim=args.feature_dim,
            input_shape=input_dim
        )lip_features is None:
        
        logging.info(f"处理后的特征形状: {processed_features.shape}")
    except Exception as e:
        logging.error(f"处理特征时出错: {e}")征形状: {clip_features.shape}")
        import traceback
        logging.error(traceback.format_exc())（如果需要）
        returnif args.save_clip_features:
    features_path = args.output.replace(".pt", "_clip_features.pt") if args.output else os.path.join(
    # 加载VAE模型gs.image_path) if os.path.isfile(args.image_path) else args.image_path,
    logging.info("加载VAE模型...")    "clip_features.pt"
    try:
        # 创建自定义scale参数以匹配z_dimve(clip_features, clip_features_path)
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921入维度
        ] tuple(map(int, args.input_dim.split(',')))
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160eturn
        ]
        
        # 如果z_dim与默认值不一致，进行截断或填充
        if args.z_dim < len(mean):o":
            mean = mean[:args.z_dim]es.shape[1]
            std = std[:args.z_dim]prod([int(d) for d in args.input_dim.split(',')])
        elif args.z_dim > len(mean):
            # 填充额外的维度
            mean = mean + [0.0] * (args.z_dim - len(mean))
            std = std + [1.0] * (args.z_dim - len(std))        logging.info(f"CLIP维度与目标维度完全匹配({feat_dim})，使用reshape模式")
        _dim % 3 == 0 or feat_dim % 4 == 0 or feat_dim % 5 == 0:
        vae = WanVAE(mode = "reshape"
            z_dim=args.z_dim,P维度({feat_dim})能被常见通道数整除，使用reshape模式")
            vae_pth=args.vae_path,
            dtype=torch.float16,ode = "project"
            device=args.device       logging.info(f"CLIP维度({feat_dim})不适合直接重塑，使用project模式")
        )
    except Exception as e:_features(
        logging.error(f"加载VAE模型时出错: {e}")s, 
        import traceback
        logging.error(traceback.format_exc())rget_dim=args.feature_dim,
        return        input_shape=input_dim
    
    # 使用VAE编码特征
    logging.info("使用VAE编码特征...")logging.info(f"处理后的特征形状: {processed_features.shape}")
    try:n as e:
        # 准备列表形式的输入
        if processed_features.shape[0] == 1:ceback
            # 单个样本
            inputs = [processed_features.squeeze(0).to(args.device)]n
        else:
            # 多个样本
            inputs = [processed_features[i].to(args.device) for i in range(processed_features.shape[0])]ing.info("加载VAE模型...")
        
        # 使用VAE编码
        encoded_features = vae.encode(inputs)    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        , -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        # 将结果转换为张量
        if len(encoded_features) == 1:
            encoded_features = encoded_features[0].8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        else:2.8251, 1.9160
            encoded_features = torch.stack(encoded_features)
            
        logging.info(f"编码后的特征形状: {encoded_features.shape if isinstance(encoded_features, torch.Tensor) else [f.shape for f in encoded_features]}")n(mean):
    except Exception as e:
        logging.error(f"编码特征时出错: {e}")rgs.z_dim]
        import traceback
        logging.error(traceback.format_exc())an = mean + [0.0] * (args.z_dim - len(mean))
        return        std = std + [1.0] * (args.z_dim - len(std))
    
    # 确定输出文件路径
    if args.output is None:
        if os.path.isdir(args.image_path):
            output_dir = args.image_path
            output_filename = "vae_encoded_features.pt"evice=args.device
        else:
            output_dir = os.path.dirname(args.image_path)
            output_filename = f"{os.path.basename(args.image_path).split('.')[0]}_vae_encoded.pt"
        args.output = os.path.join(output_dir, output_filename)    import traceback
    g.error(traceback.format_exc())
    # 保存编码后的特征return
    try:
        torch.save({
            'features': encoded_features.cpu() if isinstance(encoded_features, torch.Tensor) else [f.cpu() for f in encoded_features],
            'image_paths': image_paths,
            'vae_path': args.vae_path,ape[0] == 1:
            'z_dim': args.z_dim,s.squeeze(0).to(args.device)]
            'input_dim': args.input_dim
        }, args.output).device) for i in range(processed_features.shape[0])]
        logging.info(f"VAE编码特征已保存到: {args.output}")
    except Exception as e:puts)
        logging.error(f"保存编码特征时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())coded_features = encoded_features[0]
        return    else:
    encoded_features)
    elapsed_time = time.time() - start_time
    logging.info(f"处理完成，耗时: {elapsed_time:.2f} 秒")        logging.info(f"编码后的特征形状: {encoded_features.shape if isinstance(encoded_features, torch.Tensor) else [f.shape for f in encoded_features]}")

if __name__ == "__main__":gging.error(f"编码特征时出错: {e}")
    main()        import traceback

        logging.error(traceback.format_exc())
        return
    
    if args.output is None:
        if os.path.isdir(args.image_path):
            output_dir = args.image_path
            output_filename = "vae_encoded_features.pt"
        else:
            output_dir = os.path.dirname(args.image_path)
            output_filename = f"{os.path.basename(args.image_path).split('.')[0]}_vae_encoded.pt"
        args.output = os.path.join(output_dir, output_filename)
    
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
