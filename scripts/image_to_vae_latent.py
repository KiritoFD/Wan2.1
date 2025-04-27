#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一脚本：从图像提取CLIP特征，然后通过VAE进行编码
直接使用wan.modules.clip.py和wan.modules.vae.py中的组件

输出数据格式说明：
- features: torch.Tensor 或 list of torch.Tensor，形状为 [16, 1, 32, 32]
  - 16: z_dim，潜在空间维度，对应于VAE的特征通道数
  - 1: 时间维度，表示单帧图像
  - 32, 32: 空间维度，可通过--unified_latent_size参数修改
- image_paths: list of str，图像文件路径列表，与features对应
- metadata: dict，包含处理参数和版本信息
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
import functools
import warnings
from tqdm import tqdm

# 忽略PIL警告
warnings.filterwarnings("ignore", category=UserWarning, module='PIL.TiffImagePlugin')

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel

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
                        help="潜在空间维度，默认值16，对应Wan模型的VAE特征通道数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--input_size", type=str, default=None,
                        help="调整输入图像大小，格式为'H,W'")
    parser.add_argument("--unified_size", type=str, default=None,
                        help="强制统一所有图像到相同的尺寸，格式为'H,W'")
    parser.add_argument("--unified_latent_size", type=str, default="32,32",
                        help="VAE编码后的潜在向量尺寸，格式为'H,W'，默认32x32。所有输出将强制调整为此尺寸")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="处理批次大小")
    parser.add_argument("--clip_batch_size", type=int, default=32,
                        help="CLIP批处理大小")
    parser.add_argument("--save_clip_features", action="store_true",
                        help="是否保存中间CLIP特征")
    parser.add_argument("--no_progress", action="store_true",
                        help="不显示进度条")
    parser.add_argument("--images_per_file", type=int, default=500,
                        help="每个输出文件包含的最大图像数量，超过此数量将自动分批保存"),
    parser.add_argument("--auto_split", action="store_true",
                        help="当图像数量超过每文件最大数量时自动分批")
    parser.add_argument("--max_resolution", type=int, default=1024,
                        help="处理图像的最大分辨率，超过此值的图像将被自动缩小")
    parser.add_argument("--force_cpu_vae", action="store_true",
                        help="强制在CPU上运行VAE编码（处理超大图像时使用）")
    parser.add_argument("--cpu_batch_size", type=int, default=1,
                        help="CPU处理时的批次大小")
    parser.add_argument("--memory_efficient", action="store_true",
                        help="启用更严格的内存管理（处理大型图像集）")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细的处理信息")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="日志输出间隔（每处理多少张图像输出一次）")
    # 忽略未使用的参数警告
    parser.add_argument("--unused", type=str, default=None, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    # 优先使用位置参数中的image_path
    if args.image_path is None and args.image_path_arg is None:
        parser.error("必须提供图像路径")
    elif args.image_path is None:
        args.image_path = args.image_path_arg
    # 验证统一尺寸格式
    if args.unified_size and args.input_size:
        parser.error("不能同时指定 --unified_size 和 --input_size")
    # 解析统一尺寸
    if args.unified_size:
        try:
            h, w = map(int, args.unified_size.split(','))
            args.unified_size = (h, w)
            # 如果指定了统一尺寸，也设置input_size以保持兼容性
            args.input_size = f"{h},{w}"
        except:
            parser.error(f"无效的统一尺寸格式: {args.unified_size}, 应为'H,W'")
            args.input_size = f"{h},{w}"
    # 解析统一潜在向量尺寸
    if args.unified_latent_size:
        try:
            h, w = map(int, args.unified_latent_size.split(','))
            args.unified_latent_size = (h, w)
        except Exception:
            logging.warning(f"无效的统一潜在向量尺寸格式: {args.unified_latent_size}, 使用默认值32,32")
            args.unified_latent_size = (32, 32)
    else:
        # 始终设置默认值，确保输出一致性
        args.unified_latent_size = (32, 32)

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

def check_image_size(img, max_resolution=1024):
    """检查并调整图像大小，确保不超过最大分辨率"""
    w, h = img.size
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        logging.warning(f"图像尺寸过大 ({w}x{h})，已自动调整为 ({new_w}x{new_h})")
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def process_images_with_clip(clip_model, image_paths, device, batch_size=32, input_size=None, show_progress=True):
    """使用CLIP模型处理图像并提取特征"""
    all_features = []
    
    # 创建进度条
    pbar = tqdm(total=len(image_paths), desc="处理图像", disable=not show_progress)
    
    # 分批处理图像
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []
        valid_paths = []  # 跟踪成功加载的图像路径
        # 加载图像
        for img_path in batch_paths:
            try:
                # 加载图像
                img = Image.open(img_path).convert("RGB")
                # 如果指定了输入大小，调整图像尺寸
                if input_size:
                    img = img.resize(input_size, Image.Resampling.LANCZOS)
                # 转换为张量 - 归一化到[-1, 1]以匹配VAE预期
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2 - 1
                images.append(img_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                logging.error(f"处理图像 {img_path} 时出错: {e}")
                continue
        # 更新进度条
        pbar.update(len(batch_paths))
        if not images:    
            continue
        # 准备CLIP输入 - 直接在目标设备上创建批量
        videos = [img.unsqueeze(1).to(device) for img in images]  # 添加时间维度
        with torch.no_grad():
            # 提取CLIP特征
            features = clip_model.visual(videos)
            # 如果特征是三维的 [batch_size, seq_len, embedding_dim]，转换为二维 [batch_size, embedding_dim]
            if len(features.shape) == 3:
                features = features.mean(dim=1)
            all_features.append((features.cpu(), valid_paths))
    # 关闭进度条
    pbar.close()
    
    # 合并所有批次的特征和路径
    if all_features:
        all_feature_tensors = []
        all_valid_paths = []
        for features, paths in all_features:
            all_feature_tensors.append(features)
            all_valid_paths.extend(paths)
        return torch.cat(all_feature_tensors, dim=0), all_valid_paths
    else:
        return None, []

def fix_amp_warnings():
    """修复torch.cuda.amp.autocast弃用警告"""
    old_autocast = torch.cuda.amp.autocast
    def new_autocast(*args, **kwargs):
        if 'enabled' in kwargs:
            enabled = kwargs.pop('enabled')
            if not enabled:
                return torch.autocast('cuda', enabled=False, *args, **kwargs)
        return torch.autocast('cuda', *args, **kwargs)
    torch.cuda.amp.autocast = new_autocast

def process_image_batch(image_paths_batch, vae, clip_model, args, input_size=None, batch_index=None):
    """处理一批图像并返回VAE编码后的特征"""
    start_time = time.time()
    batch_prefix = f"批次 {batch_index}" if batch_index is not None else ""

    # 使用CLIP处理图像
    if args.verbose:
        logging.info(f"{batch_prefix}：使用CLIP提取图像特征...")
    clip_features, valid_paths = process_images_with_clip(
        clip_model, image_paths_batch, args.device, 
        args.clip_batch_size, input_size,
        show_progress=not args.no_progress
    )
    if clip_features is None:
        logging.error(f"{batch_prefix}：无法提取CLIP特征")
        return None, None
    image_paths_batch = valid_paths
    # 更新图像路径为有效路径
    if args.verbose:
        logging.info(f"{batch_prefix}：成功处理图像：{len(image_paths_batch)} 张")
    else:
        logging.info(f"{batch_prefix}：CLIP特征处理完成，有效图像 {len(image_paths_batch)} 张")
    
    if args.save_clip_features:
        # 保存CLIP特征（如果需要）
        clip_features_path = args.output.replace(".pt", f"_clip_features.pt") if args.output else os.path.join(
            os.path.dirname(args.image_path) if os.path.isfile(args.image_path) else args.image_path,
            "clip_features.pt"
        )
        # 将特征移至CPU后保存，然后释放
        clip_features_cpu = clip_features.cpu()
        torch.save(clip_features_cpu, clip_features_path)
        del clip_features_cpu
        logging.info(f"{batch_prefix}：CLIP特征已保存到: {clip_features_path}")
    del clip_features
    # 释放CLIP特征以节省内存
    torch.cuda.empty_cache()

    # 处理图像并转换为VAE可接受的输入格式 - 分批处理以减少内存占用
    try:
        processed_images = []
        image_dimensions = [] if args.verbose else None  # 只在详细模式下跟踪尺寸
        sub_batch_size = min(10, len(image_paths_batch)) if args.memory_efficient else min(20, len(image_paths_batch))
        process_pbar = tqdm(total=len(image_paths_batch), desc=f"{batch_prefix}：处理图像特征", disable=not args.verbose and args.no_progress)
        # 创建进度条
        for i in range(0, len(image_paths_batch), sub_batch_size):
            sub_batch_paths = image_paths_batch[i:i + sub_batch_size]
            sub_processed_images = []
            # 子批次处理
            for j, img_path in enumerate(sub_batch_paths):
                try:
                    # 加载并预处理图像
                    img = Image.open(img_path).convert("RGB")
                    if args.unified_size:
                        # 检查图像大小，超过最大分辨率则自动缩小
                        img = check_image_size(img, args.max_resolution)
                        # 强制统一尺寸
                        h, w = args.unified_size
                        img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
                    elif input_size:
                        # 使用输入尺寸
                        h, w = input_size
                        img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
                    else:
                        # 原始尺寸
                        h, w = img.height, img.width
                        img_resized = img
                    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor * 2 - 1
                    img_tensor = img_tensor.unsqueeze(1)  # 添加时间维度 [C, T, H, W]
                    sub_processed_images.append(img_tensor)
                    # 只在详细模式下记录图像尺寸
                    if args.verbose:
                        final_size = (img_resized.height, img_resized.width)
                        estimated_memory = img_resized.height * img_resized.width * 3 * 4 * 2  # 每像素3通道，float32，编解码阶段约2倍内存
                        image_dimensions.append({
                            'path': img_path,
                            'original': (img.height, img.width),
                            'resized': final_size,
                            'estimated_memory_mb': estimated_memory / (1024 * 1024)
                        })
                except Exception as e:
                    logging.error(f"处理图像 {img_path} 时出错: {e}")
                    continue
                img_idx = i + j
                if args.verbose and img_idx > 0 and img_idx % args.log_interval == 0:
                    logging.info(f"{batch_prefix}：已处理 {img_idx}/{len(image_paths_batch)} 张图像")
                # 更新进度条
                process_pbar.update(1)
            # 将当前子批次的处理结果添加到主列表
            processed_images.extend(sub_processed_images)    
        process_pbar.close()
        # 清理子批次变量
        del sub_processed_images
        if args.verbose and image_dimensions:        
            largest_image = max(image_dimensions, key=lambda x: x['estimated_memory_mb'])
            logging.info(f"{batch_prefix}：最大图像: {largest_image['path']} - 原始尺寸: {largest_image['original']}, 调整后: {largest_image['resized']}, 估计内存: {largest_image['estimated_memory_mb']:.2f}MB")
        # 强制垃圾回收，确保释放所有未使用的内存
        import gc
        if args.memory_efficient or i % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        # 关闭进度条
        process_pbar.close()
    except Exception as e:
        logging.error(f"{batch_prefix}：处理特征时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

    # 使用VAE编码特征
    if args.verbose:
        logging.info(f"{batch_prefix}：使用VAE编码特征...")
    else:
        logging.info(f"{batch_prefix}：开始VAE编码，共 {len(processed_images)} 张图像")
    
    try:
        # 确定是否使用CPU进行VAE编码
        use_cpu = args.force_cpu_vae
        vae_device = torch.device('cpu') if use_cpu else args.device
        
        # 保存原始VAE设备以便后续恢复
        if use_cpu:
            vae_original_device = vae.device
            vae.model = vae.model.to(torch.device('cpu'))
            vae.device = torch.device('cpu')
            
        if args.verbose and use_cpu:
            logging.info("使用CPU处理VAE编码 (可能较慢但内存安全)")
            
        # 批量处理
        if not args.no_progress:
            pbar = tqdm(total=len(processed_images), desc=f"{batch_prefix}：VAE编码")
        encoded_features = []
        valid_indices = list(range(len(processed_images)))  # 记录成功处理的索引
        
        # 分批进行VAE编码，每次编码完成后立即释放输入
        for i, inp in enumerate(processed_images):
            try:
                # 迁移到目标设备
                inp_device = inp.to(vae_device)
                
                # 尝试编码
                with torch.no_grad():
                    feat = vae.encode([inp_device])[0]
                    
                    # 确保feat的z_dim为16，与Wan模型匹配
                    if feat.shape[0] != args.z_dim:
                        logging.warning(f"VAE编码特征通道数 {feat.shape[0]} 与预期的 {args.z_dim} 不匹配")
                    
                    # 始终调整潜在向量大小，确保输出一致性
                    feat = F.interpolate(feat, size=args.unified_latent_size, 
                                       mode='bilinear', align_corners=False)
                
                # CPU特殊处理 - 保持在CPU上
                if not use_cpu:
                    feat = feat.cpu()  # 编码后立即移回CPU以释放GPU内存
                
                encoded_features.append(feat)
                
                # 释放当前输入张量
                del inp_device
                
                # 每处理几张图像清理一次缓存
                if not use_cpu and (i+1) % 5 == 0:
                    torch.cuda.empty_cache()
                    
                # 周期性日志，避免过多输出
                if args.verbose and i > 0 and i % args.log_interval == 0:
                    logging.info(f"{batch_prefix}：已编码 {i}/{len(processed_images)} 张图像")
                
            except torch.cuda.OutOfMemoryError:
                # GPU内存不足，尝试在CPU上处理此图像
                logging.warning(f"{batch_prefix}：GPU内存不足，尝试在CPU上处理图像 {i}")
                try:
                    # 临时将VAE移至CPU
                    temp_vae_device = vae.device
                    vae.model = vae.model.to(torch.device('cpu'))
                    vae.device = torch.device('cpu')
                    
                    # 在CPU上编码
                    inp_cpu = inp.to(torch.device('cpu'))
                    with torch.no_grad():
                        feat = vae.encode([inp_cpu])[0]
                        
                        # 始终调整潜在向量大小，确保输出一致性
                        feat = F.interpolate(feat, size=args.unified_latent_size, 
                                           mode='bilinear', align_corners=False)
                    
                    encoded_features.append(feat)
                    
                    # 恢复VAE设备
                    vae.model = vae.model.to(temp_vae_device)
                    vae.device = temp_vae_device
                    
                    del inp_cpu
                except Exception as e:
                    logging.error(f"在CPU上处理图像 {i} 失败: {e}")
                    valid_indices.remove(i)
                    continue
            except Exception as e:
                logging.error(f"{batch_prefix}：处理图像 {i} 失败: {e}")
                valid_indices.remove(i)
                continue
            
            # 更新进度条
            if not args.no_progress:
                pbar.update(1)
        
        # 关闭进度条
        if not args.no_progress:
            pbar.close()
            
        # 如果使用了CPU，将VAE模型恢复到原始设备
        if use_cpu:
            vae.model = vae.model.to(vae_original_device)
            vae.device = vae_original_device
        
        # 释放原始图像数据
        del processed_images
        if not use_cpu:
            torch.cuda.empty_cache()
        
        # 确认所有特征尺寸一致并输出结果形状
        if encoded_features and isinstance(encoded_features, list) and len(encoded_features) > 0:
            expected_shape = (args.z_dim, 1, args.unified_latent_size[0], args.unified_latent_size[1])
            all_same = all(feat.shape == expected_shape for feat in encoded_features)
            if not all_same:
                logging.warning("发现特征形状不一致，进行修正...")
                corrected_features = []
                for feat in encoded_features:
                    # 调整空间维度
                    if feat.shape[-2:] != (args.unified_latent_size[0], args.unified_latent_size[1]):
                        feat = F.interpolate(feat, size=args.unified_latent_size, 
                                            mode='bilinear', align_corners=False)
                    # 确保时间维度正确
                    if feat.shape[1] != 1:
                        feat = feat.mean(dim=1, keepdim=True)
                    corrected_features.append(feat)
                encoded_features = corrected_features
                logging.info(f"所有特征已调整为统一形状: {expected_shape}")
        
        # 简化特征形状输出
        if args.verbose:
            if isinstance(encoded_features, list):
                sample_shapes = [f.shape for f in encoded_features[:3]]
                logging.info(f"{batch_prefix}：编码后的特征形状示例: {sample_shapes}{'...' if len(encoded_features) > 3 else ''}")
            else:
                logging.info(f"{batch_prefix}：编码后的特征形状: {encoded_features.shape}")
    except Exception as e:
        logging.error(f"{batch_prefix}：编码特征时出错: {e}")
        if args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        return None, None
    
    elapsed_time = time.time() - start_time
    logging.info(f"{batch_prefix}：处理完成，耗时: {elapsed_time:.2f} 秒")
    
    # 确保有效路径与特征数量匹配
    if len(valid_indices) < len(image_paths_batch):
        valid_paths = [image_paths_batch[i] for i in valid_indices]
    else:
        valid_paths = image_paths_batch
    
    return encoded_features, valid_paths

def main():
    # 修复torch.cuda.amp.autocast弃用警告
    fix_amp_warnings()
    # 解析命令行参数
    args = parse_args()
    
    # 配置日志级别 - 非verbose模式下减少输出
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    # 对于关键信息强制使用INFO级别
    logging.info(f"开始处理 - {'详细模式' if args.verbose else '精简模式'}")
    
    # 打印GPU内存信息
    if args.device.startswith('cuda') and torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        logging.info(f"GPU: {gpu_props.name}, 总内存: {gpu_props.total_memory / (1024**3):.2f} GB")
        if args.max_resolution > 1536 and not args.force_cpu_vae:
            logging.warning(f"最大分辨率设置较高 ({args.max_resolution})，对于 {gpu_props.total_memory / (1024**3):.1f}GB 的GPU可能导致内存不足")
    # 检查输入路径
    if not os.path.exists(args.image_path):
        logging.error(f"输入路径不存在: {args.image_path}")
        return
        
    # 检查模型文件
    for path, name in [(args.vae_path, "VAE"), (args.clip_path, "CLIP")]:
        if not os.path.exists(path):
            logging.error(f"{name}模型文件不存在: {path}")
            return
    # 对于大量文件，只打印总数而不是每个路径
    image_paths = gather_image_paths(args.image_path)
    if not image_paths:
        logging.error(f"没有找到有效的图像文件")
        return
    logging.info(f"找到 {len(image_paths)} 个图像文件" + (f"，包括: {image_paths[:5]}..." if args.verbose and len(image_paths) > 5 else ""))
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
    
    # 检查并加载VAE模型
    logging.info("加载VAE模型...")
    try:
        vae = WanVAE(
            z_dim=args.z_dim,
            vae_pth=args.vae_path,
            dtype=torch.float16,
            device=args.device,
        )
    except Exception as e:
        logging.error(f"加载VAE模型时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 确定是否需要分批处理
    need_split = args.auto_split and len(image_paths) > args.images_per_file
    if need_split:
        num_batches = (len(image_paths) + args.images_per_file - 1) // args.images_per_file
        logging.info(f"图像数量 ({len(image_paths)}) 超过每批次最大数量 ({args.images_per_file})，将分为 {num_batches} 批处理")
    
    # 处理每一批图像
    for batch_idx in range(num_batches) if need_split else [0]:
        start_idx = batch_idx * args.images_per_file if need_split else 0
        end_idx = min(start_idx + args.images_per_file, len(image_paths)) if need_split else len(image_paths)
        batch_paths = image_paths[start_idx:end_idx]
        
        # 构建智能化输出文件名，保留原文件名或目录名信息
        if args.output:
            # 如果用户指定了输出路径
            if need_split:
                output_base = os.path.splitext(args.output)[0]
                output_ext = os.path.splitext(args.output)[1]
                batch_output = f"{output_base}_batch{batch_idx + 1}{output_ext}"
            else:
                batch_output = args.output
        else:
            # 如果没有指定输出路径，使用输入路径的名称部分
            if os.path.isdir(args.image_path):
                # 目录输入 - 使用目录名作为基础
                dir_name = os.path.basename(os.path.normpath(args.image_path))
                output_dir = args.image_path
                if need_split:
                    output_filename = f"{dir_name}_vae_encoded_batch{batch_idx + 1}.pt"
                else:
                    output_filename = f"{dir_name}_vae_encoded.pt"
            else:
                # 单文件输入 - 使用文件名作为基础
                output_dir = os.path.dirname(args.image_path)
                base_filename = os.path.basename(args.image_path).split('.')[0]
                if need_split:
                    output_filename = f"{base_filename}_vae_encoded_batch{batch_idx + 1}.pt"
                else:
                    output_filename = f"{base_filename}_vae_encoded.pt"
            
            batch_output = os.path.join(output_dir, output_filename)
        
        # 准备元数据
        metadata = {
            "version": "1.0.0",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "z_dim": args.z_dim,
            "vae_path": os.path.basename(args.vae_path),
            "clip_path": os.path.basename(args.clip_path),
            "input_size": args.input_size,
            "unified_size": str(args.unified_size) if args.unified_size else None,
            "unified_latent_size": str(args.unified_latent_size),
            "batch_info": f"Batch {batch_idx + 1}/{num_batches}" if need_split else "全量处理",
            "format_description": f"VAE编码后的潜在向量，形状固定为 [{args.z_dim}, 1, {args.unified_latent_size[0]}, {args.unified_latent_size[1]}]",
            "source": args.image_path,  # 添加源路径信息
            "file_count": len(batch_paths)  # 添加文件数量信息
        }
        
        # 解析输入尺寸
        input_size = None
        if args.input_size:
            try:
                h, w = map(int, args.input_size.split(','))
                input_size = (w, h)  # PIL使用(width, height)格式
            except:
                logging.warning(f"无效的输入尺寸格式: {args.input_size}, 将使用原始尺寸")

        # 处理一批图像
        encoded_features, valid_paths = process_image_batch(batch_paths, vae, clip_model, args, input_size, batch_idx + 1)
        if encoded_features is None or not valid_paths:
            logging.error(f"批次 {batch_idx + 1} 处理失败，跳过")
            # 强制清理内存
            torch.cuda.empty_cache()
            continue
            
        # 保存编码后的特征（确保移至CPU再保存）
        try:
            # 如果是列表，确保所有张量都在CPU上
            if isinstance(encoded_features, list):
                if len(encoded_features) == 1:
                    save_features = encoded_features[0].cpu()
                else:
                    # 将特征列表转换为完整的张量
                    # 这样.pt文件转为文本后不会显示省略号
                    try:
                        save_features = torch.stack([feat.cpu() for feat in encoded_features])
                    except:
                        # 如果无法堆叠(可能形状不一)，保留列表形式
                        logging.warning("无法将所有特征堆叠为单一张量，保留列表格式")
                        save_features = [feat.cpu() for feat in encoded_features]
            else:
                save_features = encoded_features.cpu()
                
            # 完全释放GPU上的特征
            del encoded_features
            torch.cuda.empty_cache()
            
            # 保存数据
            save_data = {
                'features': save_features,
                'image_paths': valid_paths,
                'metadata': metadata 
            }
            
            # 打印样本特征形状信息，不含省略号
            if args.verbose:
                if isinstance(save_features, list):
                    shape_info = f"特征列表: {len(save_features)}项，样本形状: {save_features[0].shape}"
                else:
                    shape_info = f"特征张量形状: {save_features.shape}"
                logging.info(f"保存的特征信息 - {shape_info}")
                
            # 保存特征和元数据
            torch.save(save_data, batch_output)
            logging.info(f"VAE编码特征已保存到: {batch_output}")
            
            # 释放内存
            del save_features, save_data
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"保存批次 {batch_idx + 1} 编码特征时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return
    
    # 在脚本结束时添加明确的输出格式说明
    logging.info("\n输出数据格式说明：")
    logging.info("====================")
    logging.info(f"features: torch.Tensor 或 torch.stack([tensor, ...])，形状为 [N, {args.z_dim}, 1, {args.unified_latent_size[0]}, {args.unified_latent_size[1]}]")
    logging.info("  - N: 图像数量")
    logging.info("  - 16: z_dim，潜在空间维度，对应VAE特征通道数")
    logging.info("  - 1: 时间维度，表示单帧图像")
    logging.info(f"  - {args.unified_latent_size[0]}x{args.unified_latent_size[1]}: 空间维度，所有输出均为此尺寸")
    logging.info("注意: 此输出格式与Wan2.1的image2video模型期望的VAE编码格式匹配")
    logging.info("====================\n")
    logging.info("提示: 保存的.pt文件包含完整张量数据，如需查看数值请使用torch.load()加载")

if __name__ == "__main__":
    main()