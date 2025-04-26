#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一脚本：从图像提取CLIP特征，然后通过VAE进行编码
直接使用wan.modules.clip.py和wan.modules.vae.py中的组件

输出数据格式说明：
- features: torch.Tensor 或 list of torch.Tensor，形状为 [z_dim, 1, H, W]，VAE编码后的特征
- image_paths: list of str，图像文件路径列表
- metadata: dict，包含处理参数和版本信息的元数据字典
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
                        help="潜在空间维度")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="计算设备")
    parser.add_argument("--input_size", type=str, default=None,
                        help="调整输入图像大小，格式为'H,W'")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="处理批次大小")
    parser.add_argument("--clip_batch_size", type=int, default=32,
                        help="CLIP批处理大小")
    parser.add_argument("--save_clip_features", action="store_true",
                        help="是否保存中间CLIP特征")
    parser.add_argument("--no_progress", action="store_true",
                        help="不显示进度条")
    parser.add_argument("--images_per_file", type=int, default=500,
                        help="每个输出文件包含的最大图像数量，超过此数量将自动分批保存")
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
    
    args = parser.parse_args()
    
    # 优先使用位置参数中的image_path
    if args.image_path is None and args.image_path_arg is None:
        parser.error("必须提供图像路径")
    elif args.image_path is None:
        args.image_path = args.image_path_arg

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
        
        # 提取CLIP特征
        with torch.no_grad():
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
    
    # 添加对装饰器用法的支持
    def patched_decorator(func=None, **kwargs):
        if func is not None:
            # 被用作装饰器
            @functools.wraps(func)
            def wrapper(*args, **kw):
                with torch.autocast('cuda'):
                    return func(*args, **kw)
            return wrapper
        # 被用作上下文管理器
        return torch.autocast('cuda', **kwargs)
    
    # 替换amp.autocast装饰器
    import torch.cuda.amp as amp
    amp.autocast = patched_decorator

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
    
    # 更新图像路径为有效路径
    image_paths_batch = valid_paths
    if args.verbose:
        logging.info(f"{batch_prefix}：成功处理图像：{len(image_paths_batch)} 张")
    else:
        logging.info(f"{batch_prefix}：CLIP特征处理完成，有效图像 {len(image_paths_batch)} 张")
    
    # 保存CLIP特征（如果需要）
    if args.save_clip_features:
        output_suffix = f"_batch{batch_index}" if batch_index is not None else ""
        clip_features_path = args.output.replace(".pt", f"{output_suffix}_clip_features.pt") if args.output else os.path.join(
            os.path.dirname(args.image_path) if os.path.isfile(args.image_path) else args.image_path,
            f"clip_features{output_suffix}.pt"
        )
        # 将特征移至CPU后保存，然后释放
        clip_features_cpu = clip_features.cpu()
        torch.save(clip_features_cpu, clip_features_path)
        del clip_features_cpu
        logging.info(f"{batch_prefix}：CLIP特征已保存到: {clip_features_path}")
    
    # 释放CLIP特征以节省内存
    del clip_features
    torch.cuda.empty_cache()
    
    # 处理图像并转换为VAE可接受的输入格式 - 分批处理以减少内存占用
    if args.verbose:
        logging.info(f"{batch_prefix}：处理图像特征以适应VAE编码器...")
    
    try:
        # 优化：分批处理图像预处理，每批处理完后释放内存
        processed_images = []
        image_dimensions = [] if args.verbose else None  # 只在详细模式下跟踪尺寸
        sub_batch_size = min(10, len(image_paths_batch)) if args.memory_efficient else min(20, len(image_paths_batch))
        
        # 创建进度条
        process_pbar = tqdm(total=len(image_paths_batch), 
                           desc=f"{batch_prefix}：处理图像特征", 
                           disable=not args.verbose and args.no_progress)
        
        # 子批次处理
        for i in range(0, len(image_paths_batch), sub_batch_size):
            sub_batch_paths = image_paths_batch[i:i + sub_batch_size]
            sub_processed_images = []
            
            for j, img_path in enumerate(sub_batch_paths):
                try:
                    # 加载并预处理图像
                    img = Image.open(img_path).convert("RGB")
                    
                    # 检查图像大小，超过最大分辨率则自动缩小
                    img = check_image_size(img, args.max_resolution)
                    
                    if input_size:
                        h, w = input_size
                    else:
                        h, w = img.height, img.width
                    
                    # 计算合适的VAE尺寸
                    aspect_ratio = h / w
                    vae_stride = vae.stride if hasattr(vae, 'stride') else [4, 8, 8]
                    patch_size = [1, 1, 1]  # 默认值，如果vae.patch_size不可用
                    if hasattr(vae, 'patch_size'):
                        patch_size = vae.patch_size
                    
                    # 类似于image2video.py中的计算方式
                    max_area = h * w  # 使用原始图像面积
                    lat_h = round(np.sqrt(max_area * aspect_ratio) // vae_stride[1] // patch_size[1] * patch_size[1])
                    lat_w = round(np.sqrt(max_area / aspect_ratio) // vae_stride[2] // patch_size[2] * patch_size[2])
                    h = lat_h * vae_stride[1]
                    w = lat_w * vae_stride[2]
                    
                    # 只在详细模式下记录图像尺寸
                    if args.verbose:
                        # 记录最终尺寸和预估内存需求
                        final_size = (h, w)
                        estimated_memory = h * w * 3 * 4 * 2  # 每像素3通道，float32，编解码阶段约2倍内存
                        
                        image_dimensions.append({
                            'path': img_path,
                            'original': (img.height, img.width),
                            'resized': final_size,
                            'estimated_memory_mb': estimated_memory / (1024 * 1024)
                        })
                        
                        # 只对特别大的图像发出警告
                        if estimated_memory > 0.75 * torch.cuda.get_device_properties(0).total_memory and not args.force_cpu_vae:
                            logging.warning(f"图像 {img_path} 尺寸过大 ({h}x{w})，可能导致内存不足")
                    
                    # 调整图像大小并准备VAE输入
                    img_resized = img.resize((w, h))
                    img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor * 2 - 1
                    img_tensor = img_tensor.unsqueeze(1)  # 添加时间维度 [C, T, H, W]
                    sub_processed_images.append(img_tensor)
                    
                    # 释放原始图像内存
                    del img, img_resized
                    
                    # 周期性日志，避免过多输出
                    img_idx = i + j
                    if args.verbose and img_idx > 0 and img_idx % args.log_interval == 0:
                        logging.info(f"{batch_prefix}：已处理 {img_idx}/{len(image_paths_batch)} 张图像")
                except Exception as e:
                    logging.error(f"处理图像 {img_path} 时出错: {e}")
                    continue
                
                # 更新进度条
                process_pbar.update(1)
            
            # 将当前子批次的处理结果添加到主列表
            processed_images.extend(sub_processed_images)
            
            # 清理子批次变量
            del sub_processed_images
            
            # 定期强制垃圾回收
            if args.memory_efficient or i % 100 == 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
        # 关闭进度条
        process_pbar.close()
        
        # 只在详细模式下输出图像尺寸统计
        if args.verbose and image_dimensions:
            largest_image = max(image_dimensions, key=lambda x: x['estimated_memory_mb'])
            logging.info(f"{batch_prefix}：最大图像: {largest_image['path']} - 原始尺寸: {largest_image['original']}, 调整后: {largest_image['resized']}, 估计内存: {largest_image['estimated_memory_mb']:.2f}MB")
        
        # 强制垃圾回收，确保释放所有未使用的内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"{batch_prefix}：处理特征时出错: {e}")
        if args.verbose:
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
        
        if use_cpu and args.verbose:
            logging.info("使用CPU处理VAE编码 (可能较慢但内存安全)")
            # 将VAE模型移至CPU
            vae_original_device = vae.device
            vae.model = vae.model.to(torch.device('cpu'))
            vae.device = torch.device('cpu')
        
        # 批量处理
        if not args.no_progress:
            pbar = tqdm(total=len(processed_images), desc=f"{batch_prefix}：VAE编码")
            encoded_features = []
            failed_indices = []  # 跟踪失败的图像索引
            
            # 分批进行VAE编码，每次编码完成后立即释放输入
            for i, inp in enumerate(processed_images):
                try:
                    # 迁移到目标设备
                    inp_device = inp.to(vae_device)
                    
                    # 尝试编码
                    with torch.no_grad():
                        feat = vae.encode([inp_device])[0]
                        
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
                        
                        encoded_features.append(feat)
                        
                        # 恢复VAE设备
                        vae.model = vae.model.to(temp_vae_device)
                        vae.device = temp_vae_device
                        
                        del inp_cpu
                    except Exception as e:
                        logging.error(f"在CPU上处理图像 {i} 失败: {e}")
                        failed_indices.append(i)
                except Exception as e:
                    if args.verbose:
                        logging.error(f"{batch_prefix}：处理图像 {i} 失败: {e}")
                    failed_indices.append(i)
                
                pbar.update(1)
            
            # 如果有失败的图像，从valid_paths列表中删除
            if failed_indices:
                logging.warning(f"有 {len(failed_indices)} 张图像处理失败，将从结果中排除")
                valid_paths = [p for i, p in enumerate(image_paths_batch) if i not in failed_indices]
            
            pbar.close()
            
            # 如果所有图像都处理失败，返回错误
            if len(encoded_features) == 0:
                logging.error("所有图像处理失败")
                return None, None
        else:
            # 无进度条模式简化处理逻辑
            encoded_features = []
            valid_indices = []
            cpu_batch_size = args.cpu_batch_size if use_cpu else 5
            
            for i in range(0, len(processed_images), cpu_batch_size):
                try:
                    sub_batch = [img.to(vae_device) for img in processed_images[i:i + cpu_batch_size]]
                    with torch.no_grad():
                        sub_encoded = vae.encode(sub_batch)
                    
                    # 如果结果是列表，扩展现有列表；否则添加到列表
                    if isinstance(sub_encoded, list):
                        encoded_features.extend([feat.cpu() if not use_cpu else feat for feat in sub_encoded])
                    else:
                        encoded_features.append(sub_encoded.cpu() if not use_cpu else sub_encoded)
                    
                    # 记录成功处理的索引
                    valid_indices.extend(range(i, min(i + cpu_batch_size, len(processed_images))))
                    
                    # 清理当前子批次
                    del sub_batch, sub_encoded
                    if not use_cpu:
                        torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"处理批次 {i//cpu_batch_size} 失败: {e}")
            
            # 如果有图像处理成功
            if valid_indices:
                valid_paths = [image_paths_batch[i] for i in valid_indices]
            else:
                logging.error("所有图像处理失败")
                return None, None
        
        # 如果使用了CPU，将VAE模型恢复到原始设备
        if use_cpu:
            vae.model = vae.model.to(vae_original_device)
            vae.device = vae_original_device
        
        # 释放原始图像数据
        del processed_images
        if not use_cpu:
            torch.cuda.empty_cache()
        
        # 简化特征形状输出
        if args.verbose:
            logging.info(f"{batch_prefix}：编码后的特征形状: {[f.shape for f in encoded_features][:3]}{'...' if len(encoded_features) > 3 else ''}")
    except Exception as e:
        logging.error(f"{batch_prefix}：编码特征时出错: {e}")
        if args.verbose:
            import traceback
            logging.error(traceback.format_exc())
        return None, None
    
    elapsed_time = time.time() - start_time
    logging.info(f"{batch_prefix}：处理完成，耗时: {elapsed_time:.2f} 秒")
    
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
    start_time = time.time()
    
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
    
    # 解析输入尺寸
    input_size = None
    if args.input_size:
        try:
            h, w = map(int, args.input_size.split(','))
            input_size = (w, h)  # PIL使用(width, height)格式
            logging.info(f"将调整输入图像尺寸为: {h}x{w}")
        except:
            logging.warning(f"无法解析输入尺寸: {args.input_size}，将使用原始图像尺寸")
    
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
    
    # 加载VAE模型
    logging.info("加载VAE模型...")
    try:
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
    
    # 确定是否需要分批处理
    need_split = args.auto_split and len(image_paths) > args.images_per_file
    
    if need_split:
        # 分批处理图像
        num_batches = (len(image_paths) + args.images_per_file - 1) // args.images_per_file
        logging.info(f"图像数量 ({len(image_paths)}) 超过每批次最大数量 ({args.images_per_file})，将分为 {num_batches} 批处理")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.images_per_file
            end_idx = min(start_idx + args.images_per_file, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            logging.info(f"处理批次 {batch_idx + 1}/{num_batches}，包含 {len(batch_paths)} 张图像")
            
            # 处理一批图像
            encoded_features, valid_paths = process_image_batch(batch_paths, vae, clip_model, args, input_size, batch_idx + 1)
            
            if encoded_features is None or not valid_paths:
                logging.error(f"批次 {batch_idx + 1} 处理失败，跳过")
                # 强制清理内存
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                continue
            
            # 确定输出文件路径
            if args.output:
                # 如果指定了输出路径，根据批次索引生成不同的文件名
                output_base = os.path.splitext(args.output)[0]
                output_ext = os.path.splitext(args.output)[1]
                batch_output = f"{output_base}_batch{batch_idx + 1}{output_ext}"
            else:
                # 如果没有指定输出路径，使用默认路径
                if os.path.isdir(args.image_path):
                    output_dir = args.image_path
                else:
                    output_dir = os.path.dirname(args.image_path)
                batch_output = os.path.join(output_dir, f"vae_encoded_features_batch{batch_idx + 1}.pt")
            
            # 准备元数据
            metadata = {
                "version": "1.0.0",
                "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "z_dim": args.z_dim,
                "vae_path": os.path.basename(args.vae_path),
                "clip_path": os.path.basename(args.clip_path),
                "input_size": args.input_size,
                "batch_info": f"Batch {batch_idx + 1}/{num_batches}",
                # 减少元数据中的特征形状信息
                "feature_shape": str([f.shape for f in encoded_features[:3]]).replace("torch.Size", "") + 
                               ("..." if len(encoded_features) > 3 else "")
            }
            
            # 保存编码特征
            try:
                # 移到CPU再保存
                if isinstance(encoded_features, list):
                    if len(encoded_features) == 1:
                        save_features = encoded_features[0].cpu()
                    else:
                        save_features = [feat.cpu() for feat in encoded_features]
                else:
                    save_features = encoded_features.cpu()
                
                # 完全释放GPU上的特征
                del encoded_features
                torch.cuda.empty_cache()
                    
                torch.save({
                    'features': save_features,
                    'image_paths': valid_paths,
                    'metadata': metadata
                }, batch_output)
                logging.info(f"批次 {batch_idx + 1} VAE编码特征已保存到: {batch_output}")
                
                # 清理GPU和CPU内存
                del save_features
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                logging.error(f"保存批次 {batch_idx + 1} 编码特征时出错: {e}")
                import traceback
                logging.error(traceback.format_exc())
    else:
        # 单批次处理所有图像
        encoded_features, valid_paths = process_image_batch(image_paths, vae, clip_model, args, input_size)
        
        if encoded_features is None or not valid_paths:
            logging.error("处理失败")
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
        
        # 准备元数据
        metadata = {
            "version": "1.0.0",
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "z_dim": args.z_dim,
            "vae_path": os.path.basename(args.vae_path),
            "clip_path": os.path.basename(args.clip_path),
            "input_size": args.input_size,
            "feature_shape": [f.shape for f in encoded_features] if isinstance(encoded_features, list) else encoded_features.shape
        }
        
        # 保存编码后的特征（确保移至CPU再保存）
        try:
            # 如果是列表，确保所有张量都在CPU上
            if isinstance(encoded_features, list):
                if len(encoded_features) == 1:
                    save_features = encoded_features[0].cpu()
                else:
                    save_features = [feat.cpu() for feat in encoded_features]
            else:
                save_features = encoded_features.cpu()
                
            torch.save({
                'features': save_features,
                'image_paths': valid_paths,
                'metadata': metadata
            }, args.output)
            logging.info(f"VAE编码特征已保存到: {args.output}")
        except Exception as e:
            logging.error(f"保存编码特征时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return
    
    # 总结统计 - 始终显示
    elapsed_time = time.time() - start_time
    logging.info(f"所有处理完成，总耗时: {elapsed_time:.2f} 秒，处理 {len(image_paths)} 张图像")

if __name__ == "__main__":
    main()
