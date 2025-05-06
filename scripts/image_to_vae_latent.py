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
import gc
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
    parser.add_argument("--clip_path", type=str, default="Wan2.1-T2V-14B\clip_vit_h_14.pth",
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
                logging.debug(f"尝试加载图像: {img_path}")
                img = Image.open(img_path).convert("RGB")
                logging.debug(f"成功加载图像: {img_path}, 尺寸: {img.size}")
                
                # 如果指定了输入大小，调整图像尺寸
                if input_size:
                    img = img.resize(input_size, Image.Resampling.LANCZOS)
                    logging.debug(f"调整图像大小至: {input_size}")
                
                # 检查图像是否有效
                if img.size[0] <= 0 or img.size[1] <= 0:
                    raise ValueError(f"图像尺寸无效: {img.size}")
                
                # 转换为张量 - 归一化到[-1, 1]以匹配VAE预期
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor * 2 - 1
                
                # 检查张量是否有效
                if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
                    raise ValueError(f"图像张量包含无效值(NaN或Inf)")
                    
                images.append(img_tensor)
                valid_paths.append(img_path)
                logging.debug(f"图像 {img_path} 成功转换为张量，形状: {img_tensor.shape}")
            except Exception as e:
                logging.error(f"处理图像 {img_path} 时出错: {str(e)}", exc_info=True)
                continue
                
        # 更新进度条
        pbar.update(len(batch_paths))
        if not images:    
            continue
            
        # 准备CLIP输入 - 直接在目标设备上创建批量
        try:
            # 打印CLIP模型信息
            if i == 0:  # 仅在第一批次打印
                logging.info(f"CLIP模型信息: 类型={type(clip_model).__name__}, 设备={device}")
                try:
                    logging.info(f"视觉编码器: {type(clip_model.visual_encoder).__name__}")
                except:
                    logging.info("无法获取视觉编码器信息")
                    
            logging.debug(f"开始处理批次 {i//batch_size}, 包含 {len(images)} 张图像")
            
            # 添加时间维度并移动到设备
            videos = []
            for idx, img in enumerate(images):
                try:
                    video = img.unsqueeze(1).to(device)  # 添加时间维度
                    videos.append(video)
                except Exception as dev_err:
                    logging.error(f"将图像 {valid_paths[idx]} 移动到设备时失败: {str(dev_err)}", exc_info=True)
            
            if not videos:
                logging.error(f"批次 {i//batch_size} 中没有可用图像")
                continue
                
            # 检查输入形状
            logging.debug(f"输入到CLIP的第一个张量形状: {videos[0].shape}")
            
            with torch.no_grad():
                try:
                    # 尝试一次处理一张图像，找出问题图像
                    batch_features = []
                    batch_valid_paths = []
                    
                    for j, (video, path) in enumerate(zip(videos, valid_paths)):
                        try:
                            # 将单张图像作为列表传递给CLIP
                            logging.debug(f"处理单张图像: {path}")
                            feature = clip_model.visual([video])
                            logging.debug(f"CLIP特征提取成功: {path}, 特征形状: {feature.shape}")
                            
                            # 如果特征是三维的，转换为二维
                            if len(feature.shape) == 3:
                                feature = feature.mean(dim=1)
                                
                            batch_features.append(feature.cpu())
                            batch_valid_paths.append(path)
                        except Exception as single_err:
                            logging.error(f"处理图片 {path} 时出错: {str(single_err)}", exc_info=True)
                    
                    # 如果成功处理了任何图像
                    if batch_features:
                        # 合并所有特征
                        features = torch.cat(batch_features, dim=0)
                        all_features.append((features, batch_valid_paths))
                        logging.debug(f"批次 {i//batch_size} 成功处理 {len(batch_features)} 张图像")
                    else:
                        logging.error(f"批次 {i//batch_size} 所有图像处理失败")
                        
                except Exception as clip_err:
                    logging.error(f"CLIP特征提取错误: {str(clip_err)}", exc_info=True)
        except Exception as batch_err:
            logging.error(f"批次处理错误: {str(batch_err)}", exc_info=True)
    
    # 关闭进度条
    pbar.close()
    
    # 合并所有批次的特征和路径
    if all_features:
        all_feature_tensors = []
        all_valid_paths = []
        for features, paths in all_features:
            all_feature_tensors.append(features)
            all_valid_paths.extend(paths)
        
        try:
            # 合并所有特征
            logging.info(f"合并所有CLIP特征，总共 {len(all_feature_tensors)} 批次，{len(all_valid_paths)} 张图像")
            clip_features = torch.cat(all_feature_tensors, dim=0)
            logging.info(f"合并后特征形状: {clip_features.shape}")
            
            # 在这里添加保存CLIP特征的代码
            clip_output = os.path.join(os.path.dirname(image_paths[0]), "clip_features.pt")
            torch.save({
                'clip_features': clip_features,
                'image_paths': all_valid_paths,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, clip_output)
            logging.info(f"CLIP特征已保存到: {clip_output}")
            
            return clip_features, all_valid_paths
        except Exception as save_err:
            logging.error(f"保存或合并CLIP特征时出错: {str(save_err)}", exc_info=True)
            
            # 尝试保存未合并的特征
            try:
                emergency_output = os.path.join(os.path.dirname(image_paths[0]), "clip_features_emergency.pt")
                torch.save({
                    'clip_features_list': all_feature_tensors, 
                    'image_paths': all_valid_paths
                }, emergency_output)
                logging.info(f"紧急保存未合并的CLIP特征到: {emergency_output}")
            except Exception as emergency_err:
                logging.error(f"紧急保存也失败: {str(emergency_err)}", exc_info=True)
            
            # 即使合并失败，也尝试返回未合并的特征
            return torch.cat(all_feature_tensors, dim=0), all_valid_paths
    else:
        logging.error("未能提取任何CLIP特征，请检查图像文件和CLIP模型路径")
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

def safe_save_pt(data, file_path):
    """安全保存PT文件，避免数据损坏"""
    try:
        # 使用临时文件先保存，成功后再替换原文件
        temp_file = file_path + ".tmp"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 使用原子操作保存数据
        torch.save(data, temp_file)
        
        # 验证保存的文件是否可以正常加载
        try:
            test_load = torch.load(temp_file, map_location='cpu')
            # 验证数据完整性
            if 'features' not in test_load and 'features_chunk' not in test_load:
                raise ValueError("保存的数据缺少features或features_chunk字段")
        except Exception as e:
            logging.error(f"验证保存文件失败: {e}")
            raise
        
        # 成功验证后，替换原文件
        if os.path.exists(file_path):
            os.remove(file_path)
        os.rename(temp_file, file_path)
        
        return True
    except Exception as e:
        logging.error(f"安全保存文件失败: {e}")
        # 如果临时文件存在，尝试删除
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False

def process_image_batch(image_paths_batch, vae, clip_model, args, input_size=None, batch_index=None):
    """处理一批图像并返回VAE编码后的特征"""
    start_time = time.time()
    batch_prefix = f"批次 {batch_index}" if batch_index is not None else ""
    
    # 计算批次大小，根据可用内存动态调整
    clip_batch_size = min(args.clip_batch_size, 8) if args.device.startswith('cuda') else args.clip_batch_size
    
    # 使用CLIP处理图像 - 优化批次处理
    if args.verbose:
        logging.info(f"{batch_prefix}：使用CLIP提取图像特征...")
    else:
        # 始终打印此日志以便调试
        logging.info(f"{batch_prefix}：开始CLIP特征提取，共 {len(image_paths_batch)} 张图像")
    
    # 确保模型在正确的设备上
    try:
        # 检查CLIP模型
        logging.info("检查CLIP模型详细信息:")
        logging.info(f"CLIP模型类型: {type(clip_model).__name__}")
        
        if hasattr(clip_model, 'visual_encoder'):
            logging.info(f"视觉编码器类型: {type(clip_model.visual_encoder).__name__}")
            device_info = f"当前CLIP模型设备: {next(clip_model.visual_encoder.parameters()).device}"
            logging.info(f"{batch_prefix}: {device_info}")
        else:
            logging.warning("CLIP模型没有visual_encoder属性")
            # 尝试打印内部结构
            logging.info(f"CLIP模型属性: {dir(clip_model)}")
    except Exception as e:
        logging.warning(f"无法确定CLIP模型设备或结构: {str(e)}", exc_info=True)
    
    # 验证一些样本图片是否能被正确加载
    sample_paths = image_paths_batch[:min(3, len(image_paths_batch))]
    for path in sample_paths:
        try:
            img = Image.open(path).convert("RGB")
            logging.info(f"样本图片 {path} 尺寸: {img.size}, 格式: {img.format}, 模式: {img.mode}")
        except Exception as e:
            logging.error(f"加载样本图片 {path} 失败: {str(e)}", exc_info=True)
    
    # 使用CLIP提取特征
    logging.info(f"{batch_prefix}: 开始提取CLIP特征 - 批次大小: {clip_batch_size}")
    clip_features, valid_paths = process_images_with_clip(
        clip_model, image_paths_batch, args.device, 
        clip_batch_size, input_size, show_progress=not args.no_progress
    )
    
    if clip_features is None or not valid_paths:
        logging.error(f"{batch_prefix}：CLIP特征提取失败，跳过此批次")
        return None, None

    # ...existing code for VAE processing...

def main():
    # 修复torch.cuda.amp.autocast弃用警告
    fix_amp_warnings()
    # 解析命令行参数
    args = parse_args()
    
    # 导入gc模块，确保在所有函数中都可访问
    import gc
    
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
    
    # 构建输出文件名时避免重复计算
    def get_output_path(args, batch_idx=0, need_split=False):
        if args.output:
            if need_split:
                output_base = os.path.splitext(args.output)[0]
                output_ext = os.path.splitext(args.output)[1]
                return f"{output_base}_batch{batch_idx + 1}{output_ext}"
            else:
                return args.output
        else:
            if os.path.isdir(args.image_path):
                dir_name = os.path.basename(os.path.normpath(args.image_path))
                output_dir = args.image_path
                if need_split:
                    return os.path.join(output_dir, f"{dir_name}_vae_encoded_batch{batch_idx + 1}.pt")
                else:
                    return os.path.join(output_dir, f"{dir_name}_vae_encoded.pt")
            else:
                output_dir = os.path.dirname(args.image_path)
                base_filename = os.path.basename(args.image_path).split('.')[0]
                if need_split:
                    return os.path.join(output_dir, f"{base_filename}_vae_encoded_batch{batch_idx + 1}.pt")
                else:
                    return os.path.join(output_dir, f"{base_filename}_vae_encoded.pt")
    
    # 解析输入尺寸
    input_size = None
    if args.input_size:
        try:
            h, w = map(int, args.input_size.split(','))
            input_size = (w, h)  # PIL使用(width, height)格式
        except:
            logging.warning(f"无效的输入尺寸格式: {args.input_size}, 将使用原始尺寸")
    
    # 处理每一批图像
    for batch_idx in range(num_batches) if need_split else [0]:
        start_idx = batch_idx * args.images_per_file if need_split else 0
        end_idx = min(start_idx + args.images_per_file, len(image_paths)) if need_split else len(image_paths)
        batch_paths = image_paths[start_idx:end_idx]
        
        # 使用函数获取输出路径
        batch_output = get_output_path(args, batch_idx, need_split)
        
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
            "source": args.image_path,
            "file_count": len(batch_paths)
        }

        # 处理一批图像
        encoded_features, valid_paths = process_image_batch(batch_paths, vae, clip_model, args, input_size, batch_idx + 1)
        if encoded_features is None or not valid_paths:
            logging.error(f"批次 {batch_idx + 1} 处理失败，跳过")
            # 强制清理内存
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        # 改进的保存逻辑，使用多种策略防止文件损坏
        try:
            # 分块保存大型特征
            if isinstance(encoded_features, list) and len(encoded_features) > 100:
                logging.info(f"特征数量较多 ({len(encoded_features)}), 采用分块保存策略")
                
                # 准备保存数据
                features_data = None
                
                # 先尝试堆叠成单一张量
                try:
                    # 确保所有特征形状一致
                    expected_shape = encoded_features[0].shape
                    if not all(feat.shape == expected_shape for feat in encoded_features):
                        logging.warning("特征形状不一致，无法堆叠为单一张量")
                        raise ValueError("特征形状不一致")
                    
                    # 尝试将所有特征堆叠为单一张量
                    features_tensor = torch.stack([feat.cpu() for feat in encoded_features])
                    
                    # 保存完整数据
                    complete_data = {
                        'features': features_tensor,
                        'image_paths': valid_paths,
                        'metadata': metadata
                    }
                    
                    # 安全保存文件
                    save_success = safe_save_pt(complete_data, batch_output)
                    if not save_success:
                        raise ValueError("保存单一张量失败，尝试分块保存")
                        
                    # 释放内存
                    del features_tensor, complete_data
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                except Exception as stack_err:
                    logging.warning(f"堆叠特征失败: {stack_err}，尝试分块保存")
                    
                    # 元数据和路径
                    metadata_data = {
                        'metadata': metadata,
                        'image_paths': valid_paths
                    }
                    
                    # 使用临时文件
                    temp_file = batch_output + ".tmp"
                    torch.save(metadata_data, temp_file)
                    
                    # 分块保存特征
                    chunk_size = 20  # 更小的块大小，降低内存需求
                    num_chunks = (len(encoded_features) + chunk_size - 1) // chunk_size
                    
                    logging.info(f"开始分块保存，共 {num_chunks} 个块")
                    
                    # 分块保存，每个块单独保存
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min((chunk_idx + 1) * chunk_size, len(encoded_features))
                        
                        # 当前块的特征
                        chunk_features = encoded_features[start_idx:end_idx]
                        chunk_data = {
                            'features_chunk': [feat.cpu() for feat in chunk_features],
                            'chunk_idx': chunk_idx,
                            'chunk_start': start_idx,
                            'chunk_end': end_idx,
                            'total_chunks': num_chunks
                        }
                        
                        # 保存当前块
                        chunk_file = f"{batch_output}.chunk{chunk_idx}"
                        torch.save(chunk_data, chunk_file)
                        
                        # 释放内存
                        del chunk_features, chunk_data
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                        logging.info(f"保存块 {chunk_idx+1}/{num_chunks}")
                    
                    # 创建索引文件
                    index_data = {
                        'metadata': metadata,
                        'image_paths': valid_paths,
                        'is_chunked': True,
                        'total_chunks': num_chunks,
                        'chunk_prefix': batch_output,
                        'features_count': len(encoded_features)
                    }
                    
                    # 保存索引文件
                    if safe_save_pt(index_data, batch_output):
                        logging.info(f"分块保存成功，索引文件: {batch_output}, 块数: {num_chunks}")
                    else:
                        raise ValueError("保存索引文件失败")
                        
                    # 清理临时文件
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            else:
                # 常规保存方式 - 适用于少量特征
                if isinstance(encoded_features, list):
                    if len(encoded_features) == 1:
                        save_features = encoded_features[0].cpu()
                    else:
                        try:
                            save_features = torch.stack([feat.cpu() for feat in encoded_features])
                        except Exception as e:
                            logging.warning(f"无法堆叠特征: {e}，保留列表格式")
                            save_features = [feat.cpu() for feat in encoded_features]
                else:
                    save_features = encoded_features.cpu()
                
                # 保存完整数据
                save_data = {
                    'features': save_features,
                    'image_paths': valid_paths,
                    'metadata': metadata 
                }
                
                # 安全保存文件
                if safe_save_pt(save_data, batch_output):
                    logging.info(f"VAE编码特征已保存到: {batch_output}")
                else:
                    raise ValueError("保存文件失败")
            
            # 释放内存
            del encoded_features, valid_paths
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"保存批次 {batch_idx + 1} 编码特征时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            
            # 紧急备份保存 - 尝试最简化的保存方式
            try:
                emergency_output = f"{batch_output}.emergency.pt"
                logging.info(f"尝试紧急备份保存到: {emergency_output}")
                
                # 尽可能简化数据
                if isinstance(encoded_features, list):
                    simple_features = [f.cpu().detach() for f in encoded_features]
                else:
                    simple_features = encoded_features.cpu().detach()
                
                # 使用最简单的方式保存
                simple_data = {'features': simple_features, 'paths': valid_paths}
                torch.save(simple_data, emergency_output)
                logging.info(f"紧急备份保存成功: {emergency_output}")
            except Exception as backup_err:
                logging.error(f"紧急备份保存也失败: {backup_err}")
            
            continue  # 继续处理下一批
    
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

# 新增一个读取功能，同时支持常规和分块保存格式
def load_vae_features(file_path):
    """加载VAE特征，支持常规和分块保存格式"""
    try:
        # 加载文件
        data = torch.load(file_path, map_location='cpu')
        
        # 检查是否为分块格式
        if 'is_chunked' in data and data['is_chunked']:
            logging.info(f"检测到分块格式，共 {data['total_chunks']} 块")
            
            # 存储所有特征
            all_features = [None] * data['features_count']
            paths = data['image_paths']
            metadata = data['metadata']
            
            # 加载所有块
            chunk_prefix = data['chunk_prefix']
            for i in range(data['total_chunks']):
                chunk_file = f"{chunk_prefix}.chunk{i}"
                if not os.path.exists(chunk_file):
                    logging.error(f"块文件不存在: {chunk_file}")
                    continue
                
                # 加载当前块
                chunk_data = torch.load(chunk_file, map_location='cpu')
                chunk_features = chunk_data['features_chunk']
                start_idx = chunk_data['chunk_start']
                
                # 填充特征数组
                for j, feat in enumerate(chunk_features):
                    all_features[start_idx + j] = feat
            
            # 检查是否所有特征都已加载
            if None in all_features:
                missing = all_features.count(None)
                logging.warning(f"有 {missing}/{len(all_features)} 个特征缺失")
                # 去除缺失的特征
                all_features = [f for f in all_features if f is not None]
            
            # 返回加载的数据
            return {'features': all_features, 'image_paths': paths, 'metadata': metadata}
        else:
            # 常规格式，直接返回
            return data
            
    except Exception as e:
        logging.error(f"加载特征文件失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# 对主脚本添加一个新参数，用于测试加载保存的文件
if __name__ == "__main__":
    main()