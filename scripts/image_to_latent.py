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
    # 修改风格转换相关参数，使其默认启用
    parser.add_argument("--skip_style_transfer", action="store_true",
                        help="跳过风格转换处理")
    parser.add_argument("--style_model", type=str, default=None,
                        help="风格转换模型路径或名称，默认使用预设模型")
    parser.add_argument("--style_strength", type=float, default=0.8,
                        help="风格转换强度 (0.0-1.0)，默认0.8")
    parser.add_argument("--styled_output", type=str, default=None,
                        help="风格转换后的输出路径，默认自动生成")
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
    
    # 分批处理CLIP特征提取，减少内存占用
    all_clip_features = []
    all_valid_paths = []
    
    # 创建进度条
    pbar = tqdm(total=len(image_paths_batch), desc=f"{batch_prefix}：CLIP特征提取", disable=not args.verbose and args.no_progress)
    
    for i in range(0, len(image_paths_batch), clip_batch_size):
        # 每批次处理
        batch_paths = image_paths_batch[i:i + clip_batch_size]
        try:
            mini_features, mini_paths = process_images_with_clip(
                clip_model, batch_paths, args.device, 
                clip_batch_size, input_size,
                show_progress=False  # 使用外部进度条
            )
            if mini_features is not None:
                all_clip_features.append(mini_features)
                all_valid_paths.extend(mini_paths)
            
            # 更新总进度条
            pbar.update(len(batch_paths))
            
            # 立即清理内存
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()
                
        except Exception as e:
            logging.error(f"{batch_prefix}：处理CLIP批次 {i//clip_batch_size} 时出错: {e}")
            # 继续处理下一批
            pbar.update(len(batch_paths))
    
    pbar.close()
    
    # 合并所有CLIP特征
    if all_clip_features:
        try:
            clip_features = torch.cat(all_clip_features, dim=0)
            valid_paths = all_valid_paths
            
            # 立即清理中间结果
            del all_clip_features, all_valid_paths
        except Exception as e:
            logging.error(f"{batch_prefix}：合并CLIP特征时出错: {e}")
            return None, None
    else:
        logging.error(f"{batch_prefix}：无法提取任何CLIP特征")
        return None, None
    
    image_paths_batch = valid_paths  # 更新为有效路径
    
    # 输出处理信息
    if args.verbose:
        logging.info(f"{batch_prefix}：成功处理图像：{len(image_paths_batch)} 张")
    else:
        logging.info(f"{batch_prefix}：CLIP特征处理完成，有效图像 {len(image_paths_batch)} 张")
    
    # 保存CLIP特征（如果需要）
    if args.save_clip_features:
        clip_features_path = args.output.replace(".pt", f"_clip_features.pt") if args.output else os.path.join(
            os.path.dirname(args.image_path) if os.path.isfile(args.image_path) else args.image_path,
            "clip_features.pt"
        )
        # CPU保存，节省显存
        torch.save(clip_features.cpu(), clip_features_path)
        logging.info(f"{batch_prefix}：CLIP特征已保存到: {clip_features_path}")
    
    # 释放CLIP特征以节省内存
    del clip_features
    torch.cuda.empty_cache()

    # 处理图像并转换为VAE可接受的输入格式 - 使用更小批次以减少内存占用
    try:
        # 减小子批量大小以降低内存使用
        sub_batch_size = min(4, len(image_paths_batch)) if args.device.startswith('cuda') else min(8, len(image_paths_batch))
        if args.memory_efficient:
            sub_batch_size = max(1, sub_batch_size // 2)  # 更保守的设置
            
        # 使用生成器而非保存所有处理图像，逐批次处理
        def image_batch_generator(paths, batch_size):
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_tensors = []
                valid_indices = []
                
                for j, img_path in enumerate(batch_paths):
                    try:
                        # 加载并预处理图像
                        img = Image.open(img_path).convert("RGB")
                        if args.unified_size:
                            img = check_image_size(img, args.max_resolution)
                            h, w = args.unified_size
                            img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
                        elif input_size:
                            h, w = input_size
                            img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
                        else:
                            img_resized = check_image_size(img, args.max_resolution)
                            
                        # 转换为张量
                        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
                        img_tensor = img_tensor * 2 - 1
                        img_tensor = img_tensor.unsqueeze(1)  # 添加时间维度 [C, T, H, W]
                        batch_tensors.append(img_tensor)
                        valid_indices.append(i + j)
                        
                        # 立即释放内存
                        del img, img_resized
                        
                    except Exception as e:
                        logging.error(f"处理图像 {img_path} 时出错: {e}")
                        continue
                
                if batch_tensors:
                    yield batch_tensors, valid_indices
                    
                # 每批次结束后清理内存
                del batch_tensors
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()

        # 使用VAE编码特征 - 优化为流水线处理
        if args.verbose:
            logging.info(f"{batch_prefix}：使用VAE编码特征，采用流水线处理...")
        else:
            logging.info(f"{batch_prefix}：开始VAE编码，共 {len(image_paths_batch)} 张图像")
        
        # 确定是否使用CPU进行VAE编码
        use_cpu = args.force_cpu_vae
        vae_device = torch.device('cpu') if use_cpu else args.device
        
        # 保存原始VAE设备
        if use_cpu:
            vae_original_device = vae.device
            vae.model = vae.model.to(torch.device('cpu'))
            vae.device = torch.device('cpu')
        
        # 创建进度条
        vae_pbar = tqdm(total=len(image_paths_batch), desc=f"{batch_prefix}：VAE编码", disable=args.no_progress)
        
        # 分批处理和编码
        encoded_features = []
        all_valid_indices = []
        
        for batch_tensors, valid_indices in image_batch_generator(image_paths_batch, sub_batch_size):
            try:
                # 编码当前批次
                with torch.no_grad():
                    # 将批次移动到设备
                    device_tensors = [tensor.to(vae_device) for tensor in batch_tensors]
                    
                    # VAE编码
                    batch_features = vae.encode(device_tensors)
                    
                    # 对每个特征调整大小
                    for i, feat in enumerate(batch_features):
                        # 调整潜在向量大小
                        feat = F.interpolate(
                            feat, 
                            size=args.unified_latent_size, 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # 移回CPU
                        if not use_cpu:
                            feat = feat.cpu()
                        
                        encoded_features.append(feat)
                
                # 记录有效索引
                all_valid_indices.extend(valid_indices)
                
                # 更新进度条
                vae_pbar.update(len(batch_tensors))
                
                # 立即释放内存
                del device_tensors, batch_features
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError:
                # 如果GPU内存不足，尝试在CPU上处理
                logging.warning(f"{batch_prefix}：GPU内存不足，尝试在CPU上处理批次")
                try:
                    # 临时将VAE移至CPU
                    temp_vae_device = vae.device
                    vae.model = vae.model.to(torch.device('cpu'))
                    vae.device = torch.device('cpu')
                    
                    with torch.no_grad():
                        # 在CPU上处理
                        cpu_tensors = [tensor.to('cpu') for tensor in batch_tensors]
                        batch_features = vae.encode(cpu_tensors)
                        
                        # 调整大小并保存
                        for i, feat in enumerate(batch_features):
                            feat = F.interpolate(
                                feat, 
                                size=args.unified_latent_size, 
                                mode='bilinear', 
                                align_corners=False
                            )
                            encoded_features.append(feat)
                    
                    # 恢复VAE设备
                    vae.model = vae.model.to(temp_vae_device)
                    vae.device = temp_vae_device
                    
                    # 记录有效索引
                    all_valid_indices.extend(valid_indices)
                    
                    # 更新进度条
                    vae_pbar.update(len(batch_tensors))
                    
                    # 清理内存
                    del cpu_tensors, batch_features
                    gc.collect()
                    
                except Exception as e:
                    logging.error(f"{batch_prefix}：在CPU上处理批次时失败: {e}")
                    # 继续处理下一批
            
            except Exception as e:
                logging.error(f"{batch_prefix}：处理VAE批次时出错: {e}")
                # 继续处理下一批
        
        vae_pbar.close()
        
        # 如果使用了CPU，将VAE模型恢复到原始设备
        if use_cpu:
            vae.model = vae.model.to(vae_original_device)
            vae.device = vae_original_device
        
        # 如果没有成功处理任何图像，返回错误
        if not encoded_features:
            logging.error(f"{batch_prefix}：无法成功编码任何图像")
            return None, None
        
        # 获取有效路径
        valid_paths = [image_paths_batch[i] for i in all_valid_indices]
        
        # 检查并确保所有特征形状一致
        if len(encoded_features) > 1:
            expected_shape = (args.z_dim, 1, args.unified_latent_size[0], args.unified_latent_size[1])
            
            # 验证并修复形状
            for i in range(len(encoded_features)):
                feat = encoded_features[i]
                if feat.shape != expected_shape:
                    # 修复形状
                    if feat.shape[0] != expected_shape[0]:
                        logging.warning(f"特征 {i} 的通道数不匹配，无法修复")
                        continue
                    
                    # 确保时间维度正确
                    if feat.shape[1] != 1:
                        feat = feat.mean(dim=1, keepdim=True)
                    
                    # 确保空间维度正确
                    feat = F.interpolate(
                        feat, 
                        size=(args.unified_latent_size[0], args.unified_latent_size[1]),
                        mode='bilinear',
                        align_corners=False
                    )
                    encoded_features[i] = feat
        
    except Exception as e:
        logging.error(f"{batch_prefix}：处理图像时出错: {e}")
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
    processed_outputs = []  # 跟踪所有处理后的文件路径
    
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
    
    # 在处理完成后，显示最终输出文件信息
    if processed_outputs:
        logging.info("\n处理结果摘要:")
        for idx, output_file in enumerate(processed_outputs):
            logging.info(f"[{idx+1}/{len(processed_outputs)}] {output_file}")
    
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

def apply_style_transfer(features_path, style_model_path=None, output_path=None, strength=0.8, device=None):
    """
    应用风格转换模型处理VAE编码特征
    
    参数:
        features_path: VAE编码特征文件路径
        style_model_path: 风格转换模型路径或模型名称，默认使用预设模型
        output_path: 输出文件路径，若为None则自动生成
        strength: 风格转换强度，0-1之间，默认0.8
        device: 计算设备，默认为自动选择
    """
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # 重要调试输出 - 显示Python路径
    logging.info("DEBUG - Python路径:")
    for p in sys.path:
        logging.info(f"  - {p}")
    
    # 固定模型根目录
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'style_transfer'))
    logging.info(f"DEBUG - 风格模型目录: {MODELS_DIR}")
    
    # 检查并创建模型目录
    if not os.path.exists(MODELS_DIR):
        logging.warning(f"风格模型目录不存在，正在创建: {MODELS_DIR}")
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            logging.info(f"成功创建目录: {MODELS_DIR}")
        except Exception as e:
            logging.error(f"创建目录失败: {e}")
    else:
        logging.info(f"风格模型目录已存在")
        # 列出目录内容
        try:
            files = os.listdir(MODELS_DIR)
            if files:
                logging.info(f"目录中的文件: {', '.join(files)}")
            else:
                logging.warning("目录为空，没有可用的模型文件")
        except Exception as e:
            logging.error(f"无法列出目录内容: {e}")
    
    # 默认模型文件
    DEFAULT_MODEL = "best_model.pth"
    
    # 处理模型路径
    if style_model_path is None:
        # 使用默认模型
        style_model_path = os.path.join(MODELS_DIR, DEFAULT_MODEL)
        logging.info(f"使用默认风格模型路径: {style_model_path}")
        # 检查默认模型是否存在
        if not os.path.exists(style_model_path):
            logging.error(f"默认风格模型不存在: {style_model_path}")
            logging.info("尝试在当前目录查找模型文件...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(current_dir, DEFAULT_MODEL)
            if os.path.exists(alt_path):
                logging.info(f"在当前目录找到模型: {alt_path}")
                style_model_path = alt_path
            else:
                logging.error("在所有可能的位置都未找到模型文件")
                return None
    elif not os.path.exists(style_model_path):
        # 尝试将输入视为模型名称，在预设目录中查找
        logging.info(f"指定模型不存在，尝试在预设目录查找: {style_model_path}")
        potential_path = os.path.join(MODELS_DIR, style_model_path)
        logging.info(f"尝试路径: {potential_path}")
        
        if os.path.exists(potential_path):
            style_model_path = potential_path
            logging.info(f"找到模型: {style_model_path}")
        else:
            # 尝试添加.pth后缀再查找
            potential_path_with_ext = os.path.join(MODELS_DIR, f"{style_model_path}.pth")
            logging.info(f"尝试带扩展名路径: {potential_path_with_ext}")
            
            if os.path.exists(potential_path_with_ext):
                style_model_path = potential_path_with_ext
                logging.info(f"找到带扩展名的模型: {style_model_path}")
            else:
                logging.error(f"找不到风格模型: {style_model_path}")
                # 最后一搏：搜索所有包含关键词的文件
                try:
                    all_files = []
                    for root, _, files in os.walk(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))):
                        for file in files:
                            if file.endswith('.pth'):
                                all_files.append(os.path.join(root, file))
                    
                    if all_files:
                        logging.info(f"系统中找到以下.pth文件:")
                        for f in all_files:
                            logging.info(f"  - {f}")
                    else:
                        logging.warning("系统中未找到任何.pth文件")
                except Exception as e:
                    logging.error(f"搜索文件时出错: {e}")
                
                return None

    # 记录详细信息
    logging.info(f"最终选择的模型路径: {style_model_path}")
    if os.path.exists(style_model_path):
        try:
            file_size_mb = os.path.getsize(style_model_path) / (1024*1024)
            logging.info(f"模型文件大小: {file_size_mb:.2f} MB")
        except:
            pass
    else:
        logging.error(f"严重错误：最终选定的模型文件不存在!")
    
    try:
        logging.info("尝试导入风格转换API...")
        from style_transfer.style_transfer_api import StyleTransferAPI
        logging.info("成功导入StyleTransferAPI")
    except ImportError as e:
        logging.error(f"导入风格转换模块失败: {e}")
        logging.info("尝试检查style_transfer模块是否存在...")
        
        # 检查模块目录
        module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'style_transfer'))
        if os.path.exists(module_dir):
            logging.info(f"style_transfer目录存在: {module_dir}")
            try:
                files = os.listdir(module_dir)
                logging.info(f"目录内容: {', '.join(files)}")
            except:
                pass
        else:
            logging.error(f"style_transfer目录不存在: {module_dir}")
        
        return None
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info(f"开始应用风格转换，模型: {style_model_path}, 强度: {strength}")
    
    # 加载VAE编码特征
    features_data = load_vae_features(features_path)
    if features_data is None:
        logging.error("无法加载VAE特征")
        return None
    
    # 准备输出路径
    if output_path is None:
        base, ext = os.path.splitext(features_path)
        style_name = os.path.splitext(os.path.basename(style_model_path))[0]
        output_path = f"{base}_style_{style_name}{ext}"
    
    # 初始化风格转换模型
    try:
        style_api = StyleTransferAPI(style_model_path, device=device)
        logging.info(f"风格转换模型已加载: {style_model_path}")
    except Exception as e:
        logging.error(f"加载风格转换模型失败: {e}")
        return None
    
    # 应用风格转换
    try:
        features = features_data['features']
        paths = features_data.get('image_paths', [])
        metadata = features_data.get('metadata', {})
        
        # 处理不同的特征格式
        if isinstance(features, list):
            logging.info(f"处理列表格式特征，共 {len(features)} 个样本")
            styled_features = []
            # 批量处理，降低内存占用
            batch_size = 8 if device == "cuda" else 4
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size]
                # 将列表中的张量堆叠成批次
                if all(f.dim() == 4 for f in batch):  # [C,T,H,W]
                    batch_tensor = torch.stack(batch)
                    styled_batch = style_api.transfer_style(batch_tensor, strength=strength)
                    styled_features.extend(list(styled_batch))
                else:
                    # 单独处理每个样本
                    for f in batch:
                        styled = style_api.transfer_style(f, strength=strength)
                        styled_features.append(styled)
                        
                logging.info(f"已处理 {min(i+batch_size, len(features))}/{len(features)} 个样本")
        else:
            logging.info(f"处理张量格式特征，形状: {features.shape}")
            styled_features = style_api.transfer_style(features, strength=strength)
        
        # 更新元数据
        new_metadata = metadata.copy() if metadata else {}
        new_metadata.update({
            "style_processed": True,
            "style_model": os.path.basename(style_model_path),
            "style_strength": strength,
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 保存结果
        output_data = {
            'features': styled_features,
            'image_paths': paths,
            'metadata': new_metadata,
            'original_features_path': features_path
        }
        
        logging.info(f"保存风格转换结果到: {output_path}")
        torch.save(output_data, output_path)
        return output_path
        
    except Exception as e:
        logging.error(f"应用风格转换时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

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