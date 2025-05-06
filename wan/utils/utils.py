# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
from typing import Union, List, Tuple, Optional

import imageio
import torch
import torchvision
import numpy as np
from PIL import Image

__all__ = ['cache_video', 'cache_image', 'str2bool', 'save_video', 'load_video']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


def save_video(tensor: torch.Tensor,
               save_file: str,
               fps: int = 16,
               nrow: int = 1,
               normalize: bool = True,
               value_range: Tuple[float, float] = (-1, 1),
               retry: int = 5) -> Optional[str]:
    """
    保存视频张量到文件。
    
    参数:
        tensor (torch.Tensor): 要保存的视频张量，形状为 [B, C, T, H, W]
        save_file (str): 保存路径
        fps (int, optional): 帧率. 默认为 16.
        nrow (int, optional): 如果批次大于1，每行的图像数. 默认为 1.
        normalize (bool, optional): 是否归一化. 默认为 True.
        value_range (tuple, optional): 归一化的值范围. 默认为 (-1, 1).
        retry (int, optional): 重试次数. 默认为 5.
    
    返回:
        str: 保存的文件路径，如果失败返回None
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(save_file)), exist_ok=True)
    return cache_video(tensor, save_file, fps, '.mp4', nrow, normalize, value_range, retry)


def load_video(video_path: str, 
               target_frames: Optional[int] = None, 
               start_frame: int = 0,
               target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    加载视频文件到张量
    
    参数:
        video_path (str): 视频文件路径
        target_frames (int, optional): 目标帧数，如果指定，将均匀采样到这个帧数
        start_frame (int, optional): 起始帧. 默认为 0.
        target_size (tuple, optional): 目标大小 (H, W)
    
    返回:
        torch.Tensor: 视频张量，形状为 [C, T, H, W]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    reader = imageio.get_reader(video_path)
    
    # 获取视频信息
    frames = []
    for i, frame in enumerate(reader):
        if i < start_frame:
            continue
        frames.append(frame)
    reader.close()
    
    # 如果需要特定帧数进行重采样
    if target_frames is not None and len(frames) > 0:
        # 均匀采样
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    if not frames:
        raise ValueError(f"未能加载任何帧: {video_path}")
    
    # 将帧列表转换为张量
    video_array = np.stack(frames, axis=0)
    
    # PIL调整大小
    if target_size is not None:
        resized_frames = []
        for frame in video_array:
            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize(target_size[::-1])  # PIL要求 (W, H)
            resized_frames.append(np.array(pil_img))
        video_array = np.stack(resized_frames, axis=0)
    
    # 转为torch张量并调整通道顺序 [T, H, W, C] -> [C, T, H, W]
    video_tensor = torch.from_numpy(video_array).float().permute(3, 0, 1, 2)
    
    # 归一化到 [0, 1]
    video_tensor = video_tensor / 255.0
    
    return video_tensor
