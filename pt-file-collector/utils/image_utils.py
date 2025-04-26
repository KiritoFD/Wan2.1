#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像处理工具模块 - 处理图像哈希、质量评估等
"""

import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image

def calculate_image_hash(img_data: bytes) -> Optional[str]:
    """计算图片的感知哈希值，用于相似性比较"""
    try:
        img = Image.open(io.BytesIO(img_data))
        # 转换为灰度图并缩放到8x8
        img = img.convert("L").resize((8, 8), Image.LANCZOS)
        
        # 计算平均值
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        
        # 生成hash值 (0和1组成的64位字符串)
        hash_value = ''.join('1' if pixel >= avg else '0' for pixel in pixels)
        return hash_value
    except Exception as e:
        print(f"计算图片哈希值时出错: {str(e)}")
        return None

def hamming_distance(hash1: Optional[str], hash2: Optional[str]) -> int:
    """计算两个哈希值之间的汉明距离"""
    if hash1 is None or hash2 is None:
        return 100
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def estimate_image_quality(img_data: bytes) -> float:
    """估计图片的质量分数，考虑分辨率和文件大小"""
    try:
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        resolution_score = width * height / 1000000  # 百万像素
        filesize_score = len(img_data) / 100000     # 文件大小(100KB为单位)
        
        # 组合得分 (70%分辨率 + 30%文件大小)
        quality_score = 0.7 * resolution_score + 0.3 * filesize_score
        return quality_score
    except Exception:
        return 0

def get_image_info(img_data: bytes) -> Dict[str, Any]:
    """获取图片的基本信息"""
    try:
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        format_name = img.format or "UNKNOWN"
        mode = img.mode
        return {
            "width": width,
            "height": height,
            "format": format_name,
            "mode": mode,
            "size_bytes": len(img_data),
            "aspect_ratio": width / height if height > 0 else 0
        }
    except Exception as e:
        print(f"获取图片信息时出错: {str(e)}")
        return {}

def is_valid_image(img_data: bytes, min_size: int, min_resolution: Tuple[int, int]) -> bool:
    """检查图片是否符合最低质量要求"""
    if len(img_data) < min_size:
        return False
        
    try:
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        if width < min_resolution[0] or height < min_resolution[1]:
            return False
        return True
    except:
        return False

def generate_filename_from_content(img_data: bytes, prefix: str = "img", 
                                  source: str = "", metadata: Dict = None) -> str:
    """根据图片内容生成文件名"""
    try:
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        format_name = img.format.lower() if img.format else "jpg"
        
        parts = [prefix]
        
        # 添加源信息
        if source:
            parts.append(source)
            
        # 添加元数据
        if metadata:
            if "title" in metadata and metadata["title"]:
                title = metadata["title"].replace(" ", "_")[:20]
                parts.append(title)
                
            if "artist" in metadata and metadata["artist"]:
                artist = metadata["artist"].replace(" ", "_")[:15]
                parts.append(artist)
        
        # 添加分辨率
        parts.append(f"{width}x{height}")
        
        # 添加哈希的前8位，确保唯一性
        img_hash = calculate_image_hash(img_data)
        if img_hash:
            parts.append(img_hash[:8])
            
        # 组合文件名
        filename = "_".join(parts) + f".{format_name}"
        
        # 确保文件名合法且不过长
        filename = "".join(c for c in filename if c.isalnum() or c in "_.-")
        if len(filename) > 255:
            filename = filename[:240] + "." + format_name
            
        return filename
    except Exception as e:
        print(f"生成文件名时出错: {str(e)}")
        return f"{prefix}_{hash(img_data) % 10000:04d}.jpg"
