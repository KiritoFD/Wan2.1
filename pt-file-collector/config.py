#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置管理模块 - 集中管理所有配置项
"""

import os
import argparse
from typing import List, Tuple, Dict, Any, Optional

# 默认配置
CONFIG = {
    # 搜索引擎和图片源配置
    "sources": [
        "google", "bing", "flickr", "bilibili", "pixiv", 
        "twitter", "pinterest", "unsplash", "500px", "instagram"
    ],
    
    # 线程和并发配置
    "max_threads": 32,
    "download_timeout": 15,
    "request_timeout": 30,
    
    # 图片质量和去重配置
    "min_image_size": 10000,
    "min_resolution": (200, 200),
    "duplicate_threshold": 90,
    "enable_deduplication": True,
    "sort_by_quality": True,
    
    # 网络请求配置
    "retry_count": 3,           # 增加重试次数
    "retry_delay": (2, 5),      # 增加重试延迟
    "page_delay": (3, 6),       # 增加页面请求间隔
    
    # 错误处理配置
    "fallback_to_alternatives": True,  # 当主要方法失败时使用备选方法
    "skip_problematic_sources": True,  # 跳过持续失败的源
    
    # 代理配置 (可选，根据需要启用)
    "use_proxy": False,
    "proxies": {
        "http": "",
        "https": ""
        # 例如: "http": "http://127.0.0.1:7890"
    },
    
    # 图片命名配置
    "use_smart_naming": True,
    "image_prefix": "img",
    "name_by_source": True,
    
    # 用户代理配置
    "user_agents": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
    ]
}

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从多个图片源搜索并下载图片')
    parser.add_argument('-q', '--query', type=str, required=True, 
                        help='搜索关键词')
    parser.add_argument('-n', '--number', type=int, default=50,
                        help=f'要下载的图片数量 (默认: 50)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='保存图片的目录 (默认: 以关键词命名)')
    parser.add_argument('-s', '--sources', type=str, nargs='+',
                        choices=CONFIG["sources"],
                        default=None,
                        help=f'指定要使用的图片源 (默认: 使用所有图片源)')
    parser.add_argument('-t', '--threads', type=int, default=CONFIG["max_threads"],
                        help=f'最大下载线程数 (默认: {CONFIG["max_threads"]})')
    parser.add_argument('-d', '--no-dedup', action='store_true',
                        help='禁用图片去重功能')
    parser.add_argument('-r', '--min-resolution', type=str, 
                        default=f"{CONFIG['min_resolution'][0]}x{CONFIG['min_resolution'][1]}",
                        help=f'最小图片分辨率，格式为"宽x高" (默认: {CONFIG["min_resolution"][0]}x{CONFIG["min_resolution"][1]})')
    parser.add_argument('-m', '--min-size', type=int, default=CONFIG["min_image_size"],
                        help=f'最小文件大小，单位为字节 (默认: {CONFIG["min_image_size"]})')
    parser.add_argument('--prefix', type=str, default=CONFIG["image_prefix"],
                        help=f'图片文件名前缀 (默认: {CONFIG["image_prefix"]})')
    parser.add_argument('--no-smart-naming', action='store_true',
                        help='禁用智能命名功能')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.threads:
        CONFIG["max_threads"] = args.threads
        
    if args.min_size:
        CONFIG["min_image_size"] = args.min_size
        
    if args.no_dedup:
        CONFIG["enable_deduplication"] = False
        
    if args.min_resolution:
        try:
            width, height = map(int, args.min_resolution.lower().split('x'))
            CONFIG["min_resolution"] = (width, height)
        except:
            print(f"警告: 无效的分辨率格式 '{args.min_resolution}'，使用默认值 {CONFIG['min_resolution']}")
    
    if args.prefix:
        CONFIG["image_prefix"] = args.prefix
    
    if args.no_smart_naming:
        CONFIG["use_smart_naming"] = False
    
    return args
