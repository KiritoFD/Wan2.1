#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量下载Google图像的脚本
使用google-images-download库
"""

import os
import sys
import argparse
import logging
import json
import time
from pathlib import Path
from google_images_download import google_images_download

def parse_args():
    parser = argparse.ArgumentParser(description="从Google图片批量下载图像")
    parser.add_argument("keywords", type=str, nargs="?", default=None,
                      help="要搜索的关键词，多个关键词用逗号分隔")
    parser.add_argument("--keywords", dest="keywords_arg", type=str, default=None,
                      help="要搜索的关键词，多个关键词用逗号分隔")
    parser.add_argument("--keywords_from_file", "-kf", type=str, default=None,
                      help="从文件导入关键词（每行一个）")
    parser.add_argument("--output", "-o", type=str, default="downloads",
                      help="下载图片的输出目录")
    parser.add_argument("--limit", "-l", type=int, default=100,
                      help="每个关键词下载的图片数量")
    parser.add_argument("--format", "-f", type=str, default=None,
                      choices=["jpg", "gif", "png", "bmp", "svg", "webp", "ico", "raw"],
                      help="图片格式")
    parser.add_argument("--color", "-co", type=str, default=None,
                      choices=["red", "orange", "yellow", "green", "teal", "blue", 
                               "purple", "pink", "white", "gray", "black", "brown"],
                      help="图片颜色过滤")
    parser.add_argument("--size", "-s", type=str, default=None,
                      help="图片尺寸，例如：large, medium, icon, 或 >400*300")
    parser.add_argument("--type", "-t", type=str, default=None,
                      choices=["face", "photo", "clip-art", "line-drawing", "animated"],
                      help="图片类型")
    parser.add_argument("--time", "-w", type=str, default=None,
                      choices=["past-24-hours", "past-7-days", "past-month", "past-year"],
                      help="图片上传时间")
    parser.add_argument("--delay", "-d", type=float, default=0.5,
                      help="下载两张图片之间的延迟时间（秒）")
    parser.add_argument("--prefix", "-pr", type=str, default=None,
                      help="图片名称前缀")
    parser.add_argument("--proxy", "-px", type=str, default=None,
                      help="代理服务器设置，格式为'IP:Port'")
    parser.add_argument("--specific_site", "-ss", type=str, default=None,
                      help="仅从特定网站下载图片")
    parser.add_argument("--safe_search", "-sa", action="store_true",
                      help="开启安全搜索")
    parser.add_argument("--no_download", "-nd", action="store_true",
                      help="不下载图片，只打印URL")
    parser.add_argument("--print_urls", "-p", action="store_true",
                      help="打印图片URL")
    parser.add_argument("--print_size", "-ps", action="store_true",
                      help="打印图片尺寸")
    parser.add_argument("--extract_metadata", "-e", action="store_true",
                      help="提取并保存图片元数据")
    parser.add_argument("--thumbnail", "-th", action="store_true",
                      help="下载缩略图")
    parser.add_argument("--language", "-la", type=str, default="English",
                      help="搜索语言")
    parser.add_argument("--chromedriver", "-cd", type=str, default=None,
                      help="chromedriver路径")
    parser.add_argument("--config_file", "-cf", type=str, default=None,
                      help="配置文件路径")
    parser.add_argument("--threads", type=int, default=4,
                      help="下载使用的线程数")
    parser.add_argument("--silent", "-sil", action="store_true",
                      help="静默模式，不打印通知信息")
    
    args = parser.parse_args()
    
    # 优先使用位置参数中的keywords
    if args.keywords is None and args.keywords_arg is None and args.keywords_from_file is None and args.config_file is None:
        parser.error("必须提供关键词参数、关键词文件或配置文件")
    elif args.keywords is None and args.keywords_arg is not None:
        args.keywords = args.keywords_arg
        
    return args

def build_arguments(args):
    """构建google_images_download接受的参数字典"""
    arguments = {
        "keywords": args.keywords,
        "output_directory": args.output,
        "limit": args.limit,
        "delay": args.delay,
        "print_urls": args.print_urls,
        "print_size": args.print_size,
        "extract_metadata": args.extract_metadata,
        "no_download": args.no_download,
        "safe_search": args.safe_search,
    }
    
    # 添加可选参数
    if args.keywords_from_file:
        arguments["keywords_from_file"] = args.keywords_from_file
    if args.format:
        arguments["format"] = args.format
    if args.color:
        arguments["color"] = args.color
    if args.size:
        arguments["size"] = args.size
    if args.type:
        arguments["type"] = args.type
    if args.time:
        arguments["time"] = args.time
    if args.prefix:
        arguments["prefix"] = args.prefix
    if args.proxy:
        arguments["proxy"] = args.proxy
    if args.specific_site:
        arguments["specific_site"] = args.specific_site
    if args.thumbnail:
        arguments["thumbnail"] = True
    if args.language:
        arguments["language"] = args.language
    if args.chromedriver:
        arguments["chromedriver"] = args.chromedriver
    if args.silent:
        arguments["silent_mode"] = True
        
    return arguments

def load_config_file(config_path):
    """从配置文件加载参数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置文件时出错: {e}")
        sys.exit(1)

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析命令行参数
    args = parse_args()
    
    # 检查库是否已安装
    try:
        import google_images_download
    except ImportError:
        logging.error("找不到google_images_download库，请使用pip安装: pip install google_images_download")
        sys.exit(1)
    
    start_time = time.time()
    
    # 创建下载器实例
    downloader = google_images_download.googleimagesdownload()
    
    # 从配置文件加载或使用命令行参数
    if args.config_file:
        arguments = load_config_file(args.config_file)
        logging.info(f"从配置文件加载参数: {args.config_file}")
    else:
        arguments = build_arguments(args)
    
    # 输出将要下载的内容信息
    keywords = arguments.get("keywords") or "从文件加载的关键词"
    limit = arguments.get("limit", 100)
    output_dir = arguments.get("output_directory", "downloads")
    
    if not arguments.get("silent_mode"):
        logging.info(f"准备下载关键词 '{keywords}' 的图片")
        logging.info(f"每个关键词下载 {limit} 张图片")
        logging.info(f"输出目录: {output_dir}")
    
    try:
        # 执行下载
        paths = downloader.download(arguments)
        
        # 计算下载总数
        total_images = 0
        if paths:
            for keyword in paths:
                if isinstance(paths[keyword], list):
                    total_images += len(paths[keyword])
        
        elapsed_time = time.time() - start_time
        if not arguments.get("silent_mode"):
            logging.info(f"下载完成! 共下载 {total_images} 张图片，耗时 {elapsed_time:.2f} 秒")
            
        # 返回下载的文件路径
        return paths
        
    except Exception as e:
        logging.error(f"下载过程中出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()
