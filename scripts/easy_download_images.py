#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用simple_image_download库的图像下载脚本
接收命令行参数：关键词和下载数量
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="下载Google图像")
    parser.add_argument("keywords", type=str, help="要搜索的关键词，多个关键词用逗号分隔")
    parser.add_argument("count", type=int, help="要下载的图片数量")
    parser.add_argument("--output", "-o", type=str, default="simple_images", help="下载图片的输出目录")
    parser.add_argument("--extensions", "-e", type=str, default=".jpg,.png,.ico,.gif,.jpeg", 
                      help="要下载的图片扩展名，用逗号分隔")
    args = parser.parse_args()
    
    try:
        # 检查库是否已安装
        try:
            from simple_image_download import simple_image_download
        except ImportError:
            logging.error("找不到simple_image_download库，请使用pip安装: pip install simple-image-download")
            logging.error("安装命令: pip install simple-image-download")
            return
        
        # 创建下载器实例 - 注意这里需要调用函数
        response = simple_image_download.simple_image_download()
        
        # 设置输出目录
        response.directory = args.output
        
        # 处理扩展名
        extensions = args.extensions.split(',')
        
        # 处理关键词列表
        keywords = [k.strip() for k in args.keywords.split(',')]
        
        total_images = 0
        for keyword in keywords:
            logging.info(f"正在下载关键词 '{keyword}' 的 {args.count} 张图片...")
            
            # 执行下载
            paths = response.download(keyword, args.count, extensions=extensions)
            
            keyword_count = len(paths)
            total_images += keyword_count
            logging.info(f"关键词 '{keyword}' 下载了 {keyword_count} 张图片")
        
        logging.info(f"下载完成! 共下载 {total_images} 张图片")
        logging.info(f"图片保存在: {os.path.abspath(os.path.join(os.getcwd(), args.output))}")
        
    except Exception as e:
        logging.error(f"下载过程中出错: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
