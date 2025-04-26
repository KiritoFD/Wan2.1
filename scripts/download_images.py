#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的Google图像下载脚本
接收命令行参数：关键词和下载数量
"""

import os
import sys
import argparse
import logging

def main():
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="下载Google图像")
    parser.add_argument("keywords", type=str, help="要搜索的关键词，多个关键词用逗号分隔")
    parser.add_argument("count", type=int, help="要下载的图片数量")
    parser.add_argument("--output", "-o", type=str, default="downloads", help="下载图片的输出目录")
    parser.add_argument("--format", "-f", type=str, default=None, 
                       choices=["jpg", "gif", "png", "bmp", "svg", "webp", "ico", "raw"],
                       help="图片格式")
    args = parser.parse_args()
    
    try:
        # 检查库是否已安装
        try:
            import google_images_download
        except ImportError:
            logging.error("找不到google_images_download库，请使用pip安装: pip install google_images_download")
            return
        
        # 创建下载器实例 - 修复实例化方式
        try:
            # 尝试新的API方式
            downloader = google_images_download.google_images_download()
        except AttributeError:
            try:
                # 尝试直接导入类
                from google_images_download import google_images_download as gid
                downloader = gid()
            except Exception as e:
                logging.error(f"无法实例化下载器: {e}")
                logging.error("请尝试重新安装库: pip install --upgrade google_images_download")
                return
        
        # 设置下载参数
        arguments = {
            "keywords": args.keywords,
            "limit": args.count,
            "output_directory": args.output
        }
        
        # 添加可选的格式参数
        if args.format:
            arguments["format"] = args.format
            
        logging.info(f"准备下载关键词 '{args.keywords}' 的 {args.count} 张图片...")
        
        # 执行下载
        try:
            paths = downloader.download(arguments)
            
            # 计算下载结果
            total_images = 0
            for keyword in paths:
                if isinstance(paths[keyword], list):
                    keyword_count = len(paths[keyword])
                    total_images += keyword_count
                    logging.info(f"关键词 '{keyword}' 下载了 {keyword_count} 张图片")
            
            logging.info(f"下载完成! 共下载 {total_images} 张图片")
            logging.info(f"图片保存在: {os.path.abspath(args.output)}")
            
        except Exception as e:
            logging.error(f"下载过程失败: {e}")
            
    except Exception as e:
        logging.error(f"下载过程中出错: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
