#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级Google图像批量下载脚本
支持自动重试、多关键词批处理、结果统计等功能
"""

import os
import sys
import argparse
import logging
import json
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 导入之前创建的下载脚本
from scripts.download_google_images import build_arguments, load_config_file

def parse_args():
    parser = argparse.ArgumentParser(description="高级Google图像批量下载工具")
    parser.add_argument("--keywords_file", "-kf", type=str, default=None,
                      help="包含关键词列表的文件（每行一个）")
    parser.add_argument("--output", "-o", type=str, default="downloads",
                      help="下载图片的输出目录")
    parser.add_argument("--limit", "-l", type=int, default=100,
                      help="每个关键词下载的图片数量")
    parser.add_argument("--format", "-f", type=str, default="jpg",
                      choices=["jpg", "gif", "png", "bmp", "svg", "webp", "ico", "raw"],
                      help="图片格式")
    parser.add_argument("--max_workers", "-mw", type=int, default=3,
                      help="同时处理的关键词数量")
    parser.add_argument("--retry", "-r", type=int, default=3,
                      help="下载失败时的重试次数")
    parser.add_argument("--delay_min", type=float, default=1.0,
                      help="下载间隔的最小延迟（秒）")
    parser.add_argument("--delay_max", type=float, default=3.0,
                      help="下载间隔的最大延迟（秒）")
    parser.add_argument("--proxy", "-px", type=str, default=None,
                      help="代理服务器设置，格式为'IP:Port'")
    parser.add_argument("--config_file", "-cf", type=str, default=None,
                      help="基础配置文件路径（会被其他参数覆盖）")
    
    args = parser.parse_args()
    
    if args.keywords_file is None and args.config_file is None:
        parser.error("必须提供关键词文件或配置文件")
        
    return args

def read_keywords_file(file_path):
    """从文件中读取关键词列表"""
    keywords = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keywords.append(line)
        return keywords
    except Exception as e:
        logging.error(f"读取关键词文件时出错: {e}")
        return []

def download_for_keyword(keyword, base_args, retry_count=3, delay_range=(1, 3)):
    """下载单个关键词的图片，支持重试"""
    from google_images_download import google_images_download
    
    # 创建新的参数字典
    args = base_args.copy()
    args["keywords"] = keyword
    args["delay"] = random.uniform(delay_range[0], delay_range[1])
    
    # 创建下载器实例
    downloader = google_images_download.googleimagesdownload()
    
    for attempt in range(retry_count + 1):
        try:
            paths = downloader.download(args)
            # 计算下载的图片数量
            count = 0
            if paths and keyword in paths:
                count = len(paths[keyword]) if isinstance(paths[keyword], list) else 0
            
            return {
                "keyword": keyword,
                "success": True,
                "count": count,
                "paths": paths.get(keyword, []) if paths else []
            }
        except Exception as e:
            if attempt < retry_count:
                # 指数退避策略
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                return {
                    "keyword": keyword,
                    "success": False,
                    "error": str(e),
                    "count": 0,
                    "paths": []
                }

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
    
    # 加载基础配置
    base_args = {}
    if args.config_file:
        base_args = load_config_file(args.config_file)
    
    # 更新配置
    base_args["output_directory"] = args.output
    base_args["limit"] = args.limit
    base_args["format"] = args.format
    if args.proxy:
        base_args["proxy"] = args.proxy
    
    # 读取关键词
    keywords = []
    if args.keywords_file:
        keywords = read_keywords_file(args.keywords_file)
        
    if not keywords:
        logging.error("没有找到有效的关键词")
        sys.exit(1)
    
    logging.info(f"准备下载 {len(keywords)} 个关键词的图片，每个关键词 {args.limit} 张")
    
    # 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    
    # 创建结果文件
    results_file = os.path.join(args.output, "download_results.json")
    log_file = os.path.join(args.output, "download_log.txt")
    
    # 设置文件日志
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # 使用线程池并行下载
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 创建下载任务
        future_to_keyword = {
            executor.submit(
                download_for_keyword, 
                keyword, 
                base_args, 
                args.retry,
                (args.delay_min, args.delay_max)
            ): keyword for keyword in keywords
        }
        
        # 使用tqdm显示进度
        with tqdm(total=len(keywords), desc="下载进度") as pbar:
            for future in as_completed(future_to_keyword):
                keyword = future_to_keyword[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 记录结果
                    status = "成功" if result["success"] else "失败"
                    message = f"关键词 '{keyword}' {status}，下载了 {result['count']} 张图片"
                    if not result["success"]:
                        message += f"，错误: {result['error']}"
                    logging.info(message)
                    
                except Exception as e:
                    results.append({
                        "keyword": keyword,
                        "success": False,
                        "error": str(e),
                        "count": 0,
                        "paths": []
                    })
                    logging.error(f"处理关键词 '{keyword}' 时出错: {e}")
                
                pbar.update(1)
    
    # 统计结果
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count
    total_images = sum(r["count"] for r in results)
    
    # 保存结果到json文件
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_keywords": len(keywords),
                "success_keywords": success_count,
                "failed_keywords": failed_count,
                "total_images": total_images,
                "elapsed_time": time.time() - start_time
            },
            "details": results
        }, f, ensure_ascii=False, indent=2)
    
    elapsed_time = time.time() - start_time
    logging.info(f"下载完成! 成功处理 {success_count}/{len(keywords)} 个关键词")
    logging.info(f"共下载 {total_images} 张图片，耗时 {elapsed_time:.2f} 秒")
    logging.info(f"详细结果保存在: {results_file}")

if __name__ == "__main__":
    main()
