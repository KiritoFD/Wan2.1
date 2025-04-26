#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多源图片搜索与下载工具 - 主程序入口

用途: 从多个图片源(Google, Bing, Flickr, Bilibili, Pixiv等)搜索并下载图片。
支持命令行参数，可自定义关键词、数量和保存路径。
增强功能：图片去重、图片质量排序、多线程下载和优化的搜索方法。
"""

import os
import sys
import time
import importlib
from typing import Dict, List, Optional
import importlib.util

# 确保必要的目录结构存在
def ensure_directory_structure():
    """确保必要的目录结构存在"""
    dirs = [
        'sources',
        'core',
        'utils'
    ]
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            # 创建__init__.py文件使其成为包
            with open(os.path.join(directory, "__init__.py"), "w") as f:
                f.write("# 自动生成的包初始化文件")

# 在导入之前确保目录结构存在
ensure_directory_structure()

# 尝试导入配置和工具，如果不存在则使用内置的默认配置
try:
    from config import CONFIG, parse_arguments
except ImportError:
    # 内置默认配置
    CONFIG = {
        "sources": [
            "google", "bing", "flickr", "bilibili", "pixiv", 
            "twitter", "pinterest", "unsplash", "500px", "instagram"
        ],
        "max_threads": 10,
        "download_timeout": 15,
        "request_timeout": 30,
        "min_image_size": 10000,
        "min_resolution": (200, 200),
        "duplicate_threshold": 90,
        "enable_deduplication": True,
        "sort_by_quality": True
    }
    
    # 内置的参数解析函数
    def parse_arguments():
        import argparse
        parser = argparse.ArgumentParser(description='从多个图片源搜索并下载图片')
        parser.add_argument('-q', '--query', type=str, required=True, help='搜索关键词')
        parser.add_argument('-n', '--number', type=int, default=50, help='要下载的图片数量')
        parser.add_argument('-o', '--output', type=str, default=None, help='保存图片的目录')
        parser.add_argument('-s', '--sources', type=str, nargs='+', choices=CONFIG["sources"], default=None, 
                            help='指定要使用的图片源')
        return parser.parse_args()

try:
    from core.downloader import ImageDownloader
except ImportError:
    # 创建简单的下载器类以保证程序可以运行
    class ImageDownloader:
        def __init__(self, save_dir):
            self.save_dir = save_dir
        
        def download_batch(self, url_list, start_count=0, max_count=None):
            print(f"警告: 核心下载器不可用。请确保创建了必要的模块文件。")
            return start_count

# 尝试导入或创建源基类
try:
    from sources.source_base import ImageSource
except ImportError:
    # 创建基本的源抽象类
    class ImageSource:
        def __init__(self, name):
            self.name = name
        
        def search(self, query, limit=100):
            return []
        
        def download(self, query, num_images, save_dir, start_count=0):
            print(f"警告: {self.name} 源模块不可用。请创建适当的源文件。")
            return start_count

# 确保必要的源模块文件存在
def ensure_source_modules():
    """确保所有需要的源模块文件存在"""
    sources_dir = "sources"
    
    # 确保源基类文件存在
    base_file = os.path.join(sources_dir, "source_base.py")
    if not os.path.exists(base_file):
        with open(base_file, "w") as f:
            f.write('''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片源基类 - 定义所有图片源的通用接口和方法
"""

class ImageSource:
    """图片源抽象基类"""
    
    def __init__(self, name):
        self.name = name
    
    def search(self, query, limit=100):
        """搜索图片并返回URL列表"""
        print(f"{self.name} 源的search方法未实现")
        return []
    
    def download(self, query, num_images, save_dir, start_count=0):
        """搜索并下载图片"""
        print(f"{self.name} 源的download方法未实现")
        return start_count
''')
    
    # 为每个源创建一个基本文件，如果不存在的话
    for source_name in CONFIG["sources"]:
        source_file = os.path.join(sources_dir, f"{source_name}.py")
        
        # 特殊处理500px，因为文件名不能以数字开头
        if source_name == "500px":
            source_file = os.path.join(sources_dir, "px500.py")
            class_name = "Px500ImageSource"
        else:
            class_name = f"{source_name.capitalize()}ImageSource"
        
        if not os.path.exists(source_file):
            with open(source_file, "w") as f:
                f.write(f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
{source_name.capitalize()} 图片源模块
"""

from .source_base import ImageSource

class {class_name}(ImageSource):
    def __init__(self):
        super().__init__("{source_name}")
    
    def search(self, query, limit=100):
        """从{source_name.capitalize()}搜索图片并返回URL列表"""
        print(f"正在从{source_name.capitalize()}搜索图片...")
        # 这里应该实现实际的搜索逻辑
        return []
    
    def download(self, query, num_images, save_dir, start_count=0):
        """从{source_name.capitalize()}搜索并下载图片"""
        print(f"\\n===== 从{source_name.capitalize()}下载图片 =====")
        # 这里应该实现实际的下载逻辑
        # 但现在只返回起始计数
        return start_count
''')

# 在导入之前确保源模块文件存在
ensure_source_modules()

def create_directories(directory: str) -> str:
    """创建保存图片的目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def download_images_from_multiple_sources(
    query: str, 
    num_images: int, 
    output_dir: str, 
    sources: Optional[List[str]] = None
) -> int:
    """从多个来源下载指定数量的图片"""
    output_dir = create_directories(output_dir)
    
    # 默认使用所有图片源
    if sources is None:
        sources = CONFIG["sources"]
    
    # 计算每个源应该下载的图片数量
    sources_count = len(sources)
    base_images_per_source = num_images // sources_count if sources_count > 0 else 0
    extra_images = num_images % sources_count if sources_count > 0 else 0
    
    print(f"开始从多个来源下载 '{query}' 的图片...")
    
    total_count = 0
    source_modules = {}
    
    # 动态加载源模块
    for source in sources:
        try:
            # 处理特殊情况：500px
            module_name = "sources.px500" if source == "500px" else f"sources.{source}"
            
            # 尝试导入对应的源模块
            module = importlib.import_module(module_name)
            
            # 获取源类，约定每个源模块都有一个与模块同名的类
            class_name = "Px500ImageSource" if source == "500px" else source.capitalize() + "ImageSource"
            
            try:
                source_class = getattr(module, class_name)
                
                # 实例化源对象
                source_modules[source] = source_class()
            except AttributeError:
                print(f"模块 {module_name} 中找不到类 {class_name}")
                sources_count -= 1
        except ImportError as e:
            print(f"加载图片源 '{source}' 时出错: {str(e)}")
            sources_count -= 1
    
    if sources_count == 0:
        print("没有可用的图片源，下载终止")
        return 0
    
    # 重新计算每个可用源的数量分配
    base_images_per_source = num_images // sources_count
    extra_images = num_images % sources_count
    
    # 从各个源下载图片
    for i, (source_name, source) in enumerate(source_modules.items()):
        # 计算当前源应下载的图片数量
        current_target = base_images_per_source + (1 if i < extra_images else 0)
        if current_target <= 0:
            continue
            
        try:
            # 使用源对象下载图片
            new_count = source.download(query, current_target, output_dir, start_count=total_count)
            downloaded = new_count - total_count
            total_count = new_count
        except Exception as e:
            print(f"从 {source_name} 下载图片时出错: {str(e)}")
    
    # 如果下载数量不够，尝试补充
    if total_count < num_images and "google" in source_modules:
        print(f"\n===== 继续从Google下载更多图片 =====")
        try:
            google_source = source_modules["google"]
            final_count = google_source.download(
                query + " HD", 
                num_images - total_count, 
                output_dir, 
                start_count=total_count
            )
            print(f"额外下载了 {final_count - total_count} 张图片")
            total_count = final_count
        except Exception as e:
            print(f"下载额外图片时出错: {str(e)}")
        
    print(f"\n下载完成！总共下载了 {total_count} 张 '{query}' 的图片到 {output_dir}")
    return total_count

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置参数
    search_query = args.query
    target_count = args.number
    save_directory = args.output if args.output else search_query.replace(" ", "_") + "_images"
    sources = args.sources
    
    # 显示配置信息
    print("图片下载配置:")
    print(f"- 搜索关键词: {search_query}")
    print(f"- 目标图片数: {target_count}")
    print(f"- 保存目录: {save_directory}")
    print(f"- 图片源: {', '.join(sources if sources else CONFIG['sources'])}")
    print(f"- 多线程下载: {CONFIG['max_threads']} 线程")
    print(f"- 图片去重: {'启用' if CONFIG['enable_deduplication'] else '禁用'}")
    print(f"- 最小分辨率: {CONFIG['min_resolution'][0]}x{CONFIG['min_resolution'][1]}")
    print(f"- 最小文件大小: {CONFIG['min_image_size']} 字节")
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行下载
    download_images_from_multiple_sources(search_query, target_count, save_directory, sources)
    
    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n总运行时间: {total_time:.1f} 秒")

if __name__ == "__main__":
    main()
