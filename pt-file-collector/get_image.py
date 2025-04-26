#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_image.py - 多源图片搜索与下载工具

用途: 从多个图片源(Google, Bing, Flickr, Bilibili, Pixiv等)搜索并下载图片。
支持命令行参数，可自定义关键词、数量和保存路径。
增强功能：图片去重、图片质量排序、多线程下载和优化的搜索方法。
"""

import os
import time
import random
import requests
import argparse
import sys
import json
import hashlib
import threading
import queue
import io
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse, urlencode
import re
from PIL import Image

# ===== 全局控制参数 =====
# 搜索引擎和图片源配置
DEFAULT_SOURCES = [
    "google", "bing", "flickr", "bilibili", "pixiv", 
    "twitter", "pinterest", "unsplash", "500px", "instagram"
]

# 线程和并发配置
MAX_THREADS = 10          # 最大下载线程数
DOWNLOAD_TIMEOUT = 15     # 下载超时时间(秒)
REQUEST_TIMEOUT = 30      # 网络请求超时时间(秒)

# 图片质量和去重配置
MIN_IMAGE_SIZE = 10000    # 最小文件大小(字节)
MIN_RESOLUTION = (200, 200)  # 最小分辨率(宽, 高)
DUPLICATE_THRESHOLD = 90  # 去重相似度阈值(0-100)
ENABLE_DEDUPLICATION = True  # 是否启用图片去重
SORT_BY_QUALITY = True    # 是否按质量排序下载

# 网络请求配置
RETRY_COUNT = 2           # 请求失败重试次数
RETRY_DELAY = (1, 3)      # 重试延迟时间范围(秒)
PAGE_DELAY = (1, 3)       # 页面间请求延迟时间范围(秒)

# 图片命名配置
USE_SMART_NAMING = True   # 是否使用智能命名
IMAGE_PREFIX = "img"      # 图片文件名前缀
NAME_BY_SOURCE = True     # 是否在文件名中包含来源信息

# ===== 功能函数 =====

def create_directories(directory):
    """创建保存图片的目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def get_user_agent():
    """随机选择一个User-Agent，避免被网站封锁"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
    ]
    return random.choice(user_agents)

def download_image(img_url, save_path, img_count):
    """下载单张图片"""
    try:
        headers = {"User-Agent": get_user_agent()}
        response = requests.get(img_url, headers=headers, stream=True, timeout=10)
        
        if response.status_code == 200:
            # 检查是否是图片内容
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image'):
                print(f"跳过非图片内容: {img_url}, Content-Type: {content_type}")
                return False
                
            # 确定文件扩展名
            extension = 'jpg'  # 默认扩展名
            if 'image/jpeg' in content_type:
                extension = 'jpg'
            elif 'image/png' in content_type:
                extension = 'png'
            elif 'image/gif' in content_type:
                extension = 'gif'
            elif 'image/webp' in content_type:
                extension = 'webp'
                
            file_path = os.path.join(save_path, f"image_{img_count}.{extension}")
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"成功下载图片 {img_count}: {img_url}")
            return True
        else:
            print(f"下载失败，状态码 {response.status_code}: {img_url}")
            return False
            
    except Exception as e:
        print(f"下载图片 {img_url} 时出错: {str(e)}")
        return False

def calculate_image_hash(img_data):
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

def hamming_distance(hash1, hash2):
    """计算两个哈希值之间的汉明距离"""
    if hash1 is None or hash2 is None:
        return 100
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def estimate_image_quality(img_data):
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

class ImageDownloader:
    """图片下载管理器，支持多线程下载、去重和质量排序"""
    
    def __init__(self, save_dir, max_threads=MAX_THREADS, source_name=None, metadata=None):
        self.save_dir = save_dir
        self.max_threads = max_threads
        self.source_name = source_name
        self.metadata = metadata or {}
        self.image_hashes = {}  # 存储已下载图片的哈希值
        self.download_count = 0
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_threads)
        
    def download_batch(self, url_list, start_count=0, max_count=None):
        """批量下载图片，支持去重和限制数量"""
        if not url_list:
            return 0
            
        # 创建任务队列
        image_queue = queue.Queue()
        for url in url_list:
            image_queue.put(url)
            
        # 创建结果列表
        downloaded_images = []
        futures = []
        
        # 提交下载任务
        while not image_queue.empty():
            if max_count and self.download_count - start_count >= max_count:
                break
                
            url = image_queue.get()
            future = self.thread_pool.submit(
                self._download_single_image, 
                url, 
                start_count + self.download_count + 1
            )
            futures.append(future)
            
        # 等待所有任务完成
        for future in futures:
            result = future.result()
            if result:
                downloaded_images.append(result)
                
        # 如果启用质量排序，按质量排序后保存
        if SORT_BY_QUALITY and downloaded_images:
            downloaded_images.sort(key=lambda x: x['quality'], reverse=True)
            
            # 重命名文件以反映质量排序
            for i, img_info in enumerate(downloaded_images):
                old_path = img_info['path']
                if os.path.exists(old_path):
                    # 生成新的文件名
                    new_name = self._generate_smart_filename(
                        img_info, 
                        start_count + i + 1,
                        img_info['source_url']
                    )
                    new_path = os.path.join(self.save_dir, new_name)
                    try:
                        os.rename(old_path, new_path)
                        img_info['path'] = new_path
                    except Exception as e:
                        print(f"重命名文件时出错: {str(e)}")
                        
        return self.download_count - start_count
    
    def _generate_smart_filename(self, img_info, img_count, url=""):
        """根据图片内容和来源生成智能文件名"""
        if not USE_SMART_NAMING:
            return f"{IMAGE_PREFIX}_{img_count}.{img_info['extension']}"
        
        parts = []
        
        # 添加前缀
        parts.append(IMAGE_PREFIX)
        
        # 添加序号
        parts.append(f"{img_count:04d}")
        
        # 添加来源信息
        if NAME_BY_SOURCE:
            source = self.source_name or self._detect_source_from_url(url)
            if source:
                parts.append(source)
        
        # 添加元数据 (针对pixiv)
        if url in self.metadata:
            meta = self.metadata[url]
            if 'title' in meta and meta['title']:
                # 清理标题中的非法字符
                title = re.sub(r'[\\/*?:"<>|]', '', meta['title'])
                title = title[:30]  # 限制长度
                if title:
                    parts.append(title)
            
            if 'artist' in meta and meta['artist']:
                artist = re.sub(r'[\\/*?:"<>|]', '', meta['artist'])
                artist = artist[:20]  # 限制长度
                if artist:
                    parts.append(artist)
        
        # 添加分辨率信息
        if 'resolution' in img_info:
            width, height = img_info['resolution']
            parts.append(f"{width}x{height}")
        
        # 添加扩展名
        filename = "_".join(parts) + f".{img_info['extension']}"
        
        # 确保文件名长度不超过255字符
        if len(filename) > 255:
            # 如果太长，截断中间部分
            basename, ext = os.path.splitext(filename)
            filename = basename[:240] + ext
            
        return filename
    
    def _detect_source_from_url(self, url):
        """从URL中检测图片来源"""
        url_lower = url.lower()
        if "google" in url_lower:
            return "google"
        elif "bing" in url_lower:
            return "bing"
        elif "flickr" in url_lower or "staticflickr" in url_lower:
            return "flickr"
        elif "bilibili" in url_lower or "bili" in url_lower:
            return "bili"
        elif "pixiv" in url_lower or "pximg" in url_lower:
            return "pixiv"
        elif "twimg" in url_lower or "twitter" in url_lower:
            return "twitter"
        elif "pinimg" in url_lower or "pinterest" in url_lower:
            return "pinterest"
        elif "unsplash" in url_lower:
            return "unsplash"
        elif "500px" in url_lower:
            return "500px"
        elif "instagram" in url_lower or "cdninstagram" in url_lower:
            return "instagram"
        return ""
        
    def _download_single_image(self, img_url, img_count):
        """下载单张图片，检查质量并去重"""
        try:
            headers = {"User-Agent": get_user_agent()}
            # 对某些特定网站添加Referer
            if "pixiv" in img_url.lower() or "pximg" in img_url.lower():
                headers["Referer"] = "https://www.pixiv.net/"
            elif "bilibili" in img_url.lower():
                headers["Referer"] = "https://www.bilibili.com/"
                
            response = requests.get(img_url, headers=headers, stream=True, timeout=DOWNLOAD_TIMEOUT)
            
            if response.status_code != 200:
                return None
                
            # 检查是否是图片内容
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image'):
                return None
                
            # 获取图片数据
            img_data = response.content
            
            # 检查文件大小
            if len(img_data) < MIN_IMAGE_SIZE:
                return None
                
            try:
                # 检查图片分辨率
                img = Image.open(io.BytesIO(img_data))
                width, height = img.size
                if width < MIN_RESOLUTION[0] or height < MIN_RESOLUTION[1]:
                    return None
                    
                # 计算图片哈希值用于去重
                img_hash = calculate_image_hash(img_data)
                
                # 如果启用去重，检查是否与现有图片相似
                if ENABLE_DEDUPLICATION and img_hash:
                    for existing_hash in self.image_hashes.values():
                        if hamming_distance(img_hash, existing_hash) < (64 * (100 - DUPLICATE_THRESHOLD) // 100):
                            print(f"跳过重复图片: {img_url}")
                            return None
                
                # 确定文件扩展名
                extension = 'jpg'  # 默认扩展名
                if 'image/jpeg' in content_type:
                    extension = 'jpg'
                elif 'image/png' in content_type:
                    extension = 'png'
                elif 'image/gif' in content_type:
                    extension = 'gif'
                elif 'image/webp' in content_type:
                    extension = 'webp'
                    
                # 估算图片质量
                quality = estimate_image_quality(img_data)
                
                # 临时文件名
                temp_filename = f"image_{img_count}.{extension}"
                file_path = os.path.join(self.save_dir, temp_filename)
                
                # 保存图片
                with open(file_path, 'wb') as f:
                    f.write(img_data)
                    
                # 记录图片哈希值
                with self.lock:
                    if ENABLE_DEDUPLICATION and img_hash:
                        self.image_hashes[img_url] = img_hash
                    self.download_count += 1
                    
                print(f"成功下载图片 {img_count}: {img_url}")
                
                return {
                    'url': img_url,
                    'path': file_path,
                    'extension': extension,
                    'quality': quality,
                    'resolution': (width, height),
                    'source_url': img_url
                }
            except Exception as e:
                print(f"处理图片数据时出错 {img_url}: {str(e)}")
                return None
                
        except Exception as e:
            print(f"下载图片 {img_url} 时出错: {str(e)}")
            return None

def get_google_images(query, num_images, save_dir):
    """从Google搜索图片"""
    base_url = "https://www.google.com/search"
    params = {
        "q": query,
        "tbm": "isch",
        "ijn": 0,  # 页码
        "start": 0,
        "tbs": "isz:l"  # 大尺寸图片
    }
    
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://www.google.com/"
    }
    
    img_count = 0
    image_urls = set()
    
    # 搜索多页结果
    for page in range(0, 5):  # 尝试获取5页的图片
        if img_count >= num_images:
            break
            
        params["ijn"] = page
        params["start"] = page * 100
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                print(f"获取Google搜索结果失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取图片URL
            # 查找JavaScript中的图片URL
            script_tags = soup.find_all("script")
            for script in script_tags:
                script_content = str(script)
                img_matches = re.findall(r'(?:http[s]?://[^"\']+\.(?:jpg|jpeg|png|gif|webp))', script_content, re.IGNORECASE)
                for img_url in img_matches:
                    if img_url not in image_urls and not img_url.startswith("data:"):
                        image_urls.add(img_url)
            
            # 查找常规图片标签
            img_tags = soup.find_all("img")
            for img in img_tags:
                img_url = img.get("src", "")
                if img_url and not img_url.startswith("data:") and img_url not in image_urls:
                    if img_url.startswith("http"):
                        image_urls.add(img_url)
                        
            # 查找srcset属性
            elements_with_srcset = soup.select("[srcset]")
            for element in elements_with_srcset:
                srcset = element.get("srcset", "")
                urls = re.findall(r'(https?://[^\s]+)', srcset)
                for url in urls:
                    if url and not url.startswith("data:") and url not in image_urls:
                        image_urls.add(url)
            
            print(f"Google搜索第{page+1}页，找到 {len(image_urls)} 个图片URL")
            
        except Exception as e:
            print(f"处理Google搜索结果时出错: {str(e)}")
            # 暂停一会儿，避免被Google封锁
            time.sleep(random.randint(2, 5))
            continue
            
        # 暂停一会儿，避免被Google封锁
        time.sleep(random.randint(2, 5))
    
    # 下载图片
    for url in image_urls:
        if img_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

def get_bing_images(query, num_images, save_dir, start_count=0):
    """从Bing搜索图片 - 优化版"""
    base_url = "https://www.bing.com/images/search"
    api_url = "https://www.bing.com/images/async"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://www.bing.com/images/search"
    }
    
    image_urls = set()
    
    # 首先获取初始页面以获取必要的cookie和参数
    try:
        initial_params = {
            "q": query,
            "form": "HDRSC2",
            "first": 1
        }
        
        response = requests.get(base_url, params=initial_params, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"获取Bing初始页面失败，状态码: {response.status_code}")
            return start_count
            
        # 获取IG和SFX参数，这些是Bing API请求所必需的
        ig_match = re.search(r'IG:"([^"]+)"', response.text)
        ig = ig_match.group(1) if ig_match else ""
        
        sfx_match = re.search(r'_SFX:"([^"]+)"', response.text)
        sfx = sfx_match.group(1) if sfx_match else ""
        
        if not ig or not sfx:
            print("无法从Bing页面获取必要参数")
            return start_count
            
        # 从初始页面提取图片URL
        soup = BeautifulSoup(response.text, 'html.parser')
        initial_images = _extract_bing_image_urls(soup)
        for img_url in initial_images:
            if img_url not in image_urls and not img_url.startswith("data:"):
                image_urls.add(img_url)
        
        # 然后发送异步请求获取更多图片
        for offset in range(1, 5):  # 尝试获取最多5页
            if len(image_urls) >= num_images * 2:  # 获取比需要的多一倍，以应对下载失败
                break
                
            async_params = {
                "q": query,
                "first": offset * 35 + 1,
                "count": 35,
                "relp": 35,
                "tsc": "ImageHoverTitle",
                "layout": "RowBased",
                "mmasync": 1,
                "dgState": "c*5_y*1632s1812s1736s1755s1642_i*36_d*20",  # 这些参数可能需要定期更新
                "IG": ig,
                "SFX": sfx
            }
            
            try:
                async_response = requests.get(
                    api_url, 
                    params=async_params, 
                    headers=headers, 
                    timeout=REQUEST_TIMEOUT
                )
                
                if async_response.status_code == 200:
                    async_soup = BeautifulSoup(async_response.text, 'html.parser')
                    batch_urls = _extract_bing_image_urls(async_soup)
                    
                    for img_url in batch_urls:
                        if img_url not in image_urls and not img_url.startswith("data:"):
                            image_urls.add(img_url)
                            
                    print(f"Bing搜索第{offset+1}页，当前共找到 {len(image_urls)} 个图片URL")
                    
                else:
                    print(f"获取Bing异步结果失败，状态码: {async_response.status_code}")
                    
            except Exception as e:
                print(f"处理Bing异步请求时出错: {str(e)}")
                
            time.sleep(random.uniform(PAGE_DELAY[0], PAGE_DELAY[1]))
            
    except Exception as e:
        print(f"Bing搜索出错: {str(e)}")
        
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def _extract_bing_image_urls(soup):
    """从Bing搜索结果HTML中提取图片URL"""
    urls = set()
    
    # 方法1：从图片标签获取
    for img in soup.select(".mimg"):
        src = img.get("src", "")
        data_src = img.get("data-src", "")
        
        if src and src.startswith("http"):
            urls.add(src)
        if data_src and data_src.startswith("http"):
            urls.add(data_src)
    
    # 方法2：从缩略图元数据获取原图URL
    for thumb in soup.select(".iusc"):
        m_attr = thumb.get("m", "")
        if m_attr:
            try:
                m_data = json.loads(m_attr)
                if "murl" in m_data:
                    img_url = m_data["murl"]
                    if img_url and img_url.startswith("http"):
                        urls.add(img_url)
            except json.JSONDecodeError:
                pass
    
    # 方法3：从JavaScript数据中提取
    for script in soup.find_all("script"):
        script_text = str(script)
        # 查找格式为 "murl":"http://..." 的URL
        murl_matches = re.findall(r'"murl":"([^"]+)"', script_text)
        for url in murl_matches:
            url = url.replace('\\', '')
            if url.startswith("http"):
                urls.add(url)
    
    return urls

def get_flickr_images(query, num_images, save_dir, start_count=0):
    """从Flickr搜索并下载图片"""
    base_url = "https://www.flickr.com/search/"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3"
    }
    
    img_count = start_count
    image_urls = set()
    
    # 搜索多页结果
    for page in range(1, 6):  # 尝试获取5页的图片
        if img_count - start_count >= num_images:
            break
            
        params = {
            "text": query,
            "view_all": 1,
            "page": page
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                print(f"获取Flickr搜索结果失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找图片标签
            img_tags = soup.select(".photo-list-photo-view img")
            for img in img_tags:
                img_url = img.get("src", "")
                if not img_url:
                    img_url = img.get("data-src", "")
                
                # 转换为更高分辨率的图片URL
                if img_url and "staticflickr.com" in img_url:
                    # 替换尺寸后缀，获取更大尺寸的图片
                    img_url = re.sub(r'_[sqtmnzcbo]\.', '_b.', img_url)
                    if img_url not in image_urls:
                        image_urls.add(img_url)
            
            print(f"Flickr搜索第{page}页，找到 {len(image_urls)} 个图片URL")
            
        except Exception as e:
            print(f"处理Flickr搜索结果时出错: {str(e)}")
            time.sleep(random.randint(2, 4))
            continue
            
        time.sleep(random.uniform(PAGE_DELAY[0], PAGE_DELAY[1]))
    
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_bilibili_images(query, num_images, save_dir, start_count=0):
    """从Bilibili搜索并下载图片 - 优化版"""
    base_url = "https://api.bilibili.com/x/web-interface/search/type"
    base_url_alt = "https://search.bilibili.com/all"
    headers = {
        "User-Agent": get_user_agent(),
        "Referer": "https://www.bilibili.com/",
        "Origin": "https://www.bilibili.com",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    
    image_urls = set()
    
    # 方法1: API搜索
    # 搜索多页结果
    for page in range(1, 6):  # 尝试获取5页的图片
        if len(image_urls) >= num_images * 2:  # 获取比需要的多一倍，以应对下载失败
            break
            
        params = {
            "keyword": query,
            "search_type": "photo",  # 搜索类型为相册
            "page": page,
            "order": "totalrank",  # 默认排序
            "category_id": 0,      # 全部分区
            "jsonp": "jsonp"
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("code") == 0:
                        # 提取图片URL
                        result_data = data.get("data", {}).get("result", [])
                        for item in result_data:
                            # 提取图片链接
                            if "pic" in item:
                                img_url = item["pic"]
                                if img_url and img_url.startswith("http") and img_url not in image_urls:
                                    image_urls.add(img_url)
                                    
                            # 提取封面图
                            if "cover" in item:
                                img_url = item["cover"]
                                if img_url and img_url.startswith("http") and img_url not in image_urls:
                                    image_urls.add(img_url)
                                    
                        print(f"B站API搜索第{page}页，当前共找到 {len(image_urls)} 个图片URL")
                    else:
                        print(f"B站API返回错误: {data.get('message', '未知错误')}")
                except json.JSONDecodeError:
                    print("解析B站返回的JSON数据失败")
            else:
                print(f"获取B站搜索结果失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"处理B站API搜索请求时出错: {str(e)}")
            
        time.sleep(random.uniform(PAGE_DELAY[0], PAGE_DELAY[1]))
    
    # 方法2: 网页搜索
    if len(image_urls) < num_images:
        params_alt = {
            "keyword": query,
            "from_source": "web_search"
        }
        
        try:
            response = requests.get(base_url_alt, params=params_alt, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取视频封面图片
                img_tags = soup.select("img.img-content")
                for img in img_tags:
                    img_url = img.get("src", "")
                    if not img_url.startswith("http"):
                        img_url = "https:" + img_url
                    if img_url and img_url not in image_urls:
                        image_urls.add(img_url)
                        
                # 提取用户头像
                avatar_tags = soup.select("img.bili-avatar-img")
                for img in avatar_tags:
                    img_url = img.get("src", "")
                    if not img_url.startswith("http"):
                        img_url = "https:" + img_url
                    if img_url and img_url not in image_urls:
                        image_urls.add(img_url)
                
                # 从页面脚本中提取数据
                script_tags = soup.find_all("script")
                for script in script_tags:
                    script_content = str(script)
                    # 查找图片URL
                    img_matches = re.findall(r'(https?:)?//[^"\']+\.(?:jpg|jpeg|png|gif|webp)', script_content, re.IGNORECASE)
                    for img_url in img_matches:
                        if not img_url.startswith("http"):
                            img_url = "https:" + img_url
                        if "@" in img_url:  # 这通常是B站的图片格式
                            img_url = img_url.split("@")[0]  # 去掉修饰参数获取原图
                        if img_url and img_url not in image_urls:
                            image_urls.add(img_url)
                
                print(f"B站网页搜索，当前共找到 {len(image_urls)} 个图片URL")
        except Exception as e:
            print(f"处理B站网页搜索时出错: {str(e)}")
    
    # 下载图片
    downloader = ImageDownloader(save_dir, source_name="bilibili")
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_pixiv_images(query, num_images, save_dir, start_count=0):
    """从Pixiv搜索并下载图片 - 优化版"""
    base_url = "https://www.pixiv.net/ajax/search/illustrations"
    headers = {
        "User-Agent": get_user_agent(),
        "Referer": "https://www.pixiv.net/",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cookie": "",  # 可能需要有效的cookie才能访问
        "Accept": "application/json"
    }
    
    image_urls = set()
    image_metadata = {}  # 存储图片的元数据
    
    # 搜索多页结果
    for page in range(1, 6):  # 尝试获取5页的图片
        if len(image_urls) >= num_images * 2:  # 获取比需要的多一倍，以应对下载失败
            break
            
        params = {
            "word": query,
            "order": "date_d",     # 按日期降序排序
            "mode": "all",         # 所有作品
            "p": page,             # 页码
            "s_mode": "s_tag",     # 按标签搜索
            "type": "illust",      # 插画类型
            "lang": "zh",          # 语言
            "version": "018c40fde3e57447801758ee0f51df0c507dc1a1"  # 这可能需要定期更新
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                try:
                    data = response.json()
                    if not data.get("error") and data.get("body"):
                        # 提取图片URL
                        illustrations = data.get("body", {}).get("illust", {}).get("data", [])
                        for illust in illustrations:
                            try:
                                if "url" in illust and "regular" in illust["url"]:
                                    img_url = illust["url"]["regular"]
                                    # 替换Pixiv的代理URL以获取实际图片
                                    img_url = img_url.replace("i.pximg.net", "i.pixiv.cat")
                                    
                                    if img_url and img_url not in image_urls:
                                        image_urls.add(img_url)
                                        
                                        # 保存元数据用于命名
                                        if "title" in illust and "userName" in illust:
                                            image_metadata[img_url] = {
                                                "title": illust["title"],
                                                "artist": illust["userName"],
                                                "id": illust.get("id", "")
                                            }
                            except KeyError as e:
                                print(f"处理Pixiv图片项出错: {str(e)}")
                        
                        print(f"Pixiv搜索第{page}页，当前共找到 {len(image_urls)} 个图片URL")
                    else:
                        print(f"Pixiv API返回错误: {data.get('message', '未知错误')}")
                except json.JSONDecodeError:
                    print("解析Pixiv返回的JSON数据失败")
            else:
                print(f"获取Pixiv搜索结果失败，状态码: {response.status_code}")
        except Exception as e:
            print(f"处理Pixiv搜索请求时出错: {str(e)}")
            
        time.sleep(random.uniform(PAGE_DELAY[0], PAGE_DELAY[1]))
    
    # 尝试从网页版Pixiv获取更多图片
    if len(image_urls) < num_images:
        try:
            pixiv_web_url = f"https://www.pixiv.net/tags/{query}/illustrations"
            web_response = requests.get(pixiv_web_url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if web_response.status_code == 200:
                soup = BeautifulSoup(web_response.text, 'html.parser')
                
                # 从页面脚本中提取数据
                script_tags = soup.find_all("script")
                for script in script_tags:
                    if "pixiv.context.preload" in str(script):
                        json_data = re.search(r'pixiv\.context\.preload\s*=\s*({.*?});</script>', str(script), re.DOTALL)
                        if json_data:
                            try:
                                data = json.loads(json_data.group(1))
                                for key, value in data.items():
                                    if "illust" in key and isinstance(value, dict):
                                        if "url" in value and value["url"].startswith("https://i.pximg.net"):
                                            img_url = value["url"].replace("i.pximg.net", "i.pixiv.cat")
                                            if img_url and img_url not in image_urls:
                                                image_urls.add(img_url)
                                                
                                                # 保存元数据
                                                image_metadata[img_url] = {
                                                    "title": value.get("title", ""),
                                                    "artist": value.get("userName", ""),
                                                    "id": value.get("id", "")
                                                }
                            except json.JSONDecodeError:
                                pass
                
                print(f"Pixiv网页搜索，当前共找到 {len(image_urls)} 个图片URL")
        except Exception as e:
            print(f"处理Pixiv网页请求时出错: {str(e)}")
    
    # 下载图片，需要特殊处理Pixiv的防盗链
    downloader = ImageDownloader(save_dir, source_name="pixiv", metadata=image_metadata)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_twitter_images(query, num_images, save_dir, start_count=0):
    """从Twitter/X搜索并下载图片"""
    base_url = "https://twitter.com/search"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://twitter.com/"
    }
    
    img_count = start_count
    image_urls = set()
    
    # 搜索多页结果
    for page in range(1, 4):  # 尝试获取3页的图片
        if img_count - start_count >= num_images:
            break
            
        params = {
            "q": f"{query} filter:images",
            "src": "typed_query",
            "f": "image"
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code != 200:
                print(f"获取Twitter搜索结果失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找图片标签
            img_tags = soup.find_all("img")
            for img in img_tags:
                img_url = img.get("src", "")
                
                # 过滤出Twitter图片URL (通常格式为https://pbs.twimg.com/media...)
                if img_url and "pbs.twimg.com/media" in img_url and img_url not in image_urls:
                    # 尝试获取更高分辨率的图片
                    img_url = img_url.split("?")[0] + "?format=jpg&name=large"
                    image_urls.add(img_url)
            
            print(f"Twitter搜索第{page}页，找到 {len(image_urls)} 个图片URL")
            
        except Exception as e:
            print(f"处理Twitter搜索结果时出错: {str(e)}")
            time.sleep(random.randint(2, 4))
            continue
            
        time.sleep(random.uniform(PAGE_DELAY[0], PAGE_DELAY[1]))
    
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_pinterest_images(query, num_images, save_dir, start_count=0):
    """从Pinterest搜索并下载图片"""
    base_url = "https://www.pinterest.com/search/pins/"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Referer": "https://www.pinterest.com/"
    }
    
    img_count = start_count
    image_urls = set()
    
    # Pinterest需要JavaScript渲染，这里尝试从HTML中提取数据
    params = {
        "q": query
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"获取Pinterest搜索结果失败，状态码: {response.status_code}")
            return img_count
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 从页面脚本中提取数据
        script_tags = soup.find_all("script")
        for script in script_tags:
            script_content = str(script)
            
            # 查找图片URL
            img_matches = re.findall(r'(https://i\.pinimg\.com/[^"\']+\.(?:jpg|jpeg|png|gif|webp))', script_content, re.IGNORECASE)
            for img_url in img_matches:
                # 尝试提取原始大小图片
                img_url = re.sub(r'/\d+x/|/\d+x\d+/', '/originals/', img_url)
                if img_url not in image_urls:
                    image_urls.add(img_url)
        
        print(f"Pinterest搜索，找到 {len(image_urls)} 个图片URL")
        
    except Exception as e:
        print(f"处理Pinterest搜索结果时出错: {str(e)}")
    
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_unsplash_images(query, num_images, save_dir, start_count=0):
    """从Unsplash搜索并下载高质量图片"""
    base_url = "https://unsplash.com/s/photos/"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3"
    }
    
    img_count = start_count
    image_urls = set()
    
    # 将查询词格式化为URL路径
    query_path = query.replace(" ", "-").lower()
    
    try:
        response = requests.get(f"{base_url}{query_path}", headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"获取Unsplash搜索结果失败，状态码: {response.status_code}")
            return img_count
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找图片数据
        img_tags = soup.find_all("img")
        for img in img_tags:
            src_set = img.get("srcset", "")
            if src_set:
                # 从srcset中提取最大尺寸的URL
                urls = re.findall(r'(https://[^\s]+)', src_set)
                if urls:
                    # 获取列表中最后一个URL（通常是最大尺寸）
                    img_url = urls[-1].split(" ")[0]
                    if img_url and img_url not in image_urls:
                        image_urls.add(img_url)
            else:
                img_url = img.get("src", "")
                if img_url and img_url.startswith("https://") and "unsplash.com" in img_url and img_url not in image_urls:
                    image_urls.add(img_url)
        
        # 还可以从页面的JSON数据中提取图片URL
        script_tags = soup.find_all("script", {"type": "application/json"})
        for script in script_tags:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and "photos" in data:
                    for photo in data["photos"]:
                        if "urls" in photo and "raw" in photo["urls"]:
                            img_url = photo["urls"]["raw"]
                            if img_url and img_url not in image_urls:
                                image_urls.add(img_url)
            except (json.JSONDecodeError, AttributeError):
                continue
        
        print(f"Unsplash搜索，找到 {len(image_urls)} 个图片URL")
        
    except Exception as e:
        print(f"处理Unsplash搜索结果时出错: {str(e)}")
    
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_500px_images(query, num_images, save_dir, start_count=0):
    """从500px搜索并下载高质量图片"""
    base_url = "https://500px.com/search"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3"
    }
    
    img_count = start_count
    image_urls = set()
    
    params = {
        "q": query,
        "type": "photos"
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"获取500px搜索结果失败，状态码: {response.status_code}")
            return img_count
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 从页面脚本中提取JSON数据
        script_content = ""
        for script in soup.find_all("script"):
            if "window.__PRELOADED_STATE__" in str(script):
                script_content = str(script)
                break
                
        # 提取图片URL
        img_matches = re.findall(r'(https://drscdn\.500px\.org/photo/[^"\']+\.(?:jpg|jpeg|png|webp))', script_content, re.IGNORECASE)
        for img_url in img_matches:
            # 尝试获取更高分辨率
            img_url = re.sub(r'/w%3D\d+/v2', '/w%3D2048/v2', img_url)
            if img_url not in image_urls:
                image_urls.add(img_url)
        
        print(f"500px搜索，找到 {len(image_urls)} 个图片URL")
        
    except Exception as e:
        print(f"处理500px搜索结果时出错: {str(e)}")
    
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def get_instagram_images(query, num_images, save_dir, start_count=0):
    """从Instagram搜索并下载图片（注：Instagram限制非登录访问）"""
    base_url = "https://www.instagram.com/explore/tags/"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        "Cookie": ""  # 可能需要Cookie
    }
    
    img_count = start_count
    image_urls = set()
    
    # 格式化查询词（移除空格和特殊字符）
    formatted_query = query.replace(" ", "").lower()
    
    try:
        response = requests.get(f"{base_url}{formatted_query}/", headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            print(f"获取Instagram搜索结果失败，状态码: {response.status_code}")
            return img_count
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 从页面脚本提取数据
        script_tags = soup.find_all("script", type="text/javascript")
        for script in script_tags:
            if "window._sharedData" in str(script):
                json_data = re.search(r'window\._sharedData\s*=\s*({.*?});</script>', str(script), re.DOTALL)
                if json_data:
                    try:
                        data = json.loads(json_data.group(1))
                        # 遍历结构提取图片
                        if "entry_data" in data and "TagPage" in data["entry_data"]:
                            sections = data["entry_data"]["TagPage"][0]["graphql"]["hashtag"]["edge_hashtag_to_media"]["edges"]
                            for edge in sections:
                                node = edge["node"]
                                if "display_url" in node:
                                    img_url = node["display_url"]
                                    if img_url and img_url not in image_urls:
                                        image_urls.add(img_url)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
        
        # 图片标签提取
        img_tags = soup.find_all("img")
        for img in img_tags:
            img_url = img.get("src", "")
            if img_url and "instagram" in img_url and "cdninstagram" in img_url and img_url not in image_urls:
                image_urls.add(img_url)
        
        print(f"Instagram搜索，找到 {len(image_urls)} 个图片URL")
        
    except Exception as e:
        print(f"处理Instagram搜索结果时出错: {str(e)}")
    
    # 下载图片
    downloader = ImageDownloader(save_dir)
    download_count = downloader.download_batch(list(image_urls), start_count, num_images)
    
    return start_count + download_count

def download_images_from_multiple_sources(query, num_images, output_dir, sources=None):
    """从多个来源下载指定数量的图片"""
    output_dir = create_directories(output_dir)
    
    # 默认使用所有图片源
    if sources is None:
        sources = DEFAULT_SOURCES
    
    # 计算每个源应该下载的图片数量
    sources_count = len(sources)
    base_images_per_source = num_images // sources_count
    extra_images = num_images % sources_count
    
    print(f"开始从多个来源下载 '{query}' 的图片...")
    
    total_count = 0
    
    # 从各个源下载图片
    for i, source in enumerate(sources):
        # 计算当前源应下载的图片数量
        current_target = base_images_per_source + (1 if i < extra_images else 0)
        if current_target <= 0:
            continue
            
        print(f"\n===== 从{source.capitalize()}下载图片 =====")
        
        source_time_start = time.time()
        
        if source == "google":
            new_count = get_google_images(query, current_target, output_dir)
        elif source == "bing":
            new_count = get_bing_images(query, current_target, output_dir, start_count=total_count)
        elif source == "flickr":
            new_count = get_flickr_images(query, current_target, output_dir, start_count=total_count)
        elif source == "bilibili":
            new_count = get_bilibili_images(query, current_target, output_dir, start_count=total_count)
        elif source == "pixiv":
            new_count = get_pixiv_images(query, current_target, output_dir, start_count=total_count)
        elif source == "twitter":
            new_count = get_twitter_images(query, current_target, output_dir, start_count=total_count)
        elif source == "pinterest":
            new_count = get_pinterest_images(query, current_target, output_dir, start_count=total_count)
        elif source == "unsplash":
            new_count = get_unsplash_images(query, current_target, output_dir, start_count=total_count)
        elif source == "500px":
            new_count = get_500px_images(query, current_target, output_dir, start_count=total_count)
        elif source == "instagram":
            new_count = get_instagram_images(query, current_target, output_dir, start_count=total_count)
        else:
            print(f"未知的图片源: {source}")
            continue
            
        source_time_end = time.time()
        source_duration = source_time_end - source_time_start
        
        print(f"从{source.capitalize()}下载了 {new_count - total_count} 张图片，耗时 {source_duration:.1f} 秒")
        total_count = new_count
    
    # 如果下载数量不够，尝试补充
    if total_count < num_images and "google" in sources:
        print(f"\n===== 继续从Google下载更多图片 =====")
        final_count = get_google_images(query + " HD", num_images - total_count, output_dir)
        print(f"额外下载了 {final_count - total_count} 张图片")
        total_count = final_count
        
    print(f"\n下载完成！总共下载了 {total_count} 张 '{query}' 的图片到 {output_dir}")
    return total_count


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从多个图片源搜索并下载图片')
    parser.add_argument('-q', '--query', type=str, required=True, 
                        help='搜索关键词')
    parser.add_argument('-n', '--number', type=int, default=50,
                        help='要下载的图片数量 (默认: 50)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='保存图片的目录 (默认: 以关键词命名)')
    parser.add_argument('-s', '--sources', type=str, nargs='+', 
                        choices=DEFAULT_SOURCES,
                        default=None,
                        help=f'指定要使用的图片源 (默认: 使用所有图片源)')
    parser.add_argument('-t', '--threads', type=int, default=MAX_THREADS,
                        help=f'最大下载线程数 (默认: {MAX_THREADS})')
    parser.add_argument('-d', '--no-dedup', action='store_true',
                        help='禁用图片去重功能')
    parser.add_argument('-r', '--min-resolution', type=str, default="200x200",
                        help='最小图片分辨率，格式为"宽x高" (默认: 200x200)')
    parser.add_argument('-m', '--min-size', type=int, default=MIN_IMAGE_SIZE,
                        help=f'最小文件大小，单位为字节 (默认: {MIN_IMAGE_SIZE})')
    parser.add_argument('--prefix', type=str, default=IMAGE_PREFIX,
                        help=f'图片文件名前缀 (默认: {IMAGE_PREFIX})')
    parser.add_argument('--no-smart-naming', action='store_true',
                        help='禁用智能命名功能')
                        
    args = parser.parse_args()
    
    # 处理线程数
    MAX_THREADS = args.threads
    
    # 处理最小文件大小
    MIN_IMAGE_SIZE = args.min_size
    
    # 处理去重设置
    ENABLE_DEDUPLICATION = not args.no_dedup
    
    # 处理图片前缀
    IMAGE_PREFIX = args.prefix
    
    # 处理智能命名设置
    USE_SMART_NAMING = not args.no_smart_naming
    
    # 处理图片前缀
    IMAGE_PREFIX = args.prefix
    
    USE_SMART_NAMING = not args.no_smart_naming
    
    return args

if __name__ == "__main__":
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
    print(f"- 图片源: {', '.join(sources if sources else DEFAULT_SOURCES)}")
    print(f"- 多线程下载: {MAX_THREADS} 线程")
    print(f"- 图片去重: {'启用' if ENABLE_DEDUPLICATION else '禁用'}")
    print(f"- 最小分辨率: {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]}")
    print(f"- 最小文件大小: {MIN_IMAGE_SIZE} 字节")
    
    # 记录开始时间
    start_time = time.time()
    
    # 执行下载
    download_images_from_multiple_sources(search_query, target_count, save_directory, sources)
    
    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n总运行时间: {total_time:.1f} 秒")