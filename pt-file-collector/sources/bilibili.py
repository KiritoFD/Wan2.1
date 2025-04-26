#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bilibili 图片源模块
"""

import os
import time
import random
import requests
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Set, Any

from .source_base import ImageSource

# 尝试导入配置，如果找不到则使用默认值
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CONFIG
    from core.downloader import ImageDownloader
except ImportError:
    CONFIG = {"request_timeout": 30, "page_delay": (1, 3)}
    # 创建一个简单的下载器类
    class ImageDownloader:
        def __init__(self, save_dir, source_name=None, metadata=None):
            self.save_dir = save_dir
            self.source_name = source_name
            self.metadata = metadata or {}
            
        def download_batch(self, url_list, start_count=0, max_count=None):
            print(f"将下载 {len(url_list)} 个图片URL")
            return start_count

class BilibiliImageSource(ImageSource):
    def __init__(self):
        super().__init__("bilibili")
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Referer": "https://www.bilibili.com/",
            "Origin": "https://www.bilibili.com",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            # 增加更多请求头，使请求看起来更像浏览器
            "Sec-Ch-Ua": '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Connection": "keep-alive",
            # 添加一些必要的cookie(这里示例)
            "Cookie": "buvid3=51337E46-CE8F-D3E9-8D47-204B7CAF7D8937326infoc; innersign=0"
        }
        self.api_url = "https://api.bilibili.com/x/web-interface/search/type"
        self.web_url = "https://search.bilibili.com/all"
        self.metadata = {}  # 存储图片元数据
    
    def search(self, query, limit=100):
        """从Bilibili搜索图片并返回URL列表"""
        print(f"正在从Bilibili搜索图片: {query}")
        image_urls = set()
        
        # 方法1: API搜索
        self._search_from_api(query, image_urls, limit)
        
        # 方法2: 网页搜索，如果API方法找到的图片不够
        if len(image_urls) < limit:
            self._search_from_web(query, image_urls, limit)
        
        print(f"B站搜索完成，总共找到 {len(image_urls)} 个图片URL")
        return list(image_urls)
    
    def _search_from_api(self, query, image_urls, limit):
        """使用B站API搜索图片"""
        # 减慢请求速度，增加页面间隔
        for page in range(1, 4):  # 减少页数，避免触发反爬限制
            if len(image_urls) >= limit * 2:  # 获取比需要的多一倍，以应对下载失败
                break
                
            params = {
                "keyword": query,
                "search_type": "photo",  # 搜索类型为相册
                "page": page,
                "order": "totalrank",    # 默认排序
                "category_id": 0,        # 全部分区
                "jsonp": "jsonp"
            }
            
            try:
                # 增加请求间隔
                self._wait_between_requests((5, 10))
                
                response = self._make_request(self.api_url, params)
                # 如果遇到412状态码，尝试使用备用方法
                if not response:
                    print(f"API请求失败，尝试使用网页搜索方法...")
                    break
                    
                data = response.json()
                if data.get("code") != 0:
                    print(f"B站API返回错误: {data.get('message', '未知错误')}")
                    continue
                    
                # 提取图片URL
                result_data = data.get("data", {}).get("result", [])
                for item in result_data:
                    # 提取图片链接和元数据
                    item_id = item.get("id", "")
                    title = item.get("title", "")
                    author = item.get("author", "")
                    
                    # 提取图片链接
                    if "pic" in item:
                        img_url = item["pic"]
                        if img_url and img_url.startswith("http") and img_url not in image_urls:
                            image_urls.add(img_url)
                            self.metadata[img_url] = {
                                "id": item_id,
                                "title": self._clean_title(title),
                                "author": author
                            }
                            
                    # 提取封面图
                    if "cover" in item:
                        img_url = item["cover"]
                        if img_url and img_url.startswith("http") and img_url not in image_urls:
                            image_urls.add(img_url)
                            self.metadata[img_url] = {
                                "id": item_id,
                                "title": self._clean_title(title),
                                "author": author
                            }
                
                print(f"B站API搜索第{page}页，当前共找到 {len(image_urls)} 个图片URL")
                
            except Exception as e:
                print(f"处理B站API搜索请求时出错: {str(e)}")
                
            # 增加请求间隔
            self._wait_between_requests((5, 10))
    
    def _search_from_web(self, query, image_urls, limit):
        """从B站网页版搜索图片 - 增强版"""
        # 尝试从多个页面搜索
        for page in range(1, 4):
            params = {
                "keyword": query,
                "from_source": "web_search",
                "page": page
            }
            
            try:
                # 增加请求间隔
                self._wait_between_requests((5, 8))
                
                # 使用更常见的浏览器类型访问
                temp_headers = self.headers.copy()
                temp_headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
                
                # 直接使用requests而不是_make_request，以便自定义头
                try:
                    response = requests.get(
                        self.web_url,
                        params=params,
                        headers=temp_headers,
                        timeout=30
                    )
                    
                    if response.status_code == 412:
                        print("遇到反爬虫措施，尝试备用方法...")
                        self._try_alternative_search(query, image_urls, limit)
                        break
                        
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # 提取视频封面图片
                        img_tags = soup.select("img.img-content, img.bili-video-card__cover")
                        for img in img_tags:
                            img_url = img.get("src", "") or img.get("data-src", "")
                            if not img_url:
                                continue
                                
                            if not img_url.startswith("http"):
                                img_url = "https:" + img_url
                                
                            if img_url and img_url not in image_urls:
                                # 尝试获取视频标题
                                title_tag = img.find_parent().find_parent().select_one(".bili-video-card__info--tit")
                                title = title_tag.text.strip() if title_tag else ""
                                
                                # 尝试获取UP主
                                up_tag = img.find_parent().find_parent().select_one(".bili-video-card__info--author")
                                author = up_tag.text.strip() if up_tag else ""
                                
                                image_urls.add(img_url)
                                self.metadata[img_url] = {
                                    "title": self._clean_title(title),
                                    "author": author
                                }
                                
                        # 提取用户头像
                        avatar_tags = soup.select("img.bili-avatar-img")
                        for img in avatar_tags:
                            img_url = img.get("src", "")
                            if not img_url.startswith("http"):
                                img_url = "https:" + img_url
                            if img_url and img_url not in image_urls:
                                # 尝试获取用户名
                                name_tag = img.find_parent().find_next("span", class_="bili-username")
                                author = name_tag.text.strip() if name_tag else ""
                                
                                image_urls.add(img_url)
                                self.metadata[img_url] = {
                                    "title": "用户头像",
                                    "author": author
                                }
                        
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
                    print(f"网页请求失败: {str(e)}")
            
            except Exception as e:
                print(f"处理B站网页搜索时出错: {str(e)}")
                
            # 增加请求间隔
            self._wait_between_requests((5, 10))
    
    def _try_alternative_search(self, query, image_urls, limit):
        """当常规B站搜索失败时使用备用方法"""
        print("尝试使用备用方法搜索B站图片...")
        
        # 使用B站视频搜索获取封面图
        try:
            video_url = f"https://api.bilibili.com/x/web-interface/search/all/v2?keyword={query}&page=1"
            
            temp_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
                "Referer": "https://www.bilibili.com/",
            }
            
            response = requests.get(video_url, headers=temp_headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0:
                    # 从各种内容中提取图片
                    result_data = []
                    
                    # 提取视频搜索结果
                    for item in data.get("data", {}).get("result", []):
                        if item.get("result_type") == "video":
                            result_data.extend(item.get("data", []))
                    
                    # 从视频结果中提取图片
                    for item in result_data:
                        # 提取视频封面
                        if "pic" in item:
                            img_url = item["pic"]
                            if img_url and img_url.startswith("http") and img_url not in image_urls:
                                image_urls.add(img_url)
                                
                                # 保存元数据
                                self.metadata[img_url] = {
                                    "id": item.get("bvid", ""),
                                    "title": self._clean_title(item.get("title", "")),
                                    "author": item.get("author", "")
                                }
                    
                    print(f"备用方法找到 {len(image_urls)} 个图片URL")
        except Exception as e:
            print(f"备用搜索方法失败: {str(e)}")
    
    def download(self, query, num_images, save_dir, start_count=0):
        """从Bilibili搜索并下载图片"""
        print(f"\n===== 从Bilibili下载图片 =====")
        
        # 搜索图片
        image_urls = self.search(query, num_images * 2)
        
        # 下载图片
        if image_urls:
            downloader = ImageDownloader(save_dir, source_name="bilibili", metadata=self.metadata)
            new_count = downloader.download_batch(image_urls, start_count, num_images)
            return new_count
        
        return start_count
    
    def _clean_title(self, title):
        """清理标题中的HTML标签和特殊字符"""
        # 移除HTML标签
        title = re.sub(r'<[^>]+>', '', title)
        # 移除特殊字符
        title = re.sub(r'[\\/*?:"<>|]', '', title)
        # 限制长度
        return title[:50] if title else ""
