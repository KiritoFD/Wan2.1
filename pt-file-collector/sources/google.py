#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Google 图片源模块
"""

import os
import time
import random
import requests
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Set

from .source_base import ImageSource

# 尝试导入配置
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CONFIG
    from core.downloader import ImageDownloader
except ImportError:
    CONFIG = {"request_timeout": 30, "page_delay": (1, 3)}
    # 创建简单的下载器类
    class ImageDownloader:
        def __init__(self, save_dir, source_name=None, metadata=None):
            self.save_dir = save_dir
        def download_batch(self, url_list, start_count=0, max_count=None):
            return start_count

class GoogleImageSource(ImageSource):
    def __init__(self):
        super().__init__("google")
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
            "Referer": "https://www.google.com/"
        }
        self.base_url = "https://www.google.com/search"
        self.metadata = {}  # 存储图片元数据
    
    def search(self, query, limit=100):
        """从Google搜索图片并返回URL列表"""
        print(f"正在从Google搜索图片: {query}")
        image_urls = set()
        
        # 尝试使用不同的方法搜索Google图片
        success = False
        
        # 方法1: 直接搜索
        success = self._search_direct_method(query, image_urls, limit)
        
        # 方法2: 如果直接方法失败，尝试使用Google图片搜索镜像
        if not success or len(image_urls) < limit // 2:
            self._search_alternative_method(query, image_urls, limit)
        
        print(f"Google搜索完成，总共找到 {len(image_urls)} 个图片URL")
        return list(image_urls)

    def _search_direct_method(self, query, image_urls, limit):
        """使用直接方法从Google搜索图片"""
        success = False
        # 搜索多页结果
        for page in range(0, 3):  # 减少页数以避免过多失败
            if len(image_urls) >= limit:
                break
                
            params = {
                "q": query,
                "tbm": "isch",  # 图片搜索
                "ijn": page,     # 页码
                "start": page * 100,
                "tbs": "isz:l"  # 大尺寸图片
            }
            
            try:
                # 设置更长的超时时间和更多的重试次数
                response = self._make_request(
                    self.base_url, 
                    params, 
                    timeout=60  # 延长超时时间
                )
                if not response:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 提取图片URL
                # 查找JavaScript中的图片URL
                self._extract_from_scripts(soup, image_urls)
                
                # 查找常规图片标签
                self._extract_from_img_tags(soup, image_urls)
                    
                # 查找srcset属性
                self._extract_from_srcsets(soup, image_urls)
                
                print(f"Google搜索第{page+1}页，找到 {len(image_urls)} 个图片URL")
                success = True  # 至少有一个请求成功
                
            except Exception as e:
                print(f"处理Google搜索结果时出错: {str(e)}")
            
            # 暂停较长时间，避免被Google封锁
            self._wait_between_requests((5, 10))
        
        return success

    def _search_alternative_method(self, query, image_urls, limit):
        """使用替代方法从Google获取图片"""
        # 尝试使用Google图片的备用API或方法
        alternative_urls = [
            # 使用Bing作为备用搜索引擎
            f"https://www.bing.com/images/search?q={query}&form=HDRSC2",
            # 可以添加其他备用源
        ]
        
        for alt_url in alternative_urls:
            if len(image_urls) >= limit:
                break
                
            try:
                print(f"尝试使用替代方法搜索图片: {alt_url}")
                response = requests.get(
                    alt_url,
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 提取图片URL (通用方法)
                    img_tags = soup.find_all("img")
                    for img in img_tags:
                        for attr in ["src", "data-src"]:
                            img_url = img.get(attr, "")
                            if img_url and img_url.startswith("http") and not img_url.startswith("data:"):
                                image_urls.add(img_url)
                    
                    print(f"替代搜索方法找到 {len(image_urls)} 个图片URL")
                    
            except Exception as e:
                print(f"替代搜索方法出错: {str(e)}")
                
            self._wait_between_requests((3, 6))
    
    def _extract_from_scripts(self, soup, image_urls):
        """从脚本标签中提取图片URL"""
        script_tags = soup.find_all("script")
        for script in script_tags:
            script_content = str(script)
            # 查找图片URL
            img_matches = re.findall(r'(?:http[s]?://[^"\']+\.(?:jpg|jpeg|png|gif|webp))', script_content, re.IGNORECASE)
            for img_url in img_matches:
                if img_url not in image_urls and not img_url.startswith("data:"):
                    image_urls.add(img_url)
                    
                    # 尝试提取图片标题和来源
                    try:
                        title_match = re.search(rf'{re.escape(img_url)}[^"]*?"([^"]+)"', script_content)
                        site_match = re.search(rf'{re.escape(img_url)}[^"]*?"([^"]+\.[^"]+)"', script_content)
                        
                        title = title_match.group(1) if title_match else ""
                        site = site_match.group(1) if site_match else ""
                        
                        self.metadata[img_url] = {
                            "title": title,
                            "site": site
                        }
                    except:
                        pass
    
    def _extract_from_img_tags(self, soup, image_urls):
        """从img标签中提取图片URL"""
        img_tags = soup.find_all("img")
        for img in img_tags:
            img_url = img.get("src", "")
            if img_url and not img_url.startswith("data:") and img_url not in image_urls:
                if img_url.startswith("http"):
                    image_urls.add(img_url)
                    
                    # 尝试提取alt文本作为标题
                    alt_text = img.get("alt", "")
                    if alt_text:
                        self.metadata[img_url] = {"title": alt_text}
    
    def _extract_from_srcsets(self, soup, image_urls):
        """从srcset属性中提取图片URL"""
        elements_with_srcset = soup.select("[srcset]")
        for element in elements_with_srcset:
            srcset = element.get("srcset", "")
            urls = re.findall(r'(https?://[^\s]+)', srcset)
            for url in urls:
                if url and not url.startswith("data:") and url not in image_urls:
                    image_urls.add(url)
    
    def download(self, query, num_images, save_dir, start_count=0):
        """从Google搜索并下载图片"""
        print(f"\n===== 从Google下载图片 =====")
        
        # 搜索图片
        image_urls = self.search(query, num_images * 2)
        
        # 下载图片
        if image_urls:
            downloader = ImageDownloader(save_dir, source_name="google", metadata=self.metadata)
            new_count = downloader.download_batch(image_urls, start_count, num_images)
            return new_count
        
        return start_count
