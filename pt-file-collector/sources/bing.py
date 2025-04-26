#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bing 图片源模块
"""

import os
import time
import random
import requests
import json
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
    class ImageDownloader:
        def __init__(self, save_dir, source_name=None, metadata=None):
            self.save_dir = save_dir
        def download_batch(self, url_list, start_count=0, max_count=None):
            return start_count

class BingImageSource(ImageSource):
    def __init__(self):
        super().__init__("bing")
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
            "Referer": "https://www.bing.com/images/search"
        }
        self.base_url = "https://www.bing.com/images/search"
        self.api_url = "https://www.bing.com/images/async"
        self.metadata = {}  # 存储图片元数据
    
    def search(self, query, limit=100):
        """从Bing搜索图片并返回URL列表"""
        print(f"正在从Bing搜索图片: {query}")
        image_urls = set()
        
        # 首先获取初始页面以获取必要的cookie和参数
        try:
            initial_params = {
                "q": query,
                "form": "HDRSC2",
                "first": 1
            }
            
            response = self._make_request(self.base_url, initial_params)
            if not response:
                return []
                
            # 获取IG和SFX参数，这些是Bing API请求所必需的
            ig_match = re.search(r'IG:"([^"]+)"', response.text)
            ig = ig_match.group(1) if ig_match else ""
            
            sfx_match = re.search(r'_SFX:"([^"]+)"', response.text)
            sfx = sfx_match.group(1) if sfx_match else ""
            
            if not ig or not sfx:
                print("无法从Bing页面获取必要参数")
                return []
                
            # 从初始页面提取图片URL
            soup = BeautifulSoup(response.text, 'html.parser')
            initial_images = self._extract_image_urls(soup)
            for img_url in initial_images:
                if img_url not in image_urls and not img_url.startswith("data:"):
                    image_urls.add(img_url)
            
            # 然后发送异步请求获取更多图片
            for offset in range(1, 5):  # 尝试获取最多5页
                if len(image_urls) >= limit:
                    break
                    
                async_params = {
                    "q": query,
                    "first": offset * 35 + 1,
                    "count": 35,
                    "relp": 35,
                    "tsc": "ImageHoverTitle",
                    "layout": "RowBased",
                    "mmasync": 1,
                    "dgState": "c*5_y*1632s1812s1736s1755s1642_i*36_d*20", 
                    "IG": ig,
                    "SFX": sfx
                }
                
                try:
                    async_response = self._make_request(self.api_url, async_params)
                    if not async_response:
                        continue
                        
                    async_soup = BeautifulSoup(async_response.text, 'html.parser')
                    batch_urls = self._extract_image_urls(async_soup)
                    
                    for img_url in batch_urls:
                        if img_url not in image_urls and not img_url.startswith("data:"):
                            image_urls.add(img_url)
                            
                    print(f"Bing搜索第{offset+1}页，当前共找到 {len(image_urls)} 个图片URL")
                    
                except Exception as e:
                    print(f"处理Bing异步请求时出错: {str(e)}")
                    
                self._wait_between_requests()
                
        except Exception as e:
            print(f"Bing搜索出错: {str(e)}")
            
        print(f"Bing搜索完成，总共找到 {len(image_urls)} 个图片URL")
        return list(image_urls)
    
    def _extract_image_urls(self, soup):
        """从Bing搜索结果HTML中提取图片URL"""
        urls = set()
        
        # 方法1：从图片标签获取
        for img in soup.select(".mimg, .rms_img"):
            src = img.get("src", "")
            data_src = img.get("data-src", "")
            
            if src and src.startswith("http"):
                urls.add(src)
                # 尝试获取图片标题和源网站
                try:
                    parent = img.find_parent("a")
                    if parent:
                        title = parent.get("t", "") or parent.get("title", "")
                        href = parent.get("href", "")
                        
                        if title and src not in self.metadata:
                            self.metadata[src] = {"title": title}
                        if href:
                            self.metadata.setdefault(src, {})["site"] = href
                except:
                    pass
                
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
                            # 保存元数据
                            if "t" in m_data and img_url not in self.metadata:
                                self.metadata[img_url] = {"title": m_data["t"]}
                            if "purl" in m_data:
                                self.metadata.setdefault(img_url, {})["site"] = m_data["purl"]
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
                    
                    # 尝试提取标题
                    title_match = re.search(rf'"t":"([^"]+)"[^"]*?"murl":"{re.escape(url)}"', script_text)
                    if title_match and url not in self.metadata:
                        self.metadata[url] = {"title": title_match.group(1)}
        
        return urls
    
    def download(self, query, num_images, save_dir, start_count=0):
        """从Bing搜索并下载图片"""
        print(f"\n===== 从Bing下载图片 =====")
        
        # 搜索图片
        image_urls = self.search(query, num_images * 2)
        
        # 下载图片
        if image_urls:
            downloader = ImageDownloader(save_dir, source_name="bing", metadata=self.metadata)
            new_count = downloader.download_batch(image_urls, start_count, num_images)
            return new_count
        
        return start_count
