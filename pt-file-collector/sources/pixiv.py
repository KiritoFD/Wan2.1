#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pixiv 图片源模块
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

class PixivImageSource(ImageSource):
    def __init__(self):
        super().__init__("pixiv")
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Referer": "https://www.pixiv.net/",
            "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cookie": "",  # 可能需要有效的cookie才能访问
            "Accept": "application/json"
        }
        self.base_url = "https://www.pixiv.net/ajax/search/illustrations"
        self.metadata = {}  # 存储图片元数据
    
    def search(self, query, limit=100):
        """从Pixiv搜索图片并返回URL列表"""
        print(f"正在从Pixiv搜索图片: {query}")
        image_urls = set()
        
        # 搜索多页结果
        for page in range(1, 6):  # 尝试获取5页的图片
            if len(image_urls) >= limit:
                break
                
            params = {
                "word": query,
                "order": "date_d",     # 按日期降序排序
                "mode": "all",         # 所有作品
                "p": page,             # 页码
                "s_mode": "s_tag",     # 按标签搜索
                "type": "illust",      # 插画类型
                "lang": "zh"           # 语言
            }
            
            try:
                response = self._make_request(self.base_url, params)
                if not response:
                    continue
                    
                data = response.json()
                if data.get("error") or not data.get("body"):
                    print(f"Pixiv API返回错误: {data.get('message', '未知错误')}")
                    continue
                    
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
                                    self.metadata[img_url] = {
                                        "title": illust["title"],
                                        "artist": illust["userName"],
                                        "id": illust.get("id", "")
                                    }
                    except KeyError as e:
                        print(f"处理Pixiv图片项出错: {str(e)}")
                
                print(f"Pixiv搜索第{page}页，当前共找到 {len(image_urls)} 个图片URL")
                
            except Exception as e:
                print(f"处理Pixiv搜索请求时出错: {str(e)}")
                
            self._wait_between_requests()
        
        # 尝试从网页版Pixiv获取更多图片
        if len(image_urls) < limit:
            self._search_from_web(query, image_urls, limit)
            
        print(f"Pixiv搜索完成，总共找到 {len(image_urls)} 个图片URL")
        return list(image_urls)
    
    def _search_from_web(self, query, image_urls, limit):
        """从Pixiv网页获取图片 - 增强版"""
        try:
            # 使用替代方法，例如通过其他网站获取Pixiv图片
            urls_to_try = [
                f"https://www.pixiv.net/tags/{query}/illustrations",
                # 可能的替代URL
                f"https://www.pixiv.net/search.php?word={query}&s_mode=s_tag"
            ]
            
            for pixiv_web_url in urls_to_try:
                try:
                    # 使用特殊处理
                    enhanced_headers = self.headers.copy()
                    enhanced_headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
                    
                    web_response = requests.get(
                        pixiv_web_url,
                        headers=enhanced_headers,
                        timeout=60  # 延长超时时间
                    )
                    
                    if web_response.status_code != 200:
                        continue
                        
                    soup = BeautifulSoup(web_response.text, 'html.parser')
                    
                    # 从页面提取图片URLs (多种方法)
                    # 方法1: 从JSON数据中提取
                    extracted = self._extract_from_json(soup, image_urls)
                    
                    # 方法2: 从页面元素中提取
                    if not extracted:
                        extracted = self._extract_from_elements(soup, image_urls)
                    
                    if extracted:
                        print(f"Pixiv网页搜索成功，当前共找到 {len(image_urls)} 个图片URL")
                        break
                        
                except Exception as e:
                    print(f"此Pixiv URL处理失败 {pixiv_web_url}: {str(e)}")
                    
                # 增加请求间隔
                self._wait_between_requests((5, 8))
                    
        except Exception as e:
            print(f"处理Pixiv网页请求时出错: {str(e)}")

    def _extract_from_json(self, soup, image_urls):
        """从JSON数据中提取图片URL"""
        extracted = False
        try:
            # 查找所有脚本标签
            script_tags = soup.find_all("script")
            for script in script_tags:
                script_content = str(script)
                # 查找包含图片数据的JSON
                for json_pattern in [
                    r'pixiv\.context\.preload\s*=\s*({.*?});</script>', 
                    r'window\.__PRELOAD_STATE__\s*=\s*({.*?});</script>'
                ]:
                    json_data = re.search(json_pattern, script_content, re.DOTALL)
                    if json_data:
                        try:
                            data = json.loads(json_data.group(1))
                            # 递归搜索JSON中的图片URL
                            self._search_json_for_images(data, image_urls)
                            extracted = True
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            print(f"JSON提取错误: {str(e)}")
        return extracted

    def _search_json_for_images(self, data, image_urls):
        """递归搜索JSON中的图片URL"""
        if isinstance(data, dict):
            for key, value in data.items():
                # 查找可能包含图片URL的键
                if isinstance(value, str) and ("i.pximg.net" in value or "pixiv.net" in value) and value.endswith((".jpg", ".png", ".gif")):
                    img_url = value.replace("i.pximg.net", "i.pixiv.cat")
                    if img_url not in image_urls:
                        image_urls.add(img_url)
                        # 尝试提取相关元数据
                        if "title" in data:
                            self.metadata[img_url] = {"title": data["title"]}
                        if "userName" in data:
                            self.metadata.setdefault(img_url, {})["artist"] = data["userName"]
                elif isinstance(value, (dict, list)):
                    self._search_json_for_images(value, image_urls)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._search_json_for_images(item, image_urls)

    def _extract_from_elements(self, soup, image_urls):
        """从页面元素中提取图片URL"""
        extracted = False
        try:
            # 查找所有可能包含图片的元素
            for img_selector in [
                "img._2WwRD0o", # Pixiv图片类名
                "img[src*='i.pximg.net']",
                "img[data-src*='i.pximg.net']"
            ]:
                img_tags = soup.select(img_selector)
                for img in img_tags:
                    for attr in ["src", "data-src"]:
                        img_url = img.get(attr, "")
                        if img_url and "i.pximg.net" in img_url:
                            img_url = img_url.replace("i.pximg.net", "i.pixiv.cat")
                            if img_url not in image_urls:
                                image_urls.add(img_url)
                                
                                # 尝试提取元数据
                                title_tag = img.get("alt", "")
                                self.metadata[img_url] = {"title": title_tag}
                                extracted = True
        except Exception as e:
            print(f"元素提取错误: {str(e)}")
        return extracted

    def download(self, query, num_images, save_dir, start_count=0):
        """从Pixiv搜索并下载图片"""
        print(f"\n===== 从Pixiv下载图片 =====")
        
        # 搜索图片
        image_urls = self.search(query, num_images * 2)
        
        # 下载图片
        if image_urls:
            # 为下载器提供特殊的Referer头以绕过防盗链
            downloader = ImageDownloader(save_dir, source_name="pixiv", metadata=self.metadata)
            new_count = downloader.download_batch(image_urls, start_count, num_images)
            return new_count
        
        return start_count
