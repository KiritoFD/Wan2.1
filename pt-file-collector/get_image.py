#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
get_image.py - 多源图片搜索与下载工具

用途: 从多个图片源(Google, Bing, Flickr, Bilibili, Pixiv)搜索并下载图片。
支持命令行参数，可自定义关键词、数量和保存路径。
"""

import os
import time
import random
import requests
import argparse
import sys
from bs4 import BeautifulSoup
from urllib.parse import quote_plus, urlparse
import re
import json

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
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
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
    """从Bing搜索图片"""
    base_url = "https://www.bing.com/images/search"
    headers = {
        "User-Agent": get_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3"
    }
    
    img_count = start_count
    image_urls = set()
    
    # 搜索多页结果
    for page in range(0, 5):  # 尝试获取5页的图片
        if img_count - start_count >= num_images:
            break
            
        params = {
            "q": query,
            "form": "HDRSC2",
            "first": page * 35,
            "tsc": "ImageHoverTitle"
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"获取Bing搜索结果失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找JavaScript中的图片URL
            script_tags = soup.find_all("script")
            for script in script_tags:
                script_content = str(script)
                img_matches = re.findall(r'"murl":"([^"]+)"', script_content)
                for img_url in img_matches:
                    img_url = img_url.replace('\\', '')
                    if img_url not in image_urls and not img_url.startswith("data:"):
                        image_urls.add(img_url)
            
            print(f"Bing搜索第{page+1}页，找到 {len(image_urls)} 个图片URL")
            
        except Exception as e:
            print(f"处理Bing搜索结果时出错: {str(e)}")
            time.sleep(random.randint(2, 5))
            continue
            
        time.sleep(random.randint(1, 3))
    
    # 下载图片
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

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
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
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
            
        time.sleep(random.randint(1, 3))
    
    # 下载图片
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

def get_bilibili_images(query, num_images, save_dir, start_count=0):
    """从Bilibili搜索并下载图片"""
    base_url = "https://api.bilibili.com/x/web-interface/search/type"
    headers = {
        "User-Agent": get_user_agent(),
        "Referer": "https://www.bilibili.com/",
        "Origin": "https://www.bilibili.com",
    }
    
    img_count = start_count
    image_urls = set()
    
    # 搜索多页结果
    for page in range(1, 6):  # 尝试获取5页的图片
        if img_count - start_count >= num_images:
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
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"获取Bilibili搜索结果失败，状态码: {response.status_code}")
                continue
                
            data = response.json()
            if data.get("code") != 0:
                print(f"Bilibili API返回错误: {data.get('message')}")
                continue
                
            # 提取图片URL
            result_data = data.get("data", {}).get("result", [])
            for item in result_data:
                # 提取图片链接
                if "pic" in item:
                    img_url = item["pic"]
                    if img_url and img_url not in image_urls:
                        image_urls.add(img_url)
                        
                # 提取封面图
                if "cover" in item:
                    img_url = item["cover"]
                    if img_url and img_url not in image_urls:
                        image_urls.add(img_url)
            
            print(f"Bilibili搜索第{page}页，找到 {len(image_urls)} 个图片URL")
            
        except Exception as e:
            print(f"处理Bilibili搜索结果时出错: {str(e)}")
            time.sleep(random.randint(2, 4))
            continue
            
        time.sleep(random.randint(1, 3))
    
    # 下载图片
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

def get_pixiv_images(query, num_images, save_dir, start_count=0):
    """从Pixiv搜索并下载图片（注意：需要特殊处理以访问Pixiv）"""
    # 由于Pixiv有较强的访问限制，这里使用其公开的接口搜索图片
    base_url = "https://www.pixiv.net/ajax/search/illustrations"
    headers = {
        "User-Agent": get_user_agent(),
        "Referer": "https://www.pixiv.net/",
        "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cookie": ""  # 可能需要有效的cookie才能访问
    }
    
    img_count = start_count
    image_urls = set()
    
    # 搜索多页结果
    for page in range(1, 4):  # 尝试获取3页的图片
        if img_count - start_count >= num_images:
            break
            
        params = {
            "word": query,
            "order": "date_d",  # 按日期降序排序
            "mode": "all",      # 所有作品
            "p": page,          # 页码
            "s_mode": "s_tag",  # 按标签搜索
            "type": "illust",   # 插画类型
            "lang": "zh"        # 语言
        }
        
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"获取Pixiv搜索结果失败，状态码: {response.status_code}")
                continue
                
            try:
                data = response.json()
                if data.get("error") or not data.get("body"):
                    print(f"Pixiv API返回错误: {data.get('message', '未知错误')}")
                    continue
                    
                # 提取图片URL
                illustrations = data.get("body", {}).get("illust", {}).get("data", [])
                for illust in illustrations:
                    if "url" in illust and "regular" in illust["url"]:
                        img_url = illust["url"]["regular"]
                        # 替换Pixiv的代理URL以获取实际图片
                        img_url = img_url.replace("i.pximg.net", "i.pixiv.cat")
                        if img_url and img_url not in image_urls:
                            image_urls.add(img_url)
                
                print(f"Pixiv搜索第{page}页，找到 {len(image_urls)} 个图片URL")
            
            except json.JSONDecodeError:
                print("解析Pixiv返回的JSON数据失败")
                continue
                
        except Exception as e:
            print(f"处理Pixiv搜索结果时出错: {str(e)}")
            time.sleep(random.randint(2, 4))
            continue
            
        time.sleep(random.randint(2, 4))
    
    # 下载图片，需要特殊处理Pixiv的防盗链
    pixiv_headers = headers.copy()
    pixiv_headers["Referer"] = "https://www.pixiv.net/"
    
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
            
        try:
            # 对Pixiv图片使用特殊的下载处理
            headers = {"User-Agent": get_user_agent(), "Referer": "https://www.pixiv.net/"}
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            
            if response.status_code == 200:
                # 检查是否是图片内容
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image'):
                    print(f"跳过非图片内容: {url}")
                    continue
                    
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
                    
                file_path = os.path.join(save_dir, f"image_{img_count+1}.{extension}")
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"成功下载Pixiv图片 {img_count+1}: {url}")
                img_count += 1
            else:
                print(f"下载Pixiv图片失败，状态码 {response.status_code}: {url}")
                
        except Exception as e:
            print(f"下载Pixiv图片 {url} 时出错: {str(e)}")
            
        time.sleep(random.randint(1, 3))
    
    return img_count

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
            response = requests.get(base_url, params=params, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"获取Twitter搜索结果失败，状态码: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找图片标签
            img_tags = soup.find_all("img")
            for img in img_tags:
                img_url = img.get("src", "")
                
                # 过滤出Twitter图片URL (通常格式为https://pbs.twimg.com/media/...)
                if img_url and "pbs.twimg.com/media" in img_url and img_url not in image_urls:
                    # 尝试获取更高分辨率的图片
                    img_url = img_url.split("?")[0] + "?format=jpg&name=large"
                    image_urls.add(img_url)
            
            print(f"Twitter搜索第{page}页，找到 {len(image_urls)} 个图片URL")
            
        except Exception as e:
            print(f"处理Twitter搜索结果时出错: {str(e)}")
            time.sleep(random.randint(2, 4))
            continue
            
        time.sleep(random.randint(2, 4))
    
    # 下载图片
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

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
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
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
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

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
        response = requests.get(f"{base_url}{query_path}", headers=headers, timeout=30)
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
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

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
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
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
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

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
        response = requests.get(f"{base_url}{formatted_query}/", headers=headers, timeout=30)
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
    for url in image_urls:
        if img_count - start_count >= num_images:
            break
        if download_image(url, save_dir, img_count + 1):
            img_count += 1
    
    return img_count

def download_images_from_multiple_sources(query, num_images, output_dir, sources=None):
    """从多个来源下载指定数量的图片"""
    output_dir = create_directories(output_dir)
    
    # 默认使用所有图片源
    if sources is None:
        sources = ["google", "bing", "flickr", "bilibili", "pixiv", "twitter", "pinterest", "unsplash", "500px", "instagram"]
    
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
            
        print(f"从{source.capitalize()}下载了 {new_count - total_count} 张图片")
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
                        choices=['google', 'bing', 'flickr', 'bilibili', 'pixiv', 
                                'twitter', 'pinterest', 'unsplash', '500px', 'instagram'], 
                        default=None,
                        help='指定要使用的图片源 (默认: 使用所有图片源)')
    
    return parser.parse_args()

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
    print(f"- 图片源: {', '.join(sources)}")
    
    # 执行下载
    download_images_from_multiple_sources(search_query, target_count, save_directory, sources)