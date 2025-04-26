#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图片源基类 - 定义所有图片源的通用接口和方法
"""

import os
import time
import random
import requests
from typing import Dict, Any, Optional, List

# 尝试导入配置
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import CONFIG
except ImportError:
    CONFIG = {
        "request_timeout": 30,
        "page_delay": (1, 3),
        "retry_count": 2,
        "retry_delay": (1, 3),
        "user_agents": [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0"
        ]
    }

class ImageSource:
    """图片源抽象基类"""
    
    def __init__(self, name):
        self.name = name
        self.headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3",
        }
    
    def search(self, query: str, limit: int = 100) -> List[str]:
        """搜索图片并返回URL列表"""
        print(f"{self.name} 源的search方法未实现")
        return []
    
    def download(self, query: str, num_images: int, save_dir: str, start_count: int = 0) -> int:
        """搜索并下载图片"""
        print(f"{self.name} 源的download方法未实现")
        return start_count
    
    def _get_random_user_agent(self) -> str:
        """获取随机用户代理"""
        user_agents = CONFIG.get("user_agents", [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        ])
        return random.choice(user_agents)
    
    def _make_request(self, url: str, params: Optional[Dict] = None, 
                     method: str = "GET", timeout: Optional[int] = None) -> Optional[requests.Response]:
        """发送网络请求，带有重试机制"""
        timeout = timeout or CONFIG.get("request_timeout", 30)
        retry_count = CONFIG.get("retry_count", 2)
        
        for attempt in range(retry_count + 1):
            try:
                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        params=params,
                        headers=self.headers,
                        timeout=timeout
                    )
                else:
                    response = requests.post(
                        url,
                        json=params,
                        headers=self.headers,
                        timeout=timeout
                    )
                
                if response.status_code == 200:
                    return response
                
                print(f"{self.name} 请求失败，状态码: {response.status_code}")
                
            except Exception as e:
                print(f"{self.name} 请求出错 (尝试 {attempt+1}/{retry_count+1}): {str(e)}")
            
            # 重试前等待
            if attempt < retry_count:
                retry_delay = random.uniform(
                    CONFIG.get("retry_delay", (1, 3))[0], 
                    CONFIG.get("retry_delay", (1, 3))[1]
                )
                time.sleep(retry_delay)
        
        return None
    
    def _wait_between_requests(self, delay_range: Optional[tuple] = None) -> None:
        """请求间隔等待
        
        Args:
            delay_range: 自定义延迟范围元组(最小秒数, 最大秒数)
        """
        delay_range = delay_range or CONFIG.get("page_delay", (1, 3))
        time.sleep(random.uniform(delay_range[0], delay_range[1]))
