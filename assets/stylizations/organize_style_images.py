import os
import shutil
import re

def organize_style_images():
    """按照图片名中的style_标记将图片分类到不同文件夹"""
    # 当前目录作为源目录
    source_dir = "."
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    print("正在搜索包含'style_'的图片...")
    
    # 查找所有包含'style_'的图片并按样式分组
    style_images = {}
    for file in os.listdir(source_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions) and "style_" in file:
            # 提取样式名: 'style_'之后直到下一个'_'或文件扩展名
            match = re.search(r'style_([^_\.]+)', file)
            if match:
                style_name = match.group(1)
                if style_name not in style_images:
                    style_images[style_name] = []
                style_images[style_name].append(file)
    
    if not style_images:
        print("没有找到包含'style_'的图片文件")
        return
    
    print(f"找到 {sum(len(files) for files in style_images.values())} 个图片，分属 {len(style_images)} 种样式")
    
    # 为每个样式创建目录并移动文件
    for style_name, files in style_images.items():
        style_dir = os.path.join(source_dir, f"style_{style_name}")
        if not os.path.exists(style_dir):
            os.makedirs(style_dir)
            print(f"创建文件夹: {style_dir}")
        
        for file in files:
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(style_dir, file)
            shutil.move(source_path, destination_path)
            print(f"移动: {file} -> {style_dir}")
    
    print("文件分类完成！所有原始文件已移动到对应文件夹")

if __name__ == "__main__":
    organize_style_images()
