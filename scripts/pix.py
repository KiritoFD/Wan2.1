import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from PIL import Image
from scipy import ndimage

def pixel_art_transform(
    image_path, 
    output_path, 
    target_size=(64, 64), 
    num_colors=16, 
    edge_strength=1.0,
    preserve_aspect_ratio=True,
    detail_level=0.5,
    custom_palette=None
):
    """
    将图片转换为像素风风格，保持边缘和视觉特征。
    
    参数：
    - image_path: 输入图片路径
    - output_path: 输出图片路径
    - target_size: 目标分辨率 (宽度, 高度)
    - num_colors: 颜色调节数量
    - edge_strength: 边缘强度，0.0-2.0之间
    - preserve_aspect_ratio: 是否保持纵横比
    - detail_level: 细节保留程度，0.0-1.0之间
    - custom_palette: 自定义调色板，格式为 [(B,G,R), (B,G,R), ...]
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图片，请检查路径是否正确！")
    
    # 计算工作尺寸
    h, w = img.shape[:2]
    
    # 保持纵横比
    if preserve_aspect_ratio:
        if w > h:
            new_w = target_size[0]
            new_h = int(h * (new_w / w))
            if new_h == 0: new_h = 1
        else:
            new_h = target_size[1]
            new_w = int(w * (new_h / h))
            if new_w == 0: new_w = 1
        resize_dims = (new_w, new_h)
    else:
        resize_dims = target_size
    
    # 计算合适的工作尺寸，保留更多细节
    work_size = (int(resize_dims[0] * (1 + detail_level)), 
                int(resize_dims[1] * (1 + detail_level)))
    
    # 缩放到工作尺寸，使用INTER_AREA获得更好的降采样效果
    work_img = cv2.resize(img, work_size, interpolation=cv2.INTER_AREA)
    
    # 步骤1: 边缘检测和增强
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    
    # 使用双边滤波保持边缘的同时减少纹理噪声
    smooth = cv2.bilateralFilter(work_img, 9, 75, 75)
    
    # 自适应边缘检测
    edges = cv2.Canny(gray, 100, 200)
    
    # 边缘膨胀，确保边缘被保留
    kernel = np.ones((2, 2), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 创建边缘掩码 - 边缘附近的区域会保留更多细节
    edge_mask = cv2.GaussianBlur(dilated_edges.astype(float), (7, 7), 0) * edge_strength
    edge_mask = np.clip(edge_mask, 0, 255) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=-1)  # 转换为3通道
    
    # 步骤2: 区域分割和区域内颜色简化
    # 使用分水岭算法或超像素分割会更复杂，这里使用简化版本
    
    # 使用Mean Shift进行颜色分割
    shifted = cv2.pyrMeanShiftFiltering(smooth, 10, 25)
    
    # 步骤3: 颜色量化
    pixels = shifted.reshape(-1, 3)  # 展平
    
    if custom_palette is not None:
        # 使用自定义调色板
        custom_palette = np.array(custom_palette, dtype=np.float32)
        distances = np.sqrt(((pixels[:, np.newaxis, :] - custom_palette[np.newaxis, :, :])**2).sum(axis=2))
        nearest_indices = np.argmin(distances, axis=1)
        quantized_colors = custom_palette[nearest_indices]
    else:
        # 使用K-Means聚类
        kmeans = KMeans(n_clusters=num_colors, random_state=0, n_init=10).fit(pixels)
        quantized_colors = kmeans.cluster_centers_[kmeans.labels_]
    
    quantized_img = quantized_colors.reshape(shifted.shape).astype(np.uint8)
    
    # 步骤4: 智能细节合并 - 在边缘处保留更多细节
    # 边缘处使用原始图像和量化图像的混合
    enhanced = work_img * edge_mask + quantized_img * (1 - edge_mask * 0.5)
    enhanced = enhanced.astype(np.uint8)
    
    # 步骤5: 下采样到目标像素尺寸，使像素感更明显
    pixel_art = cv2.resize(enhanced, resize_dims, interpolation=cv2.INTER_AREA)
    
    # 锐化边缘，增强像素感
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    pixel_art = cv2.filter2D(pixel_art, -1, kernel)
    
    # 步骤6: 最终缩放 - 用最近邻插值放大以保持像素边缘锐利
    scale_factor = 4  # 可以根据需要调整
    final_size = (resize_dims[0] * scale_factor, resize_dims[1] * scale_factor)
    final_img = cv2.resize(pixel_art, final_size, interpolation=cv2.INTER_NEAREST)
    
    # 保存结果
    cv2.imwrite(output_path, final_img)
    return final_img

def generate_retro_palette(style="gameboy"):
    """
    生成预设的复古游戏调色板。
    
    参数：
    - style: 复古风格，可选 "gameboy", "nes", "cga", "c64", "zx"
    
    返回：
    - BGR格式的调色板列表
    """
    palettes = {
        "gameboy": [
            (155, 188, 15),   # 浅绿
            (139, 172, 15),   # 中绿
            (48, 98, 48),     # 深绿
            (15, 56, 15)      # 最深绿
        ],
        "nes": [
            (0, 0, 0),        # 黑色
            (255, 255, 255),  # 白色
            (216, 40, 0),     # 红色
            (76, 220, 72),    # 绿色
            (20, 20, 255),    # 蓝色
            (0, 255, 255),    # 青色
            (200, 68, 228),   # 品红
            (255, 230, 0)     # 黄色
        ],
        "cga": [
            (0, 0, 0),        # 黑色
            (0, 255, 255),    # 青色
            (255, 0, 255),    # 品红
            (255, 255, 255)   # 白色
        ]
    }
    
    if style in palettes:
        # 转换为BGR格式
        return [(b, g, r) for (r, g, b) in palettes[style]]
    else:
        raise ValueError(f"未知风格: {style}。可用选项: {list(palettes.keys())}")

# 批量处理函数
def batch_process_images(input_dir, output_dir, **kwargs):
    """
    批量处理文件夹中的所有图片。
    
    参数：
    - input_dir: 输入图片文件夹
    - output_dir: 输出图片文件夹
    - **kwargs: pixel_art_transform函数的其他参数
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 支持的图片格式
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 查找所有图片
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"pixel_{filename}")
            
            try:
                pixel_art_transform(input_path, output_path, **kwargs)
                print(f"处理成功: {filename}")
            except Exception as e:
                print(f"处理失败: {filename}，错误: {str(e)}")

# 添加命令行功能
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='将图片转换为像素风格。')
    parser.add_argument('input', help='输入图片路径或包含图片的文件夹路径')
    parser.add_argument('--out-dir', default='pix_out', help='输出目录，默认为pix_out')
    parser.add_argument('--size', type=int, default=64, help='目标像素尺寸，默认为64')
    parser.add_argument('--colors', type=int, default=16, help='颜色数量，默认为16')
    parser.add_argument('--edge', type=float, default=1.0, help='边缘强度(0.0-2.0)，默认为1.0')
    parser.add_argument('--detail', type=float, default=0.5, help='细节保留程度(0.0-1.0)，默认为0.5')
    parser.add_argument('--style', choices=['gameboy', 'nes', 'cga'], help='使用预设复古风格')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"创建输出目录: {args.out_dir}")
    
    # 参数验证
    if args.edge < 0.0 or args.edge > 2.0:
        print("错误: 边缘强度必须在0.0到2.0之间")
        sys.exit(1)
    
    if args.detail < 0.0 or args.detail > 1.0:
        print("错误: 细节保留程度必须在0.0到1.0之间")
        sys.exit(1)
    
    # 准备参数
    params = {
        'target_size': (args.size, args.size),
        'num_colors': args.colors,
        'edge_strength': args.edge,
        'detail_level': args.detail
    }
    
    # 使用预设风格
    if args.style:
        try:
            params['custom_palette'] = generate_retro_palette(args.style)
            print(f"使用 {args.style} 风格调色板")
        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
    
    # 处理输入路径
    if os.path.isdir(args.input):
        # 批量处理整个目录
        print(f"批量处理目录: {args.input}")
        batch_process_images(args.input, args.out_dir, **params)
        print(f"完成! 所有图片已保存到 {args.out_dir}")
    elif os.path.isfile(args.input):
        # 处理单个文件
        try:
            filename = os.path.basename(args.input)
            output_path = os.path.join(args.out_dir, f"pixel_{filename}")
            print(f"处理图片: {args.input}")
            pixel_art_transform(args.input, output_path, **params)
            print(f"完成! 图片已保存到 {output_path}")
        except Exception as e:
            print(f"处理失败: {str(e)}")
            sys.exit(1)
    else:
        print(f"错误: 输入路径不存在 - {args.input}")
        sys.exit(1)

# 示例命令行用法:
# python pix.py input.jpg --colors 8 --size 32 --dithering
# python pix.py input_folder --out-dir my_pixel_art --style gameboy