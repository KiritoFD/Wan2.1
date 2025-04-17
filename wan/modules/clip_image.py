import os
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# 导入Wan2.1的CLIP模块
from clip import CLIPModel
from easydict import EasyDict

def parse_args():
    parser = argparse.ArgumentParser(description='提取图像的CLIP特征')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出特征保存路径')
    parser.add_argument('--checkpoint_dir', type=str, default='Wan2.1-T2V-14B', help='模型权重目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批处理大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备(cuda:0, cpu)')
    return parser.parse_args()

def load_images(image_path, batch_size=1):
    """加载图像，支持单张图像或整个文件夹"""
    if os.path.isfile(image_path):
        img = Image.open(image_path).convert('RGB')
        return [[img]], [os.path.basename(image_path)]
    
    image_files = [f for f in os.listdir(image_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    
    batches = []
    filenames = []
    
    for i in range(0, len(image_files), batch_size):
        batch = []
        batch_files = image_files[i:i+batch_size]
        for img_file in batch_files:
            img_path = os.path.join(image_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                batch.append(img)
                filenames.append(img_file)
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
        
        if batch:
            batches.append(batch)
    
    return batches, filenames

def process_batch(clip_model, images, device):
    """处理一批图像并提取CLIP特征"""
    # 使用模型自带的transforms进行预处理
    tensor_list = []
    for img in images:
        # 使用clip_model的transforms预处理图像
        tensor = clip_model.transforms(img).unsqueeze(0).to(device)  # [1, C, H, W]
        tensor_list.append(tensor)
    
    # 合并为批次
    batch_tensor = torch.cat(tensor_list, dim=0)  # [B, C, H, W]
    
    # 直接使用底层ViT模型而不是通过visual方法
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=clip_model.dtype):
            features = clip_model.model.visual(batch_tensor, use_31_block=True)
    
    return features

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    # 配置CLIP模型
    config = EasyDict()
    config.clip_dtype = torch.float16  # 使用半精度以提高性能
    
    # 初始化CLIP模型
    print("初始化CLIP模型...")
    clip_model = CLIPModel(
        dtype=config.clip_dtype,
        device=device,
        checkpoint_path=os.path.join(args.checkpoint_dir, "clip_vit_h_14.pth"),
        tokenizer_path=os.path.join(args.checkpoint_dir, "xlm_roberta_tokenizer")
    )
    
    # 加载图像
    print("加载图像...")
    batches, filenames = load_images(args.input, args.batch_size)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 处理图像
    print("提取CLIP特征...")
    features = {}
    for i, batch in enumerate(tqdm(batches)):
        batch_features = process_batch(clip_model, batch, device)
        
        # 保存每个图像的特征
        for j, feature in enumerate(batch_features):
            idx = i * args.batch_size + j
            if idx < len(filenames):
                filename = filenames[idx]
                features[filename] = feature.cpu().numpy()
    
    # 保存特征
    output_path = os.path.join(args.output, "clip_features.npz")
    print(f"保存特征到 {output_path}")
    np.savez_compressed(output_path, **features)
    
    # 输出特征形状信息
    if features:
        for name, feature in list(features.items())[:1]:  # 只显示第一个
            print(f"特征形状: {feature.shape} (第一个图像: {name})")
            if feature.ndim >= 2:
                if feature.ndim == 3:
                    print(f"特征类型: 序列长度 {feature.shape[1]}，特征维度 {feature.shape[2]}")
                else:
                    print(f"特征类型: 维度 {feature.shape}")
            else:
                print(f"特征类型: 标量值")
    else:
        print("未能提取任何特征")
    
    print("处理完成!")

if __name__ == "__main__":
    main()