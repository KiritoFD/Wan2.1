# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import sys
import os.path as osp
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import glob

# 添加项目根路径到系统路径
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, load_video
from wan.image2video import WanI2V

def load_image(image_path, image_size=(224, 224)):
    """加载并处理单张图片"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加批次维度

def test_dreambooth_model(args):
    """测试微调后的DreamBooth模型"""
    # 1. 初始化模型
    print(f"初始化T2V-1.3B模型...")
    cfg = WAN_CONFIGS['t2v-1.3B']
    model = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
    )
    
    # 2. 加载微调后的模型权重
    print(f"加载微调模型权重：{args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=f"cuda:{args.device_id}")
    
    # 处理不同形式的模型权重
    if "model" in checkpoint:
        print("加载DiT模型权重")
        model.model.load_state_dict(checkpoint["model"])
    elif "unet" in checkpoint:
        print("加载UNet模型权重 (旧版本)")
        # 假设这是旧版本格式的权重
        model.model.load_state_dict(checkpoint["unet"])
    else:
        print("无法识别的模型格式，尝试直接加载")
        model.model.load_state_dict(checkpoint)
    
    # 获取模型中保存的unique_token
    unique_token = checkpoint.get("unique_token", args.unique_token)
    print(f"使用的唯一标记: {unique_token}")
    
    # 3. 初始化CLIP模型用于图像编码
    print(f"初始化CLIP模型...")
    # 创建临时I2V模型以获取CLIP功能
    temp_i2v = WanI2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
    )
    # 获取CLIP模型
    clip_model = temp_i2v.clip
    del temp_i2v
    
    # 4. 加载测试图像
    if args.image_path:
        print(f"加载测试图像：{args.image_path}")
        test_image = load_image(args.image_path).to(f"cuda:{args.device_id}")
        
        # 使用CLIP编码图像
        with torch.no_grad():
            clip_features = clip_model.visual(test_image)
    else:
        print("不使用图像条件")
        clip_features = None
    
    # 5. 准备测试提示词
    if args.prompt:
        test_prompt = args.prompt
    else:
        test_prompt = f"a {unique_token} person walking on a beach"
    
    if "[unique_token]" in test_prompt:
        test_prompt = test_prompt.replace("[unique_token]", unique_token)
    
    print(f"使用测试提示词：{test_prompt}")
    
    # 6. 设置生成参数
    W, H = [int(x) for x in args.resolution.split("*")]
    
    # 7. 生成视频
    print("生成测试视频...")
    with torch.no_grad():
        video = model.generate(
            test_prompt,
            size=(W, H),
            frame_num=args.frame_num,
            shift=args.shift_scale,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            clip_fea=clip_features
        )
    
    # 8. 保存生成的视频
    test_video_path = args.output_path
    save_video(
        tensor=video[None],
        save_file=test_video_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"测试视频已保存至 {test_video_path}")
    
    # 批量生成多个提示词（如果有）
    if args.prompt_list:
        print(f"从文件加载提示词列表：{args.prompt_list}")
        with open(args.prompt_list, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"找到 {len(prompts)} 个提示词，开始批量生成...")
        os.makedirs(args.batch_output_dir, exist_ok=True)
        
        for i, prompt in enumerate(tqdm(prompts)):
            if "[unique_token]" in prompt:
                prompt = prompt.replace("[unique_token]", unique_token)
            
            output_path = os.path.join(args.batch_output_dir, f"gen_{i:03d}.mp4")
            
            with torch.no_grad():
                video = model.generate(
                    prompt,
                    size=(W, H),
                    frame_num=args.frame_num,
                    shift=args.shift_scale,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    seed=args.seed + i,
                    clip_fea=clip_features
                )
            
            save_video(
                tensor=video[None],
                save_file=output_path,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        
        print(f"批量生成完成，视频已保存到: {args.batch_output_dir}")
    
    # 9. 批量处理目录中的所有图片（如果指定了多图像目录）
    if args.batch_image_dir:
        print(f"从目录加载多个图像：{args.batch_image_dir}")
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            image_files.extend(glob.glob(os.path.join(args.batch_image_dir, f"*.{ext}")))
        
        print(f"找到 {len(image_files)} 张图片，开始批量生成...")
        os.makedirs(args.batch_output_dir, exist_ok=True)
        
        for i, img_path in enumerate(tqdm(image_files)):
            # 加载并编码图像
            img_tensor = load_image(img_path).to(f"cuda:{args.device_id}")
            with torch.no_grad():
                img_features = clip_model.visual(img_tensor)
            
            # 生成提示词
            if args.image_prompt:
                cur_prompt = args.image_prompt.replace("[unique_token]", unique_token)
            else:
                cur_prompt = test_prompt
            
            # 文件名
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            output_path = os.path.join(args.batch_output_dir, f"{img_name}.mp4")
            
            # 生成视频
            with torch.no_grad():
                video = model.generate(
                    cur_prompt,
                    size=(W, H),
                    frame_num=args.frame_num,
                    shift=args.shift_scale,
                    sampling_steps=args.sampling_steps,
                    guide_scale=args.guide_scale,
                    seed=args.seed + i,
                    clip_fea=img_features
                )
            
            # 保存视频
            save_video(
                tensor=video[None],
                save_file=output_path,
                fps=16,
                nrow=1,
                normalize=True,
                value_range=(-1, 1)
            )
        
        print(f"批量图像处理完成，视频已保存到: {args.batch_output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="测试基于图片的DreamBooth微调模型")
    
    # 模型参数
    parser.add_argument("--ckpt_dir", type=str, required=True, help="预训练模型检查点目录")
    parser.add_argument("--model_path", type=str, required=True, help="微调后的模型路径")
    parser.add_argument("--device_id", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--unique_token", type=str, default="sks", help="唯一标记，如果模型中没有保存")
    
    # 输入参数
    parser.add_argument("--image_path", type=str, default=None, help="测试图像路径，可选")
    parser.add_argument("--batch_image_dir", type=str, default=None, help="批量处理的图像目录")
    parser.add_argument("--prompt", type=str, default=None, help="测试提示词")
    parser.add_argument("--prompt_list", type=str, default=None, help="提示词列表文件路径，用于批量生成")
    parser.add_argument("--image_prompt", type=str, default=None, help="批量图像处理使用的提示词模板")
    
    # 输出参数
    parser.add_argument("--output_path", type=str, default="test_output.mp4", help="输出视频路径")
    parser.add_argument("--batch_output_dir", type=str, default="batch_output", help="批量生成输出目录")
    
    # 生成参数
    parser.add_argument("--resolution", type=str, default="480*832", help="视频分辨率")
    parser.add_argument("--frame_num", type=int, default=16, help="生成的视频帧数")
    parser.add_argument("--sampling_steps", type=int, default=30, help="采样步数")
    parser.add_argument("--guide_scale", type=float, default=6.0, help="引导尺度")
    parser.add_argument("--shift_scale", type=float, default=8.0, help="位移尺度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_dreambooth_model(args)
