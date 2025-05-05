# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 添加项目根路径到系统路径
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, load_video

class CustomVideoDataset(Dataset):
    """自定义视频数据集，用于DreamBooth式微调"""
    def __init__(self, video_dir, prompt, transform=None, unique_token="sks"):
        """
        参数:
            video_dir: 包含视频文件的目录
            prompt: 用于训练的提示词模板，如 "a [unique_token] person"
            transform: 视频预处理函数
            unique_token: 用来表示目标概念的唯一标记
        """
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
        self.prompt = prompt
        self.transform = transform
        self.unique_token = unique_token
        
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        # 加载视频
        video_frames = load_video(video_path)
        
        if self.transform:
            video_frames = self.transform(video_frames)
            
        # 替换提示词中的占位符
        prompt = self.prompt.replace("[unique_token]", self.unique_token)
        
        return {
            "video": video_frames,
            "prompt": prompt
        }

def finetune_t2v_dreambooth(args):
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
    
    # 2. 准备数据集
    print(f"准备自定义数据集...")
    dataset = CustomVideoDataset(
        video_dir=args.video_dir,
        prompt=args.prompt_template,
        unique_token=args.unique_token
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 3. 设置优化器 - 只优化UNet部分
    # 对于DreamBooth风格微调，通常只微调UNet部分
    trainable_params = []
    for name, param in model.pipe.unet.named_parameters():
        if "down_blocks" in name or "mid_block" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
            
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 4. 训练循环
    print(f"开始微调...")
    model.pipe.unet.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            videos = batch["video"].to(f"cuda:{args.device_id}")
            prompts = batch["prompt"]
            
            # Forward pass (使用模型内部训练逻辑)
            loss = model.training_step(videos, prompts)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}, 平均损失: {avg_epoch_loss}")
        
        # 每个epoch保存一次检查点
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"dreambooth_epoch_{epoch+1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            
            # 保存模型状态
            model_state = {
                "unet": model.pipe.unet.state_dict(),
                "epoch": epoch,
                "unique_token": args.unique_token
            }
            torch.save(model_state, save_path)
            print(f"模型检查点已保存至 {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "dreambooth_final.pt")
    model_state = {
        "unet": model.pipe.unet.state_dict(),
        "epoch": args.num_epochs,
        "unique_token": args.unique_token
    }
    torch.save(model_state, final_save_path)
    print(f"最终模型已保存至 {final_save_path}")
    
    # 生成测试视频
    print("生成测试视频...")
    model.pipe.unet.eval()
    test_prompt = args.test_prompt.replace("[unique_token]", args.unique_token)
    W, H = [int(x) for x in args.resolution.split("*")]
    
    with torch.no_grad():
        video = model.generate(
            test_prompt,
            size=(W, H),
            shift=args.shift_scale,
            sampling_steps=args.sd_steps,
            guide_scale=args.guide_scale,
            seed=args.seed
        )
    
    test_video_path = os.path.join(args.output_dir, "test_generation.mp4")
    save_video(
        tensor=video[None],
        save_file=test_video_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    print(f"测试视频已保存至 {test_video_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth-style fine-tuning for Wan2.1 T2V-1.3B model")
    parser.add_argument("--ckpt_dir", type=str, default="cache", help="预训练模型检查点目录")
    parser.add_argument("--video_dir", type=str, required=True, help="训练视频数据集目录")
    parser.add_argument("--output_dir", type=str, default="dreambooth_output", help="输出目录")
    parser.add_argument("--prompt_template", type=str, default="a [unique_token] person", 
                        help="提示词模板，使用[unique_token]作为占位符")
    parser.add_argument("--unique_token", type=str, default="sks", help="用于表示目标概念的唯一标记")
    parser.add_argument("--device_id", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器工作线程数")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮次")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--save_every", type=int, default=5, help="每隔多少轮保存一次检查点")
    
    # 测试生成参数
    parser.add_argument("--test_prompt", type=str, default="a [unique_token] person on the beach", 
                        help="测试提示词，使用[unique_token]作为占位符")
    parser.add_argument("--resolution", type=str, default="480*832", help="视频分辨率")
    parser.add_argument("--sd_steps", type=int, default=30, help="扩散步数")
    parser.add_argument("--guide_scale", type=float, default=6.0, help="引导尺度")
    parser.add_argument("--shift_scale", type=float, default=8.0, help="位移尺度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    finetune_t2v_dreambooth(args)
