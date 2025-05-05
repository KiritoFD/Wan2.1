# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import os
import os.path as osp
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import glob
from PIL import Image
import torchvision.transforms as transforms

# 添加项目根路径到系统路径
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, load_video
from wan.modules.clip import CLIPModel

class CustomImageVideoDataset(Dataset):
    """自定义图片-视频数据集，用于基于图片的DreamBooth式微调"""
    def __init__(self, image_dir, video_dir=None, prompt_template="a [unique_token]", 
                 unique_token="sks", image_size=(224, 224)):
        """
        参数:
            image_dir: 包含参考图片的目录
            video_dir: 可选，包含视频文件的目录
            prompt_template: 用于训练的提示词模板，如 "a [unique_token] person"
            unique_token: 用来表示目标概念的唯一标记
            image_size: 输入图像的大小
        """
        self.image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            self.image_files.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
        
        self.video_files = []
        if video_dir:
            for ext in ['mp4', 'avi', 'mov']:
                self.video_files.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
                
        self.prompt_template = prompt_template
        self.unique_token = unique_token
        
        # 图像预处理变换
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"找到 {len(self.image_files)} 张图片和 {len(self.video_files)} 个视频")
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到任何图片文件")
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图片
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # 加载视频（如果有）
        video_tensor = None
        if self.video_files and idx < len(self.video_files):
            video_path = self.video_files[idx]
            video_tensor = load_video(video_path)
        
        # 替换提示词中的占位符
        prompt = self.prompt_template.replace("[unique_token]", self.unique_token)
        
        return {
            "image": image_tensor,
            "video": video_tensor,
            "prompt": prompt,
            "image_path": image_path
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
    
    # 初始化Wan2.1自带的CLIP模型用于图像编码
    print(f"初始化CLIP模型...")
    from wan.image2video import WanI2V
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
    # 获取CLIP模型并删除临时I2V模型以节省内存
    clip_model = temp_i2v.clip
    del temp_i2v
    
    # 2. 准备数据集
    print(f"准备自定义数据集...")
    dataset = CustomImageVideoDataset(
        image_dir=args.image_dir,
        video_dir=args.video_dir if args.use_video else None,
        prompt_template=args.prompt_template,
        unique_token=args.unique_token
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 3. 设置优化器 - 优化DiT模型部分
    # 对于DreamBooth风格微调，我们针对DiT模型进行微调
    trainable_params = []
    
    # 首先冻结所有参数
    for param in model.model.parameters():
        param.requires_grad = False
        
    # 选择性地解冻DiT中的特定层
    for name, param in model.model.named_parameters():
        # 主要微调中间层的交叉注意力机制和自注意力机制
        if "blocks" in name:
            block_num = int(name.split(".")[1])  # 获取block的编号
            # 只微调部分块，通常是中间几层
            if block_num >= args.start_block and block_num <= args.end_block:
                if "self_attn" in name or "cross_attn" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                # 也可以选择性地微调MLP部分
                elif args.tune_mlp and "mlp" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
    
    # 打印可训练参数统计
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params_count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params_count:,} ({trainable_params_count/total_params*100:.2f}%)")
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 准备先验保留(prior-preservation)样本
    # 如果启用了先验保留，则需要生成类别样本
    if args.prior_preservation:
        print("生成先验保留样本...")
        # 保存原始训练状态
        model.model.eval()
        
        prior_samples = []
        class_prompt = args.class_prompt
        
        # 生成先验保留样本
        for i in range(args.num_class_images):
            with torch.no_grad():
                W, H = [int(x) for x in args.resolution.split("*")]
                video = model.generate(
                    class_prompt,
                    size=(W, H),
                    shift=args.shift_scale,
                    sampling_steps=args.sd_steps,
                    guide_scale=args.guide_scale,
                    seed=args.base_seed + i
                )
                
                # 编码为潜在表示
                latents = model.vae.encode(video.unsqueeze(0).to(model.device))
                prior_samples.append(latents)
                
                # 保存生成的样本供参考
                if i < 3:  # 只保存前几个样本
                    save_video(
                        tensor=video[None],
                        save_file=os.path.join(args.output_dir, f"prior_sample_{i}.mp4"),
                        fps=16,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1)
                    )
        
        # 将先验样本转换为张量
        prior_samples = torch.cat(prior_samples, dim=0)
        print(f"生成了 {len(prior_samples)} 个先验保留样本")
    
    # 学习率调度器
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.num_epochs,
            eta_min=args.min_lr
        )
    
    # 4. 训练循环
    print(f"开始微调...")
    model.model.train()  # 设置为训练模式
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_instance_loss = 0.0
        epoch_prior_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(f"cuda:{args.device_id}")
            prompts = batch["prompt"]
            
            # 使用CLIP编码图像
            with torch.no_grad():
                clip_features = clip_model.visual(images)
            
            # 生成视频或使用提供的视频
            if batch["video"] is not None and args.use_video:
                videos = batch["video"].to(f"cuda:{args.device_id}")
            else:
                # 使用模型生成对应图片的视频表示
                W, H = [int(x) for x in args.resolution.split("*")]
                # 在训练初期使用生成的视频，之后逐渐过渡到噪声
                if epoch < args.image_init_epochs:
                    with torch.no_grad():
                        # 使用图像生成视频作为初始化
                        videos = []
                        for i in range(images.size(0)):
                            video = model.generate(
                                prompts[i],
                                size=(W, H),
                                shift=args.shift_scale,
                                sampling_steps=args.sd_steps // 2,  # 减少步数加快生成
                                guide_scale=args.guide_scale,
                                n_prompt="",
                                seed=batch_idx * images.size(0) + i,
                                offload_model=False
                            )
                            videos.append(video)
                        videos = torch.stack(videos)
                else:
                    # 使用随机噪声作为起点
                    videos = torch.randn(
                        images.size(0), 3, args.frame_num, H, W, 
                        device=f"cuda:{args.device_id}"
                    )
            
            # 实例损失计算
            # 编码输入视频
            latents = model.vae.encode(videos)
            
            # 生成噪声和时间步
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, model.scheduler.num_train_timesteps, 
                                      (images.size(0),), 
                                      device=model.device)
            
            # 添加噪声
            noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)
            
            # 编码文本提示
            text_embeddings = model.t5_encode(prompts)
            
            # 预测噪声（使用图像特征进行条件生成）
            noise_pred = model.model(noisy_latents, timesteps, text_embeddings, 
                                     clip_fea=clip_features)
            
            # 计算实例损失
            instance_loss = torch.nn.functional.mse_loss(noise_pred[0], noise)
            
            # 先验保留损失计算
            prior_loss = 0.0
            if args.prior_preservation and len(prior_samples) > 0:
                # 随机选择一些先验样本
                num_prior = min(images.size(0), prior_samples.shape[0])
                prior_indices = torch.randperm(prior_samples.shape[0])[:num_prior]
                prior_latents = prior_samples[prior_indices]
                
                # 为先验样本添加噪声
                prior_noise = torch.randn_like(prior_latents)
                prior_timesteps = torch.randint(0, model.scheduler.num_train_timesteps, 
                                              (num_prior,), 
                                              device=model.device)
                prior_noisy = model.scheduler.add_noise(prior_latents, prior_noise, prior_timesteps)
                
                # 编码类别提示
                class_embeddings = model.t5_encode([args.class_prompt] * num_prior)
                
                # 预测噪声 - 注意不使用图像特征
                prior_noise_pred = model.model(prior_noisy, prior_timesteps, class_embeddings)
                
                # 计算先验损失
                prior_loss = torch.nn.functional.mse_loss(prior_noise_pred[0], prior_noise)
            
            # 图像相似性损失（确保生成的视频与参考图像相似）
            img_sim_loss = 0.0
            if args.use_img_sim_loss and epoch >= args.image_init_epochs:
                # 解码潜在表示获取视频
                pred_video = model.vae.decode(noisy_latents - noise_pred[0])
                
                # 取第一帧与参考图像比较
                first_frame = pred_video[:, :, 0]  # B x 3 x H x W
                
                # 调整参考图像大小以匹配生成的帧
                resized_images = torch.nn.functional.interpolate(
                    images, size=(first_frame.shape[2], first_frame.shape[3]))
                
                # 计算L1相似度损失
                img_sim_loss = torch.nn.functional.l1_loss(first_frame, resized_images)
            
            # 组合损失
            loss = (instance_loss + 
                   args.prior_weight * prior_loss + 
                   args.img_sim_weight * img_sim_loss)
            
            # 添加正则化损失
            if args.use_reg_loss:
                reg_loss = 0.0
                # 特征正则化
                if hasattr(model.model, "get_last_hidden_state"):
                    # 假设模型有获取隐藏状态的方法
                    hidden_states = model.model.get_last_hidden_state()
                    # L2正则化
                    reg_loss += args.reg_weight * torch.mean(torch.norm(hidden_states, dim=-1))
                
                # 添加到总损失
                loss += reg_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_instance_loss += instance_loss.item()
            if args.prior_preservation:
                epoch_prior_loss += prior_loss.item()
                
            # 更新进度条
            progress_bar.set_postfix({
                "loss": loss.item(),
                "inst_loss": instance_loss.item(),
                "prior_loss": prior_loss.item() if args.prior_preservation else 0,
                "img_sim": img_sim_loss.item() if args.use_img_sim_loss else 0
            })
        
        # 更新学习率
        if args.use_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"当前学习率: {current_lr:.8f}")
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_instance_loss = epoch_instance_loss / len(dataloader)
        avg_prior_loss = epoch_prior_loss / len(dataloader) if args.prior_preservation else 0
        
        print(f"Epoch {epoch+1}, 平均损失: {avg_epoch_loss:.6f}, "
              f"实例损失: {avg_instance_loss:.6f}, "
              f"先验损失: {avg_prior_loss:.6f}")
        
        # 每个epoch保存一次检查点
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"dreambooth_epoch_{epoch+1}.pt")
            os.makedirs(args.output_dir, exist_ok=True)
            
            # 保存模型状态
            model_state = {
                "model": model.model.state_dict(),
                "epoch": epoch,
                "unique_token": args.unique_token,
                "loss": avg_epoch_loss
            }
            torch.save(model_state, save_path)
            print(f"模型检查点已保存至 {save_path}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "dreambooth_final.pt")
    model_state = {
        "model": model.model.state_dict(),
        "epoch": args.num_epochs,
        "unique_token": args.unique_token,
        "loss": avg_epoch_loss
    }
    torch.save(model_state, final_save_path)
    print(f"最终模型已保存至 {final_save_path}")
    
    # 生成测试视频
    print("生成测试视频...")
    model.model.eval()  # 设置为评估模式
    
    # 加载一张图片用于测试
    test_image = next(iter(dataloader))["image"][0:1].to(model.device)
    test_prompt = args.test_prompt.replace("[unique_token]", args.unique_token)
    
    # 编码测试图像
    with torch.no_grad():
        test_clip_features = clip_model.visual(test_image)
    
    W, H = [int(x) for x in args.resolution.split("*")]
    
    with torch.no_grad():
        # 使用自定义生成函数，考虑CLIP特征
        video = model.generate(
            test_prompt,
            size=(W, H),
            shift=args.shift_scale,
            sampling_steps=args.sd_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            clip_fea=test_clip_features  # 添加CLIP特征作为条件
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
    parser = argparse.ArgumentParser(description="图片条件的DreamBooth风格微调 - Wan2.1 T2V-1.3B模型")
    parser.add_argument("--ckpt_dir", type=str, default="cache", help="预训练模型检查点目录")
    parser.add_argument("--image_dir", type=str, required=True, help="训练用参考图片目录")
    parser.add_argument("--video_dir", type=str, default=None, help="可选的训练视频数据集目录")
    parser.add_argument("--output_dir", type=str, default="dreambooth_output", help="输出目录")
    parser.add_argument("--prompt_template", type=str, default="a [unique_token] person", 
                        help="提示词模板，使用[unique_token]作为占位符")
    parser.add_argument("--unique_token", type=str, default="sks", help="用于表示目标概念的唯一标记")
    parser.add_argument("--device_id", type=int, default=0, help="GPU设备ID")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器工作线程数")
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮次")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="最小学习率(用于调度器)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--save_every", type=int, default=5, help="每隔多少轮保存一次检查点")
    
    # 图像到视频相关参数
    parser.add_argument("--use_video", action="store_true", help="是否使用提供的视频作为训练数据")
    parser.add_argument("--frame_num", type=int, default=16, help="生成的视频帧数")
    parser.add_argument("--image_init_epochs", type=int, default=10, 
                        help="使用图像生成的视频进行初始化的轮数")
    parser.add_argument("--use_img_sim_loss", action="store_true", 
                        help="是否使用图像相似度损失")
    parser.add_argument("--img_sim_weight", type=float, default=0.5, 
                        help="图像相似度损失的权重")
    
    # DiT微调相关参数
    parser.add_argument("--start_block", type=int, default=10, help="开始微调的DiT块索引")
    parser.add_argument("--end_block", type=int, default=20, help="结束微调的DiT块索引")
    parser.add_argument("--tune_mlp", action="store_true", help="是否微调MLP部分")
    parser.add_argument("--clip_grad", action="store_true", help="是否进行梯度裁剪")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪的最大范数")
    parser.add_argument("--use_scheduler", action="store_true", help="是否使用学习率调度器")
    
    # DreamBooth风格微调参数
    parser.add_argument("--prior_preservation", action="store_true", 
                        help="是否使用先验保留损失(DreamBooth的关键特性)")
    parser.add_argument("--prior_weight", type=float, default=1.0, 
                        help="先验保留损失的权重")
    parser.add_argument("--class_prompt", type=str, default="a person", 
                        help="用于生成先验类别样本的提示词")
    parser.add_argument("--num_class_images", type=int, default=5, 
                        help="要生成的类别图像数量")
    parser.add_argument("--use_reg_loss", action="store_true", 
                        help="是否使用正则化损失")
    parser.add_argument("--reg_weight", type=float, default=0.01, 
                        help="正则化损失的权重")
    parser.add_argument("--base_seed", type=int, default=42, 
                        help="用于生成类别样本的基础种子")
    
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
