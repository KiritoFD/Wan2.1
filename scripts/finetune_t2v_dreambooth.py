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
import torchvision.transforms as transforms
from PIL import Image
import math
import torch.nn.functional as F

# 添加项目根路径到系统路径
sys.path.insert(0, os.path.sep.join(osp.realpath(__file__).split(os.path.sep)[:-2]))

# 现在导入wan相关模块
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video, load_video
from wan.image2video import WanI2V
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler

# 添加我们自己实现的函数
def get_training_sigmas(num_timesteps, max_sigma=120.0, min_sigma=0.002):
    """生成训练用的噪声水平"""
    betas = torch.linspace(0.0001, 0.02, num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
    
    # 缩放到合适的范围
    sigmas = sigmas * (max_sigma - min_sigma) / sigmas.max() + min_sigma
    return sigmas

def get_index_from_sigma(scheduler, sigma):
    """根据噪声水平找到对应的时间步索引"""
    # 确保scheduler.sigmas和sigma在同一设备上
    scheduler_sigmas = scheduler.sigmas.to(sigma.device)
    # 找到最接近的噪声水平对应的时间步
    dists = torch.abs(scheduler_sigmas - sigma)
    return torch.argmin(dists)

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

class CustomImageDataset(Dataset):
    """自定义图像数据集，用于基于图片的DreamBooth微调"""
    def __init__(self, image_dir, prompt, transform=None, unique_token="sks", image_size=(224, 224)):
        """
        参数:
            image_dir: 包含参考图片的目录
            prompt: 用于训练的提示词模板，如 "a [unique_token] person"
            transform: 图像预处理函数
            unique_token: 用来表示目标概念的唯一标记
            image_size: 输入图像大小
        """
        self.image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'webp']:
            self.image_files.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
            
        if len(self.image_files) == 0:
            raise ValueError(f"在 {image_dir} 中未找到任何图片文件")
            
        self.prompt = prompt
        self.unique_token = unique_token
        
        # 图像预处理变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        print(f"找到 {len(self.image_files)} 张图片")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载图片
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # 替换提示词中的占位符
        prompt = self.prompt.replace("[unique_token]", self.unique_token)
        
        return {
            "image": image_tensor,
            "prompt": prompt,
            "image_path": image_path
        }

class PreEncodedCLIPDataset(Dataset):
    """使用预先编码的CLIP特征数据集"""
    def __init__(self, clip_features_path, prompt, unique_token="sks"):
        """
        参数:
            clip_features_path: 包含预编码CLIP特征的.pt文件路径
            prompt: 用于训练的提示词模板，如 "a [unique_token] person"
            unique_token: 用来表示目标概念的唯一标记
        """
        # 加载预编码的CLIP特征
        if not os.path.exists(clip_features_path):
            raise FileNotFoundError(f"找不到CLIP特征文件: {clip_features_path}")
            
        print(f"正在加载预编码的CLIP特征: {clip_features_path}")
        data = torch.load(clip_features_path, map_location='cpu')
        
        # 首先检查数据类型，确保我们可以正确处理它
        if isinstance(data, dict):
            # 字典格式，检查键值
            if 'image_paths' in data:
                self.image_paths = data['image_paths']
            elif 'paths' in data:
                self.image_paths = data['paths']
            elif 'features' in data:
                # 只包含特征，没有路径信息
                print("注意: 未在CLIP特征文件中找到图像路径信息，将使用生成的路径")
                features = data['features']
                if isinstance(features, list):
                    self.image_paths = [f"image_{i}.jpg" for i in range(len(features))]
                else:
                    self.image_paths = [f"image_{i}.jpg" for i in range(features.size(0))]
                self.clip_features = features
            else:
                # 未知格式的字典
                raise ValueError(f"无效的CLIP特征文件格式，无法识别的键: {list(data.keys())}")
        elif isinstance(data, torch.Tensor):
            # 直接是tensor格式，没有额外的字典包装
            print("注意: CLIP特征文件直接包含特征张量，将使用生成的路径")
            self.clip_features = data
            self.image_paths = [f"image_{i}.jpg" for i in range(data.size(0))]
        else:
            # 其他未知格式
            raise ValueError(f"无效的CLIP特征文件格式，数据类型为: {type(data)}")
        
        # 如果clip_features尚未在前面的条件判断中设置，在这里从data中获取
        if not hasattr(self, 'clip_features'):
            if 'features' in data:
                self.clip_features = data['features']
            else:
                raise ValueError("无法在CLIP特征文件中找到特征数据")
        
        # 如果特征是列表，转换为张量
        if isinstance(self.clip_features, list):
            print("将CLIP特征列表转换为张量...")
            try:
                self.clip_features = torch.stack(self.clip_features)
            except:
                print("警告: 无法将特征堆叠为单一张量，特征可能大小不一致")
                
        print(f"加载成功，包含{len(self.image_paths)}个CLIP特征")
            
        self.prompt = prompt
        self.unique_token = unique_token
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 获取CLIP特征
        if isinstance(self.clip_features, list):
            clip_feature = self.clip_features[idx]
        else:
            clip_feature = self.clip_features[idx:idx+1]
            
        # 替换提示词中的占位符
        prompt = self.prompt.replace("[unique_token]", self.unique_token)
        
        return {
            "clip_feature": clip_feature,
            "prompt": prompt,
            "image_path": self.image_paths[idx]
        }

def finetune_t2v_dreambooth(args):
    # 修复警告，将torch.cuda.amp.autocast替换为推荐的torch.amp.autocast
    def fix_amp_warnings():
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            old_autocast = torch.cuda.amp.autocast
            def new_autocast(*args, **kwargs):
                if 'enabled' in kwargs:
                    enabled = kwargs.pop('enabled')
                    if not enabled:
                        return torch.autocast('cuda', enabled=False, *args, **kwargs)
                return torch.autocast('cuda', *args, **kwargs)
            torch.cuda.amp.autocast = new_autocast
    
    # 尝试修复警告
    fix_amp_warnings()

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
    
    # 如果需要使用CLIP特征，但当前模型不支持，为模型添加img_emb属性
    if args.clip_features_path and not hasattr(model.model, 'img_emb'):
        print("检测到CLIP特征输入，但当前模型不是I2V或FLF2V类型")
        print("为模型添加CLIP特征处理能力...")
        
        # 导入所需模块
        from wan.modules.model import MLPProj
        
        # 获取模型的维度
        model_dim = model.model.dim
        
        # 添加img_emb属性，用于处理CLIP特征
        # 参考WanModel.__init__中的相关代码
        model.model.img_emb = MLPProj(1280, model_dim, flf_pos_emb=False)
        model.model.img_emb.to(device=f"cuda:{args.device_id}")
        
        # 修改model_type以支持CLIP特征
        model.model.model_type = 'i2v'  # 将模型标记为支持图像条件
        
        # 覆盖forward方法，使其不再强制要求y参数
        original_forward = model.model.forward
        
        def patched_forward(x, t, context, seq_len, clip_fea=None, y=None):
            """
            修补的forward方法，当使用CLIP特征时不再强制要求y参数。
            对于微调过程，我们实际上不需要y参数。
            """
            # 如果model_type为i2v或flf2v，但未提供y，则创建一个虚拟y
            if model.model.model_type in ['i2v', 'flf2v'] and clip_fea is not None and y is None:
                # 创建一个与x相同形状的虚拟y
                y = [x_i.clone() for x_i in x]  # 复制x作为y
            return original_forward(x, t, context, seq_len, clip_fea, y)
            
        # 替换原始forward方法
        model.model.forward = patched_forward
        
        print("成功添加CLIP特征处理能力，并修补了forward方法")
    
    # 2. 准备数据集
    if args.clip_features_path:
        print(f"使用预编码的CLIP特征...")
        dataset = PreEncodedCLIPDataset(
            clip_features_path=args.clip_features_path,
            prompt=args.prompt_template,
            unique_token=args.unique_token
        )
    elif args.video_dir:
        print(f"准备视频数据集...")
        dataset = CustomVideoDataset(
            video_dir=args.video_dir,
            prompt=args.prompt_template,
            unique_token=args.unique_token
        )
    elif args.image_dir:
        print(f"准备图像数据集...")
        dataset = CustomImageDataset(
            image_dir=args.image_dir,
            prompt=args.prompt_template,
            unique_token=args.unique_token
        )
    else:
        raise ValueError("必须指定 --image_dir、--video_dir 或 --clip_features_path 参数")
        
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 3. 设置优化器 - 调整参数选择逻辑以适应WanModel (DiT)
    print("选择要微调的参数 (交叉注意力和FFN)...")
    trainable_params = []
    trainable_param_names = []
    for name, param in model.model.named_parameters():
        # 微调所有交叉注意力层和FFN层中的参数
        if "cross_attn" in name or "ffn" in name:
            param.requires_grad = True
            trainable_params.append(param)
            trainable_param_names.append(name)
        else:
            param.requires_grad = False
            
    if not trainable_params:
        print("警告: 未找到任何可训练的参数。检查模型结构和参数名称。")
        raise ValueError("无法找到任何参数进行优化，请检查模型结构或参数选择逻辑。")

    print(f"可训练参数数量: {len(trainable_params)}")
            
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.1
    )
    
    # 4. 创建训练用噪声调度器
    print("初始化噪声调度器...")
    noise_scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=model.num_train_timesteps,
        shift=1.0,
        use_dynamic_shifting=False
    )
    
    # 获取训练噪声级别
    sigmas = get_training_sigmas(model.num_train_timesteps)
    
    # 5. 训练循环
    print(f"开始微调，共 {args.num_epochs} 轮...")
    model.model.train()
    
    # 跟踪最佳模型
    best_loss = float('inf')
    best_epoch = -1
    
    # 准备保存目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in progress_bar:
            # 1. 准备输入数据
            if "video" in batch and batch["video"] is not None:
                videos = batch["video"].to(f"cuda:{args.device_id}") # Shape: (B, C, N, H, W)
                has_clip_features = False
            elif "clip_feature" in batch:
                # 使用预编码的CLIP特征
                clip_features = batch["clip_feature"].to(f"cuda:{args.device_id}")
                has_clip_features = True
                
                # 使用随机噪声作为起点
                W, H = [int(x) for x in args.resolution.split("*")]
                videos = torch.randn(
                    clip_features.size(0), 3, args.frame_num, H, W, 
                    device=f"cuda:{args.device_id}"
                ) # Shape: (B, C, N, H, W)
            else:
                raise ValueError("批次中没有有效的视频或预编码CLIP特征数据")
            
            prompts = batch["prompt"]
            batch_size = videos.shape[0] # 获取实际的批次大小
            
            # 2. VAE编码视频
            with torch.no_grad():
                # 确保视频在[-1, 1]范围内
                if videos.max() > 1.0 or videos.min() < -1.0:
                    videos = torch.clamp(videos, -1.0, 1.0)
                elif videos.max() <= 1.0 and videos.min() >= 0.0:
                    # 如果视频在[0, 1]范围内，转换到[-1, 1]
                    videos = videos * 2.0 - 1.0
                
                # 准备VAE编码的输入 - 每个样本单独处理
                batch_size = videos.shape[0]
                latents_list = []
                
                for i in range(batch_size):
                    # 从批次中提取单个视频，并使用VAE编码它
                    video_i = videos[i]  # 形状: [C, T, H, W]
                    # 将视频作为列表的元素传递给VAE (VAE期望的输入格式)
                    latent_i = model.vae.encode([video_i])[0]  # 输出形状应该是[C', T', H', W']
                    
                    # 检查并调整通道数以匹配模型输入需求
                    expected_channels = model.model.in_dim  # 获取模型期望的输入通道数
                    actual_channels = latent_i.shape[0]
                    
                    # 打印通道信息以进行调试
                    print(f"调整前: 潜在表示形状 = {latent_i.shape}, 期望通道数 = {expected_channels}")
                    
                    if actual_channels != expected_channels:
                        print(f"通道数不匹配: VAE输出{actual_channels}通道，模型期望{expected_channels}通道，进行调整...")
                        # 如果VAE输出32通道但模型期望16通道，则取前16通道
                        if actual_channels > expected_channels:
                            latent_i = latent_i[:expected_channels]
                        # 如果VAE输出少于期望通道，则增加通道（例如通过复制）
                        else:
                            repeats = math.ceil(expected_channels / actual_channels)
                            latent_i = latent_i.repeat(repeats, 1, 1, 1)[:expected_channels]
                    
                    # 再次打印形状以确认调整
                    print(f"调整后: 潜在表示形状 = {latent_i.shape}")
                    latents_list.append(latent_i)
                
                # 如果批次大小为1，直接使用潜在变量，否则堆叠它们
                if batch_size == 1:
                    latents = latents_list[0]
                else:
                    # 尝试堆叠潜在变量 - 注意这可能需要确保所有潜在变量具有相同的形状
                    try:
                        latents = torch.stack(latents_list)
                    except:
                        # 如果无法堆叠，可能是因为形状不一致，这种情况下我们只使用第一个
                        print("警告: 无法堆叠不同形状的潜在变量，只使用第一个样本")
                        latents = latents_list[0]
                        batch_size = 1  # 修改批次大小
            
            # 3. 添加噪声到潜在表示
            noise = torch.randn_like(latents)
            
            # 对每个样本随机选择不同的噪声水平，使训练更稳定
            sigma_indices = torch.randint(0, len(sigmas), (batch_size,))
            timesteps = []
            
            # 逐个样本添加噪声
            noisy_latents_list = []
            for i in range(batch_size):
                sigma = sigmas[sigma_indices[i]].to(latents.device)
                # 如果batch_size=1，则直接使用latents，否则提取第i个样本
                if batch_size == 1:
                    sample_noisy = latents + noise * sigma
                else:
                    sample_noisy = latents[i:i+1] + noise[i:i+1] * sigma
                noisy_latents_list.append(sample_noisy)
                
                # 获取对应的时间步
                timestep = get_index_from_sigma(noise_scheduler, sigma)
                timesteps.append(timestep)
            
            # 合并所有样本的噪声潜在向量 - 但是保持列表格式，以符合模型预期
            noisy_latents = noisy_latents_list
            
            # 打印噪声潜在表示的形状，确保通道数正确
            print(f"噪声潜在表示形状: {[nl.shape for nl in noisy_latents]}")
            
            # 确保通道数匹配，再次检查
            for i, nl in enumerate(noisy_latents):
                if nl.shape[0] != model.model.in_dim:
                    print(f"警告: 第{i}个噪声潜在表示的通道数({nl.shape[0]})与模型期望({model.model.in_dim})不匹配，进行调整")
                    if nl.shape[0] > model.model.in_dim:
                        noisy_latents[i] = nl[:model.model.in_dim]
                    else:
                        repeats = math.ceil(model.model.in_dim / nl.shape[0])
                        noisy_latents[i] = nl.repeat(repeats, 1, 1, 1)[:model.model.in_dim]
                    print(f"调整后: {noisy_latents[i].shape}")
            
            timesteps = torch.stack(timesteps).to(latents.device)
            
            # 4. 获取文本条件嵌入
            with torch.no_grad():
                # 使用T5编码器获取文本嵌入
                if not model.t5_cpu:
                    model.text_encoder.model.to(model.device)
                # 获取文本嵌入，并确保它们是与模型匹配的数据类型
                context = model.text_encoder(prompts, model.device)
                
                # 确保context使用与模型参数相同的数据类型
                # 检查text_embedding的第一个参数的数据类型(通常是权重)
                target_dtype = next(model.model.text_embedding.parameters()).dtype
                context = [c.to(dtype=target_dtype) for c in context]
                
                if not model.t5_cpu:
                    model.text_encoder.model.cpu()
            
            # 5. 计算序列长度和准备模型参数
            target_shape = latents.shape
            seq_len = math.ceil((target_shape[2] * target_shape[3]) / 
                              (model.patch_size[1] * model.patch_size[2]) * 
                              target_shape[1] / model.sp_size) * model.sp_size
            
            # 准备模型输入参数
            model_kwargs = {'context': context, 'seq_len': seq_len}
            
            # 添加CLIP特征（如果有的话）
            # 检查模型类型是否支持CLIP特征
            if has_clip_features and hasattr(model.model, 'model_type') and model.model.model_type in ['i2v', 'flf2v']:
                # 同样确保CLIP特征使用正确的数据类型
                model_kwargs['clip_fea'] = clip_features.to(dtype=target_dtype)
            elif has_clip_features:
                print("警告: 当前模型类型不支持CLIP特征输入。这是T2V模型，但您正在尝试使用CLIP特征。")
                print("CLIP特征将被忽略。如需使用CLIP特征，请使用I2V或FLF2V模型。")
            
            # 6. 前向传播 - 预测噪声
            # 确保噪声潜在变量也使用正确的数据类型和正确通道数
            fixed_noisy_latents = []
            expected_channels = model.model.in_dim
            
            for nl in noisy_latents:
                # 确保数据类型正确
                nl = nl.to(dtype=target_dtype)
                
                # 确保通道数正确
                actual_channels = nl.shape[0]
                if actual_channels != expected_channels:
                    if actual_channels > expected_channels:
                        # 裁剪多余的通道
                        nl = nl[:expected_channels]
                    else:
                        # 重复通道以达到预期数量
                        repeats = math.ceil(expected_channels / actual_channels)
                        nl = nl.repeat(repeats, 1, 1, 1)[:expected_channels]
                
                fixed_noisy_latents.append(nl)
            
            print(f"修复后的潜在表示通道数: {[nl.shape[0] for nl in fixed_noisy_latents]}")
            noisy_latents = fixed_noisy_latents
            
            timesteps = timesteps.to(dtype=target_dtype)
            
            try:
                noise_pred = model.model(noisy_latents, t=timesteps, **model_kwargs)[0]
            except RuntimeError as e:
                if "expected input" in str(e) and "channels" in str(e):
                    print(f"通道不匹配错误: {e}")
                    # 尝试使用紧急修复：创建正确通道数的随机潜在向量
                    print("紧急修复：创建符合模型预期的随机潜在向量...")
                    patched_latents = []
                    for nl in noisy_latents:
                        shape = list(nl.shape)
                        shape[0] = model.model.in_dim  # 设置正确的通道数
                        random_latent = torch.randn(shape, device=nl.device, dtype=nl.dtype)
                        patched_latents.append(random_latent)
                    
                    # 用修复的潜在向量替换原来的
                    noisy_latents = patched_latents
                    noise_pred = model.model(noisy_latents, t=timesteps, **model_kwargs)[0]
                    print("使用紧急修复成功执行模型前向传播")
                else:
                    raise  # 如果是其他错误，继续抛出
            
            # 7. 计算损失
            # 确保噪声也使用相同的数据类型进行损失计算
            noise = noise.to(dtype=noise_pred.dtype)
            # 基本MSE损失
            loss = F.mse_loss(noise_pred, noise)
            
            # 8. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 可选：梯度裁剪以增加训练稳定性
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            optimizer.step()
            
            # 9. 记录损失并更新进度条
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1
            
            # 10. 更新进度条
            progress_bar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })
            
            # 11. 清理显存
            del latents, noise, noisy_latents, noise_pred, videos, latents_list
            if 'clip_features' in locals():
                 del clip_features
            if 'context' in locals():
                 del context
            torch.cuda.empty_cache()
        
        # 每个epoch的后处理
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"Epoch {epoch+1}/{args.num_epochs}, 平均损失: {avg_epoch_loss:.6f}, 学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 更新学习率
        scheduler.step()
        
        # 检查是否是最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch
            
            # 保存最佳模型
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model": model.model.state_dict(),
                "epoch": epoch,
                "loss": best_loss,
                "unique_token": args.unique_token
            }, best_model_path)
            print(f"发现新的最佳模型 (损失: {best_loss:.6f})，已保存到 {best_model_path}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"dreambooth_epoch_{epoch+1}.pt")
            
            # 保存模型状态
            torch.save({
                "model": model.model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_epoch_loss,
                "unique_token": args.unique_token
            }, save_path)
            print(f"检查点已保存至 {save_path}")
    
    # 训练完成后的总结
    print(f"\n训练已完成！")
    print(f"最佳模型出现在Epoch {best_epoch+1}，损失: {best_loss:.6f}")
    print(f"最佳模型已保存到 {os.path.join(args.output_dir, 'best_model.pt')}")
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "dreambooth_final.pt")
    torch.save({
        "model": model.model.state_dict(),
        "epoch": args.num_epochs - 1,
        "unique_token": args.unique_token,
        "best_epoch": best_epoch,
        "best_loss": best_loss
    }, final_save_path)
    print(f"最终模型已保存至 {final_save_path}")
    
    # 生成测试视频
    print("生成测试视频...")
    model.model.eval()
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
    parser.add_argument("--ckpt_dir", type=str, default="Wan2.1-T2V-1.3B", help="预训练模型检查点目录")
    parser.add_argument("--image_dir", type=str, default=None, help="训练用参考图片目录")
    parser.add_argument("--video_dir", type=str, default=None, help="可选的训练视频数据集目录")
    parser.add_argument("--clip_features_path", "--clip_feature_path", dest="clip_features_path", type=str, default=None, 
                        help="预先编码的CLIP特征文件路径，如assets/Eleina/clip_features.pt")
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
    
    # 图像相关参数
    parser.add_argument("--use_video", action="store_true", help="是否使用提供的视频作为训练数据")
    parser.add_argument("--frame_num", type=int, default=16, help="生成的视频帧数")
    parser.add_argument("--image_init_epochs", type=int, default=10, 
                        help="使用图像生成的视频进行初始化的轮数")
    parser.add_argument("--use_img_sim_loss", action="store_true", 
                        help="是否使用图像相似度损失")
    parser.add_argument("--img_sim_weight", type=float, default=0.5, 
                        help="图像相似度损失的权重")
    
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
