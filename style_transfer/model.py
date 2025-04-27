#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型：基于AAE(Adversarial Autoencoder)的方法
学习VAE潜在空间中两种风格之间的映射
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


class StyleDataset(Dataset):
    """加载.pt文件中的VAE编码特征"""
    def __init__(self, data_path, squeeze_time=True):
        self.path = data_path
        
        try:
            # 加载数据
            logging.info(f"加载数据: {data_path}")
            data = torch.load(data_path, map_location='cpu')
            
            # 处理features
            if 'features' not in data:
                keys = list(data.keys()) if isinstance(data, dict) else []
                raise ValueError(f"数据中找不到'features'键，可用键: {keys}")
                
            features = data['features']
            
            # 处理不同形式的特征
            if isinstance(features, torch.Tensor):
                if features.dim() == 4:  # [16, 1, 32, 32]
                    self.features = features.unsqueeze(0)
                else:
                    self.features = features
            elif isinstance(features, list):
                try:
                    self.features = torch.stack(features)
                except:
                    # 处理维度不一致的特征
                    sample = features[0]
                    consistent_features = [f for f in features if f.shape == sample.shape]
                    if not consistent_features:
                        raise ValueError("无法找到形状一致的特征")
                    self.features = torch.stack(consistent_features)
            else:
                raise ValueError(f"不支持的features类型: {type(features)}")
            
            # 记录原始形状
            self.original_shape = self.features.shape
            
            # 去除时间维度（如果需要）
            if squeeze_time and self.features.shape[2] == 1:
                self.features = self.features.squeeze(2)
            
            # 其他元数据
            self.image_paths = data.get('image_paths', None)
            self.metadata = data.get('metadata', None)
            
            logging.info(f"数据集加载完成: {len(self.features)}个样本")
            
        except Exception as e:
            logging.error(f"加载数据集错误: {e}")
            raise

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
    
    @property
    def feature_shape(self):
        return self.features.shape[1:]


# 基本模块

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels, use_norm=True):
        super(ResidualBlock, self).__init__()
        
        norm_layer = nn.InstanceNorm2d if use_norm else nn.Identity
        
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            norm_layer(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)


class Attention(nn.Module):
    """自注意力机制"""
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.query = nn.Conv2d(channels, channels//8, 1)
        self.key = nn.Conv2d(channels, channels//8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, h, w = x.size()
        
        q = self.query(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h*w)
        v = self.value(x).view(b, -1, h*w)
        
        attention = F.softmax(torch.bmm(q, k), dim=2)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(b, c, h, w)
        
        return x + self.gamma * out


# 主要模型组件

class Encoder(nn.Module):
    """编码器：将输入特征映射到潜在空间"""
    def __init__(self, in_channels=16, latent_dim=256, use_attention=True):
        super(Encoder, self).__init__()
        
        # 初始处理
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        
        # 下采样路径
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(128)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            ResidualBlock(256)
        )
        
        # 注意力（可选）
        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(256)
            
        # 全局池化和映射
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, latent_dim)
        )
        
    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        x = self.pool(x)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    """解码器：将潜在向量解码为特征"""
    def __init__(self, out_channels=16, latent_dim=256):
        super(Decoder, self).__init__()
        
        # 潜在空间到特征图
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        
        # 上采样路径
        self.initial = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        
        self.up1 = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )
        
        self.up2 = nn.Sequential(
            ResidualBlock(128),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        
        # 输出层
        self.output = nn.Conv2d(64, out_channels, 3, 1, 1)
        
    def forward(self, x):
        x = self.fc(x).view(-1, 256, 8, 8)
        x = self.initial(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.output(x)
        return x


class Discriminator(nn.Module):
    """判别器：区分潜在向量的风格来源"""
    def __init__(self, latent_dim=256):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.model(x)


class StyleMapper(nn.Module):
    """风格映射网络：在潜在空间转换风格"""
    def __init__(self, latent_dim=256, depth=3):
        super(StyleMapper, self).__init__()
        
        layers = [
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2, True)
        ]
        
        for _ in range(depth-1):
            layers.extend([
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2, True)
            ])
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class StyleTransferAAE:
    """风格转换模型主类"""
    def __init__(self, device='cuda', latent_dim=256):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # 内部状态
        self.is_trained = False
        self.history = {'recon_loss': [], 'disc_loss': [], 'gen_loss': []}
        
    def build_models(self, in_channels=16):
        """构建模型组件"""
        # 创建模型组件
        self.encoder = Encoder(in_channels, self.latent_dim).to(self.device)
        self.decoder = Decoder(in_channels, self.latent_dim).to(self.device)
        self.discriminator = Discriminator(self.latent_dim).to(self.device)
        self.mapper = StyleMapper(self.latent_dim).to(self.device)
        
        # 损失函数
        self.recon_loss = nn.MSELoss()
        self.content_loss = nn.L1Loss()
        
        # 优化器
        lr = 2e-4
        beta1, beta2 = 0.5, 0.999
        
        self.opt_E = optim.Adam(self.encoder.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.decoder.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_Dis = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_M = optim.Adam(self.mapper.parameters(), lr=lr, betas=(beta1, beta2))
        
        logging.info(f"模型构建完成，设备: {self.device}")
        
    def train(self, dataloader_a, dataloader_b, epochs=100, save_dir=None, lambda_gp=10.0):
        """训练模型（使用WGAN-GP训练方式）"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # 设为训练模式
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.mapper.train()
        
        # 创建无限循环迭代器
        def infinite_loader(loader):
            while True:
                for data in loader:
                    yield data
                    
        iter_a = infinite_loader(dataloader_a)
        iter_b = infinite_loader(dataloader_b)
        
        # 每个epoch的步数
        steps_per_epoch = min(len(dataloader_a), len(dataloader_b))
        
        # 梯度惩罚函数
        def gradient_penalty(real, fake):
            alpha = torch.rand(real.size(0), 1, device=self.device)
            mixed = alpha * real + (1 - alpha) * fake
            mixed.requires_grad_(True)
            
            pred_mixed = self.discriminator(mixed)
            gradients = torch.autograd.grad(
                outputs=pred_mixed, inputs=mixed,
                grad_outputs=torch.ones_like(pred_mixed),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            
            gradient_norm = gradients.norm(2, dim=1)
            penalty = ((gradient_norm - 1)**2).mean()
            return penalty
        
        # 开始训练
        logging.info(f"开始训练，共{epochs}个周期")
        start_time = time.time()
        
        for epoch in range(epochs):
            # 累积损失
            epoch_recon = 0
            epoch_disc = 0
            epoch_gen = 0
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
            for _ in pbar:
                # 获取数据
                real_a = next(iter_a).to(self.device)
                real_b = next(iter_b).to(self.device)
                
                # --------------------
                # 1. 训练判别器
                # --------------------
                self.opt_Dis.zero_grad()
                
                # 获取真假样本
                with torch.no_grad():
                    z_a = self.encoder(real_a)
                    z_b = self.encoder(real_b)
                    z_ab = self.mapper(z_a)
                
                # 判别真假样本
                pred_real = self.discriminator(z_b)
                pred_fake = self.discriminator(z_ab.detach())
                
                # WGAN-GP损失
                d_loss = torch.mean(pred_fake) - torch.mean(pred_real)
                gp = gradient_penalty(z_b, z_ab.detach())
                d_total = d_loss + lambda_gp * gp
                
                d_total.backward()
                self.opt_Dis.step()
                
                # --------------------
                # 2. 训练自编码器和映射器
                # --------------------
                # 每隔5步训练一次生成器
                if _ % 5 == 0:
                    self.opt_E.zero_grad()
                    self.opt_D.zero_grad()
                    self.opt_M.zero_grad()
                    
                    # 编码和重建
                    z_a = self.encoder(real_a)
                    z_b = self.encoder(real_b)
                    
                    recon_a = self.decoder(z_a)
                    recon_b = self.decoder(z_b)
                    
                    # 映射风格
                    z_ab = self.mapper(z_a)
                    fake_b = self.decoder(z_ab)
                    
                    # 重建损失
                    loss_recon = (
                        self.recon_loss(recon_a, real_a) + 
                        self.recon_loss(recon_b, real_b)
                    )
                    
                    # 内容损失（循环一致性）
                    z_fake_b = self.encoder(fake_b)
                    loss_content = self.content_loss(z_fake_b, z_ab)
                    
                    # 生成器对抗损失
                    pred_gen = self.discriminator(z_ab)
                    loss_adv = -torch.mean(pred_gen)
                    
                    # 风格统计损失
                    m_b = z_b.mean(0, keepdim=True)
                    v_b = z_b.var(0, keepdim=True)
                    m_ab = z_ab.mean(0, keepdim=True)
                    v_ab = z_ab.var(0, keepdim=True)
                    loss_style = F.mse_loss(m_ab, m_b) + F.mse_loss(v_ab, v_b)
                    
                    # 总损失
                    g_total = (
                        10.0 * loss_recon + 
                        5.0 * loss_content + 
                        1.0 * loss_adv +
                        2.0 * loss_style
                    )
                    
                    g_total.backward()
                    self.opt_E.step()
                    self.opt_D.step()
                    self.opt_M.step()
                    
                    # 记录损失
                    epoch_recon += loss_recon.item()
                    epoch_gen += loss_adv.item()
                
                epoch_disc += d_loss.item()
                
                # 更新进度条
                pbar.set_postfix({'d_loss': d_loss.item()})
                
            # 计算周期平均损失
            avg_recon = epoch_recon / (steps_per_epoch / 5)
            avg_disc = epoch_disc / steps_per_epoch
            avg_gen = epoch_gen / (steps_per_epoch / 5)
            
            # 记录历史
            self.history['recon_loss'].append(avg_recon)
            self.history['disc_loss'].append(avg_disc)
            self.history['gen_loss'].append(avg_gen)
            
            # 日志输出
            logging.info(
                f"Epoch {epoch+1}/{epochs}, "
                f"重建: {avg_recon:.4f}, "
                f"判别器: {avg_disc:.4f}, "
                f"生成器: {avg_gen:.4f}"
            )
            
            # 保存检查点
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_{epoch+1}.pt"))
                
        # 训练完成
        self.is_trained = True
        elapsed = time.time() - start_time
        logging.info(f"训练完成，用时: {elapsed:.2f}秒")
        
        # 保存最终模型
        if save_dir:
            self.save_model(os.path.join(save_dir, "final_model.pt"))
            self.plot_history(save_dir)
            
        return self.history
            
    def transfer_style(self, x, add_time_dim=False):
        """将输入从风格A转换为风格B"""
        # 确保为评估模式
        self.encoder.eval()
        self.mapper.eval()
        self.decoder.eval()
        
        # 处理单个样本
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        # 转换
        with torch.no_grad():
            x = x.to(self.device)
            z = self.encoder(x)
            z_mapped = self.mapper(z)
            result = self.decoder(z_mapped)
            
            if add_time_dim:
                result = result.unsqueeze(2)
                
        return result.cpu()
    
    def train_with_gradient_penalty(self, dataloader_a, dataloader_b, num_epochs=100, save_dir=None, 
                                   lambda_gp=10.0, n_critic=5, eval_interval=None, **kwargs):
        """
        使用梯度惩罚的WGAN训练方式 (向后兼容接口)
        
        Args:
            dataloader_a: 风格A的数据加载器
            dataloader_b: 风格B的数据加载器
            num_epochs: 训练轮数
            save_dir: 保存模型的目录
            lambda_gp: 梯度惩罚系数
            n_critic: 每训练生成器一次，训练判别器的次数
            eval_interval: 评估间隔
            **kwargs: 额外参数传递给train方法
        """
        # 直接调用train方法，它已经实现了WGAN-GP训练
        # 但需要增强稳定性，所以我们添加额外的梯度裁剪参数
        return self.train(dataloader_a=dataloader_a,
                         dataloader_b=dataloader_b,
                         epochs=num_epochs,
                         save_dir=save_dir,
                         lambda_gp=lambda_gp,
                         **kwargs)  # 传递所有额外参数
    
    # 添加梯度裁剪辅助函数，防止梯度爆炸
    def _clip_gradients(self, optimizer, max_norm=1.0):
        """对优化器中的参数进行梯度裁剪，防止梯度爆炸"""
        for group in optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm)
            
    def train_stable(self, dataloader_a, dataloader_b, epochs=100, save_dir=None, lambda_gp=2.0, 
                     clip_value=1.0, lr_decay=0.995, start_epoch=0):
        """增强稳定性的训练方法，专为风格向量映射设计"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # 设为训练模式
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.mapper.train()
        
        # 创建无限循环迭代器
        def infinite_loader(loader):
            while True:
                for data in loader:
                    yield data
                    
        iter_a = infinite_loader(dataloader_a)
        iter_b = infinite_loader(dataloader_b)
        
        # 每个epoch的步数
        steps_per_epoch = min(len(dataloader_a), len(dataloader_b))
        
        # 学习率调度器 - 使用稍慢的衰减率
        lr_scheduler_e = torch.optim.lr_scheduler.ExponentialLR(self.opt_E, gamma=lr_decay)
        lr_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.opt_D, gamma=lr_decay)
        lr_scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(self.opt_Dis, gamma=lr_decay)
        lr_scheduler_m = torch.optim.lr_scheduler.ExponentialLR(self.opt_M, gamma=lr_decay)

        # 设置初始较小学习率，让训练更稳定 - 只在开始新训练时执行
        if start_epoch == 0:
            for param_group in self.opt_Dis.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
        
        # 修改版梯度惩罚函数
        def wasserstein_gp(real_samples, fake_samples):
            batch_size = real_samples.size(0)
            
            # 生成在真假样本之间插值的随机点
            epsilon = torch.rand(batch_size, 1, device=self.device)
            interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
            interpolated.requires_grad_(True)
            
            # 判别器输出
            d_interpolated = self.discriminator(interpolated)
            
            # 创建全1张量供梯度计算
            grad_outputs = torch.ones_like(d_interpolated, device=self.device)
            
            # 计算梯度
            gradients = torch.autograd.grad(
                outputs=d_interpolated,
                inputs=interpolated,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
            
            # 计算梯度范数
            gradients = gradients.view(batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
            return gradient_penalty
        
        # 开始训练
        logging.info(f"开始稳定训练模式，共{epochs}个周期，从第{start_epoch+1}轮开始，梯度裁剪值={clip_value}")
        start_time = time.time()
        
        best_loss = float('inf')
        no_improve_epochs = 0
        
        # 从指定epoch开始训练
        for epoch in range(start_epoch, start_epoch + epochs):
            # 累积损失
            epoch_recon = 0
            epoch_disc = 0
            epoch_gen = 0
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{start_epoch+epochs}")
            for step in pbar:
                # 获取数据
                real_a = next(iter_a).to(self.device)
                real_b = next(iter_b).to(self.device)
                
                # --------------------
                # 1. 训练判别器
                # --------------------
                self.opt_Dis.zero_grad()
                
                # 编码样本
                with torch.no_grad():
                    z_a = self.encoder(real_a)
                    z_b = self.encoder(real_b)
                    z_ab = self.mapper(z_a)  # 从A映射到B的潜在表示
                
                # 真假样本判别
                d_real = self.discriminator(z_b)  # 真样本得分
                d_fake = self.discriminator(z_ab.detach())  # 假样本得分
                
                # Wasserstein损失
                d_loss = torch.mean(d_fake) - torch.mean(d_real)
                
                # 梯度惩罚 - 只在步骤为偶数时计算以减少计算开销
                if step % 2 == 0:
                    gp = wasserstein_gp(z_b, z_ab.detach())
                    d_loss = d_loss + gp
                
                # 反向传播
                d_loss.backward()
                self._clip_gradients(self.opt_Dis, clip_value)
                self.opt_Dis.step()
                
                # --------------------
                # 2. 训练自编码器和映射器
                # --------------------
                # 限制判别器训练频率，避免其过于强大
                if step % 4 == 0:
                    self.opt_E.zero_grad()
                    self.opt_D.zero_grad()
                    self.opt_M.zero_grad()
                    
                    # 获取潜在表示
                    z_a = self.encoder(real_a)
                    z_b = self.encoder(real_b)
                    
                    # 自编码重建
                    recon_a = self.decoder(z_a)
                    recon_b = self.decoder(z_b)
                    
                    # 重建损失 - 强调重建准确性
                    loss_recon = (
                        self.recon_loss(recon_a, real_a) + 
                        self.recon_loss(recon_b, real_b)
                    )
                    
                    # 风格映射
                    z_ab = self.mapper(z_a)  # A->B映射
                    fake_b = self.decoder(z_ab)  # 生成B风格图像
                    
                    # 循环一致性
                    z_fake_b = self.encoder(fake_b)
                    loss_cycle = self.content_loss(z_fake_b, z_ab)
                    
                    # 使用维度统计进行风格匹配 (更稳定的实现)
                    stats_b = {
                        'mean': z_b.mean(dim=0),
                        'std': z_b.std(dim=0) + 1e-5
                    }
                    
                    stats_ab = {
                        'mean': z_ab.mean(dim=0),
                        'std': z_ab.std(dim=0) + 1e-5
                    }
                    
                    loss_style = (
                        F.mse_loss(stats_ab['mean'], stats_b['mean']) + 
                        F.mse_loss(stats_ab['std'], stats_b['std'])
                    )
                    
                    # 对抗损失 - 使用更稳定的公式
                    d_fake_gen = self.discriminator(z_ab)
                    adv_weight = min(1.0, epoch * 0.1)  # 逐步增加对抗权重
                    loss_adv = -torch.mean(d_fake_gen) * adv_weight
                    
                    # 总损失 - 精心调整权重
                    g_total = (
                        12.0 * loss_recon +    # 重建损失
                        5.0 * loss_cycle +     # 循环一致性损失
                        2.0 * loss_style +     # 风格统计损失
                        adv_weight * loss_adv  # 动态调整的对抗损失
                    )
                    
                    # 反向传播
                    g_total.backward()
                    
                    # 应用梯度裁剪
                    self._clip_gradients(self.opt_E, clip_value)
                    self._clip_gradients(self.opt_D, clip_value)
                    self._clip_gradients(self.opt_M, clip_value)
                    
                    # 更新参数
                    self.opt_E.step()
                    self.opt_D.step()
                    self.opt_M.step()
                    
                    # 记录损失
                    epoch_recon += loss_recon.item()
                    epoch_gen += loss_adv.item() if isinstance(loss_adv, torch.Tensor) else 0
                
                # 记录判别器损失
                epoch_disc += d_loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'd_loss': f"{d_loss.item():.3f}",
                    'recon': f"{loss_recon.item():.3f}" if 'loss_recon' in locals() else 'N/A'
                })
            
            # 计算平均损失
            avg_recon = epoch_recon / (steps_per_epoch / 4)  # 调整为实际更新次数
            avg_disc = epoch_disc / steps_per_epoch
            avg_gen = epoch_gen / (steps_per_epoch / 4)
            
            # 记录到历史
            self.history['recon_loss'].append(avg_recon)
            self.history['disc_loss'].append(avg_disc)
            self.history['gen_loss'].append(avg_gen)
            
            # 日志输出
            logging.info(
                f"Epoch {epoch+1}/{start_epoch+epochs}, "
                f"重建: {avg_recon:.4f}, "
                f"判别器: {avg_disc:.4f}, "
                f"生成器: {avg_gen:.4f}"
            )
            
            # 学习率调度
            lr_scheduler_e.step()
            lr_scheduler_d.step()
            lr_scheduler_dis.step()
            lr_scheduler_m.step()
            
            # 早停检查
            current_loss = avg_recon + abs(avg_disc) + abs(avg_gen)
            if current_loss < best_loss:
                best_loss = current_loss
                no_improve_epochs = 0
                # 保存最佳模型
                if save_dir:
                    self.save_model(os.path.join(save_dir, "best_model.pth"), current_epoch=epoch)
            else:
                no_improve_epochs += 1
            
            # 定期保存检查点，并添加当前epoch信息
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_{epoch+1}.pt"), current_epoch=epoch)
            
            # 如果连续15个epoch没有改善，提前停止
            if no_improve_epochs >= 15:
                logging.info(f"提前停止训练: 连续{no_improve_epochs}个周期无改善")
                break
        
        # 训练完成
        self.is_trained = True
        elapsed = time.time() - start_time
        logging.info(f"训练完成，用时: {elapsed:.2f}秒")
        
        # 返回历史记录和最后的epoch
        return self.history
    
    def save_model(self, path, current_epoch=None):
        """保存模型，包含当前训练状态"""
        save_dict = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapper': self.mapper.state_dict(),
            'latent_dim': self.latent_dim,
            'history': self.history,
            'is_trained': self.is_trained,
            'model_state': self.state_dict()  # 添加整体模型状态
        }
        
        # 保存当前epoch
        if current_epoch is not None:
            save_dict['epoch'] = current_epoch
            
        torch.save(save_dict, path)
        logging.info(f"模型已保存到: {path}")
        
    def save_checkpoint(self, path, current_epoch=None):
        """保存检查点，包含完整训练状态"""
        save_dict = {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapper': self.mapper.state_dict(),
            'opt_E': self.opt_E.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'opt_Dis': self.opt_Dis.state_dict(),
            'opt_M': self.opt_M.state_dict(),
            'history': self.history,
            'is_trained': self.is_trained,
            'model_state': self.state_dict()  # 添加整体模型状态
        }
        
        # 保存当前epoch
        if current_epoch is not None:
            save_dict['epoch'] = current_epoch
            
        torch.save(save_dict, path)
        logging.info(f"检查点已保存到: {path}")
        
    def load_model(self, path, device=None):
        """加载模型，包括训练状态"""
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        state = torch.load(path, map_location=self.device)
        
        # 获取潜在维度
        self.latent_dim = state.get('latent_dim', 256)
        
        # 创建模型（如果未创建）
        if not hasattr(self, 'encoder') or not hasattr(self, 'decoder'):
            self.build_models()
            
        # 加载参数
        self.encoder.load_state_dict(state['encoder'])
        self.decoder.load_state_dict(state['decoder'])
        self.discriminator.load_state_dict(state['discriminator'])
        
        if 'mapper' in state:
            self.mapper.load_state_dict(state['mapper'])
            
        # 加载历史和状态
        if 'history' in state:
            self.history = state['history']
            
        self.is_trained = state.get('is_trained', True)
        
        # 加载当前epoch信息（如果有）
        current_epoch = state.get('epoch', 0)
        
        # 设为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.mapper.eval()
        
        logging.info(f"模型已从 {path} 加载")
        if current_epoch > 0:
            logging.info(f"继续训练将从第 {current_epoch+1} 轮开始")
            
        return current_epoch  # 返回当前epoch，便于继续训练

    def state_dict(self):
        """获取模型的状态字典"""
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapper': self.mapper.state_dict()
        }

    def load_state_dict(self, state_dict):
        """从状态字典加载模型"""
        if 'encoder' in state_dict:
            self.encoder.load_state_dict(state_dict['encoder'])
        if 'decoder' in state_dict:
            self.decoder.load_state_dict(state_dict['decoder'])
        if 'discriminator' in state_dict:
            self.discriminator.load_state_dict(state_dict['discriminator'])
        if 'mapper' in state_dict:
            self.mapper.load_state_dict(state_dict['mapper'])