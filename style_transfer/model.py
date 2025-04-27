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
        return self.train(
            dataloader_a=dataloader_a,
            dataloader_b=dataloader_b,
            epochs=num_epochs,
            save_dir=save_dir,
            lambda_gp=lambda_gp,
            **kwargs  # 传递所有额外参数
        )
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapper': self.mapper.state_dict(),
            'latent_dim': self.latent_dim,
            'history': self.history,
            'is_trained': self.is_trained
        }, path)
        logging.info(f"模型已保存到: {path}")
        
    def save_checkpoint(self, path):
        """保存检查点"""
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'mapper': self.mapper.state_dict(),
            'opt_E': self.opt_E.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'opt_Dis': self.opt_Dis.state_dict(),
            'opt_M': self.opt_M.state_dict(),
            'history': self.history,
            'is_trained': self.is_trained
        }, path)
        
    def load_model(self, path, device=None):
        """加载模型"""
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
        
        # 设为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.mapper.eval()
        
        logging.info(f"模型已从 {path} 加载")
        
    def plot_history(self, save_dir=None):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))
        
        # 重建损失
        plt.subplot(1, 2, 1)
        plt.plot(self.history['recon_loss'])
        plt.title('重建损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 对抗损失
        plt.subplot(1, 2, 2)
        plt.plot(self.history['disc_loss'], label='判别器')
        plt.plot(self.history['gen_loss'], label='生成器')
        plt.title('对抗损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "training_history.png"))
            plt.close()
        else:
            plt.show()