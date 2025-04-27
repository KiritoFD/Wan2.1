#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
风格转换模型：基于AAE(Adversarial Autoencoder)的方法学习VAE潜在空间中两种风格之间的映射

输入数据形状: [N, 16, 1, 32, 32]
- N: 样本数
- 16: VAE潜在空间维度
- 1: 时间维度
- 32x32: 空间维度
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


class StyleDataset(Dataset):
    """
    风格数据集：加载.pt文件中的VAE编码特征
    
    Args:
        data_path (str): .pt文件路径
        squeeze_time (bool): 是否去除时间维度(通常为1)
    """
    def __init__(self, data_path, squeeze_time=True):
        data = torch.load(data_path)
        self.path = data_path
        
        # 处理features，根据不同的格式进行适配
        if 'features' in data:
            features = data['features']
            
            # 如果features是单个张量
            if isinstance(features, torch.Tensor):
                if features.dim() == 4:  # [16, 1, 32, 32]
                    # 添加批次维度
                    self.features = features.unsqueeze(0)
                else:
                    # 假设已有批次维度 [N, 16, 1, 32, 32]
                    self.features = features
            # 如果features是张量列表
            elif isinstance(features, list):
                # 堆叠为单个张量
                self.features = torch.stack(features)
            else:
                raise ValueError(f"不支持的features类型: {type(features)}")
        else:
            raise ValueError(f"数据中找不到'features'键")
        
        # 记录原始形状
        self.original_shape = self.features.shape
        
        # 可选去除时间维度（通常为1）
        if squeeze_time and self.features.shape[2] == 1:
            self.features = self.features.squeeze(2)
        
        # 获取图像路径和元数据，如果有
        self.image_paths = data.get('image_paths', None)
        self.metadata = data.get('metadata', None)
        
        logging.info(f"加载数据集: {data_path}")
        logging.info(f"  - 特征形状: {self.features.shape}")
        logging.info(f"  - 样本数量: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
    
    @property
    def feature_shape(self):
        return self.features.shape[1:]
    
    def get_metadata(self):
        return self.metadata
    
    def get_paths(self):
        return self.image_paths


class Encoder(nn.Module):
    """
    编码器：将输入的VAE潜在向量映射到新的潜在空间
    
    输入形状: [N, 16, 32, 32] (已去除时间维度)
    输出形状: [N, latent_dim]
    """
    def __init__(self, in_channels=16, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 降采样: 32x32 -> 16x16
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 降采样: 16x16 -> 8x8
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # 自适应平均池化，保证输出大小固定
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层，输出潜在向量
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x):
        # 输入形状: [N, 16, 32, 32]
        x = self.conv1(x)          # -> [N, 64, 16, 16]
        x = self.conv2(x)          # -> [N, 256, 8, 8]
        x = self.adaptive_pool(x)  # -> [N, 256, 1, 1]
        x = torch.flatten(x, 1)    # -> [N, 256]
        x = self.fc(x)             # -> [N, latent_dim]
        return x


class Decoder(nn.Module):
    """
    解码器：将潜在空间映射回VAE潜在向量
    
    输入形状: [N, latent_dim]
    输出形状: [N, 16, 32, 32]
    """
    def __init__(self, out_channels=16, latent_dim=128):
        super(Decoder, self).__init__()
        
        # 全连接层，从潜在向量生成特征图
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 上采样: 8x8 -> 16x16
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 上采样: 16x16 -> 32x32
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
            # 不使用激活函数，因为VAE输出范围不确定
        )

    def forward(self, x):
        # 输入形状: [N, latent_dim]
        x = self.fc(x)                            # -> [N, 256*8*8]
        x = x.view(-1, 256, 8, 8)                 # -> [N, 256, 8, 8]
        x = self.conv1(x)                         # -> [N, 128, 16, 16]
        x = self.conv2(x)                         # -> [N, out_channels, 32, 32]
        return x


class Discriminator(nn.Module):
    """
    判别器：区分潜在向量的来源（风格A或风格B）
    
    输入形状: [N, latent_dim]
    输出形状: [N, 1]
    """
    def __init__(self, latent_dim=128):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class StyleTransferAAE:
    """
    基于AAE的风格转换模型
    
    使用对抗性自编码器学习两种不同风格之间的映射
    """
    def __init__(self, device='cuda', latent_dim=128):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # 初始化模型
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        
        # 优化器
        self.optimizer_E = None
        self.optimizer_D = None
        self.optimizer_Dis = None
        
        # 损失函数
        self.reconstruction_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        
        # 训练状态
        self.is_trained = False
        self.training_history = {
            'recon_loss': [],
            'disc_loss': [],
            'gen_loss': []
        }
        
        logging.info(f"初始化StyleTransferAAE模型, 设备: {self.device}, 潜在维度: {latent_dim}")

    def build_models(self, in_channels=16):
        """构建所有模型组件"""
        self.encoder = Encoder(in_channels=in_channels, latent_dim=self.latent_dim).to(self.device)
        self.decoder = Decoder(out_channels=in_channels, latent_dim=self.latent_dim).to(self.device)
        self.discriminator = Discriminator(latent_dim=self.latent_dim).to(self.device)
        
        # 初始化优化器
        self.optimizer_E = optim.Adam(self.encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.decoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_Dis = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        logging.info(f"构建模型完成, 输入通道数: {in_channels}")

    def train(self, dataloader_a, dataloader_b, num_epochs=100, save_dir=None):
        """
        训练模型
        
        Args:
            dataloader_a: 风格A的数据加载器
            dataloader_b: 风格B的数据加载器
            num_epochs: 训练轮数
            save_dir: 保存模型的目录
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        logging.info(f"开始训练, 训练轮数: {num_epochs}")
        start_time = time.time()
        
        # 确保模型处于训练模式
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        
        # 创建无限循环的数据迭代器
        def infinite_dataloader(dataloader):
            while True:
                for data in dataloader:
                    yield data
        
        # 创建两个风格的无限迭代器
        iter_a = infinite_dataloader(dataloader_a)
        iter_b = infinite_dataloader(dataloader_b)
        
        # 每个epoch的步数为较短的dataloader的长度
        steps_per_epoch = min(len(dataloader_a), len(dataloader_b))
        
        for epoch in range(num_epochs):
            epoch_recon_loss = 0
            epoch_disc_loss = 0
            epoch_gen_loss = 0
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch [{epoch+1}/{num_epochs}]")
            for _ in pbar:
                # 获取下一批数据
                real_a = next(iter_a).to(self.device)
                real_b = next(iter_b).to(self.device)
                
                batch_size = real_a.size(0)
                
                # ---------------------
                # 训练自编码器（重建损失）
                # ---------------------
                self.optimizer_E.zero_grad()
                self.optimizer_D.zero_grad()

                # 编码和解码
                latent_a = self.encoder(real_a)
                reconstructed_a = self.decoder(latent_a)
                loss_reconstruction = self.reconstruction_loss(reconstructed_a, real_a)

                # 反向传播
                loss_reconstruction.backward()
                self.optimizer_E.step()
                self.optimizer_D.step()
                
                # ---------------------
                # 训练判别器
                # ---------------------
                self.optimizer_Dis.zero_grad()

                # 真实分布（来自风格B的潜在空间）
                latent_b = self.encoder(real_b).detach()
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)

                # 判别真实分布
                pred_real = self.discriminator(latent_b)
                loss_real = self.adversarial_loss(pred_real, real_labels)

                # 判别生成分布
                pred_fake = self.discriminator(latent_a.detach())
                loss_fake = self.adversarial_loss(pred_fake, fake_labels)

                # 总判别器损失
                loss_discriminator = (loss_real + loss_fake) / 2
                loss_discriminator.backward()
                self.optimizer_Dis.step()

                # ---------------------
                # 训练生成器（欺骗判别器）
                # ---------------------
                self.optimizer_E.zero_grad()

                # 让生成的潜在空间被误认为是真实的
                pred_fake = self.discriminator(latent_a)
                loss_generator = self.adversarial_loss(pred_fake, real_labels)

                # 反向传播
                loss_generator.backward()
                self.optimizer_E.step()
                
                # 累积损失
                epoch_recon_loss += loss_reconstruction.item()
                epoch_disc_loss += loss_discriminator.item()
                epoch_gen_loss += loss_generator.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'recon': loss_reconstruction.item(),
                    'disc': loss_discriminator.item(), 
                    'gen': loss_generator.item()
                })
            
            # 计算平均损失
            avg_recon_loss = epoch_recon_loss / steps_per_epoch
            avg_disc_loss = epoch_disc_loss / steps_per_epoch
            avg_gen_loss = epoch_gen_loss / steps_per_epoch
            
            # 记录训练历史
            self.training_history['recon_loss'].append(avg_recon_loss)
            self.training_history['disc_loss'].append(avg_disc_loss)
            self.training_history['gen_loss'].append(avg_gen_loss)
            
            # 输出信息
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], "
                         f"重建损失: {avg_recon_loss:.4f}, "
                         f"判别器损失: {avg_disc_loss:.4f}, "
                         f"生成器损失: {avg_gen_loss:.4f}")
            
            # 保存检查点
            if save_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        self.is_trained = True
        elapsed_time = time.time() - start_time
        logging.info(f"训练完成，耗时: {elapsed_time:.2f} 秒")
        
        # 保存最终模型
        if save_dir:
            self.save_model(save_dir)
            self.plot_training_history(save_dir)
            
        return self.training_history

    def transfer_style(self, input_tensor):
        """
        将输入张量从风格A转换为风格B
        
        Args:
            input_tensor: 输入张量，形状为 [N, 16, 32, 32] 或 [16, 32, 32]
            
        Returns:
            风格转换后的张量
        """
        if not self.is_trained:
            logging.warning("模型尚未训练，转换结果可能不准确")
        
        # 确保模型处于评估模式
        self.encoder.eval()
        self.decoder.eval()
        
        # 处理单个样本的情况
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
            
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            latent = self.encoder(input_tensor)
            output = self.decoder(latent)
            
        return output.cpu()

    def save_model(self, save_dir):
        """保存完整模型"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "style_transfer_model.pth")
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'latent_dim': self.latent_dim,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }, model_path)
        
        logging.info(f"模型已保存到: {model_path}")

    def save_checkpoint(self, checkpoint_path):
        """保存训练检查点"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_E_state_dict': self.optimizer_E.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_Dis_state_dict': self.optimizer_Dis.state_dict(),
            'training_history': self.training_history,
        }, checkpoint_path)
        
        logging.info(f"检查点已保存到: {checkpoint_path}")

    def load_model(self, model_path, device=None):
        """加载模型"""
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取潜在维度
        self.latent_dim = checkpoint.get('latent_dim', 128)
        
        # 构建模型（如果尚未构建）
        if self.encoder is None or self.decoder is None or self.discriminator is None:
            self.build_models()
            
        # 加载模型参数
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        if 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
        # 加载训练历史和状态
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        self.is_trained = checkpoint.get('is_trained', True)
        
        # 将模型设置为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        
        logging.info(f"模型已从 {model_path} 加载")

    def plot_training_history(self, save_dir=None):
        """绘制训练历史"""
        plt.figure(figsize=(12, 5))
        
        # 绘制重建损失
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['recon_loss'], label='Reconstruction Loss')
        plt.title('Reconstruction Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制对抗损失
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['disc_loss'], label='Discriminator Loss')
        plt.plot(self.training_history['gen_loss'], label='Generator Loss')
        plt.title('Adversarial Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "training_history.png"))
            plt.close()
        else:
            plt.show()
