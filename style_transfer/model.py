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
        self.path = data_path
        
        try:
            # 尝试加载文件的不同方式
            try:
                # 主要加载方式
                logging.info(f"尝试加载数据: {data_path}")
                data = torch.load(data_path)
                logging.info(f"数据加载成功，文件大小: {os.path.getsize(data_path) / (1024*1024):.2f}MB")
            except Exception as e:
                # 如果主要方式失败，尝试备用加载方式
                logging.warning(f"常规加载失败，尝试使用备用方式: {str(e)}")
                
                # 使用map_location确保加载到CPU，避免CUDA内存问题
                data = torch.load(data_path, map_location='cpu')
            
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
                    logging.info(f"发现特征列表，包含 {len(features)} 个张量")
                    try:
                        self.features = torch.stack(features)
                    except Exception as stack_err:
                        # 如果无法直接堆叠，检查维度不一致问题
                        logging.warning(f"无法直接堆叠特征: {stack_err}")
                        
                        # 转换为统一大小
                        if len(features) > 0:
                            sample = features[0]
                            expected_shape = list(sample.shape)
                            consistent_features = []
                            
                            for i, feat in enumerate(features):
                                if feat.shape == sample.shape:
                                    consistent_features.append(feat)
                                else:
                                    logging.warning(f"跳过形状不一致的特征 {i}: {feat.shape} != {expected_shape}")
                            
                            if consistent_features:
                                self.features = torch.stack(consistent_features)
                                logging.info(f"成功堆叠 {len(consistent_features)}/{len(features)} 个形状一致的特征")
                            else:
                                raise ValueError(f"无法找到形状一致的特征进行堆叠")
                        else:
                            raise ValueError("特征列表为空")
                else:
                    raise ValueError(f"不支持的features类型: {type(features)}")
            else:
                # 尝试检查文件包含的内容
                keys = list(data.keys()) if isinstance(data, dict) else []
                raise ValueError(f"数据中找不到'features'键，可用键: {keys}")
            
            # 记录原始形状
            self.original_shape = self.features.shape
            logging.info(f"加载的特征形状: {self.original_shape}")
            
            # 可选去除时间维度（通常为1）
            if squeeze_time and self.features.shape[2] == 1:
                self.features = self.features.squeeze(2)
                logging.info(f"去除时间维度后的形状: {self.features.shape}")
            
            # 获取图像路径和元数据，如果有
            self.image_paths = data.get('image_paths', None)
            self.metadata = data.get('metadata', None)
            
            logging.info(f"数据集加载完成: {data_path}")
            logging.info(f"  - 特征形状: {self.features.shape}")
            logging.info(f"  - 样本数量: {len(self.features)}")
            
        except Exception as e:
            logging.error(f"加载数据集 {data_path} 时出错: {e}")
            import traceback
            logging.error(traceback.format_exc())
            raise

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
    def __init__(self, device='cuda', latent_dim=256):  # 增大默认潜在空间维度
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # 初始化模型
        self.encoder = None
        self.decoder = None
        self.discriminator = None
        self.style_mapper = None  # 新增风格映射网络
        
        # 优化器
        self.optimizer_E = None
        self.optimizer_D = None
        self.optimizer_Dis = None
        self.optimizer_Map = None  # 新增映射网络的优化器
        
        # 损失函数
        self.reconstruction_loss = nn.MSELoss()
        self.content_loss = nn.L1Loss()  # 内容损失
        self.style_loss = nn.MSELoss()   # 风格损失
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # 带logits的BCE，更稳定
        
        # 训练参数
        self.lambda_recon = 10.0     # 重建损失权重
        self.lambda_content = 5.0    # 内容损失权重
        self.lambda_adv = 1.0        # 对抗损失权重
        
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
        # 构建更强大的编码器
        self.encoder = EnhancedEncoder(
            in_channels=in_channels, 
            latent_dim=self.latent_dim,
            use_attention=True  # 使用注意力机制
        ).to(self.device)
        
        # 构建更强大的解码器
        self.decoder = EnhancedDecoder(
            out_channels=in_channels, 
            latent_dim=self.latent_dim,
            use_residual=True  # 使用残差连接
        ).to(self.device)
        
        # 风格映射网络 - 在潜在空间中学习风格映射
        self.style_mapper = StyleMapper(
            latent_dim=self.latent_dim,
            depth=4  # 增加深度
        ).to(self.device)
        
        # 增强的判别器
        self.discriminator = EnhancedDiscriminator(
            latent_dim=self.latent_dim,
            use_spectral_norm=True  # 使用谱归一化
        ).to(self.device)
        
        # 优化器使用更优的配置
        lr = 3e-4  # 使用稍低的学习率，提高稳定性
        beta1, beta2 = 0.5, 0.999  # 标准GAN的beta参数
        
        self.optimizer_E = optim.AdamW(self.encoder.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-4)
        self.optimizer_D = optim.AdamW(self.decoder.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-4)
        self.optimizer_Map = optim.AdamW(self.style_mapper.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-4)
        self.optimizer_Dis = optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=1e-4)
        
        # 使用学习率调度器
        self.scheduler_E = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_E, T_max=50)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=50)
        self.scheduler_Map = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_Map, T_max=50)
        self.scheduler_Dis = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_Dis, T_max=50)
        
        logging.info(f"构建增强模型完成, 输入通道数: {in_channels}, 潜在空间维度: {self.latent_dim}")

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
                # 1. 训练自编码器 - 重建和内容损失
                # ---------------------
                self.optimizer_E.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_Map.zero_grad()

                # 编码
                latent_a = self.encoder(real_a)
                latent_b = self.encoder(real_b)
                
                # 风格映射
                mapped_latent_a = self.style_mapper(latent_a)  # A->B
                
                # 解码
                recon_a = self.decoder(latent_a)       # A->A
                recon_b = self.decoder(latent_b)       # B->B
                fake_b = self.decoder(mapped_latent_a)  # A->B
                
                # 重建损失
                loss_recon_a = self.reconstruction_loss(recon_a, real_a)
                loss_recon_b = self.reconstruction_loss(recon_b, real_b)
                loss_recon = loss_recon_a + loss_recon_b
                
                # 重新编码生成的B风格特征，计算循环一致性损失
                latent_fake_b = self.encoder(fake_b)
                loss_content = self.content_loss(latent_fake_b, mapped_latent_a)
                
                # 总重建损失
                total_recon_loss = self.lambda_recon * loss_recon + self.lambda_content * loss_content
                
                # 反向传播
                total_recon_loss.backward()
                self.optimizer_E.step()
                self.optimizer_D.step()
                self.optimizer_Map.step()
                
                # ---------------------
                # 2. 训练判别器
                # ---------------------
                self.optimizer_Dis.zero_grad()

                # 获取真假样本
                latent_b_real = self.encoder(real_b).detach()
                fake_latent_b = self.style_mapper(latent_a.detach())
                
                # 真样本的判别
                pred_real = self.discriminator(latent_b_real)
                # 假样本的判别
                pred_fake = self.discriminator(fake_latent_b.detach())
                
                # 使用hinge loss或WGAN-GP损失可以提高稳定性
                loss_d_real = torch.mean(F.relu(1.0 - pred_real))
                loss_d_fake = torch.mean(F.relu(1.0 + pred_fake))
                loss_discriminator = loss_d_real + loss_d_fake
                
                # 反向传播
                loss_discriminator.backward()
                self.optimizer_Dis.step()

                # ---------------------
                # 3. 训练生成器 - 对抗损失和风格损失
                # ---------------------
                self.optimizer_E.zero_grad()
                self.optimizer_Map.zero_grad()

                # 重新计算映射的潜在向量
                latent_a_gen = self.encoder(real_a)
                mapped_latent_a_gen = self.style_mapper(latent_a_gen)
                
                # 判别器对生成的风格B的预测
                pred_gen = self.discriminator(mapped_latent_a_gen)
                
                # 生成器对抗损失 - 欺骗判别器
                loss_generator = -torch.mean(pred_gen)  # WGAN风格的损失
                
                # 风格损失 - 让生成的特征与目标风格匹配
                target_style_stats = calc_style_statistics(latent_b)
                gen_style_stats = calc_style_statistics(mapped_latent_a_gen)
                loss_style = style_distance(gen_style_stats, target_style_stats)
                
                # 总生成器损失
                total_gen_loss = self.lambda_adv * loss_generator + loss_style
                
                # 反向传播
                total_gen_loss.backward()
                self.optimizer_E.step()
                self.optimizer_Map.step()
                
                # 累积损失
                epoch_recon_loss += total_recon_loss.item()
                epoch_disc_loss += loss_discriminator.item()
                epoch_gen_loss += total_gen_loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'recon': total_recon_loss.item(),
                    'disc': loss_discriminator.item(), 
                    'gen': total_gen_loss.item()
                })
            
            # 更新学习率
            self.scheduler_E.step()
            self.scheduler_D.step()
            self.scheduler_Map.step()
            self.scheduler_Dis.step()
            
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

    def transfer_style(self, input_tensor, add_time_dim=False):
        """
        将输入张量从风格A转换为风格B
        
        Args:
            input_tensor: 输入张量，形状为 [N, 16, 32, 32] 或 [16, 32, 32]
            add_time_dim: 是否在输出中添加时间维度
            
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
            mapped_latent = self.style_mapper(latent)  # 风格映射
            output = self.decoder(mapped_latent)
            
            # 如果需要添加时间维度
            if add_time_dim:
                output = output.unsqueeze(2)  # [N, C, H, W] -> [N, C, 1, H, W]
                
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


# 新增更强大的编码器
class EnhancedEncoder(nn.Module):
    def __init__(self, in_channels=16, latent_dim=256, use_attention=True):
        super(EnhancedEncoder, self).__init__()
        
        # 激活函数
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
        # 初始卷积
        self.conv_in = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.norm_in = nn.InstanceNorm2d(64)
        
        # 下采样块
        self.down1 = ResidualDownBlock(64, 128)  # 32x32 -> 16x16
        self.down2 = ResidualDownBlock(128, 256)  # 16x16 -> 8x8
        self.down3 = ResidualDownBlock(256, 512)  # 8x8 -> 4x4
        
        # 注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = SelfAttention(512)
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 映射到潜在空间
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        # 初始处理
        x = self.act(self.norm_in(self.conv_in(x)))
        
        # 下采样
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # 注意力
        if self.use_attention:
            x = self.attention(x)
        
        # 全局池化和映射
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        
        return x


# 新增更强大的解码器
class EnhancedDecoder(nn.Module):
    def __init__(self, out_channels=16, latent_dim=256, use_residual=True):
        super(EnhancedDecoder, self).__init__()
        
        # 潜在空间 -> 4x4 特征图
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.norm_fc = nn.BatchNorm2d(512)
        
        # 上采样块
        self.up1 = ResidualUpBlock(512, 256)  # 4x4 -> 8x8
        self.up2 = ResidualUpBlock(256, 128)  # 8x8 -> 16x16
        self.up3 = ResidualUpBlock(128, 64)   # 16x16 -> 32x32
        
        # AdaIN自适应实例归一化
        self.use_residual = use_residual
        if use_residual:
            self.res1 = AdaINResBlock(256)
            self.res2 = AdaINResBlock(128)
            self.res3 = AdaINResBlock(64)
        
        # 输出层
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # 从潜在空间重建特征图
        x = self.fc(x).view(-1, 512, 4, 4)
        x = F.relu(self.norm_fc(x))
        
        # 上采样
        x = self.up1(x)
        if self.use_residual:
            x = self.res1(x)
            
        x = self.up2(x)
        if self.use_residual:
            x = self.res2(x)
            
        x = self.up3(x)
        if self.use_residual:
            x = self.res3(x)
        
        # 输出
        x = self.conv_out(x)
        
        return x


# 风格映射网络 - MLP形式
class StyleMapper(nn.Module):
    def __init__(self, latent_dim=256, depth=4, width_factor=2):
        super(StyleMapper, self).__init__()
        
        hidden_dim = latent_dim * width_factor
        
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dim))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 中间层
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mapping(x)


# 增强的判别器
class EnhancedDiscriminator(nn.Module):
    def __init__(self, latent_dim=256, use_spectral_norm=True):
        super(EnhancedDiscriminator, self).__init__()
        
        def get_norm_layer(dim):
            if use_spectral_norm:
                return nn.utils.spectral_norm(nn.Linear(dim, dim))
            else:
                return nn.Identity()
        
        # 更深的判别器网络
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            get_norm_layer(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 512),
            get_norm_layer(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            get_norm_layer(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 1)  # 输出原始logits，不用sigmoid
        )
        
    def forward(self, x):
        return self.model(x)


# 残差下采样块
class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDownBlock, self).__init__()
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        # 残差分支
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        residual = x
        
        # 主分支
        out = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        out = F.leaky_relu(self.norm2(self.conv2(out)), 0.2)
        
        # 残差连接
        out = out + self.shortcut(residual)
        
        return out


# 残差上采样块
class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualUpBlock, self).__init__()
        
        # 主分支
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        # 残差分支
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        residual = x
        
        # 主分支
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        
        # 残差连接
        out = out + self.shortcut(residual)
        
        return out


# 自适应实例归一化残差块
class AdaINResBlock(nn.Module):
    def __init__(self, channels):
        super(AdaINResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.instance_norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.instance_norm2 = nn.InstanceNorm2d(channels, affine=False)
        
    def forward(self, x):
        residual = x
        
        # 主分支
        out = F.relu(self.instance_norm1(self.conv1(x)))
        out = self.instance_norm2(self.conv2(out))
        
        # 残差连接
        out = out + residual
        out = F.relu(out)
        
        return out


# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 线性映射
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C/8
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C/8 x HW
        
        # 注意力图
        energy = torch.bmm(proj_query, proj_key)  # B x HW x HW
        attention = F.softmax(energy, dim=2)
        
        # 加权值
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


# 计算风格特征的统计特性(均值和方差)
def calc_style_statistics(features):
    mean = features.mean(0, keepdim=True)
    std = features.std(0, keepdim=True)
    return {'mean': mean, 'std': std}


# 计算两个风格特征的距离
def style_distance(style_a, style_b):
    mean_loss = F.mse_loss(style_a['mean'], style_b['mean'])
    std_loss = F.mse_loss(style_a['std'], style_b['std'])
    return mean_loss + std_loss
