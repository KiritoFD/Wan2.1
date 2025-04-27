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


class StyleMapper(nn.Module):
    """
    风格映射网络 - MLP形式
    
    输入形状: [N, latent_dim]
    输出形状: [N, latent_dim]
    """
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


class StyleTransferAAE:
    """
    基于AAE的风格转换模型
    
    使用对抗性自编码器学习两种不同风格之间的映射
    """
    def __init__(self, device='cuda', latent_dim=256, use_flash_attn=True):  # 增大默认潜在空间维度
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # 检查Flash Attention可用性
        flash_available, flash_version = check_flash_attn_available()
        if use_flash_attn and flash_available:
            self.use_flash_attn = True
            logging.info(f"启用FlashAttention优化，版本:{flash_version}")
        else:
            self.use_flash_attn = False
            if use_flash_attn:
                logging.warning("FlashAttention不可用，回退到标准注意力机制。可通过pip install flash-attn安装")
        
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
            use_attention=True,  # 使用注意力机制
            use_flash_attn=self.use_flash_attn  # 传递FlashAttention参数
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
        self.style_mapper.train()
        
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
                total_recon_loss.backward(retain_graph=True)  # 添加retain_graph=True
                self.optimizer_E.step()
                self.optimizer_D.step()
                self.optimizer_Map.step()
                
                # ---------------------
                # 2. 训练判别器
                # ---------------------
                self.optimizer_Dis.zero_grad()

                # 获取真假样本 - 使用detach()切断梯度流
                latent_b_real = self.encoder(real_b).detach()  # 确保完全切断梯度
                fake_latent_b = self.style_mapper(latent_a.detach()).detach()  # 双重detach确保安全
                
                # 真样本的判别
                pred_real = self.discriminator(latent_b_real)
                # 假样本的判别
                pred_fake = self.discriminator(fake_latent_b)
                
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

                # 重新计算映射的潜在向量 - 使用新的forward pass避免梯度问题
                latent_a_gen = self.encoder(real_a)
                mapped_latent_a_gen = self.style_mapper(latent_a_gen)
                
                # 判别器对生成的风格B的预测
                pred_gen = self.discriminator(mapped_latent_a_gen)
                
                # 生成器对抗损失 - 欺骗判别器
                loss_generator = -torch.mean(pred_gen)  # WGAN风格的损失
                
                # 风格损失 - 让生成的特征与目标风格匹配
                target_style_stats = calc_style_statistics(latent_b.detach())  # 使用detach避免重复反向传播
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
    
    def save_checkpoint(self, checkpoint_path):
        """保存训练检查点"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'style_mapper_state_dict': self.style_mapper.state_dict(),
            'optimizer_E_state_dict': self.optimizer_E.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_Dis_state_dict': self.optimizer_Dis.state_dict(),
            'optimizer_Map_state_dict': self.optimizer_Map.state_dict(),
            'scheduler_E_state_dict': self.scheduler_E.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'scheduler_Dis_state_dict': self.scheduler_Dis.state_dict(),
            'scheduler_Map_state_dict': self.scheduler_Map.state_dict(),
            'training_history': self.training_history,
            'latent_dim': self.latent_dim
        }, checkpoint_path)
        
        logging.info(f"检查点已保存到: {checkpoint_path}")
    
    def save_model(self, save_dir):
        """保存完整模型"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "style_transfer_model.pth")
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'style_mapper_state_dict': self.style_mapper.state_dict(),
            'latent_dim': self.latent_dim,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }, model_path)
        
        # 保存配置信息
        config_path = os.path.join(save_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"latent_dim: {self.latent_dim}\n")
            f.write(f"lambda_recon: {self.lambda_recon}\n")
            f.write(f"lambda_content: {self.lambda_content}\n")
            f.write(f"lambda_adv: {self.lambda_adv}\n")
            
        logging.info(f"模型已保存到: {model_path}")
        logging.info(f"配置信息已保存到: {config_path}")
    
    def load_model(self, model_path, device=None):
        """加载已保存的模型"""
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取潜在维度
        self.latent_dim = checkpoint.get('latent_dim', self.latent_dim)
        
        # 检查是否需要构建模型
        if not all([self.encoder, self.decoder, self.discriminator, self.style_mapper]):
            logging.info("模型组件未初始化，正在从头构建...")
            # 从保存的状态字典推断输入通道数
            encoder_state_dict = checkpoint['encoder_state_dict']
            first_layer_weight = encoder_state_dict.get('conv_in.weight')
            if first_layer_weight is not None:
                in_channels = first_layer_weight.size(1)
            else:
                in_channels = 16  # 默认通道数
            
            # 构建模型组件
            self.build_models(in_channels=in_channels)
        
        # 加载状态字典
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'style_mapper_state_dict' in checkpoint:
            self.style_mapper.load_state_dict(checkpoint['style_mapper_state_dict'])
        
        # 加载训练历史
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        self.is_trained = checkpoint.get('is_trained', True)
        
        # 将模型设置为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.style_mapper.eval()
        
        logging.info(f"模型已从 {model_path} 加载完成")
        return self
    
    def load_checkpoint(self, checkpoint_path, device=None, resume_training=True):
        """加载检查点继续训练"""
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取潜在维度
        self.latent_dim = checkpoint.get('latent_dim', self.latent_dim)
        
        # 检查是否需要构建模型
        if not all([self.encoder, self.decoder, self.discriminator, self.style_mapper]):
            logging.info("模型组件未初始化，正在从头构建...")
            # 从保存的状态字典推断输入通道数
            encoder_state_dict = checkpoint['encoder_state_dict']
            first_layer_weight = encoder_state_dict.get('conv_in.weight')
            if first_layer_weight is not None:
                in_channels = first_layer_weight.size(1)
            else:
                in_channels = 16  # 默认通道数
            
            # 构建模型组件
            self.build_models(in_channels=in_channels)
        
        # 加载模型状态
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.style_mapper.load_state_dict(checkpoint['style_mapper_state_dict'])
        
        if resume_training:
            # 加载优化器状态
            self.optimizer_E.load_state_dict(checkpoint['optimizer_E_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.optimizer_Dis.load_state_dict(checkpoint['optimizer_Dis_state_dict'])
            self.optimizer_Map.load_state_dict(checkpoint['optimizer_Map_state_dict'])
            
            # 加载学习率调度器状态
            if 'scheduler_E_state_dict' in checkpoint:
                self.scheduler_E.load_state_dict(checkpoint['scheduler_E_state_dict'])
                self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
                self.scheduler_Dis.load_state_dict(checkpoint['scheduler_Dis_state_dict'])
                self.scheduler_Map.load_state_dict(checkpoint['scheduler_Map_state_dict'])
        
        # 加载训练历史
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logging.info(f"检查点已从 {checkpoint_path} 加载完成")
        return self
    
    def plot_training_history(self, save_dir=None):
        """绘制训练历史"""
        if not self.training_history or all(len(v) == 0 for v in self.training_history.values()):
            logging.warning("没有训练历史数据可供绘制")
            return
            
        plt.figure(figsize=(15, 5))
        
        # 重建损失
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['recon_loss'])
        plt.title('重建损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 判别器损失
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['disc_loss'])
        plt.title('判别器损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 生成器损失
        plt.subplot(1, 3, 3)
        plt.plot(self.training_history['gen_loss'])
        plt.title('生成器损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "training_history.png"))
            plt.close()
            logging.info(f"训练历史图已保存到: {os.path.join(save_dir, 'training_history.png')}")
        else:
            plt.show()

    def transfer_style(self, input_tensor, add_time_dim=False):
        """
        将输入张量从风格A转换为风格B
        
        Args:
            input_tensor: 输入张量，形状为 [N, C, H, W] 或 [C, H, W]
            add_time_dim: 是否在输出中添加时间维度
            
        Returns:
            转换后的张量
        """
        if not self.is_trained:
            logging.warning("模型尚未训练，转换效果可能不理想")
        
        self.encoder.eval()
        self.decoder.eval()
        self.style_mapper.eval()
        
        # 处理输入张量的维度
        if input_tensor.dim() == 3:  # [C, H, W]
            input_tensor = input_tensor.unsqueeze(0)  # [1, C, H, W]
            
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            
            # 编码
            latent = self.encoder(input_tensor)
            
            # 映射风格
            mapped_latent = self.style_mapper(latent)
            
            # 解码
            output = self.decoder(mapped_latent)
            
            if add_time_dim:
                output = output.unsqueeze(2)  # [N, C, 1, H, W]
        
        return output.cpu()
    
    def evaluate(self, test_dataloader_a, test_dataloader_b=None, save_dir=None, n_samples=5):
        """
        评估模型的风格转换效果
        
        Args:
            test_dataloader_a: 风格A的测试数据加载器
            test_dataloader_b: 风格B的测试数据加载器 (可选)
            save_dir: 保存评估结果的目录
            n_samples: 用于可视化的样本数量
            
        Returns:
            dict: 包含评估指标的字典
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.encoder.eval()
        self.decoder.eval()
        self.style_mapper.eval()
        
        metrics = {
            'recon_loss_a': 0.0,  # 风格A重建损失
            'transfer_fidelity': 0.0,  # 转换保真度
            'cycle_consistency': 0.0,  # 循环一致性
        }
        
        # 获取测试样本
        test_a_samples = []
        for i, batch in enumerate(test_dataloader_a):
            if i >= n_samples:
                break
            test_a_samples.append(batch)
        
        if len(test_a_samples) > 0:
            test_a_samples = torch.cat(test_a_samples, dim=0)[:n_samples].to(self.device)
        else:
            logging.warning("没有测试样本可用于评估")
            return metrics
        
        test_b_samples = None
        if test_dataloader_b:
            test_b_samples = []
            for i, batch in enumerate(test_dataloader_b):
                if i >= n_samples:
                    break
                test_b_samples.append(batch)
            
            if len(test_b_samples) > 0:
                test_b_samples = torch.cat(test_b_samples, dim=0)[:n_samples].to(self.device)
        
        with torch.no_grad():
            # 1. 评估重建质量
            latent_a = self.encoder(test_a_samples)
            recon_a = self.decoder(latent_a)
            metrics['recon_loss_a'] = F.mse_loss(recon_a, test_a_samples).item()
            
            # 2. 风格转换
            mapped_latent_a = self.style_mapper(latent_a)
            fake_b = self.decoder(mapped_latent_a)
            
            # 3. 计算循环一致性
            latent_fake_b = self.encoder(fake_b)
            metrics['transfer_fidelity'] = F.l1_loss(latent_fake_b, mapped_latent_a).item()
            
            # 4. 循环转换回风格A (如果有风格B样本)
            if test_b_samples is not None:
                latent_b = self.encoder(test_b_samples)
                recon_b = self.decoder(latent_b)
                
                # 评估风格B重建质量
                metrics['recon_loss_b'] = F.mse_loss(recon_b, test_b_samples).item()
                
                # B->A 转换
                # 这里假设有一个逆映射, 或者可以训练第二个映射器
                # 但由于当前模型只关注 A->B, 这里我们只演示一个方向
            
            # 可视化结果
            if save_dir:
                fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
                
                # 单样本情况特殊处理
                if n_samples == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(len(test_a_samples)):
                    # 原始样本A
                    self._visualize_feature(test_a_samples[i], axes[i, 0], "风格A原图")
                    # 重建样本A
                    self._visualize_feature(recon_a[i], axes[i, 1], "A重建")
                    # A转B的结果
                    self._visualize_feature(fake_b[i], axes[i, 2], "A→B转换")
                    
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "style_transfer_evaluation.png"))
                plt.close()
                
                with open(os.path.join(save_dir, "evaluation_metrics.txt"), "w") as f:
                    for metric_name, metric_value in metrics.items():
                        f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        logging.info("评估完成")
        logging.info(f"重建损失A: {metrics['recon_loss_a']:.4f}")
        logging.info(f"转换保真度: {metrics['transfer_fidelity']:.4f}")
        
        return metrics
    
    def _visualize_feature(self, tensor, ax, title):
        """可视化VAE潜在特征"""
        # 对于VAE潜在特征，我们通常只能可视化通道
        if tensor.dim() == 3:  # [C, H, W]
            feature_map = tensor.detach().cpu()
            
            # 归一化特征以便于可视化
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # 显示前几个通道的平均值
            num_channels_to_show = min(16, feature_map.shape[0])
            avg_map = feature_map[:num_channels_to_show].mean(dim=0)
            ax.imshow(avg_map, cmap='viridis')
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, "无法可视化", ha='center', va='center')
    
    def train_with_gradient_penalty(self, dataloader_a, dataloader_b, num_epochs=100, save_dir=None, 
                                   lambda_gp=10.0, n_critic=5, eval_interval=10):
        """
        使用梯度惩罚的改进训练方法
        
        Args:
            dataloader_a: 风格A的数据加载器
            dataloader_b: 风格B的数据加载器
            num_epochs: 训练轮数
            save_dir: 保存模型的目录
            lambda_gp: 梯度惩罚系数
            n_critic: 每训练生成器一次，训练判别器n_critic次
            eval_interval: 每隔多少轮进行一次评估
        """
        if save_dir:
            os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
            
        logging.info(f"开始WGAN-GP训练, 训练轮数: {num_epochs}")
        start_time = time.time()
        
        # 确保模型处于训练模式
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()
        self.style_mapper.train()
        
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
        
        # 创建验证数据加载器
        val_loader_a = None
        val_loader_b = None
        
        if hasattr(dataloader_a.dataset, 'dataset'):  # 检查是否为SubsetDataLoader
            # 创建验证数据加载器
            full_dataset_a = dataloader_a.dataset.dataset
            full_dataset_b = dataloader_b.dataset.dataset
            
            # 取5%的数据作为验证集
            val_size = int(0.05 * len(full_dataset_a))
            val_indices = torch.randperm(len(full_dataset_a))[:val_size]
            
            val_batch_size = min(dataloader_a.batch_size, val_size)
            val_loader_a = DataLoader([full_dataset_a[i] for i in val_indices], batch_size=val_batch_size)
            val_loader_b = DataLoader([full_dataset_b[i] for i in val_indices], batch_size=val_batch_size)
        
        def compute_gradient_penalty(discriminator, real_samples, fake_samples):
            """计算WGAN-GP梯度惩罚"""
            # 在真实样本和生成样本之间随机插值
            alpha = torch.rand(real_samples.size(0), 1, device=self.device)
            interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
            
            # 计算判别器对插值样本的输出
            d_interpolates = discriminator(interpolates)
            
            # 创建填充的梯度输出
            fake = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
            
            # 计算梯度
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # 计算梯度惩罚
            gradients = gradients.view(gradients.size(0), -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            return gradient_penalty
        
        for epoch in range(num_epochs):
            epoch_recon_loss = 0
            epoch_disc_loss = 0
            epoch_gen_loss = 0
            
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch [{epoch+1}/{num_epochs}]")
            for step in pbar:
                # ---------------------
                # 1. 训练判别器
                # ---------------------
                # 每n_critic步训练一次判别器
                train_disc = (step % n_critic == 0)
                
                # 获取下一批数据
                real_a = next(iter_a).to(self.device)
                real_b = next(iter_b).to(self.device)
                
                batch_size = real_a.size(0)
                
                # 冻结生成器，训练判别器
                for param in self.encoder.parameters():
                    param.requires_grad = False
                for param in self.decoder.parameters():
                    param.requires_grad = False
                for param in self.style_mapper.parameters():
                    param.requires_grad = False
                for param in self.discriminator.parameters():
                    param.requires_grad = True
                
                if train_disc:
                    self.optimizer_Dis.zero_grad()
                    
                    # 获取真实样本的潜在表示
                    with torch.no_grad():
                        latent_a = self.encoder(real_a)
                        latent_b = self.encoder(real_b)
                        mapped_latent_a = self.style_mapper(latent_a)
                    
                    # 真实样本B的分数
                    pred_real = self.discriminator(latent_b)
                    # 生成样本的分数
                    pred_fake = self.discriminator(mapped_latent_a.detach())
                    
                    # WGAN损失
                    loss_d_real = -torch.mean(pred_real)
                    loss_d_fake = torch.mean(pred_fake)
                    
                    # 梯度惩罚
                    gradient_penalty = compute_gradient_penalty(
                        self.discriminator, latent_b, mapped_latent_a.detach()
                    )
                    
                    # 判别器总损失
                    loss_discriminator = loss_d_real + loss_d_fake + lambda_gp * gradient_penalty
                    
                    # 反向传播
                    loss_discriminator.backward()
                    self.optimizer_Dis.step()
                    
                    epoch_disc_loss += loss_discriminator.item()
                
                # ---------------------
                # 2. 训练生成器
                # ---------------------
                # 解冻生成器，冻结判别器
                for param in self.encoder.parameters():
                    param.requires_grad = True
                for param in self.decoder.parameters():
                    param.requires_grad = True
                for param in self.style_mapper.parameters():
                    param.requires_grad = True
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                
                self.optimizer_E.zero_grad()
                self.optimizer_D.zero_grad()
                self.optimizer_Map.zero_grad()
                
                # 重建损失和风格转换
                latent_a = self.encoder(real_a)
                latent_b = self.encoder(real_b)
                
                recon_a = self.decoder(latent_a)  # A重建
                recon_b = self.decoder(latent_b)  # B重建
                
                # 重建损失
                loss_recon_a = F.mse_loss(recon_a, real_a)
                loss_recon_b = F.mse_loss(recon_b, real_b)
                loss_recon = loss_recon_a + loss_recon_b
                
                # 风格映射和生成
                mapped_latent_a = self.style_mapper(latent_a)  # A->B映射
                fake_b = self.decoder(mapped_latent_a)  # 生成伪B
                
                # 计算循环一致性
                latent_fake_b = self.encoder(fake_b)
                loss_cycle = F.l1_loss(latent_fake_b, mapped_latent_a)
                
                # 对抗损失（仅在判别器表现足够好时应用）
                pred_fake = self.discriminator(mapped_latent_a)
                loss_gen = -torch.mean(pred_fake)
                
                # 风格一致性损失（确保fake_b与real_b有相似的统计特性）
                fake_b_stats = calc_style_statistics(latent_fake_b)
                real_b_stats = calc_style_statistics(latent_b)
                loss_style = style_distance(fake_b_stats, real_b_stats)
                
                # 总生成器损失
                total_gen_loss = (
                    self.lambda_recon * loss_recon + 
                    self.lambda_content * loss_cycle + 
                    self.lambda_adv * loss_gen + 
                    0.5 * loss_style  # 降低风格损失权重
                )
                
                # 反向传播
                total_gen_loss.backward()
                self.optimizer_E.step()
                self.optimizer_D.step()
                self.optimizer_Map.step()
                
                # 记录损失
                epoch_recon_loss += loss_recon.item()
                epoch_gen_loss += loss_gen.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'recon': loss_recon.item(),
                    'disc': loss_discriminator.item() if train_disc else 0, 
                    'gen': loss_gen.item(),
                    'style': loss_style.item()
                })
            
            # 更新学习率
            self.scheduler_E.step()
            self.scheduler_D.step()
            self.scheduler_Map.step()
            self.scheduler_Dis.step()
            
            # 计算平均损失
            avg_recon_loss = epoch_recon_loss / steps_per_epoch
            avg_disc_loss = epoch_disc_loss / (steps_per_epoch // n_critic + 1)
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
            
            # 定期评估模型
            if val_loader_a and val_loader_b and (epoch + 1) % eval_interval == 0:
                eval_dir = os.path.join(save_dir, f"eval_epoch_{epoch+1}")
                metrics = self.evaluate(val_loader_a, val_loader_b, save_dir=eval_dir)
                logging.info(f"验证指标: {metrics}")
            
            # 保存检查点
            if save_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(save_dir, "checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
                self.save_checkpoint(checkpoint_path)
        
        self.is_trained = True
        elapsed_time = time.time() - start_time
        logging.info(f"训练完成，耗时: {elapsed_time:.2f} 秒")
        
        # 保存最终模型
        if save_dir:
            self.save_model(save_dir)
            self.plot_training_history(save_dir)
            
        return self.training_history

# 新增更强大的编码器
class EnhancedEncoder(nn.Module):
    def __init__(self, in_channels=16, latent_dim=256, use_attention=True, use_flash_attn=True):
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
            self.attention = SelfAttention(512, use_flash_attn=use_flash_attn)
        
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


class EnhancedDecoder(nn.Module):
    """
    解码器：将潜在空间映射回VAE潜在向量
    
    输入形状: [N, latent_dim]
    输出形状: [N, 16, 32, 32]
    """
    def __init__(self, out_channels=16, latent_dim=256, use_residual=True):
        super(EnhancedDecoder, self).__init__()
        
        # 潜在空间 -> 4x4 特征图
        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.norm_fc = nn.BatchNorm2d(512)
        
        # 上采样块
        self.up1 = ResidualUpBlock(512, 256)  # 4x4 -> 8x8
        self.up2 = ResidualUpBlock(256, 128)  # 8x8 -> 16x16
        self.up3 = ResidualUpBlock(128, 64)   # 16x16 -> 32x32
        
        # Residual连接 - 修改为与上采样块的输出通道数匹配
        self.use_residual = use_residual
        if use_residual:
            self.res_512 = ResidualBlock(512)
            self.res_256 = ResidualBlock(256)
            self.res_128 = ResidualBlock(128)
            self.res_64 = ResidualBlock(64)
        
        # 输出层
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 从潜在空间重建特征图
        x = self.fc(x).view(-1, 512, 4, 4)
        x = F.relu(self.norm_fc(x))
        
        # 残差层应用于对应通道数的特征图
        if self.use_residual:
            x = self.res_512(x)
        
        # 上采样512->256
        x = self.up1(x)
        if self.use_residual:
            x = self.res_256(x)
        
        # 上采样256->128
        x = self.up2(x)
        if self.use_residual:
            x = self.res_128(x)
        
        # 上采样128->64
        x = self.up3(x)
        if self.use_residual:
            x = self.res_64(x)
            
        # 输出层 64->out_channels
        x = self.conv_out(x)
        
        return x


class EnhancedDiscriminator(nn.Module):
    """
    判别器：区分潜在向量的来源（风格A或风格B）
    
    输入形状: [N, latent_dim]
    输出形状: [N, 1]
    """
    def __init__(self, latent_dim=256, use_spectral_norm=True):
        super(EnhancedDiscriminator, self).__init__()
        
        def get_norm_layer(dim):
            if use_spectral_norm:
                return nn.utils.spectral_norm(nn.Linear(dim, dim))
            else:
                return nn.Identity()
        
        # 更深的判别器网络
        self.model = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 512),
            get_norm_layer(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            get_norm_layer(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            get_norm_layer(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 输出原始logits，不用sigmoid
        )

        # 添加自注意力机制
        self.self_attn = nn.MultiheadAttention(512, 4, batch_first=True)
        
    def forward(self, x):
        # 初始映射
        x = self.model[0:4](x)  # 到第一个LayerNorm
        
        # 应用自注意力
        x_reshaped = x.unsqueeze(1)  # [B, 1, 512]
        attn_out, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        x = x + attn_out.squeeze(1)  # 残差连接
        
        # 继续处理
        x = self.model[4:](x)
        
        return x


class ResidualDownBlock(nn.Module):
    """
    残差下采样块
    
    输入形状: [N, in_channels, H, W]
    输出形状: [N, out_channels, H/2, W/2]
    """
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


class ResidualBlock(nn.Module):
    """
    残差块
    
    输入形状: [N, channels, H, W]
    输出形状: [N, channels, H, W]
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        out = out + residual
        out = F.relu(out)
        
        return out


class ResidualUpBlock(nn.Module):
    """
    残差上采样块
    
    输入形状: [N, in_channels, H, W]
    输出形状: [N, out_channels, 2H, 2W]
    """
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


def calc_style_statistics(features):
    # 计算风格特征的统计特性(均值和方差)
    mean = features.mean(0, keepdim=True)
    std = features.std(0, keepdim=True)
    return {'mean': mean, 'std': std}


def style_distance(style_a, style_b):
    # 计算两个风格特征的距离
    mean_loss = F.mse_loss(style_a['mean'], style_b['mean'])
    std_loss = F.mse_loss(style_a['std'], style_b['std'])
    return mean_loss + std_loss


def check_flash_attn_available():
    """检查是否可以使用FlashAttention"""
    try:
        from flash_attn import __version__
        return True, __version__
    except ImportError:
        return False, None


class SelfAttention(nn.Module):
    def __init__(self, channels, use_flash_attn=True):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.use_flash_attn = use_flash_attn
        
        # 检测是否可以使用FlashAttention
        self.flash_attn_available = False
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_available = True
        except ImportError:
            if use_flash_attn:
                logging.warning("FlashAttention未安装，使用标准注意力机制。可通过pip install flash-attn安装")
    
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 线性映射
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C/8
        k = self.key(x).view(batch_size, -1, height * width)  # B x C/8 x HW
        v = self.value(x).view(batch_size, -1, height * width)  # B x C x HW
        
        # 使用FlashAttention或标准注意力机制
        if self.use_flash_attn and self.flash_attn_available:
            try:
                from flash_attn import flash_attn_func
                
                # 使用标准注意力，因为Flash Attention有维度问题
                energy = torch.bmm(q, k)  # B x HW x HW
                attention = F.softmax(energy, dim=2)
                attn_output = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
            except Exception as e:
                # 回退到标准注意力
                logging.warning(f"FlashAttention错误，回退到标准注意力: {e}")
                
                # 标准注意力计算
                energy = torch.bmm(q, k)  # B x HW x HW
                attention = F.softmax(energy, dim=2)
                attn_output = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
        else:
            # 标准注意力计算
            energy = torch.bmm(q, k)  # B x HW x HW
            attention = F.softmax(energy, dim=2)
            attn_output = torch.bmm(v, attention.permute(0, 2, 1))  # B x C x HW
        
        # 重塑回原始维度
        out = attn_output.view(batch_size, C, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class StyleMapper(nn.Module):
    """
    风格映射网络 - MLP形式
    
    输入形状: [N, latent_dim]
    输出形状: [N, latent_dim]
    """
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


class NormalizingFlowStyleMapper(nn.Module):
    """
    使用归一化流的风格映射器 - 可逆且强大的表达能力
    """
    def __init__(self, latent_dim=256, flow_steps=4):
        super(NormalizingFlowStyleMapper, self).__init__()
        
        # 基础映射网络
        self.base_mapper = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # 归一化流步骤
        self.flow_steps = nn.ModuleList([
            CouplingLayer(latent_dim) for _ in range(flow_steps)
        ])
    
    def forward(self, x):
        # 基础映射
        h = self.base_mapper(x)
        
        # 应用归一化流
        for flow in self.flow_steps:
            h = flow(h)
            
        return h


class CouplingLayer(nn.Module):
    """实现简单的仿射耦合层"""
    def __init__(self, latent_dim):
        super(CouplingLayer, self).__init__()
        self.split_size = latent_dim // 2
        
        # 变换网络
        self.net = nn.Sequential(
            nn.Linear(self.split_size, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, self.split_size * 2)
        )
    
    def forward(self, x):
        x1, x2 = torch.split(x, [self.split_size, self.split_size], dim=1)
        h = self.net(x1)
        shift, scale = torch.split(h, [self.split_size, self.split_size], dim=1)
        scale = torch.sigmoid(scale + 2) + 1e-5  # 确保缩放因子始终为正
        
        y1 = x1
        y2 = x2 * scale + shift
        
        return torch.cat([y1, y2], dim=1)


class EnergyDiscriminator(nn.Module):
    """
    基于能量的GAN判别器
    """
    def __init__(self, latent_dim=256, use_spectral_norm=True):
        super(EnergyDiscriminator, self).__init__()
        
        def get_norm_layer(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        
        # 更深更稳定的判别器
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            get_norm_layer(nn.Linear(latent_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            
            nn.Dropout(0.2),
            get_norm_layer(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            
            nn.Dropout(0.1),
            get_norm_layer(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            
            get_norm_layer(nn.Linear(256, 1)),
        )
        
        # 添加自注意力机制
        self.self_attn = nn.MultiheadAttention(512, 4, batch_first=True)
        
    def forward(self, x):
        # 初始映射
        x = self.model[0:4](x)  # 到第一个LayerNorm
        
        # 应用自注意力
        x_reshaped = x.unsqueeze(1)  # [B, 1, 512]
        attn_out, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        x = x + attn_out.squeeze(1)  # 残差连接
        
        # 继续处理
        x = self.model[4:](x)
        
        return x


class StyleTransferAAE:
    """
    基于AAE的风格转换模型
    
    使用对抗性自编码器学习两种不同风格之间的映射
    """
    def __init__(self, device='cuda', latent_dim=256, use_flash_attn=True):  # 增大默认潜在空间维度
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.latent_dim = latent_dim
        
        # 检查Flash Attention可用性
        flash_available, flash_version = check_flash_attn_available()
        if use_flash_attn and flash_available:
            self.use_flash_attn = True
            logging.info(f"启用FlashAttention优化，版本:{flash_version}")
        else:
            self.use_flash_attn = False
            if use_flash_attn:
                logging.warning("FlashAttention不可用，回退到标准注意力机制。可通过pip install flash-attn安装")
        
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
            use_attention=True,  # 使用注意力机制
            use_flash_attn=self.use_flash_attn  # 传递FlashAttention参数
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
        self.style_mapper.train()
        
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
                fake_b = self.decoder(mapped_latent_a)   # A->B
                
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
                total_recon_loss.backward(retain_graph=True)  # 添加retain_graph=True
                self.optimizer_E.step()
                self.optimizer_D.step()
                self.optimizer_Map.step()
                
                # ---------------------
                # 2. 训练判别器
                # ---------------------
                self.optimizer_Dis.zero_grad()

                # 获取真假样本 - 使用detach()切断梯度流
                latent_b_real = self.encoder(real_b).detach()  # 确保完全切断梯度
                fake_latent_b = self.style_mapper(latent_a.detach()).detach()  # 双重detach确保安全
                
                # 真样本的判别
                pred_real = self.discriminator(latent_b_real)
                # 假样本的判别
                pred_fake = self.discriminator(fake_latent_b)
                
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

                # 重新计算映射的潜在向量 - 使用新的forward pass避免梯度问题
                latent_a_gen = self.encoder(real_a)
                mapped_latent_a_gen = self.style_mapper(latent_a_gen)
                
                # 判别器对生成的风格B的预测
                pred_gen = self.discriminator(mapped_latent_a_gen)
                
                # 生成器对抗损失 - 欺骗判别器
                loss_generator = -torch.mean(pred_gen)  # WGAN风格的损失
                
                # 风格损失 - 让生成的特征与目标风格匹配
                target_style_stats = calc_style_statistics(latent_b.detach())  # 使用detach避免重复反向传播
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
    
    def save_checkpoint(self, checkpoint_path):
        """保存训练检查点"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'style_mapper_state_dict': self.style_mapper.state_dict(),
            'optimizer_E_state_dict': self.optimizer_E.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_Dis_state_dict': self.optimizer_Dis.state_dict(),
            'optimizer_Map_state_dict': self.optimizer_Map.state_dict(),
            'scheduler_E_state_dict': self.scheduler_E.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'scheduler_Dis_state_dict': self.scheduler_Dis.state_dict(),
            'scheduler_Map_state_dict': self.scheduler_Map.state_dict(),
            'training_history': self.training_history,
            'latent_dim': self.latent_dim
        }, checkpoint_path)
        
        logging.info(f"检查点已保存到: {checkpoint_path}")
    
    def save_model(self, save_dir):
        """保存完整模型"""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "style_transfer_model.pth")
        
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'style_mapper_state_dict': self.style_mapper.state_dict(),
            'latent_dim': self.latent_dim,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }, model_path)
        
        # 保存配置信息
        config_path = os.path.join(save_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(f"latent_dim: {self.latent_dim}\n")
            f.write(f"lambda_recon: {self.lambda_recon}\n")
            f.write(f"lambda_content: {self.lambda_content}\n")
            f.write(f"lambda_adv: {self.lambda_adv}\n")
            
        logging.info(f"模型已保存到: {model_path}")
        logging.info(f"配置信息已保存到: {config_path}")
    
    def load_model(self, model_path, device=None):
        """加载已保存的模型"""
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取潜在维度
        self.latent_dim = checkpoint.get('latent_dim', self.latent_dim)
        
        # 检查是否需要构建模型
        if not all([self.encoder, self.decoder, self.discriminator, self.style_mapper]):
            logging.info("模型组件未初始化，正在从头构建...")
            # 从保存的状态字典推断输入通道数
            encoder_state_dict = checkpoint['encoder_state_dict']
            first_layer_weight = encoder_state_dict.get('conv_in.weight')
            if first_layer_weight is not None:
                in_channels = first_layer_weight.size(1)
            else:
                in_channels = 16  # 默认通道数
            
            # 构建模型组件
            self.build_models(in_channels=in_channels)
        
        # 加载状态字典
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'style_mapper_state_dict' in checkpoint:
            self.style_mapper.load_state_dict(checkpoint['style_mapper_state_dict'])
        
        # 加载训练历史
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
            
        self.is_trained = checkpoint.get('is_trained', True)
        
        # 将模型设置为评估模式
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.style_mapper.eval()
        
        logging.info(f"模型已从 {model_path} 加载完成")
        return self
    
    def load_checkpoint(self, checkpoint_path, device=None, resume_training=True):
        """加载检查点继续训练"""
        if device:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取潜在维度
        self.latent_dim = checkpoint.get('latent_dim', self.latent_dim)
        
        # 检查是否需要构建模型
        if not all([self.encoder, self.decoder, self.discriminator, self.style_mapper]):
            logging.info("模型组件未初始化，正在从头构建...")
            # 从保存的状态字典推断输入通道数
            encoder_state_dict = checkpoint['encoder_state_dict']
            first_layer_weight = encoder_state_dict.get('conv_in.weight')
            if first_layer_weight is not None:
                in_channels = first_layer_weight.size(1)
            else:
                in_channels = 16  # 默认通道数
            
            # 构建模型组件
            self.build_models(in_channels=in_channels)
        
        # 加载模型状态
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.style_mapper.load_state_dict(checkpoint['style_mapper_state_dict'])
        
        if resume_training:
            # 加载优化器状态
            self.optimizer_E.load_state_dict(checkpoint['optimizer_E_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.optimizer_Dis.load_state_dict(checkpoint['optimizer_Dis.state_dict'])
            self.optimizer_Map.load_state_dict(checkpoint['optimizer_Map.state_dict'])
            
            # 加载学习率调度器状态
            if 'scheduler_E_state_dict' in checkpoint:
                self.scheduler_E.load_state_dict(checkpoint['scheduler_E_state_dict'])
                self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
                self.scheduler_Dis.load_state_dict(checkpoint['scheduler_Dis.state_dict'])
                self.scheduler_Map.load_state_dict(checkpoint['scheduler_Map.state_dict'])
        
        # 加载训练历史
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logging.info(f"检查点已从 {checkpoint_path} 加载完成")
        return self
    
    def plot_training_history(self, save_dir=None):
        """绘制训练历史"""
        if not self.training_history or all(len(v) == 0 for v in self.training_history.values()):
            logging.warning("没有训练历史数据可供绘制")
            return
            
        plt.figure(figsize=(15, 5))
        
        # 重建损失
        plt.subplot(1, 3, 1)
        plt.plot(self.training_history['recon_loss'])
        plt.title('重建损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 判别器损失
        plt.subplot(1, 3, 2)
        plt.plot(self.training_history['disc_loss'])
        plt.title('判别器损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        # 生成器损失
        plt.subplot(1, 3, 3)
        plt.plot(self.training_history['gen_loss'])
        plt.title('生成器损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "training_history.png"))
            plt.close()
            logging.info(f"训练历史图已保存到: {os.path.join(save_dir, 'training_history.png')}")
        else:
            plt.show()

    def transfer_style(self, input_tensor, add_time_dim=False):
        """
        将输入张量从风格A转换为风格B
        
        Args:
            input_tensor: 输入张量，形状为 [N, C, H, W] 或 [C, H, W]
            add_time_dim: 是否在输出中添加时间维度
            
        Returns:
            转换后的张量
        """
        if not self.is_trained:
            logging.warning("模型尚未训练，转换效果可能不理想")
        
        self.encoder.eval()
        self.decoder.eval()
        self.style_mapper.eval()
        
        # 处理输入张量的维度
        if input_tensor.dim() == 3:  # [C, H, W]
            input_tensor = input_tensor.unsqueeze(0)  # [1, C, H, W]
            
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            
            # 编码
            latent = self.encoder(input_tensor)
            
            # 映射风格
            mapped_latent = self.style_mapper(latent)
            
            # 解码
            output = self.decoder(mapped_latent)
            
            if add_time_dim:
                output = output.unsqueeze(2)  # [N, C, 1, H, W]
        
        return output.cpu()
    
    def evaluate(self, test_dataloader_a, test_dataloader_b=None, save_dir=None, n_samples=5):
        """
        评估模型的风格转换效果
        
        Args:
            test_dataloader_a: 风格A的测试数据加载器
            test_dataloader_b: 风格B的测试数据加载器 (可选)
            save_dir: 保存评估结果的目录
            n_samples: 用于可视化的样本数量
            
        Returns:
            dict: 包含评估指标的字典
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        self.encoder.eval()
        self.decoder.eval()
        self.style_mapper.eval()
        
        metrics = {
            'recon_loss_a': 0.0,  # 风格A重建损失
            'transfer_fidelity': 0.0,  # 转换保真度
            'cycle_consistency': 0.0,  # 循环一致性
        }
        
        # 获取测试样本
        test_a_samples = []
        for i, batch in enumerate(test_dataloader_a):
            if i >= n_samples:
                break
            test_a_samples.append(batch)
        
        if len(test_a_samples) > 0:
            test_a_samples = torch.cat(test_a_samples, dim=0)[:n_samples].to(self.device)
        else:
            logging.warning("没有测试样本可用于评估")
            return metrics
        
        test_b_samples = None
        if test_dataloader_b:
            test_b_samples = []
            for i, batch in enumerate(test_dataloader_b):
                if i >= n_samples:
                    break
                test_b_samples.append(batch)
            
            if len(test_b_samples) > 0:
                test_b_samples = torch.cat(test_b_samples, dim=0)[:n_samples].to(self.device)
        
        with torch.no_grad():
            # 1. 评估重建质量
            latent_a = self.encoder(test_a_samples)
            recon_a = self.decoder(latent_a)
            metrics['recon_loss_a'] = F.mse_loss(recon_a, test_a_samples).item()
            
            # 2. 风格转换
            mapped_latent_a = self.style_mapper(latent_a)
            fake_b = self.decoder(mapped_latent_a)
            
            # 3. 计算循环一致性
            latent_fake_b = self.encoder(fake_b)
            metrics['transfer_fidelity'] = F.l1_loss(latent_fake_b, mapped_latent_a).item()
            
            # 4. 循环转换回风格A (如果有风格B样本)
            if test_b_samples is not None:
                latent_b = self.encoder(test_b_samples)
                recon_b = self.decoder(latent_b)
                
                # 评估风格B重建质量
                metrics['recon_loss_b'] = F.mse_loss(recon_b, test_b_samples).item()
                
                # B->A 转换
                # 这里假设有一个逆映射, 或者可以训练第二个映射器
                # 但由于当前模型只关注 A->B, 这里我们只演示一个方向
            
            # 可视化结果
            if save_dir:
                fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
                
                # 单样本情况特殊处理
                if n_samples == 1:
                    axes = axes.reshape(1, -1)
                
                for i in range(len(test_a_samples)):
                    # 原始样本A
                    self._visualize_feature(test_a_samples[i], axes[i, 0], "风格A原图")
                    # 重建样本A
                    self._visualize_feature(recon_a[i], axes[i, 1], "A重建")
                    # A转B的结果
                    self._visualize_feature(fake_b[i], axes[i, 2], "A→B转换")
                    
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "style_transfer_evaluation.png"))
                plt.close()
                
                with open(os.path.join(save_dir, "evaluation_metrics.txt"), "w") as f:
                    for metric_name, metric_value in metrics.items():
                        f.write(f"{metric_name}: {metric_value:.4f}\n")
        
        logging.info("评估完成")
        logging.info(f"重建损失A: {metrics['recon_loss_a']:.4f}")
        logging.info(f"转换保真度: {metrics['transfer_fidelity']:.4f}")
        
        return metrics
    
    def _visualize_feature(self, tensor, ax, title):
        """可视化VAE潜在特征"""
        # 对于VAE潜在特征，我们通常只能可视化通道
        if tensor.dim() == 3:  # [C, H, W]
            feature_map = tensor.detach().cpu()
            
            # 归一化特征以便于可视化
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # 显示前几个通道的平均值
            num_channels_to_show = min(16, feature_map.shape[0])
            avg_map = feature_map[:num_channels_to_show].mean(dim=0)
            ax.imshow(avg_map, cmap='viridis')
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, "无法可视化", ha='center', va='center')
    
    def improve_model_structure(self):
        """
        改进模型结构以提高性能
        
        针对VAE潜在空间的特性进行优化调整
        """
        # 增加内容保留模块 - 确保风格转换时内容不变
        if not hasattr(self, 'content_extractor'):
            in_channels = self.encoder.conv_in.in_channels
            self.content_extractor = ContentExtractionModule(
                in_channels=in_channels, 
                latent_dim=self.latent_dim // 2
            ).to(self.device)
            
            # 更新优化器
            lr = 3e-4
            beta1, beta2 = 0.5, 0.999
            self.optimizer_CE = optim.AdamW(
                self.content_extractor.parameters(),
                lr=lr, 
                betas=(beta1, beta2), 
                weight_decay=1e-4
            )
            
            self.scheduler_CE = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_CE, 
                T_max=50
            )
            
            logging.info(f"增加了内容提取模块, 大小: {count_parameters(self.content_extractor) / 1e6:.2f}M 参数")
        
        # 使用归一化流增强风格映射网络
        if not isinstance(self.style_mapper, NormalizingFlowStyleMapper):
            old_mapper = self.style_mapper
            self.style_mapper = NormalizingFlowStyleMapper(
                latent_dim=self.latent_dim,
                flow_steps=4
            ).to(self.device)
            
            # 更新优化器
            self.optimizer_Map = optim.AdamW(
                self.style_mapper.parameters(),
                lr=3e-4, 
                betas=(0.5, 0.999), 
                weight_decay=1e-4
            )
            
            self.scheduler_Map = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_Map, 
                T_max=50
            )
            
            logging.info("升级风格映射为归一化流模型")
        
        # 替换为更强的判别器 - 基于能量的GAN判别器
        if not isinstance(self.discriminator, EnergyDiscriminator):
            self.discriminator = EnergyDiscriminator(
                latent_dim=self.latent_dim,
                use_spectral_norm=True
            ).to(self.device)
            
            # 更新优化器
            self.optimizer_Dis = optim.AdamW(
                self.discriminator.parameters(),
                lr=1e-5,  # 使用较小的学习率
                betas=(0.5, 0.999),
                weight_decay=1e-4
            )
            
            self.scheduler_Dis = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_Dis, 
                T_max=50
            )
            
            logging.info("升级判别器为基于能量的GAN判别器")
        
        # 增加自恢复约束
        self.lambda_cycle = 10.0  # 循环一致性损失权重
        
        # 更新训练参数 - 平衡各种损失的权重
        self.lambda_recon = 5.0    # 降低重建损失权重
        self.lambda_content = 10.0  # 增加内容保持损失权重
        self.lambda_adv = 2.0      # 适度增加对抗损失权重
        
        logging.info("模型结构改进完成，调整了损失权重")
        return True
    
    def get_normal_training_patterns(self):
        """
        返回什么样的训练模式是正常的参考指南
        """
        normal_patterns = {
            "重建损失": {
                "正常初始值": "1.5-2.5",
                "正常下降趋势": "在前10-20轮迅速下降，然后逐渐平稳到0.3-0.7",
                "警示信号": "长期停滞在高值、剧烈波动或突然增加"
            },
            "判别器损失": {
                "健康范围": "0.6-1.2之间波动",
                "收敛信号": "不会持续固定在0.5左右或在2.0附近",
                "警示信号": "长期保持在2.0附近（判别器过强）或0（判别器崩溃）"
            },
            "生成器损失": {
                "正常趋势": "呈现波动下降，最终在-0.5到0.5之间稳定",
                "收敛信号": "波动幅度减小，整体下降至稳定区间",
                "警示信号": "持续处于高正值或持续负值不变"
            },
            "视觉效果指南": {
                "内容保留": "应能看出原图结构，但风格变化明显",
                "风格转换": "应表现出目标风格的关键特征",
                "伪影情况": "无明显像素块、杂色或变形"
            },
            "训练时长参考": {
                "足够训练轮数": "通常需要100-200轮",
                "收敛指标": "验证集损失稳定10-20轮无明显变化",
                "训练提前停止": "如果重建良好且风格转换有效，可提前停止"
            }
        }
        return normal_patterns


# 新增内容提取模块，专注于保留内容特征
class ContentExtractionModule(nn.Module):
    def __init__(self, in_channels=16, latent_dim=128):
        super(ContentExtractionModule, self).__init__()
        
        # 收缩通道，捕获结构特征
        self.content_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim)
        )
    
    def forward(self, x):
        return self.content_encoder(x)


# 使用归一化流的风格映射器 - 可逆且强大的表达能力
class NormalizingFlowStyleMapper(nn.Module):
    def __init__(self, latent_dim=256, flow_steps=4):
        super(NormalizingFlowStyleMapper, self).__init__()
        
        # 基础映射网络
        self.base_mapper = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        # 归一化流步骤
        self.flow_steps = nn.ModuleList([
            CouplingLayer(latent_dim) for _ in range(flow_steps)
        ])
    
    def forward(self, x):
        # 基础映射
        h = self.base_mapper(x)
        
        # 应用归一化流
        for flow in self.flow_steps:
            h = flow(h)
            
        return h


class CouplingLayer(nn.Module):
    """实现简单的仿射耦合层"""
    def __init__(self, latent_dim):
        super(CouplingLayer, self).__init__()
        self.split_size = latent_dim // 2
        
        # 变换网络
        self.net = nn.Sequential(
            nn.Linear(self.split_size, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, self.split_size * 2)
        )
    
    def forward(self, x):
        x1, x2 = torch.split(x, [self.split_size, self.split_size], dim=1)
        h = self.net(x1)
        shift, scale = torch.split(h, [self.split_size, self.split_size], dim=1)
        scale = torch.sigmoid(scale + 2) + 1e-5  # 确保缩放因子始终为正
        
        y1 = x1
        y2 = x2 * scale + shift
        
        return torch.cat([y1, y2], dim=1)


class EnergyDiscriminator(nn.Module):
    """
    基于能量的GAN判别器
    """
    def __init__(self, latent_dim=256, use_spectral_norm=True):
        super(EnergyDiscriminator, self).__init__()
        
        def get_norm_layer(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer
        
        # 更深更稳定的判别器
        self.model = nn.Sequential(
            nn.Dropout(0.2),
            get_norm_layer(nn.Linear(latent_dim, 512)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            
            nn.Dropout(0.2),
            get_norm_layer(nn.Linear(512, 512)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(512),
            
            nn.Dropout(0.1),
            get_norm_layer(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(256),
            
            get_norm_layer(nn.Linear(256, 1)),
        )
        
        # 添加自注意力机制
        self.self_attn = nn.MultiheadAttention(512, 4, batch_first=True)
        
    def forward(self, x):
        # 初始映射
        x = self.model[0:4](x)  # 到第一个LayerNorm
        
        # 应用自注意力
        x_reshaped = x.unsqueeze(1)  # [B, 1, 512]
        attn_out, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        x = x + attn_out.squeeze(1)  # 残差连接
        
        # 继续处理
        x = self.model[4:](x)
        
        return x


def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_style_statistics(features):
    # 计算风格特征的统计特性(均值和方差)
    mean = features.mean(0, keepdim=True)
    std = features.std(0, keepdim=True)
    return {'mean': mean, 'std': std}


def style_distance(style_a, style_b):
    # 计算两个风格特征的距离
    mean_loss = F.mse_loss(style_a['mean'], style_b['mean'])
    std_loss = F.mse_loss(style_a['std'], style_b['std'])
    return mean_loss + std_loss


def check_flash_attn_available():
    """检查是否可以使用FlashAttention"""
    try:
        from flash_attn import __version__
        return True, __version__
    except ImportError:
        return False, None