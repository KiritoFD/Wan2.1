#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""VAE CycleGAN训练器实现"""

import os
import time
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from tqdm import tqdm

from .models import AdaptiveGenerator, Discriminator, weights_init_normal
from .dataset import ReplayBuffer


class VAECycleGANTrainer:
    """VAE特征向量的CycleGAN训练器，封装训练相关逻辑"""
    
    def __init__(self, config):
        """
        初始化VAE CycleGAN训练器
        
        参数:
            config: 配置对象，包含模型参数
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 初始化网络
        self.netG_A2B = AdaptiveGenerator(
            input_channels=config.input_channels, 
            output_channels=config.output_channels,
            n_residual_blocks=config.n_residual_blocks,
            base_filters=config.base_filters
        )
        
        self.netG_B2A = AdaptiveGenerator(
            input_channels=config.output_channels, 
            output_channels=config.input_channels,
            n_residual_blocks=config.n_residual_blocks,
            base_filters=config.base_filters
        )
        
        self.netD_A = Discriminator(
            input_channels=config.input_channels,
            base_filters=config.base_filters,
            n_layers=config.discriminator_layers
        )
        
        self.netD_B = Discriminator(
            input_channels=config.output_channels,
            base_filters=config.base_filters,
            n_layers=config.discriminator_layers
        )
        
        # 移动模型到指定设备
        self.netG_A2B.to(self.device)
        self.netG_B2A.to(self.device)
        self.netD_A.to(self.device)
        self.netD_B.to(self.device)
        
        # 应用权重初始化
        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)
        
        # 定义损失函数
        self.criterion_GAN = torch.nn.MSELoss()         # 对抗损失
        self.criterion_cycle = torch.nn.L1Loss()        # 循环一致性损失
        self.criterion_identity = torch.nn.L1Loss()     # 身份映射损失
        
        # 定义优化器
        self.optimizer_G = optim.Adam(
            list(self.netG_A2B.parameters()) + list(self.netG_B2A.parameters()),
            lr=config.lr, betas=(config.b1, config.b2)
        )
        self.optimizer_D_A = optim.Adam(self.netD_A.parameters(), lr=config.lr, betas=(config.b1, config.b2))
        self.optimizer_D_B = optim.Adam(self.netD_B.parameters(), lr=config.lr, betas=(config.b1, config.b2))
        
        # 定义学习率调度器
        self.lr_scheduler_G = lr_scheduler.LambdaLR(
            self.optimizer_G, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / float(config.n_epochs - config.decay_epoch)
        )
        self.lr_scheduler_D_A = lr_scheduler.LambdaLR(
            self.optimizer_D_A,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / float(config.n_epochs - config.decay_epoch)
        )
        self.lr_scheduler_D_B = lr_scheduler.LambdaLR(
            self.optimizer_D_B,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.decay_epoch) / float(config.n_epochs - config.decay_epoch)
        )
        
        # 创建经验回放缓冲区
        self.fake_A_buffer = ReplayBuffer(config.buffer_size)
        self.fake_B_buffer = ReplayBuffer(config.buffer_size)
        
        # 创建日志记录器
        self.history = {
            'train_losses': {'G': [], 'D_A': [], 'D_B': [], 'cycle': [], 'identity': []},
            'lr': {'G': [], 'D_A': [], 'D_B': []}
        }
        
        logging.info(f"模型已初始化，设备: {self.device}")
    
    def _resize_tensor(self, tensor, size):
        """调整张量的空间维度尺寸到指定大小，增强处理能力
        
        参数:
            tensor: 输入张量，形状为[B,C,H,W]
            size: 目标尺寸 (H, W)或(H_target, W_target)
            
        返回:
            调整后的张量
        """
        # 确认大小参数
        if isinstance(size, tuple) and len(size) == 2:
            target_h, target_w = size
        else:
            target_h, target_w = size, size
            
        current_h, current_w = tensor.shape[-2], tensor.shape[-1]
        
        # 如果已经匹配，直接返回
        if current_h == target_h and current_w == target_w:
            return tensor
        
        # 使用双线性插值调整尺寸
        try:
            # 保存输入形状
            original_shape = tensor.shape
            b = original_shape[0] if len(original_shape) > 3 else 1
            c = original_shape[1] if len(original_shape) > 3 else original_shape[0]
            
            # 调整尺寸
            if len(original_shape) > 3:  # 批量张量 [B,C,H,W]
                resized = F.interpolate(
                    tensor, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:  # 单个张量 [C,H,W]
                resized = F.interpolate(
                    tensor.unsqueeze(0), 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
                
            return resized
        except Exception as e:
            logging.error(f"调整张量尺寸时出错: {e}, 输入形状: {tensor.shape}, 目标尺寸: {(target_h, target_w)}")
            # 返回原始张量
            return tensor
    
    def _update_epoch_history(self, epoch_losses, batch_count):
        """计算平均损失并更新训练历史"""
        # 计算平均损失
        for k in epoch_losses.keys():
            epoch_losses[k] /= max(1, batch_count)
            self.history['train_losses'][k].append(epoch_losses[k])
        return epoch_losses
    
    def _update_learning_rates(self):
        """更新所有优化器的学习率并记录当前值"""
        # 更新学习率
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()
        
        # 记录当前学习率
        lr_g = self.optimizer_G.param_groups[0]['lr']
        lr_d_a = self.optimizer_D_A.param_groups[0]['lr']
        lr_d_b = self.optimizer_D_B.param_groups[0]['lr']
        self.history['lr']['G'].append(lr_g)
        self.history['lr']['D_A'].append(lr_d_a)
        self.history['lr']['D_B'].append(lr_d_b)
        
        return {"G": lr_g, "D_A": lr_d_a, "D_B": lr_d_b}
        
    def train(self, dataloader_A, dataloader_B):
        """训练CycleGAN模型"""
        # 确保输出目录存在
        os.makedirs(os.path.join(self.config.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "logs"), exist_ok=True)
        
        # 保存配置（如果需要）
        try:
            self.config.save(os.path.join(self.config.output_dir, "config.json"))
        except:
            pass
        
        logging.info(f"开始训练，共{self.config.n_epochs}个epoch")
        
        # 主训练循环
        for epoch in range(1, self.config.n_epochs + 1):
            start_time = time.time()
            epoch_losses = {'G': 0, 'D_A': 0, 'D_B': 0, 'cycle': 0, 'identity': 0}
            batch_count = 0
            
            # 创建进度条
            with tqdm(total=min(len(dataloader_A), len(dataloader_B)), 
                     desc=f"Epoch {epoch}/{self.config.n_epochs}") as pbar:
                
                # 训练一个epoch
                batch_count = self._train_epoch(dataloader_A, dataloader_B, epoch_losses, pbar)
            
            # 计算平均损失并更新历史
            epoch_losses = self._update_epoch_history(epoch_losses, batch_count)
            
            # 更新学习率并获取当前值
            learning_rates = self._update_learning_rates()
            
            # 计算训练时间
            epoch_time = time.time() - start_time
            
            # 打印统计信息
            logging.info(f"[Epoch {epoch}/{self.config.n_epochs}] "
                         f"Loss_G: {epoch_losses['G']:.4f}, Loss_D_A: {epoch_losses['D_A']:.4f}, "
                         f"Loss_D_B: {epoch_losses['D_B']:.4f}, Loss_cycle: {epoch_losses['cycle']:.4f}, "
                         f"Loss_identity: {epoch_losses['identity']:.4f}, "
                         f"LR: G={learning_rates['G']:.6f}, D_A={learning_rates['D_A']:.6f}, D_B={learning_rates['D_B']:.6f}, "
                         f"Time: {epoch_time:.2f}s")
            
            # 保存模型
            if epoch % self.config.save_freq == 0 or epoch == self.config.n_epochs:
                self.save_models(epoch)
                self.plot_losses(epoch)
                logging.info(f"模型已保存 (Epoch {epoch})")
            
            # 保存一些示例结果
            if epoch % self.config.sample_freq == 0:
                self.save_samples(epoch, dataloader_A, dataloader_B)
        
        # 保存最终模型和历史记录
        self.save_models("final")
        self.save_history()
        self.plot_losses("final")
        logging.info("训练完成")
        
        return self.history
    
    def _train_epoch(self, dataloader_A, dataloader_B, epoch_losses, pbar):
        """训练单个epoch"""
        batch_count = 0
        
        # 同时遍历两个数据加载器
        for batch_idx, (batch_A, batch_B) in enumerate(zip(dataloader_A, dataloader_B)):
            # 获取真实样本
            real_A = batch_A["feature"].to(self.device)
            real_B = batch_B["feature"].to(self.device)
            
            # 处理VAE特征的特殊维度：保持时间维度为1
            # VAE特征形状: [B, C, T, H, W] 或 [C, T, H, W]
            if real_A.dim() == 5:  # 批处理多帧情况
                real_A = real_A[:, :, 0]  # 只使用第一帧
            elif real_A.dim() == 4 and real_A.size(1) > 1:  # 单样本多帧情况
                real_A = real_A[:, 0].unsqueeze(1)  # 只使用第一帧，保持时间维度
            
            if real_B.dim() == 5:
                real_B = real_B[:, :, 0]  # 只使用第一帧
            elif real_B.dim() == 4 and real_B.size(1) > 1:
                real_B = real_B[:, 0].unsqueeze(1)  # 只使用第一帧，保持时间维度
            
            # 确保形状是[B,C,H,W]用于2D卷积
            if real_A.dim() == 4 and real_A.size(1) == 1:  # 如果是[B,1,H,W]
                real_A = real_A.squeeze(1)  # 变成[B,H,W]
                if real_A.dim() == 3:  # 如果只有3维，添加通道维度
                    real_A = real_A.unsqueeze(1)  # 变成[B,1,H,W]
            
            if real_B.dim() == 4 and real_B.size(1) == 1:
                real_B = real_B.squeeze(1)
                if real_B.dim() == 3:
                    real_B = real_B.unsqueeze(1)
            
            batch_size = real_A.size(0)
            
            # 检查形状，记录原始形状以便后续恢复
            A_shape = real_A.shape
            B_shape = real_B.shape
            
            # 如果高度或宽度不匹配，记录并跳过该批次
            if A_shape[-2:] != B_shape[-2:]:
                logging.warning(f"跳过形状不匹配的批次: A={A_shape}, B={B_shape}")
                continue
            
            batch_losses = self._train_batch(real_A, real_B)
            
            # 记录损失
            for k, v in batch_losses.items():
                epoch_losses[k] += v
            
            batch_count += 1
            
            # 更新进度条
            description = f"Epoch [G: {batch_losses['G']:.3f}, D_A: {batch_losses['D_A']:.3f}, D_B: {batch_losses['D_B']:.3f}]"
            pbar.set_description(description)            
            pbar.update(1)
        
        return batch_count
    
    def _train_batch(self, real_A, real_B):
        """训练单个批次"""
        batch_losses = {}
        
        # 计算判别器输出大小 - 自适应判别器输出大小
        try:
            valid_shape_A = self.netD_A(real_A).shape
            valid_shape_B = self.netD_B(real_B).shape
        except Exception as e:
            logging.error(f"判别器前向传播错误: {e}, A形状: {real_A.shape}, B形状: {real_B.shape}")
            return {'G': 0, 'D_A': 0, 'D_B': 0, 'cycle': 0, 'identity': 0}
        
        # 创建真假标签
        valid_A = torch.ones(valid_shape_A, device=self.device)
        fake_A = torch.zeros(valid_shape_A, device=self.device)
        valid_B = torch.ones(valid_shape_B, device=self.device)
        fake_B = torch.zeros(valid_shape_B, device=self.device)
        
        #----------------------
        # 训练生成器
        #----------------------
        self.optimizer_G.zero_grad()
        
        # 身份损失 (如果lambda_identity > 0)
        if self.config.lambda_identity > 0:
            try:
                # G_A2B(B) 应接近 B
                identity_B = self.netG_A2B(real_B)
                # 检查输出形状是否匹配
                if identity_B.shape[-2:] != real_B.shape[-2:]:
                    identity_B = self._resize_tensor(identity_B, real_B.shape[-2:])
                loss_identity_B = self.criterion_identity(identity_B, real_B) * self.config.lambda_identity
                
                # G_B2A(A) 应接近 A
                identity_A = self.netG_B2A(real_A)
                # 检查输出形状是否匹配
                if identity_A.shape[-2:] != real_A.shape[-2:]:
                    identity_A = self._resize_tensor(identity_A, real_A.shape[-2:])
                loss_identity_A = self.criterion_identity(identity_A, real_A) * self.config.lambda_identity
                
                loss_identity = loss_identity_A + loss_identity_B
            except RuntimeError as e:
                logging.warning(f"计算身份损失时出错: {e}, 跳过身份损失计算")
                loss_identity = torch.tensor(0.0, device=self.device)
        else:
            loss_identity = torch.tensor(0.0, device=self.device)
        
        # GAN 损失
        try:
            # 生成假样本
            fake_B = self.netG_A2B(real_A)
            # 确保形状匹配
            if fake_B.shape[-2:] != real_B.shape[-2:]:
                fake_B = self._resize_tensor(fake_B, real_B.shape[-2:])
            loss_GAN_A2B = self.criterion_GAN(self.netD_B(fake_B), valid_B)
            
            fake_A = self.netG_B2A(real_B)
            # 确保形状匹配
            if fake_A.shape[-2:] != real_A.shape[-2:]:
                fake_A = self._resize_tensor(fake_A, real_A.shape[-2:])
            loss_GAN_B2A = self.criterion_GAN(self.netD_A(fake_A), valid_A)
            
            loss_GAN = loss_GAN_A2B + loss_GAN_B2A
        except Exception as e:
            logging.error(f"计算GAN损失时出错: {e}")
            loss_GAN = torch.tensor(0.0, device=self.device)
        
        # 循环一致性损失
        try:
            # A -> B -> A 应接近原始 A
            recovered_A = self.netG_B2A(fake_B)
            if recovered_A.shape[-2:] != real_A.shape[-2:]:
                recovered_A = self._resize_tensor(recovered_A, real_A.shape[-2:])
            loss_cycle_A = self.criterion_cycle(recovered_A, real_A) * self.config.lambda_cycle
            
            # B -> A -> B 应接近原始 B
            recovered_B = self.netG_A2B(fake_A)
            if recovered_B.shape[-2:] != real_B.shape[-2:]:
                recovered_B = self._resize_tensor(recovered_B, real_B.shape[-2:])
            loss_cycle_B = self.criterion_cycle(recovered_B, real_B) * self.config.lambda_cycle
            
            loss_cycle = loss_cycle_A + loss_cycle_B
        except RuntimeError as e:
            logging.warning(f"计算循环一致性损失时出错: {e}, 跳过循环一致性损失计算")
            loss_cycle = torch.tensor(0.0, device=self.device)
        
        # 总生成器损失
        loss_G = loss_GAN + loss_cycle + loss_identity
        
        # 反向传播和优化
        loss_G.backward()
        self.optimizer_G.step()
        
        #----------------------
        # 训练判别器 A
        #----------------------
        self.optimizer_D_A.zero_grad()
        
        # 真实样本
        loss_real = self.criterion_GAN(self.netD_A(real_A), valid_A)
        
        # 假样本 (从缓冲区获取)
        fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A.detach())
        loss_fake = self.criterion_GAN(self.netD_A(fake_A_buffer), fake_A)
        
        # 总判别器损失
        loss_D_A = (loss_real + loss_fake) * 0.5
        
        # 反向传播和优化
        loss_D_A.backward()
        self.optimizer_D_A.step()
        
        #----------------------
        # 训练判别器 B
        #----------------------
        self.optimizer_D_B.zero_grad()
        
        # 真实样本
        loss_real = self.criterion_GAN(self.netD_B(real_B), valid_B)
        
        # 假样本 (从缓冲区获取)
        fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B.detach())
        loss_fake = self.criterion_GAN(self.netD_B(fake_B_buffer), fake_B)
        
        # 总判别器损失
        loss_D_B = (loss_real + loss_fake) * 0.5
        
        # 反向传播和优化
        loss_D_B.backward()
        self.optimizer_D_B.step()
        
        # 返回损失值
        return {
            'G': loss_G.item(),
            'D_A': loss_D_A.item(),
            'D_B': loss_D_B.item(),
            'cycle': loss_cycle.item(),
            'identity': loss_identity.item()
        }
    
    def save_models(self, epoch):
        """保存模型权重"""
        save_dir = os.path.join(self.config.output_dir, "models")
        os.makedirs(save_dir, exist_ok=True)            
        
        # 确定文件名前缀
        prefix = f"epoch_{epoch}" if isinstance(epoch, int) else epoch
        
        # 保存生成器
        torch.save(self.netG_A2B.state_dict(), os.path.join(save_dir, f"G_A2B_{prefix}.pth"))
        torch.save(self.netG_B2A.state_dict(), os.path.join(save_dir, f"G_B2A_{prefix}.pth"))
        
        # 保存判别器
        torch.save(self.netD_A.state_dict(), os.path.join(save_dir, f"D_A_{prefix}.pth"))
        torch.save(self.netD_B.state_dict(), os.path.join(save_dir, f"D_B_{prefix}.pth"))
    
    def load_models(self, epoch=None, path=None):
        """加载模型权重"""
        if path is not None:
            self.netG_A2B.load_state_dict(torch.load(os.path.join(path, f"G_A2B.pth"), map_location=self.device))
            self.netG_B2A.load_state_dict(torch.load(os.path.join(path, f"G_B2A.pth"), map_location=self.device))
            self.netD_A.load_state_dict(torch.load(os.path.join(path, f"D_A.pth"), map_location=self.device))
            self.netD_B.load_state_dict(torch.load(os.path.join(path, f"D_B.pth"), map_location=self.device))
            logging.info(f"从 {path} 加载模型")
        elif epoch is not None:
            save_dir = os.path.join(self.config.output_dir, "models")
            prefix = f"epoch_{epoch}" if isinstance(epoch, int) else epoch
            self.netG_A2B.load_state_dict(torch.load(os.path.join(save_dir, f"G_A2B_{prefix}.pth"), map_location=self.device))
            self.netG_B2A.load_state_dict(torch.load(os.path.join(save_dir, f"G_B2A_{prefix}.pth"), map_location=self.device))
            self.netD_A.load_state_dict(torch.load(os.path.join(save_dir, f"D_A_{prefix}.pth"), map_location=self.device))
            self.netD_B.load_state_dict(torch.load(os.path.join(save_dir, f"D_B_{prefix}.pth"), map_location=self.device))
            logging.info(f"从 epoch {epoch} 加载模型")
    
    def save_samples(self, epoch, dataloader_A, dataloader_B, num_samples=5):
        """保存样本转换结果"""
        # 确保输出目录存在
        os.makedirs(os.path.join(self.config.output_dir, "samples"), exist_ok=True)
        
        self.netG_A2B.eval()
        self.netG_B2A.eval()
        
        with torch.no_grad():
            for i, (batch_A, batch_B) in enumerate(zip(dataloader_A, dataloader_B)):
                real_A = batch_A["feature"].to(self.device)
                real_B = batch_B["feature"].to(self.device)
                
                # 处理特征维度
                real_A = self._prepare_features(real_A)
                real_B = self._prepare_features(real_B)
                
                # 转换方向A2B或B2A
                fake_B = self.netG_A2B(real_A)
                recovered_A = self.netG_B2A(fake_B)
                
                # 保存示例
                for j in range(min(num_samples, real_A.size(0))):
                    # 真实样本
                    real_A_j = real_A[j].cpu().clone()
                    real_B_j = real_B[j].cpu().clone()
                    
                    # 假样本
                    fake_B_j = fake_B[j].cpu().clone()
                    recovered_A_j = recovered_A[j].cpu().clone()
                    
                    # 转换为PIL图像并保存
                    self._save_sample_image(real_A_j, fake_B_j, recovered_A_j, 
                                            os.path.join(self.config.output_dir, "samples"), 
                                            f"epoch_{epoch}_sample_{i*num_samples+j}.png")
                
                if i * self.config.batch_size >= num_samples:
                    break
        
        self.netG_A2B.train()
        self.netG_B2A.train()
    
    def _save_sample_image(self, real, fake, recovered, save_dir, filename):
        """保存单个样本的图像
        
        参数:
            real: 真实图像张量 [C,H,W]
            fake: 生成的假图像张量 [C,H,W]
            recovered: 恢复的图像张量 [C,H,W]
            save_dir: 保存目录
            filename: 文件名
        """
        # 转换为PIL图像
        real = self._tensor_to_image(real)
        fake = self._tensor_to_image(fake)
        recovered = self._tensor_to_image(recovered)
        
        # 拼接图像
        img = torch.cat([real, fake, recovered], dim=2)
        
        # 保存图像
        img = Image.fromarray(img)
        img.save(os.path.join(save_dir, filename))
    
    def _tensor_to_image(self, tensor):
        """将张量转换为图像
        
        参数:
            tensor: 输入张量 [C,H,W] 或 [1,H,W]
        
        返回:
            处理后的图像张量 [H,W,C] 或 [H,W]
        """
        if tensor.dim() == 3 and tensor.size(0) == 1:
            # 单通道情况，去掉通道维度
            tensor = tensor.squeeze(0)
        elif tensor.dim() == 3:
            # 多通道情况，调整通道顺序
            tensor = tensor.permute(1, 2, 0)
        
        # 将值从[-1,1]缩放到[0,255]
        tensor = (tensor + 1) / 2 * 255
        
        # 转换为整数类型
        tensor = tensor.clamp(0, 255).byte()
        
        return tensor.numpy()
    
    def save_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.output_dir, "logs", "history.pt")
        torch.save(self.history, history_path)
        
    def plot_losses(self, epoch):
        """绘制损失曲线"""
        # ...existing code...
    
    def translate_domain(self, features, direction="A2B"):
        """将特征从一个域转换到另一个域"""
        # ...existing code...
    
    def translate_file(self, input_file, output_file, direction="A2B", batch_size=4):
        """转换整个文件中的特征"""
        # ...existing code...
    
def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="VAE特征向量的CycleGAN训练")
    
    # 数据参数
    parser.add_argument("--domain_a", type=str, required=True, help="域A特征文件路径")
    parser.add_argument("--domain_b", type=str, required=True, help="域B特征文件路径")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--max_samples", type=int, default=None, help="每个域使用的最大样本数，None表示全部使用")
    parser.add_argument("--shuffle", action='store_true', help="是否打乱数据集")
    parser.add_argument("--drop_last", action='store_true', help="丢弃最后不完整的批次")
    
    # 模型参数
    parser.add_argument("--input_channels", type=int, default=16, help="输入通道数")
    parser.add_argument("--output_channels", type=int, default=16, help="输出通道数")
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="残差块数量")
    parser.add_argument("--base_filters", type=int, default=64, help="基础滤波器数量")    
    parser.add_argument("--discriminator_layers", type=int, default=3, help="判别器层数")
    
    # 训练参数
    parser.add_argument("--n_epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--decay_epoch", type=int, default=100, help="学习率开始衰减的轮数")
    parser.add_argument("--lr", type=float, default=0.0002, help="学习率")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam优化器beta1参数")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam优化器beta2参数")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="循环一致性损失权重")
    parser.add_argument("--lambda_identity", type=float, default=5.0, help="身份损失权重")        
    parser.add_argument("--buffer_size", type=int, default=50, help="经验回放缓冲区大小")
    
    # 其他参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--output_dir", type=str, default="./output/vae_cyclegan", help="输出目录")
    parser.add_argument("--save_freq", type=int, default=10, help="模型保存频率（轮数）")
    parser.add_argument("--sample_freq", type=int, default=5, help="样本生成频率（轮数）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的模型路径或轮数")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="训练或测试模式")
    parser.add_argument("--config_path", type=str, default=None, help="配置文件路径，优先级高于命令行参数")
    
    # 测试参数
    parser.add_argument("--test_input", type=str, default=None, help="测试模式下的输入文件")
    parser.add_argument("--test_output", type=str, default=None, help="测试模式下的输出文件")
    parser.add_argument("--test_direction", type=str, default="A2B", choices=["A2B", "B2A"], help="测试模式下的转换方向")
    parser.add_argument("--test_batch_size", type=int, default=4, help="测试模式下的批处理大小")
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config.from_args(args)
    
    # 如果配置文件存在，则覆盖命令行参数
    if args.config_path and os.path.exists(args.config_path):
        config = Config.from_file(args.config_path)
    
    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 配置日志
    log_file = os.path.join(config.output_dir, "vae_cyclegan.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 保存配置
    config_path = os.path.join(config.output_dir, "config.json")
    config.save(config_path)
    
    # 打印配置信息
    logging.info(f"配置参数: {config}")

    # 加载数据集
    dataset_A = VAEFeatureDataset(config.domain_a_path, config.max_samples)
    dataset_B = VAEFeatureDataset(config.domain_b_path, config.max_samples)
    
    # 创建形状感知的数据加载器
    dataloader_A = ShapeAwareDataLoader(
        dataset_A, 
        batch_size=config.batch_size,
        shuffle=config.shuffle, 
        num_workers=config.num_workers,
        drop_last=config.drop_last
    )
    
    dataloader_B = ShapeAwareDataLoader(
        dataset_B, 
        batch_size=config.batch_size,
        shuffle=config.shuffle, 
        num_workers=config.num_workers,
        drop_last=config.drop_last
    )
    
    logging.info(f"数据集A: {len(dataset_A)}个样本，数据集B: {len(dataset_B)}个样本")
    
    # 初始化模型
    model = VAECycleGAN(config)
    
    # 加载预训练模型（如果指定）
    if config.resume_path:
        try:
            if config.resume_path.isdigit():
                model.load_models(epoch=int(config.resume_path))
            else:
                model.load_models(path=config.resume_path)
        except Exception as e:
            logging.error(f"加载模型失败: {e}")
            return
    
    # 训练或测试
    if args.mode == "train":
        # 训练模型
        logging.info("开始训练...")
        losses = model.train(dataloader_A, dataloader_B)
        logging.info("训练完成")
        
        # 保存最终模型
        model.save_models(epoch="final")
    else:
        # 测试模式
        logging.info("开始测试...")
        if args.test_input and args.test_output:
            # 转换整个文件
            model.translate_file(
                args.test_input, 
                args.test_output, 
                direction=args.test_direction, 
                batch_size=args.test_batch_size
            )
        else:
            # 生成一些示例
            model.save_samples("test", dataloader_A, dataloader_B, num_samples=10)
        
        logging.info("测试完成")


if __name__ == "__main__":
    main()