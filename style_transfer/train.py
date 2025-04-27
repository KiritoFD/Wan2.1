#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练风格转换模型：加载两种不同风格的VAE潜在向量，并训练AAE模型学习风格转换
"""

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import time
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# 修改导入路径，使用相对导入
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import StyleDataset, StyleTransferAAE, calc_style_statistics, style_distance


def setup_logging(log_dir):
    """设置日志配置"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练风格转换模型")
    
    parser.add_argument("--style_a", type=str, required=True,
                        help="风格A的VAE编码文件路径(.pt)")
    parser.add_argument("--style_b", type=str, required=True,
                        help="风格B的VAE编码文件路径(.pt)")
    parser.add_argument("--output_dir", type=str, default="models/style_transfer",
                        help="模型输出目录")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批次大小")
    parser.add_argument("--latent_dim", type=int, default=128,
                        help="潜在空间维度")
    parser.add_argument("--epochs", type=int, default=70,
                        help="训练轮数")
    parser.add_argument("--valid_split", type=float, default=0.1,
                        help="验证集比例")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练设备，'cuda'或'cpu'")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 添加从最优模型开始训练的选项
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从指定的检查点文件继续训练，例如'checkpoint_epoch_40.pth'")
    parser.add_argument("--best_checkpoint", action="store_true",
                        help="如果设置，则尝试自动找到最优检查点继续训练")
    parser.add_argument("--use_wgan_gp", action="store_true",
                        help="使用WGAN-GP训练方式获得更好的训练效果")
    parser.add_argument("--n_critic", type=int, default=5,
                        help="使用WGAN-GP时判别器的训练频率")
    
    return parser.parse_args()


def train_single_epoch(model, dataloader_a, dataloader_b, wgan_gp=False, n_critic=5, lambda_gp=10.0, quiet=False):
    """对模型训练单个轮次，避免使用内置训练循环
    
    Args:
        model: StyleTransferAAE模型
        dataloader_a: 风格A的数据加载器
        dataloader_b: 风格B的数据加载器
        wgan_gp: 是否使用WGAN-GP训练策略
        n_critic: 判别器迭代次数
        lambda_gp: 梯度惩罚权重
        quiet: 是否禁用进度条
        
    Returns:
        dict: 包含训练损失的字典
    """
    # 设置为训练模式
    model.encoder.train()
    model.decoder.train()
    model.discriminator.train()
    model.style_mapper.train()
    device = next(model.encoder.parameters()).device
    
    # 创建无限循环的数据迭代器
    def infinite_dataloader(dataloader):
        while True:
            for data in dataloader:
                yield data
    
    iter_a = infinite_dataloader(dataloader_a)
    iter_b = infinite_dataloader(dataloader_b)
    
    # 每个epoch的步数为较短的dataloader的长度
    steps_per_epoch = min(len(dataloader_a), len(dataloader_b))
    
    epoch_recon_loss = 0
    epoch_disc_loss = 0
    epoch_gen_loss = 0
    
    # 静默tqdm输出避免与外部训练循环冲突
    if not quiet:
        pbar = tqdm(total=steps_per_epoch, desc="Training Steps", leave=False)
    
    for step in range(steps_per_epoch):
        if wgan_gp:
            # WGAN-GP训练逻辑
            # 获取下一批数据
            real_a = next(iter_a).to(device)
            real_b = next(iter_b).to(device)
            
            # 训练判别器
            if step % n_critic == 0:
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = False
                for param in model.style_mapper.parameters():
                    param.requires_grad = False
                for param in model.discriminator.parameters():
                    param.requires_grad = True
                
                # 获取真假样本
                with torch.no_grad():
                    latent_a = model.encoder(real_a)
                    latent_b = model.encoder(real_b)
                    mapped_latent_a = model.style_mapper(latent_a)
                
                # 计算判别器损失
                pred_real = model.discriminator(latent_b)
                pred_fake = model.discriminator(mapped_latent_a.detach())
                
                loss_d_real = -torch.mean(pred_real)
                loss_d_fake = torch.mean(pred_fake)
                
                # 梯度惩罚
                alpha = torch.rand(latent_b.size(0), 1, device=device)
                interpolates = (alpha * latent_b + (1 - alpha) * mapped_latent_a.detach()).requires_grad_(True)
                d_interpolates = model.discriminator(interpolates)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(d_interpolates),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                gradients = gradients.view(gradients.size(0), -1)
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                loss_discriminator = loss_d_real + loss_d_fake + lambda_gp * gradient_penalty
                
                loss_discriminator.backward()
                model.optimizer_Dis.step()
                
                epoch_disc_loss += loss_discriminator.item()
            
            # 训练生成器
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.decoder.parameters():
                param.requires_grad = True
            for param in model.style_mapper.parameters():
                param.requires_grad = True
            for param in model.discriminator.parameters():
                param.requires_grad = False
            
            # 生成器和重建损失
            model.optimizer_E.zero_grad()
            model.optimizer_D.zero_grad()
            model.optimizer_Map.zero_grad()
            
            # 编码和重建
            latent_a = model.encoder(real_a)
            latent_b = model.encoder(real_b)
            
            recon_a = model.decoder(latent_a)
            recon_b = model.decoder(latent_b)
            
            mapped_latent_a = model.style_mapper(latent_a)
            fake_b = model.decoder(mapped_latent_a)
            
            # 重建损失
            loss_recon = model.reconstruction_loss(recon_a, real_a) + model.reconstruction_loss(recon_b, real_b)
            
            # 内容一致性损失
            latent_fake_b = model.encoder(fake_b)
            loss_content = model.content_loss(latent_fake_b, mapped_latent_a)
            
            # 生成器对抗损失
            pred_gen = model.discriminator(mapped_latent_a)
            loss_gen = -torch.mean(pred_gen)
            
            # 风格损失
            target_style_stats = calc_style_statistics(latent_b.detach())
            gen_style_stats = calc_style_statistics(mapped_latent_a)
            loss_style = style_distance(gen_style_stats, target_style_stats)
            
            # 总损失
            total_recon_loss = model.lambda_recon * loss_recon + model.lambda_content * loss_content
            total_gen_loss = model.lambda_adv * loss_gen + loss_style
            
            # 先反向传播重建损失
            total_recon_loss.backward(retain_graph=True)
            # 再反向传播生成器损失
            total_gen_loss.backward()
            
            model.optimizer_E.step()
            model.optimizer_D.step()
            model.optimizer_Map.step()
            
            epoch_recon_loss += total_recon_loss.item()
            epoch_gen_loss += total_gen_loss.item()
            
            if not quiet:
                pbar.update(1)
                pbar.set_postfix({
                    'recon': f"{total_recon_loss.item():.3f}" if 'total_recon_loss' in locals() else "N/A",
                    'disc_loss': epoch_disc_loss / (step + 1),
                    'gen_loss': epoch_gen_loss / (step + 1)
                })
        else:
            # 标准训练逻辑
            # 获取下一批数据
            real_a = next(iter_a).to(device)
            real_b = next(iter_b).to(device)
            
            # 1. 训练自编码器和内容损失
            model.optimizer_E.zero_grad()
            model.optimizer_D.zero_grad()
            model.optimizer_Map.zero_grad()
            
            # 编码
            latent_a = model.encoder(real_a)
            latent_b = model.encoder(real_b)
            
            # 风格映射
            mapped_latent_a = model.style_mapper(latent_a)
            
            # 解码
            recon_a = model.decoder(latent_a)
            recon_b = model.decoder(latent_b)
            fake_b = model.decoder(mapped_latent_a)
            
            # 重建损失
            loss_recon = model.reconstruction_loss(recon_a, real_a) + model.reconstruction_loss(recon_b, real_b)
            
            # 内容损失
            latent_fake_b = model.encoder(fake_b)
            loss_content = model.content_loss(latent_fake_b, mapped_latent_a)
            
            total_recon_loss = model.lambda_recon * loss_recon + model.lambda_content * loss_content
            total_recon_loss.backward(retain_graph=True)
            model.optimizer_E.step()
            model.optimizer_D.step()
            model.optimizer_Map.step()
            
            # 2. 训练判别器
            model.optimizer_Dis.zero_grad()
            
            # 获取真假样本
            latent_b_real = model.encoder(real_b).detach()
            fake_latent_b = model.style_mapper(latent_a.detach()).detach()
            
            # 真假样本的判别
            pred_real = model.discriminator(latent_b_real)
            pred_fake = model.discriminator(fake_latent_b)
            
            # 判别器损失
            loss_d_real = torch.mean(F.relu(1.0 - pred_real))
            loss_d_fake = torch.mean(F.relu(1.0 + pred_fake))
            loss_discriminator = loss_d_real + loss_d_fake
            
            loss_discriminator.backward()
            model.optimizer_Dis.step()
            
            # 3. 训练生成器对抗性
            model.optimizer_E.zero_grad()
            model.optimizer_Map.zero_grad()
            
            # 重新计算映射的潜在向量
            latent_a_gen = model.encoder(real_a)
            mapped_latent_a_gen = model.style_mapper(latent_a_gen)
            
            # 对抗损失
            pred_gen = model.discriminator(mapped_latent_a_gen)
            loss_generator = -torch.mean(pred_gen)
            
            # 风格损失
            target_style_stats = calc_style_statistics(latent_b.detach())
            gen_style_stats = calc_style_statistics(mapped_latent_a_gen)
            loss_style = style_distance(gen_style_stats, target_style_stats)
            
            # 总生成器损失
            total_gen_loss = model.lambda_adv * loss_generator + loss_style
            total_gen_loss.backward()
            
            model.optimizer_E.step()
            model.optimizer_Map.step()
            
            # 记录损失
            epoch_recon_loss += total_recon_loss.item()
            epoch_gen_loss += total_gen_loss.item()
            if not quiet:
                pbar.update(1)
                pbar.set_postfix({
                    'recon': f"{total_recon_loss.item():.3f}" if 'total_recon_loss' in locals() else "N/A",
                    'disc_loss': epoch_disc_loss / (step + 1),
                    'gen_loss': epoch_gen_loss / (step + 1)
                })
    
    if not quiet:
        pbar.close()
    
    # 更新学习率
    model.scheduler_E.step()
    model.scheduler_D.step()
    model.scheduler_Map.step()
    model.scheduler_Dis.step()
    
    # 计算平均损失
    avg_recon_loss = epoch_recon_loss / steps_per_epoch
    avg_disc_loss = epoch_disc_loss / steps_per_epoch
    avg_gen_loss = epoch_gen_loss / steps_per_epoch
    
    # 更新模型的训练历史
    model.training_history['recon_loss'].append(avg_recon_loss)
    model.training_history['disc_loss'].append(avg_disc_loss)
    model.training_history['gen_loss'].append(avg_gen_loss)
    
    return {
        'recon_loss': avg_recon_loss,
        'disc_loss': avg_disc_loss,
        'gen_loss': avg_gen_loss
    }


def inspect_pt_file(file_path):
    """检查.pt文件的有效性和内容"""
    try:
        # 检查文件是否存在和大小
        if not os.path.exists(file_path):
            return False, f"文件不存在: {file_path}"
        
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        if file_size < 0.001:  # 小于1KB
            return False, f"文件可能是空的: {file_path} (大小 {file_size:.2f}MB)"
        
        logging.info(f"检查文件: {file_path} (大小: {file_size:.2f}MB)")
        
        # 尝试简单加载文件
        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            return False, f"无法加载文件: {str(e)}"
        
        # 检查基本结构
        if not isinstance(data, dict):
            return False, f"文件内容不是字典类型: {type(data)}"
        
        # 检查是否包含features键
        if 'features' not in data:
            keys = list(data.keys())
            return False, f"文件中缺少'features'键，包含的键: {keys}"
        
        # 检查features的类型和形状
        features = data['features']
        if isinstance(features, torch.Tensor):
            shape = features.shape
        elif isinstance(features, list) and all(isinstance(x, torch.Tensor) for x in features):
            shape = [x.shape for x in features[:3]]
            shape = f"{shape}... (共{len(features)}项)"
        else:
            return False, f"features不是预期的张量类型: {type(features)}"
        
        # 返回验证成功和基本信息
        return True, f"文件验证通过，形状: {shape}"
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return False, f"检查文件时出错: {str(e)}\n{tb}"


def find_best_checkpoint(checkpoint_dir):
    """查找最优的检查点文件"""
    try:
        checkpoints = []
        # 遍历目录查找所有检查点
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
                epoch = int(file.split('_')[-1].split('.')[0])
                # 尝试提取验证损失（如果有）
                try:
                    checkpoint_data = torch.load(os.path.join(checkpoint_dir, file), map_location='cpu')
                    # 如果存在验证损失数据，使用它作为排序依据
                    if 'validation_loss' in checkpoint_data:
                        val_loss = checkpoint_data['validation_loss']
                        checkpoints.append((epoch, os.path.join(checkpoint_dir, file), val_loss))
                    else:
                        # 如果没有验证损失，仅使用轮次
                        checkpoints.append((epoch, os.path.join(checkpoint_dir, file), float('inf')))
                except Exception as load_err:
                    logging.warning(f"无法加载检查点文件 {file} 进行验证损失提取: {str(load_err)}")
                    # 如果无法加载检查点，仅使用轮次
                    checkpoints.append((epoch, os.path.join(checkpoint_dir, file), float('inf')))
        
        if not checkpoints:
            return None, 0, float('inf')
        
        # 优先按验证损失排序，然后按轮次排序（越大越新）
        if any(x[2] != float('inf') for x in checkpoints):
            # 如果存在验证损失数据，按损失排序
            checkpoints.sort(key=lambda x: (x[2], -x[0]))
        else:
            # 否则按轮次排序
            checkpoints.sort(key=lambda x: -x[0])
        
        # 返回最佳检查点和验证损失
        best_epoch, best_path, best_loss = checkpoints[0]
        
        return best_path, best_epoch, best_loss
    except Exception as e:
        logging.error(f"查找最优检查点时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, 0, float('inf')


def delete_old_checkpoints(checkpoint_dir, keep_path=None):
    """删除所有旧检查点，只保留最佳检查点"""
    try:
        count = 0
        for file in os.listdir(checkpoint_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
                file_path = os.path.join(checkpoint_dir, file)
                # 如果不是要保留的文件，则删除
                if keep_path != file_path:
                    os.remove(file_path)
                    count += 1
        logging.info(f"已删除 {count} 个旧检查点")
    except Exception as e:
        logging.error(f"删除旧检查点时出错: {e}")


def plot_realtime_loss(history, save_path=None, title="Training Loss"):
    """绘制实时训练损失曲线
    
    Args:
        history: 包含训练历史的字典
        save_path: 图像保存路径
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['recon_loss']) + 1)
    
    plt.plot(epochs, history['recon_loss'], 'b-', label='Reconstruction Loss')
    plt.plot(epochs, history['disc_loss'], 'r-', label='Discriminator Loss')
    plt.plot(epochs, history['gen_loss'], 'g-', label='Generator Loss')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 设置x轴为整数刻度
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def save_checkpoint_with_validation(model, checkpoint_path, val_loss=None, epoch=None):
    """保存检查点，包括验证损失数据"""
    checkpoint_data = {
        'encoder_state_dict': model.encoder.state_dict(),
        'decoder_state_dict': model.decoder.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'style_mapper_state_dict': model.style_mapper.state_dict(),
        'optimizer_E_state_dict': model.optimizer_E.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
        'optimizer_Dis_state_dict': model.optimizer_Dis.state_dict(),
        'optimizer_Map_state_dict': model.optimizer_Map.state_dict(),
        'scheduler_E_state_dict': model.scheduler_E.state_dict(),
        'scheduler_D_state_dict': model.scheduler_D.state_dict(),
        'scheduler_Dis_state_dict': model.scheduler_Dis.state_dict(),
        'scheduler_Map_state_dict': model.scheduler_Map.state_dict(),
        'training_history': model.training_history,
        'latent_dim': model.latent_dim
    }
    
    # 如果提供了验证损失，添加到检查点
    if val_loss is not None:
        checkpoint_data['validation_loss'] = val_loss
    
    # 如果提供了轮次，添加到检查点
    if epoch is not None:
        checkpoint_data['epoch'] = epoch
    
    torch.save(checkpoint_data, checkpoint_path)
    logging.info(f"检查点已保存到: {checkpoint_path}")
    if val_loss is not None:
        logging.info(f"验证损失: {val_loss:.6f}")


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    model_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(log_dir)
    logging.info(f"开始训练，参数: {args}")
    logging.info(f"日志将保存到: {log_file}")
    
    # 详细检查输入文件
    logging.info("验证输入文件...")
    for file_path in [args.style_a, args.style_b]:
        success, message = inspect_pt_file(file_path)
        if not success:
            logging.error(message)
            return
    
    # 加载数据集
    try:
        logging.info("加载风格A数据集...")
        dataset_a = StyleDataset(args.style_a)
        
        logging.info("加载风格B数据集...")
        dataset_b = StyleDataset(args.style_b)
        
        # 检查数据集特征形状
        if dataset_a.feature_shape != dataset_b.feature_shape:
            logging.warning(f"两个数据集的特征形状不一致: {dataset_a.feature_shape} vs {dataset_b.feature_shape}")
            logging.warning("这可能导致训练问题，但尝试继续...")
        
        # 分割训练集和验证集
        train_size_a = int((1 - args.valid_split) * len(dataset_a))
        valid_size_a = len(dataset_a) - train_size_a
        train_dataset_a, valid_dataset_a = random_split(dataset_a, [train_size_a, valid_size_a])
        
        train_size_b = int((1 - args.valid_split) * len(dataset_b))
        valid_size_b = len(dataset_b) - train_size_b
        train_dataset_b, valid_dataset_b = random_split(dataset_b, [train_size_b, valid_size_b])
        
        # 创建数据加载器
        train_loader_a = DataLoader(train_dataset_a, batch_size=args.batch_size, shuffle=True, num_workers=0)
        train_loader_b = DataLoader(train_dataset_b, batch_size=args.batch_size, shuffle=True, num_workers=0)
        valid_loader_a = DataLoader(valid_dataset_a, batch_size=args.batch_size, num_workers=0)
        valid_loader_b = DataLoader(valid_dataset_b, batch_size=args.batch_size, num_workers=0)
        
        logging.info(f"数据集加载完成:")
        logging.info(f"  - 风格A: 训练 {len(train_dataset_a)} 样本, 验证 {len(valid_dataset_a)} 样本")
        logging.info(f"  - 风格B: 训练 {len(train_dataset_b)} 样本, 验证 {len(valid_dataset_b)} 样本")
    except Exception as e:
        logging.error(f"加载数据集时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # 初始化模型
    try:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        model = StyleTransferAAE(device=device, latent_dim=args.latent_dim)
        
        # 获取输入通道数
        in_channels = dataset_a.feature_shape[0]
        model.build_models(in_channels=in_channels)
        
        # 处理检查点恢复
        start_epoch = 0
        best_val_loss = float('inf')
        best_checkpoint_path = None
        best_model_path = os.path.join(model_dir, "best_model.pth")
        checkpoint_loaded = False
        
        # 检查保存目录是否已经包含最佳模型文件
        if os.path.exists(best_model_path):
            logging.info(f"找到最佳模型文件: {best_model_path}")
            try:
                checkpoint_data = torch.load(best_model_path, map_location='cpu')
                best_val_loss = checkpoint_data.get('validation_loss', float('inf'))
                start_epoch = checkpoint_data.get('epoch', 0)
                # 加载最佳模型
                model.load_checkpoint(best_model_path)
                checkpoint_loaded = True
                best_checkpoint_path = best_model_path
                logging.info(f"从最佳模型继续训练，起始轮次 {start_epoch}，验证损失: {best_val_loss:.6f}")
            except Exception as e:
                logging.error(f"加载最佳模型失败: {e}")
                logging.warning("尝试从头开始训练")
                # 重置模型和状态，以便从头开始训练
                model = StyleTransferAAE(device=device, latent_dim=args.latent_dim)
                model.build_models(in_channels=in_channels)
                start_epoch = 0
                best_val_loss = float('inf')
                best_checkpoint_path = None
                checkpoint_loaded = False
        
        # 如果未找到最佳模型文件，则检查命令行参数
        if not checkpoint_loaded:
            if args.best_checkpoint:
                checkpoint_result = find_best_checkpoint(model_dir)
                if checkpoint_result:
                    checkpoint_path, best_epoch, val_loss = checkpoint_result
                    logging.info(f"找到最优检查点: {checkpoint_path}, 轮次: {best_epoch}")
                    if val_loss != float('inf'):
                        best_val_loss = val_loss
                        # 加载模型
                        try:
                            model.load_checkpoint(checkpoint_path)
                            checkpoint_loaded = True
                            start_epoch = best_epoch
                            best_checkpoint_path = checkpoint_path
                            logging.info(f"从轮次 {start_epoch} 继续训练")
                        except Exception as e:
                            logging.error(f"加载检查点失败: {e}")
                            logging.warning("将从头开始训练")
                            # 确保模型重置
                            model = StyleTransferAAE(device=device, latent_dim=args.latent_dim)
                            model.build_models(in_channels=in_channels)
                            start_epoch = 0
                            best_val_loss = float('inf')
                            best_checkpoint_path = None
            elif args.resume_from:
                checkpoint_path = os.path.join(model_dir, args.resume_from)
                if os.path.exists(checkpoint_path):
                    try:
                        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                        if 'validation_loss' in checkpoint_data:
                            best_val_loss = checkpoint_data['validation_loss']
                        # 加载检查点
                        model.load_checkpoint(checkpoint_path)
                        checkpoint_loaded = True
                        best_checkpoint_path = checkpoint_path
                        # 确定起始轮次
                        if 'epoch' in checkpoint_data:
                            start_epoch = checkpoint_data['epoch']
                            logging.info(f"从轮次 {start_epoch} 继续训练")
                        else:
                            # 从文件名中提取轮次信息
                            try:
                                start_epoch = int(args.resume_from.split('_')[-1].split('.')[0])
                            except:
                                start_epoch = 0
                    except Exception as e:
                        logging.error(f"加载检查点 {checkpoint_path} 失败: {e}")
                        logging.warning("将从头开始训练")
                        # 确保模型重置
                        model = StyleTransferAAE(device=device, latent_dim=args.latent_dim)
                        model.build_models(in_channels=in_channels)
                        start_epoch = 0
                        best_val_loss = float('inf')
                        best_checkpoint_path = None
                else:
                    logging.warning(f"检查点文件不存在: {checkpoint_path}，从头开始训练")
        
        # 验证模型加载状态
        if checkpoint_loaded:
            logging.info("检查点成功加载，验证模型状态...")
            # 执行一次简单的前向传递以确认模型可用
            try:
                model.encoder.eval()
                model.decoder.eval()
                with torch.no_grad():
                    # 从验证集中获取一个样本进行测试
                    test_sample = next(iter(valid_loader_a)).to(device)
                    latent = model.encoder(test_sample)
                    recon = model.decoder(latent)
                    logging.info(f"模型前向传递测试成功, 输入形状: {test_sample.shape}, 输出形状: {recon.shape}")
            except Exception as e:
                logging.error(f"模型验证失败: {e}")
                import traceback
                logging.error(traceback.format_exc())
                logging.warning("将尝试从头重新初始化模型")
                
                # 重新初始化模型
                model = StyleTransferAAE(device=device, latent_dim=args.latent_dim)
                model.build_models(in_channels=in_channels)
                start_epoch = 0
                best_val_loss = float('inf')
                best_checkpoint_path = None
                checkpoint_loaded = False
        
        # 确保添加明确的用户提示，表示正在开始训练
        logging.info("="*50)
        logging.info(f"开始训练过程, 从轮次 {start_epoch+1} 到 {args.epochs}")
        logging.info("="*50)
        
        # 开始训练
        total_epochs = args.epochs
        
        # 执行训练循环
        # 使用模型的内置训练方法，但进行修改以实现保存最佳模型
        if args.use_wgan_gp:
            logging.info(f"使用WGAN-GP训练方法, n_critic={args.n_critic}")
            epoch_pbar = tqdm(range(start_epoch+1, total_epochs+1), desc=f"训练进度", unit="epoch")
            plt.ion()  # 打开交互模式
            
            # 运行自定义训练循环来保存最佳模型
            for epoch in epoch_pbar:
                # 训练一个轮次
                try:
                    logging.info(f"轮次 {epoch}/{total_epochs} 开始训练...")
                    epoch_stats = train_single_epoch(
                        model=model,
                        dataloader_a=train_loader_a,
                        dataloader_b=train_loader_b,
                        wgan_gp=True,
                        n_critic=args.n_critic,
                        lambda_gp=10.0
                    )
                    desc = f"Epoch {epoch}/{total_epochs} [重建={epoch_stats['recon_loss']:.4f}, 判别器={epoch_stats['disc_loss']:.4f}, 生成器={epoch_stats['gen_loss']:.4f}]"
                    epoch_pbar.set_description(desc)
                    
                    logging.info(f"轮次 {epoch} 训练完成，损失: 重建={epoch_stats['recon_loss']:.4f}, 判别器={epoch_stats['disc_loss']:.4f}, 生成器={epoch_stats['gen_loss']:.4f}")
                    # 每5个轮次绘制一次实时损失图
                    if epoch % 5 == 0 or epoch == total_epochs:
                        plot_realtime_loss(
                            model.training_history,
                            save_path=os.path.join(args.output_dir, f"loss_realtime.png"),
                            title=f"训练损失曲线 (轮次 {epoch})"
                        )
                except Exception as e:
                    logging.error(f"轮次 {epoch} 训练失败: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
                
                # 评估模型
                try:
                    eval_dir = os.path.join(args.output_dir, f"eval_epoch_{epoch}")
                    metrics = model.evaluate(valid_loader_a, valid_loader_b, save_dir=eval_dir, n_samples=3)
                    current_val_loss = metrics.get('recon_loss_a', float('inf'))
                    
                    logging.info(f"轮次 {epoch}/{total_epochs}, 验证损失: {current_val_loss:.6f}, 当前最佳: {best_val_loss:.6f}")
                    
                    # 保存检查点
                    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth")
                    save_checkpoint_with_validation(model, checkpoint_path, current_val_loss, epoch)
                    
                    # 如果是最佳模型，更新记录并删除旧检查点
                    if current_val_loss < best_val_loss:
                        logging.info(f"找到更好的模型! 损失从 {best_val_loss:.6f} 改善到 {current_val_loss:.6f}")
                        best_val_loss = current_val_loss
                        best_checkpoint_path = checkpoint_path
                        
                        # 删除之前的检查点，只保留最佳的
                        delete_old_checkpoints(model_dir, best_checkpoint_path)
                        
                        # 额外保存一个最佳模型文件
                        save_checkpoint_with_validation(model, best_model_path, current_val_loss, epoch)
                        logging.info(f"最佳模型已保存到: {best_model_path}")
                    else:
                        # 如果不是最佳模型，删除当前检查点
                        if best_checkpoint_path and best_checkpoint_path != checkpoint_path:
                            os.remove(checkpoint_path)
                            logging.info(f"删除次优检查点: {checkpoint_path}")
                except Exception as e:
                    logging.error(f"轮次 {epoch} 评估或保存失败: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        else:
            logging.info("使用标准训练方法")
            
            # 运行自定义训练循环来保存最佳模型
            epoch_pbar = tqdm(range(start_epoch+1, total_epochs+1), desc=f"训练进度", unit="epoch")
            plt.ion()  # 打开交互模式
            
            for epoch in epoch_pbar:
                try:
                    logging.info(f"轮次 {epoch}/{total_epochs} 开始训练...")
                    epoch_stats = train_single_epoch(
                        model=model,
                        dataloader_a=train_loader_a,
                        dataloader_b=train_loader_b,
                        wgan_gp=False
                    )
                    desc = f"Epoch {epoch}/{total_epochs} [重建={epoch_stats['recon_loss']:.4f}, 判别器={epoch_stats['disc_loss']:.4f}, 生成器={epoch_stats['gen_loss']:.4f}]"
                    epoch_pbar.set_description(desc)
                    
                    logging.info(f"轮次 {epoch} 训练完成，损失: 重建={epoch_stats['recon_loss']:.4f}, 判别器={epoch_stats['disc_loss']:.4f}, 生成器={epoch_stats['gen_loss']:.4f}")
                    # 每5个轮次绘制一次实时损失图
                    if epoch % 5 == 0 or epoch == total_epochs:
                        plot_realtime_loss(
                            model.training_history,
                            save_path=os.path.join(args.output_dir, f"loss_realtime.png"),
                            title=f"训练损失曲线 (轮次 {epoch})"
                        )
                except Exception as e:
                    logging.error(f"轮次 {epoch} 训练失败: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    continue
                
                # 评估模型
                try:
                    eval_dir = os.path.join(args.output_dir, f"eval_epoch_{epoch}")
                    metrics = model.evaluate(valid_loader_a, valid_loader_b, save_dir=eval_dir, n_samples=3)
                    current_val_loss = metrics.get('recon_loss_a', float('inf'))
                    
                    logging.info(f"轮次 {epoch}/{total_epochs}, 验证损失: {current_val_loss:.6f}, 当前最佳: {best_val_loss:.6f}")
                    
                    # 保存检查点
                    checkpoint_path = os.path.join(model_dir, f"checkpoint_epoch_{epoch}.pth")
                    save_checkpoint_with_validation(model, checkpoint_path, current_val_loss, epoch)
                    
                    # 如果是最佳模型，更新记录并删除旧检查点
                    if current_val_loss < best_val_loss:
                        logging.info(f"找到更好的模型! 损失从 {best_val_loss:.6f} 改善到 {current_val_loss:.6f}")
                        best_val_loss = current_val_loss
                        best_checkpoint_path = checkpoint_path
                        
                        # 删除之前的检查点，只保留最佳的
                        delete_old_checkpoints(model_dir, best_checkpoint_path)
                        
                        # 额外保存一个最佳模型文件
                        save_checkpoint_with_validation(model, best_model_path, current_val_loss, epoch)
                        logging.info(f"最佳模型已保存到: {best_model_path}")
                    else:
                        # 如果不是最佳模型，删除当前检查点
                        if best_checkpoint_path and best_checkpoint_path != checkpoint_path:
                            os.remove(checkpoint_path)
                            logging.info(f"删除次优检查点: {checkpoint_path}")
                except Exception as e:
                    logging.error(f"轮次 {epoch} 评估或保存失败: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        # 训练完成后，保存最终模型
        logging.info("="*50)
        if best_checkpoint_path:
            # 如果找到了最佳检查点，则保存最终模型
            save_checkpoint_with_validation(model, best_model_path, best_val_loss, total_epochs)
            logging.info(f"训练完成，最佳验证损失: {best_val_loss:.6f}")
            logging.info(f"最佳检查点路径: {best_checkpoint_path}")
        else:
            logging.warning("训练完成，但未找到有效的最佳检查点")
            # 尝试至少保存最后一个模型
            try:
                final_checkpoint_path = os.path.join(model_dir, f"final_model.pth")
                save_checkpoint_with_validation(model, final_checkpoint_path, float('inf'), total_epochs)
                logging.info(f"保存了最终模型到: {final_checkpoint_path}")
            except Exception as e:
                logging.error(f"保存最终模型失败: {e}")
        logging.info("="*50)
        
    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()
