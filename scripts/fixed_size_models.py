import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from collections import defaultdict
from scripts.vae_cyclegan import VAECycleGANTrainer

class FixedSizeGenerator(nn.Module):
    """为VAE特征设计的固定尺寸生成器"""
    def __init__(self, input_shape, output_shape, base_filters=64, n_residual_blocks=9):
        super(FixedSizeGenerator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # 确定正确的输入和输出通道数
        self.in_channels = input_shape[0]
        self.out_channels = output_shape[0]
        
        # 初始特征提取
        self.initial = nn.Sequential(
            nn.Conv2d(self.in_channels, base_filters, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # 残差块处理特征
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(
                nn.Sequential(
                    nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                    nn.InstanceNorm2d(base_filters),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(base_filters, base_filters, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
                    nn.InstanceNorm2d(base_filters)
                )
            )
        self.res_blocks = nn.ModuleList(res_blocks)
        
        # 计算空间尺寸需要的上采样倍数
        self.h_scale = output_shape[2] / input_shape[2]
        self.w_scale = output_shape[3] / input_shape[3]
        
        # 创建上采样层
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(self.h_scale, self.w_scale), mode='bilinear', align_corners=False),
        )
        
        # 输出卷积层 - 调整通道数
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_filters, self.out_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.Tanh()  # 归一化输出
        )

    def forward(self, x):
        # 保存原始形状和批次大小
        orig_shape = x.shape
        batch_size = orig_shape[0]
        is_5d = len(orig_shape) == 5
        
        if is_5d:
            # 如果是5D输入 [batch, ch1, ch2, h, w]，转成4D [batch, ch1*ch2, h, w]
            x = x.reshape(batch_size, orig_shape[1] * orig_shape[2], orig_shape[3], orig_shape[4])
        
        # 初始特征提取
        x = self.initial(x)
        
        # 应用残差块
        for res_block in self.res_blocks:
            x = x + res_block(x)
        
        # 上采样到目标空间尺寸
        x = self.upsample(x)
        
        # 转换到目标通道数
        x = self.output_conv(x)
        
        # 如果输入是5D，则输出也应为5D
        if is_5d:
            # 重新整形为5D [batch, out_ch1, out_ch2, out_h, out_w]
            # 其中out_ch1和out_ch2是输出形状定义的通道和子通道
            x = x.reshape(batch_size, self.output_shape[0], self.output_shape[1], 
                        int(orig_shape[3] * self.h_scale), 
                        int(orig_shape[4] * self.w_scale))
        
        return x


class FixedSizeDiscriminator(nn.Module):
    """为VAE特征设计的固定尺寸判别器"""
    def __init__(self, input_shape, base_filters=64, n_layers=3):
        super(FixedSizeDiscriminator, self).__init__()
        self.input_shape = input_shape
        
        # 计算实际输入通道数
        in_channels = input_shape[0] * input_shape[1]  # 通道 * 子通道
        
        # 处理5维输入 [批次, 通道, 子通道, 高, 宽]
        # 实际使用的通道数是通道*子通道
        if len(input_shape) == 4:  # 如果是 [通道, 子通道, 高度, 宽度]
            height = input_shape[2] 
            width = input_shape[3]
        else:  # 如果是 [通道, 高度, 宽度]
            height = input_shape[1]
            width = input_shape[2]
        
        # 计算最终特征图的大小，确保判别器能够正确工作
        min_dim = min(height, width)
        max_layers = 0
        while min_dim > 4:
            min_dim = min_dim // 2
            max_layers += 1
        
        # 调整层数，确保不会尺寸过小
        actual_layers = min(n_layers, max_layers)
        
        # 构建判别器网络
        layers = [
            nn.Conv2d(in_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 添加中间层
        mult = 1
        for i in range(1, actual_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            layers.extend([
                nn.Conv2d(base_filters * mult_prev, base_filters * mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_filters * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # 输出层
        mult_prev = mult
        mult = min(2**actual_layers, 8)
        
        layers.extend([
            nn.Conv2d(base_filters * mult_prev, base_filters * mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_filters * mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * mult, 1, kernel_size=4, stride=1, padding=1)
        ])
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # 处理5维输入 [批次, 通道, 子通道, 高, 宽]
        if len(x.shape) == 5:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        
        return self.model(x)

# 自定义GAN损失函数，确保输入和目标尺寸匹配
class GanLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GanLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
        
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)
        
    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class FixedSizeVAECycleGANTrainer(VAECycleGANTrainer):
    """固定尺寸VAE CycleGAN训练器"""
    def __init__(self, config, fault_tolerant=False, skip_individual=False):
        # 初始化基本属性而不调用父类构造函数
        self.config = config
        self.device = torch.device(config.device)
        self.fault_tolerant = fault_tolerant  # 容错模式标志
        self.skip_individual = skip_individual  # 启用单个样本跳过模式
        
        # 统计量
        self.skipped_samples = 0
        self.skipped_batches = 0
        
        # 分析数据集中的形状
        self.analyze_dataset_shapes()
        
        # 初始化固定尺寸的模型
        self.initialize_fixed_size_models()
        
        # 设置损失函数
        self.criterion_cycle = nn.MSELoss()
        self.criterion_identity = nn.MSELoss()
        self.criterion_GAN = GanLoss().to(self.device)
        
        # 设置经验回放缓冲区
        self.fake_A_buffer = AdaptiveReplayBuffer(config.buffer_size)
        self.fake_B_buffer = AdaptiveReplayBuffer(config.buffer_size)
        
        # 设置学习率调度器
        self.schedulers = []
        self.optimizers = [self.optimizer_G, self.optimizer_D_A, self.optimizer_D_B]
        for optimizer in self.optimizers:
            self.schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(
                    optimizer, 
                    lr_lambda=LambdaLR(config.n_epochs, config.decay_epoch).step
                )
            )
        
        # 创建输出目录
        self.output_dir = config.output_dir
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        
    def analyze_dataset_shapes(self):
        """分析数据集中的形状并确定最常见的形状"""
        # 设置为完整的5维形状 [批次, 通道, 子通道, 高, 宽]
        # 定义为 [通道, 子通道, 高, 宽]，批次维度会在处理时添加
        self.shape_A = (16, 1, 80, 105)
        self.shape_B = (16, 1, 80, 105)
        
        logging.info(f"使用固定形状: A = {self.shape_A}, B = {self.shape_B} (5维张量)")

    def initialize_fixed_size_models(self):
        """初始化固定尺寸的模型"""
        # 创建固定尺寸的生成器
        self.G_AB = FixedSizeGenerator(
            input_shape=self.shape_A, 
            output_shape=self.shape_B,
            base_filters=self.config.base_filters,
            n_residual_blocks=self.config.n_residual_blocks
        ).to(self.device)
        
        self.G_BA = FixedSizeGenerator(
            input_shape=self.shape_B, 
            output_shape=self.shape_A,
            base_filters=self.config.base_filters,
            n_residual_blocks=self.config.n_residual_blocks
        ).to(self.device)
        
        # 创建固定尺寸的判别器
        self.D_A = FixedSizeDiscriminator(
            input_shape=self.shape_A,
            base_filters=self.config.base_filters,
            n_layers=self.config.discriminator_layers
        ).to(self.device)
        
        self.D_B = FixedSizeDiscriminator(
            input_shape=self.shape_B,
            base_filters=self.config.base_filters,
            n_layers=self.config.discriminator_layers
        ).to(self.device)
        
        # 初始化优化器
        self.optimizer_G = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2)
        )
        
        self.optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), 
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2)
        )
        
        self.optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), 
            lr=self.config.lr,
            betas=(self.config.b1, self.config.b2)
        )
    
    def _train_batch(self, real_A_dict, real_B_dict):
        """重写训练批次方法，能够安全处理不同形状的张量"""
        # 设置为训练模式
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
        
        try:
            # 从字典中获取特征张量
            if 'feature' in real_A_dict:
                real_A = real_A_dict['feature']
            else:
                for key in ['x', 'features', 'data']:
                    if key in real_A_dict:
                        real_A = real_A_dict[key]
                        break
                else:
                    real_A = next(iter(real_A_dict.values()))
            
            if 'feature' in real_B_dict:
                real_B = real_B_dict['feature']
            else:
                for key in ['x', 'features', 'data']:
                    if key in real_B_dict:
                        real_B = real_B_dict[key]
                        break
                else:
                    real_B = next(iter(real_B_dict.values()))
            
            # 转移数据到设备上
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            # 只在首批次记录形状信息
            if not hasattr(self, '_logged_shapes'):
                logging.info(f"原始输入形状: A = {real_A.shape}, B = {real_B.shape}")
                self._logged_shapes = True
            
            # 如果启用了细粒度容错模式并且批次大小大于1
            if self.fault_tolerant and self.skip_individual and real_A.shape[0] > 1:
                return self._train_batch_with_individual_skipping(real_A, real_B)
            
            # 处理5维输入张量 [批次, 通道, 子通道, 高, 宽]
            if len(real_A.shape) == 5:
                batch_size = real_A.shape[0]
                # 将5维张量转换为4维 [批次, 通道*子通道, 高, 宽]
                real_A = real_A.reshape(batch_size, 
                                     real_A.shape[1] * real_A.shape[2], 
                                     real_A.shape[3], 
                                     real_A.shape[4])
            
            if len(real_B.shape) == 5:
                batch_size = real_B.shape[0]
                # 将5维张量转换为4维
                real_B = real_B.reshape(batch_size, 
                                    real_B.shape[1] * real_B.shape[2], 
                                    real_B.shape[3], 
                                    real_B.shape[4])
            
            # 检查批次是否有效
            if self._is_invalid_batch(real_A) or self._is_invalid_batch(real_B):
                if self.fault_tolerant:
                    self.skipped_batches += 1
                    if self.skipped_batches % 100 == 1:
                        logging.warning(f"跳过无效批次 (已跳过 {self.skipped_batches} 批次)")
                    return self._get_default_losses()
                else:
                    logging.error(f"无效的批次形状: A = {real_A.shape}, B = {real_B.shape}")
                    raise ValueError("无效的批次形状")
            
            # 正常的批次训练逻辑
            return self._process_valid_batch(real_A, real_B)
            
        except Exception as e:
            if self.fault_tolerant:
                self.skipped_batches += 1
                if self.skipped_batches % 100 == 1:
                    logging.warning(f"批次处理异常，已跳过: {e} (已跳过 {self.skipped_batches} 批次)")
                return self._get_default_losses()
            else:
                raise e
    
    def _is_invalid_batch(self, tensor):
        """检查批次是否有效"""
        # 检查形状维度
        if len(tensor.shape) < 4:
            return True
        
        # 批次大小为0
        if tensor.shape[0] == 0:
            return True
            
        # 检查是否有NaN或inf值
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return True
            
        return False
    
    def _train_batch_with_individual_skipping(self, real_A, real_B):
        """对批次中的每个样本单独处理，跳过有问题的样本"""
        batch_size = min(real_A.shape[0], real_B.shape[0])
        
        # 累积总损失
        batch_losses = {
            "G": 0.0,
            "D_A": 0.0,
            "D_B": 0.0,
            "G_identity_A": 0.0,
            "G_identity_B": 0.0,
            "G_GAN_AB": 0.0,
            "G_GAN_BA": 0.0,
            "G_cycle_A": 0.0,
            "G_cycle_B": 0.0,
        }
        
        valid_samples = 0
        
        # 处理每个样本
        for i in range(batch_size):
            # 提取单个样本
            sample_A = real_A[i:i+1]
            sample_B = real_B[i:i+1]
            
            try:
                # 处理单个样本
                if len(sample_A.shape) == 5:
                    sample_A = sample_A.reshape(1, sample_A.shape[1] * sample_A.shape[2], 
                                              sample_A.shape[3], sample_A.shape[4])
                
                if len(sample_B.shape) == 5:
                    sample_B = sample_B.reshape(1, sample_B.shape[1] * sample_B.shape[2],
                                              sample_B.shape[3], sample_B.shape[4])
                
                # 检查单个样本是否有效
                if self._is_invalid_batch(sample_A) or self._is_invalid_batch(sample_B):
                    self.skipped_samples += 1
                    continue
                
                # 处理有效样本
                sample_losses = self._process_valid_batch(sample_A, sample_B)
                
                # 累加损失
                for key in batch_losses:
                    if key in sample_losses:
                        batch_losses[key] += sample_losses[key]
                
                valid_samples += 1
                
            except Exception as e:
                self.skipped_samples += 1
                if self.skipped_samples % 100 == 0:
                    logging.warning(f"样本处理异常，已跳过: {e} (已跳过 {self.skipped_samples} 个样本)")
        
        # 计算平均损失
        if valid_samples > 0:
            for key in batch_losses:
                batch_losses[key] /= valid_samples
        else:
            # 如果所有样本都无效，则返回默认损失
            return self._get_default_losses()
        
        return batch_losses
    
    def _process_valid_batch(self, real_A, real_B):
        """处理有效的批次"""
        # 前向传播
        fake_B = self.G_AB(real_A)
        rec_A = self.G_BA(fake_B)
        fake_A = self.G_BA(real_B)
        rec_B = self.G_AB(fake_A)
        
        # 身份映射
        same_A = self.G_BA(real_A)
        same_B = self.G_AB(real_B)
        
        # 更新生成器
        self.optimizer_G.zero_grad()
        
        # 身份损失
        loss_identity_A = self.criterion_identity(same_A, real_A) * self.config.lambda_identity
        loss_identity_B = self.criterion_identity(same_B, real_B) * self.config.lambda_identity
        
        # GAN损失
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), True)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), True)
        
        # 循环一致性损失
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.config.lambda_cycle
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.config.lambda_cycle
        
        # 总生成器损失
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B
        loss_G.backward()
        self.optimizer_G.step()
        
        # 更新判别器A
        self.optimizer_D_A.zero_grad()
        
        # 从缓冲区获取假样本
        fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A)
        
        # 真实样本损失
        pred_real_A = self.D_A(real_A)
        loss_D_real_A = self.criterion_GAN(pred_real_A, True)
        
        # 生成样本损失
        pred_fake_A = self.D_A(fake_A_buffer.detach())
        loss_D_fake_A = self.criterion_GAN(pred_fake_A, False)
        
        # 总判别器A损失
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()
        
        # 更新判别器B
        self.optimizer_D_B.zero_grad()
        
        # 从缓冲区获取假样本
        fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B)
        
        # 真实样本损失
        pred_real_B = self.D_B(real_B)
        loss_D_real_B = self.criterion_GAN(pred_real_B, True)
        
        # 生成样本损失
        pred_fake_B = self.D_B(fake_B_buffer.detach())
        loss_D_fake_B = self.criterion_GAN(pred_fake_B, False)
        
        # 总判别器B损失
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()
        
        # 返回损失值
        return {
            "G": loss_G.item(),
            "D_A": loss_D_A.item(),
            "D_B": loss_D_B.item(),
            "G_identity_A": loss_identity_A.item(),
            "G_identity_B": loss_identity_B.item(),
            "G_GAN_AB": loss_GAN_AB.item(),
            "G_GAN_BA": loss_GAN_BA.item(), 
            "G_cycle_A": loss_cycle_A.item(),
            "G_cycle_B": loss_cycle_B.item(),
        }
    
    def _get_default_losses(self):
        """返回默认的损失值，用于跳过异常批次时"""
        return {
            "G": 0.0,  
            "D_A": 0.0,
            "D_B": 0.0,
            "G_identity_A": 0.0,
            "G_identity_B": 0.0,
            "G_GAN_AB": 0.0,
            "G_GAN_BA": 0.0, 
            "G_cycle_A": 0.0,
            "G_cycle_B": 0.0,
        }

# 改进的经验回放缓冲区，能够更好地处理不同形状的张量
class AdaptiveReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        # 使用字典按形状存储缓冲区数据
        self.buffers = {}  # shape_key -> [tensors]
    
    def push_and_pop(self, data):
        """添加新数据到缓冲区并返回历史数据+新数据的混合"""
        result = data.clone()
        shape_key = tuple(data.shape)
        
        # 如果缓冲区中没有该形状的数据，则创建新的缓冲区
        if shape_key not in self.buffers:
            self.buffers[shape_key] = []
        
        buffer = self.buffers[shape_key]
        
        if len(buffer) < self.max_size:
            # 缓冲区未满，添加新数据
            buffer.append(data.clone().detach())
        else:
            # 缓冲区已满，随机替换一个数据项
            idx = np.random.randint(0, self.max_size)
            buffer[idx] = data.clone().detach()
            
            # 随机决定是否替换结果中的一个样本
            if np.random.uniform() > 0.5:
                i = np.random.randint(0, data.shape[0])
                result[i] = buffer[np.random.randint(0, self.max_size)][i]
                
        return result

# 学习率调整器
class LambdaLR:
    def __init__(self, n_epochs, decay_start_epoch):
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch):
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# 确保导入所需模块
import os
from tqdm import tqdm
