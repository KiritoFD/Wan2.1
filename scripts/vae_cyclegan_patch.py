import torch
import torch.nn as nn
import torch.nn.functional as F

# 尺寸自适应的残差块
class AdaptiveResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# 尺寸自适应的生成器
class AdaptiveGenerator(nn.Module):
    def __init__(self, input_channels, output_channels, base_filters=64, n_residual_blocks=9):
        super(AdaptiveGenerator, self).__init__()
        
        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            *[AdaptiveResidualBlock(base_filters) for _ in range(n_residual_blocks)]
        )
        
        # 输出层 - 使用 1x1 卷积将通道数调整为目标通道数
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_filters, output_channels, kernel_size=1, stride=1),
            nn.Tanh()  # 归一化输出
        )
        
        # 形状适应层 - 这将在forward中动态构建
        self.shape_adapter = None

    def forward(self, x, target_shape=None):
        # 初始特征提取
        x = self.initial(x)
        
        # 残差处理
        x = self.residual_blocks(x)
        
        # 将通道数转换为输出通道
        x = self.output_conv(x)
        
        # 如果提供了目标形状，调整输出形状
        if target_shape is not None and (x.shape[2:] != target_shape[2:]):
            x = F.interpolate(x, size=target_shape[2:], mode='bilinear', align_corners=False)
            
        return x

# 尺寸自适应的判别器
class AdaptiveDiscriminator(nn.Module):
    def __init__(self, input_channels, base_filters=64, n_layers=3):
        super(AdaptiveDiscriminator, self).__init__()
        
        # 初始卷积层
        layers = [
            nn.Conv2d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 添加中间层
        mult = 1
        mult_prev = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            layers.extend([
                nn.Conv2d(base_filters * mult_prev, base_filters * mult, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(base_filters * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # 添加输出层
        mult_prev = mult
        mult = min(2**n_layers, 8)
        layers.extend([
            nn.Conv2d(base_filters * mult_prev, base_filters * mult, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(base_filters * mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_filters * mult, 1, kernel_size=4, stride=1, padding=1)
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x_scaled = x
        # 为了确保判别器能够处理输入，我们确保它至少为4x4
        if x.size(2) < 4 or x.size(3) < 4:
            x_scaled = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.model(x_scaled)

# 替换原有 VAECycleGAN 中的生成器和判别器
def patch_vae_cyclegan(trainer):
    """替换训练器中的生成器和判别器为自适应版本"""
    # 获取原始模型的配置
    config = trainer.config
    
    # 创建新的自适应生成器
    G_AB = AdaptiveGenerator(
        input_channels=config.input_channels, 
        output_channels=config.output_channels,
        base_filters=config.base_filters,
        n_residual_blocks=config.n_residual_blocks
    )
    
    G_BA = AdaptiveGenerator(
        input_channels=config.output_channels, 
        output_channels=config.input_channels,
        base_filters=config.base_filters,
        n_residual_blocks=config.n_residual_blocks
    )
    
    # 创建新的自适应判别器
    D_A = AdaptiveDiscriminator(
        input_channels=config.input_channels,
        base_filters=config.base_filters,
        n_layers=config.discriminator_layers
    )
    
    D_B = AdaptiveDiscriminator(
        input_channels=config.output_channels,
        base_filters=config.base_filters,
        n_layers=config.discriminator_layers
    )
    
    # 将新模型移到原始模型所在的设备
    device = next(trainer.G_AB.parameters()).device
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)
    
    # 尝试加载原始模型的权重（如果形状兼容）
    try:
        G_AB.load_state_dict(trainer.G_AB.state_dict(), strict=False)
        G_BA.load_state_dict(trainer.G_BA.state_dict(), strict=False)
        D_A.load_state_dict(trainer.D_A.state_dict(), strict=False)
        D_B.load_state_dict(trainer.D_B.state_dict(), strict=False)
    except Exception as e:
        print(f"无法加载原始模型权重，使用新初始化的权重: {e}")
    
    # 替换原始模型
    trainer.G_AB = G_AB
    trainer.G_BA = G_BA
    trainer.D_A = D_A
    trainer.D_B = D_B
    
    # 重新设置优化器
    trainer.optimizer_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=config.lr, betas=(config.b1, config.b2)
    )
    trainer.optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    trainer.optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=config.lr, betas=(config.b1, config.b2))
    
    return trainer

# 安全的MSE损失函数，只是为了保险
def safe_mse_loss(input_tensor, target_tensor):
    """计算MSE损失，确保输入和目标尺寸匹配"""
    if input_tensor.shape != target_tensor.shape:
        input_tensor = F.interpolate(input_tensor, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)
        if input_tensor.shape[1] != target_tensor.shape[1]:
            if target_tensor.shape[1] % input_tensor.shape[1] == 0:
                repeat_factor = target_tensor.shape[1] // input_tensor.shape[1]
                input_tensor = input_tensor.repeat(1, repeat_factor, 1, 1)
    return F.mse_loss(input_tensor, target_tensor)
