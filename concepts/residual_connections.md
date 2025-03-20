# 残差连接 (Residual Connections)

## 概述

残差连接（Residual Connections）是深度神经网络中的一种架构设计，通过添加跳跃连接（skip connections）使信息可以直接从一个层传播到更深的层。这种设计最早由何凯明等人在论文《Deep Residual Learning for Image Recognition》(2015)中提出，引入了ResNet网络架构。

## 工作原理

残差连接的基本思想是让网络学习输入和输出之间的残差（差异），而不是直接学习从输入到输出的完整映射。具体来说：

- 若期望某层的理想映射为 H(x)
- 传统方法直接学习 H(x)
- 残差方法学习 F(x) = H(x) - x
- 最终输出为 H(x) = F(x) + x

这种结构通过将输入 x 直接添加到该层的输出上实现。

## 数学表示

对于一个输入为 x 的残差块：

```
输出 = F(x) + x
```

其中 F(x) 是残差函数，通常由多个卷积层、归一化层和激活函数组成。

## 优势

残差连接具有以下几个关键优势：

1. **缓解梯度消失/爆炸问题**：在深度网络中，梯度可以通过跳跃连接直接向后传播，减轻了梯度消失/爆炸问题。

2. **更容易训练更深的网络**：引入残差连接前，增加网络深度往往会导致性能下降，而残差连接使得非常深的网络（如ResNet-152等）可以被有效训练。

3. **提高收敛速度**：残差连接使网络可以更快地收敛，加速训练过程。

4. **便于学习恒等映射**：如果恒等映射是最优的，残差网络可以通过将权重推向零来轻松学习恒等映射。

## 在Wan2.1中的应用

在Wan2.1的视频变分自编码器中，残差连接被广泛应用于以下部分：

1. **ResidualBlock类**：实现了标准的残差块，包含两个卷积层和一个跳跃连接。

2. **AttentionBlock类**：结合了自注意力机制和残差连接，提高特征表示能力。

3. **编码器和解码器**：通过大量使用残差块构建了可扩展的深度架构。

残差连接允许Wan2.1中的变分自编码器达到很深的网络深度，提高了视频编码和解码的质量和稳定性。

## 示例代码

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            # 多个卷积、归一化和激活层
            # ...
        )
        self.shortcut = nn.Conv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        h = self.shortcut(x)  # 跳跃连接
        x = self.residual(x)  # 残差函数
        return x + h  # 残差连接
```

## 多种实现方案

### 1. 标准残差块 (ResNet风格)
```python
class StandardResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.layers(x))
```

### 2. 瓶颈残差块 (Bottleneck)
```python
class BottleneckBlock(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, 1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim)
        ) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.shortcut(x) + self.layers(x))
```

### 3. 预激活残差块 (Pre-activation)
```python
class PreActResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1)
        )
    
    def forward(self, x):
        return x + self.layers(x)
```

### 4. 密集连接 (DenseNet风格)
```python
class DenseBlock(nn.Module):
    def __init__(self, in_dim, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_dim + i * growth_rate),
                nn.ReLU(),
                nn.Conv2d(in_dim + i * growth_rate, growth_rate, 3, padding=1)
            ))
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)
```

## 优势对比
| 方案 | 计算效率 | 参数效率 | 特点 |
|------|----------|----------|------|
| 标准残差块 | 中 | 高 | 实现简单，广泛应用 |
| 瓶颈残差块 | 高 | 中 | 降低计算复杂度，适合深层网络 |
| 预激活残差块 | 中 | 高 | 改善信息流，训练更稳定 |
| 密集连接 | 低 | 中 | 最大化特征重用，适合小数据集 |

## 应用场景

残差连接在众多深度学习架构中都有广泛应用：

1. **深度CNN网络**：ResNet系列（ResNet-18/50/101/152等）
2. **图像分割网络**：U-Net++、DeepLabv3+等
3. **GAN模型**：StyleGAN、BigGAN等
4. **Transformer架构**：每个编码器/解码器层中的FFN后
5. **轻量级模型**：MobileNetV2/V3、EfficientNet等

## 实际应用建议

- 对于**浅层网络**（<10层），标准残差块通常足够
- 对于**中等深度网络**（10-50层），可选择预激活残差块提高稳定性
- 对于**超深网络**（>50层），瓶颈残差块可显著降低计算成本
- 当**特征重用**非常重要且计算资源充足时，考虑密集连接