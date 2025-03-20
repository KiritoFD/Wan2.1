# 残差链接（Residual Connection）

**概念**: 通过跳跃连接将输入直接添加到输出，缓解深度网络的梯度问题。

**公式**: `H(x) = F(x) + x` （网络学习残差F(x)，而非直接学习H(x)）

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