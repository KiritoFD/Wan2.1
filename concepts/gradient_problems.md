# 梯度爆炸与消失问题

深度神经网络训练中的两个关键挑战是梯度爆炸和梯度消失。这两个问题严重影响网络的训练稳定性和性能。

## 梯度爆炸

**问题定义**: 反向传播时梯度值异常增大，导致训练不稳定，权重更新过大，模型无法收敛。

### 全面解决方案

#### 1. 梯度裁剪/缩放
```python
# 梯度裁剪(设置阈值)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 梯度缩放(按比例缩小)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**工作原理**: 限制梯度的最大范数或绝对值，防止梯度爆炸引起的不稳定训练。

**应用场景**: 特别适用于RNN、LSTM等循环网络，或非常深的网络架构。

#### 2. 权重初始化方法
```python
# Xavier/Glorot初始化(适合tanh/sigmoid激活函数)
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

# He初始化(适合ReLU系列激活函数)
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# 正交初始化(保持梯度范数，适合RNN)
nn.init.orthogonal_(layer.weight)
```

**工作原理**: 根据网络结构和激活函数特性，合理初始化权重，使前向传播的方差保持稳定。

**选择指南**:
- Sigmoid/Tanh激活函数 → Xavier/Glorot初始化
- ReLU系列激活函数 → He初始化
- 循环神经网络 → 正交初始化

#### 3. 归一化技术
```python
# 批归一化(Batch Normalization)
self.bn = nn.BatchNorm2d(dim)

# 层归一化(Layer Normalization)
self.ln = nn.LayerNorm(dim)

# 权重归一化(Weight Normalization)
def weight_norm_linear(in_dim, out_dim):
    linear = nn.Linear(in_dim, out_dim)
    return nn.utils.weight_norm(linear)
```

**工作原理**: 通过标准化激活值或权重，控制每一层的输入分布，稳定训练过程。

**选择指南**:
- CNN → 批归一化
- RNN/Transformer → 层归一化
- 需要确定性行为 → 权重归一化

#### 4. 学习率调整
```python
# 指数衰减学习率
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# 余弦退火学习率
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# 动态学习率调整
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10
)
```

**工作原理**: 随着训练进行，动态调整学习率，控制权重更新的步长，避免梯度爆炸导致的不稳定。

**选择指南**:
- 长期训练 → 余弦退火
- 稳定收敛 → 指数衰减
- 避免过拟合 → 动态调整

## 梯度消失

**问题定义**: 梯度值在反向传播中接近零，导致深层网络权重几乎不更新，网络无法学习。

### 全面解决方案

#### 1. 激活函数选择
```python
# ReLU (简单有效，但有死神经元问题)
self.relu = nn.ReLU()

# Leaky ReLU (允许负值梯度)
self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

# ELU (指数线性单元，平滑过渡)
self.elu = nn.ELU(alpha=1.0)

# GELU (高斯误差线性单元，Transformer常用)
self.gelu = nn.GELU()

# Swish/SiLU (自门控激活函数)
self.swish = nn.SiLU()
```

**工作原理**: 选择导数不会饱和的激活函数，确保梯度能有效地反向传播。

**选择指南**:
- 一般情况 → ReLU/Leaky ReLU
- 需要平滑过渡 → ELU/SELU
- Transformer架构 → GELU
- 性能优先 → Swish/Mish

#### 2. 残差连接与长短连接
```python
# 残差连接
output = input + layers(input)

# 门控跳跃连接(Highway Networks)
class HighwayLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.H = nn.Linear(dim, dim)
        self.T = nn.Linear(dim, dim)
        
    def forward(self, x):
        H = F.relu(self.H(x))
        T = torch.sigmoid(self.T(x))
        return H * T + x * (1 - T)
```

**工作原理**: 通过跳跃连接，建立浅层和深层之间的直接通道，使梯度能够有效流动。

**选择指南**:
- 深度CNN → 标准残差连接
- 深度RNN → 门控跳跃连接
- 需要选择性特征 → Highway Networks

#### 3. 权重初始化与归一化
```python
# LSUV初始化(Layer-Sequential Unit-Variance)
def lsuv_init(model, data):
    # 实现需自定义，算法步骤：
    # 1. 正交初始化所有层
    # 2. 逐层调整权重比例使输出单位方差
    pass

# 谱归一化(Spectral Normalization)
def spectral_norm_conv(in_c, out_c, kernel_size):
    conv = nn.Conv2d(in_c, out_c, kernel_size)
    return nn.utils.spectral_norm(conv)
```

**工作原理**: 特殊的初始化和归一化方法可以控制梯度的尺度，防止梯度消失。

**应用场景**:
- LSUV适用于深度网络的稳定训练
- 谱归一化常用于GAN训练稳定

#### 4. 注意力和门控机制
```python
# LSTM门控单元(解决RNN梯度消失)
self.lstm = nn.LSTM(input_size, hidden_size)

# GRU门控单元(LSTM的简化版本)
self.gru = nn.GRU(input_size, hidden_size)

# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / q.size(-1)**0.5, dim=-1)
        return attn @ v
```

**工作原理**: 通过门控或注意力机制，选择性地传递信息，建立长距离依赖，避免梯度消失。

**选择指南**:
- 序列数据 → LSTM/GRU
- 需要并行计算 → 自注意力机制
- 超长序列 → Transformer架构

## 方案对比
| 解决方案 | 实现复杂度 | 计算开销 | 适用场景 | 优势 | 局限性 |
|----------|------------|----------|----------|------|--------|
| 梯度裁剪/缩放 | 低 | 低 | 通用，特别是RNN | 实现简单，广泛适用 | 可能妨碍收敛 |
| 合适初始化 | 低 | 低 | 所有深度网络 | 防患于未然，效果显著 | 仅影响初期训练 |
| 归一化技术 | 中 | 中 | CNN/RNN/Transformer | 稳定训练，加速收敛 | 增加计算复杂度 |
| 激活函数优化 | 低 | 低 | 几乎所有网络 | 简单有效 | 可能引入新问题 |
| 残差/跳跃连接 | 中 | 低 | 深层CNN/Transformer | 结构简单，效果显著 | 增加网络复杂性 |
| 门控机制 | 高 | 高 | 序列模型/RNN | 有效建立长距离依赖 | 训练复杂，计算量大 |
| 注意力机制 | 高 | 高 | 序列数据/图像 | 强大的表征能力 | 显存占用大 |

## 实际应用建议

1. **多措施并用**: 没有单一方法能解决所有梯度问题，通常需要结合多种技术

2. **分析问题**:
   - 训练发散/NaN → 可能是梯度爆炸，尝试梯度裁剪
   - 训练停滞/缓慢 → 可能是梯度消失，检查激活函数或添加残差连接

3. **网络架构选择**:
   - CNN → 残差连接 + 批归一化 + ReLU变体
   - RNN → LSTM/GRU + 梯度裁剪 + 层归一化
   - Transformer → 多头注意力 + 残差连接 + 层归一化 + GELU

4. **调参顺序**:
   - 先优化架构（激活函数、残差连接等）
   - 再调整优化器设置（学习率、梯度裁剪等）
   - 最后微调超参数