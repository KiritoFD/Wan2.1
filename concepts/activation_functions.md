# 激活函数（Activation Functions）

激活函数是神经网络中的非线性变换，它们使网络能够学习复杂的模式和表示。不同的激活函数具有不同的特性和适用场景。

## 常见激活函数及其特性

### 1. Sigmoid 函数
```python
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# PyTorch实现
sigmoid = nn.Sigmoid()
```

**特点**:
- 将输入映射到(0,1)区间
- 在历史上曾广泛使用
- 存在梯度消失问题
- 输出不以零为中心

**优缺点**:
- ✓ 输出可解释为概率
- ✓ 平滑且处处可导
- ✗ 存在梯度饱和问题
- ✗ 计算开销较大(指数运算)
- ✗ 不以零为中心，影响权重更新

### 2. Tanh 函数(双曲正切)
```python
def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# PyTorch实现
tanh = nn.Tanh()
```

**特点**:
- 将输入映射到(-1,1)区间
- 输出以零为中心
- 在深度较浅的网络中效果好于Sigmoid

**优缺点**:
- ✓ 输出以零为中心，有助于权重更新
- ✓ 梯度较Sigmoid强
- ✗ 仍然存在梯度饱和问题
- ✗ 计算开销大

### 3. ReLU (Rectified Linear Unit)
```python
def relu(x):
    return torch.maximum(torch.tensor(0.0), x)

# PyTorch实现
relu = nn.ReLU()
```

**特点**:
- f(x) = max(0, x)
- 简单高效
- 大多数现代神经网络的默认选择

**优缺点**:
- ✓ 计算简单高效
- ✓ 缓解梯度消失问题
- ✓ 引入稀疏性，减少过拟合
- ✗ 死神经元问题(负输入梯度为零)
- ✗ 输出不以零为中心

### 4. Leaky ReLU
```python
def leaky_relu(x, alpha=0.01):
    return torch.maximum(torch.tensor(0.0), x) + alpha * torch.minimum(torch.tensor(0.0), x)

# PyTorch实现
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
```

**特点**:
- f(x) = max(0, x) + α * min(0, x)
- α是一个小正数(通常0.01)
- 解决ReLU死神经元问题

**优缺点**:
- ✓ 解决ReLU死神经元问题
- ✓ 计算依然高效
- ✓ 负输入仍有梯度传播
- ✗ α参数需要手动设置
- ✗ 理论上可能存在梯度爆炸

### 5. PReLU (Parametric ReLU)
```python
class PReLU(nn.Module):
    def __init__(self, num_parameters=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([0.25] * num_parameters))
        
    def forward(self, x):
        return torch.maximum(torch.tensor(0.0), x) + self.alpha * torch.minimum(torch.tensor(0.0), x)

# PyTorch实现
prelu = nn.PReLU()
```

**特点**:
- Leaky ReLU的改进版，α参数可学习
- 每个特征或每个通道可以有独立的α

**优缺点**:
- ✓ 自适应负区间斜率
- ✓ 比Leaky ReLU表达能力更强
- ✗ 增加了模型参数
- ✗ 训练可能不稳定

### 6. ELU (Exponential Linear Unit)
```python
def elu(x, alpha=1.0):
    return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

# PyTorch实现
elu = nn.ELU(alpha=1.0)
```

**特点**:
- f(x) = x if x > 0 else α * (e^x - 1)
- 产生平滑的输出，包括负值

**优缺点**:
- ✓ 输出平均值接近零，有助于权重更新
- ✓ 平滑的导数，减少震荡
- ✓ 对输入扰动更鲁棒
- ✗ 计算复杂度高
- ✗ α参数需要手动设置

### 7. SELU (Scaled Exponential Linear Unit)
```python
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))

# PyTorch实现
selu = nn.SELU()
```

**特点**:
- ELU的改进版，具有自归一化特性
- 在深度网络中自动保持激活的均值和方差

**优缺点**:
- ✓ 自归一化特性，不需要批归一化
- ✓ 有助于构建更深的网络
- ✓ 理论保证收敛性
- ✗ 需要特殊的权重初始化(LeCun Normal)
- ✗ 计算复杂度高

### 8. GELU (Gaussian Error Linear Unit)
```python
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# PyTorch实现
gelu = nn.GELU()
```

**特点**:
- 结合了高斯CDF的思想
- 在Transformer等模型中广泛使用

**优缺点**:
- ✓ 平滑过渡，有助于模型收敛
- ✓ 在现代Transformer架构中表现优异
- ✓ 组合了ReLU和dropout的效果
- ✗ 计算复杂度高
- ✗ 在轻量级网络中可能不是最佳选择

### 9. Swish/SiLU (Sigmoid Linear Unit)
```python
def swish(x, beta=1.0):
    return x * torch.sigmoid(beta * x)

# PyTorch实现
silu = nn.SiLU()  # 等价于Swish当beta=1时
```

**特点**:
- f(x) = x * sigmoid(βx)
- 谷歌研究的自门控激活函数
- β为可学习参数时又称为Swish-β

**优缺点**:
- ✓ 无上界，有下界
- ✓ 平滑且非单调
- ✓ 在深度网络中表现优于ReLU
- ✗ 计算成本高
- ✗ 梯度计算较复杂

### 10. Mish
```python
def mish(x):
    return x * torch.tanh(F.softplus(x))

# PyTorch实现
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
```

**特点**:
- f(x) = x * tanh(ln(1 + e^x))
- 无上界，有下界，平滑且非单调

**优缺点**:
- ✓ 几乎处处可导
- ✓ 在许多任务上表现优于Swish和ReLU
- ✓ 有助于信息流动
- ✗ 计算复杂度高
- ✗ 训练时内存消耗大

## 激活函数比较

### 数学特性比较
| 激活函数 | 值域 | 导数 | 单调性 | 零中心 |
|---------|------|------|--------|--------|
| Sigmoid | (0, 1) | (0, 0.25) | 单调 | 否 |
| Tanh | (-1, 1) | (0, 1) | 单调 | 是 |
| ReLU | [0, ∞) | {0, 1} | 单调 | 否 |
| Leaky ReLU | (-∞, ∞) | {α, 1} | 单调 | 否 |
| ELU | (-α, ∞) | (0, 1] | 单调 | 近似 |
| SELU | (-λα, ∞) | (λα, λ] | 单调 | 是(自归一化) |
| GELU | (-∞, ∞) | 连续 | 非单调 | 近似 |
| Swish/SiLU | (-∞, ∞) | 连续 | 非单调 | 近似 |
| Mish | (-∞, ∞) | 连续 | 非单调 | 近似 |

### 计算效率比较
| 激活函数 | 计算复杂度 | 内存消耗 | GPU加速优化 |
|---------|------------|----------|-------------|
| ReLU | 非常低 | 低 | 良好 |
| Leaky ReLU | 低 | 低 | 良好 |
| Sigmoid | 中 | 低 | 有专用实现 |
| Tanh | 中 | 低 | 有专用实现 |
| PReLU | 低 | 中 | 良好 |
| ELU | 中高 | 低 | 一般 |
| SELU | 中高 | 低 | 一般 |
| GELU | 高 | 中 | 有专用实现 |
| Swish/SiLU | 中高 | 中 | 一般 |
| Mish | 高 | 高 | 一般 |

## 激活函数选择指南

### 按网络类型选择
| 网络类型 | 推荐激活函数 | 理由 |
|---------|--------------|------|
| 深度CNN | ReLU/Leaky ReLU | 计算高效，避免梯度消失 |
| 浅层网络 | Tanh | 以零为中心，有助于收敛 |
| RNN/LSTM | Tanh/LSTM门 | 控制长期依赖，防止梯度爆炸 |
| Transformer | GELU | 平滑过渡，有利于注意力机制 |
| GAN | LeakyReLU | 避免生成器梯度稀疏 |
| 轻量级移动网络 | ReLU6/h-swish | 计算效率高，量化友好 |
| 超深网络 | SELU | 自归一化特性 |
| 图像分类 | ReLU/Swish | 历史验证/新研究 |
| 目标检测 | Swish/Mish | 平滑非单调，特征丰富 |

### 按问题类型选择
| 问题类型 | 推荐激活函数 | 理由 |
|---------|--------------|------|
| 分类问题输出层 | Sigmoid/Softmax | 输出可解释为概率 |
| 回归问题输出层 | 线性/Tanh | 输出范围适合问题 |
| 对抗训练/GAN | LeakyReLU | 避免梯度稀疏 |
| 自编码器 | SELU/ELU | 自归一化特性，保持表示 |
| 需要梯度平滑 | Swish/Mish | 平滑过渡，有助于优化 |
| 强化学习 | ReLU/ELU | 稳定性与表达力平衡 |
| 需要稀疏表示 | ReLU | 自然稀疏性 |

## 激活函数在深度学习发展中的演化

早期神经网络 → Sigmoid → Tanh → ReLU → Leaky ReLU → PReLU → ELU → SELU → GELU/Swish/Mish

每一代激活函数都试图解决前代的问题:
- **Sigmoid** → 解决非线性问题，但有梯度消失和非零中心问题
- **Tanh** → 解决零中心问题，但仍有梯度消失
- **ReLU** → 解决梯度消失，但有死神经元问题
- **Leaky ReLU/PReLU** → 解决死神经元问题
- **ELU/SELU** → 提供平滑过渡，更好的收敛性
- **GELU/Swish/Mish** → 非单调性，更丰富的特征表示

## 实现自定义激活函数

### PyTorch中定义自定义激活函数
```python
# 函数式定义
def custom_activation(x, alpha=0.1, beta=1.0):
    return x * torch.sigmoid(beta * x) + alpha * torch.relu(x)

# 模块式定义
class CustomActivation(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x) + self.alpha * F.relu(x)
    
    # 可选：如果需要导出到ONNX等格式
    def symbolic(self, g, x):
        return g.op("CustomActivation", x, 
                    alpha_f=self.alpha, 
                    beta_f=self.beta)
```

### 训练中动态调整激活函数
```python
class AdaptiveActivation(nn.Module):
    def __init__(self, dim, types=['relu', 'leaky_relu', 'elu']):
        super().__init__()
        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        })
        self.weights = nn.Parameter(torch.ones(len(types)) / len(types))
        self.types = types
        
    def forward(self, x):
        out = 0
        weights = F.softmax(self.weights, dim=0)
        for i, act_type in enumerate(self.types):
            out += weights[i] * self.activations[act_type](x)
        return out
```

## 选择合适激活函数的实践建议

1. **先从标准选择开始**:
   - 大多数情况下，**ReLU**是一个安全的默认选择
   - 对于Transformer架构，**GELU**是目前最常用的

2. **考虑计算资源**: 
   - 资源受限设备→简单激活函数(ReLU, Leaky ReLU)
   - 服务器/云计算→可考虑更复杂的激活函数(GELU, Mish)

3. **实验验证**:
   - 不同激活函数对性能的影响需要通过实验验证
   - 特别是在自定义网络架构中

4. **混合使用**:
   - 同一网络不同层可以使用不同的激活函数
   - 例如，浅层使用ReLU，深层使用ELU可能效果更好

5. **注意组合效应**:
   - 激活函数与其他技术(如批归一化、跳跃连接)有交互作用
   - 需要整体考虑而非孤立选择