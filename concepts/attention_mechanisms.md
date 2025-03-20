# 注意力机制（Attention Mechanism）

**核心概念**: 注意力机制允许模型动态地关注输入数据中的不同部分，为输入的不同元素分配权重，突出重要信息。

## 多种注意力机制实现

### 1. 缩放点积注意力 (Scaled Dot-Product Attention)
```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: 查询，键，值张量
    mask: 可选遮罩张量
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights
```

**工作原理**: 计算查询和键的点积来度量相似度，除以键维度的平方根进行缩放，然后通过softmax得到注意力权重，最后用这些权重对值进行加权求和。

**使用场景**: Transformer架构的基础，适用于需要建立长距离依赖的任务。

### 2. 多头注意力 (Multi-Head Attention)
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性变换并分头
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        
        # 重整形状并线性变换
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.out_linear(out)
        
        return out
```

**工作原理**: 将输入分成多个"头"，每个头独立地执行注意力操作，然后将结果合并，允许模型同时关注不同表示子空间的信息。

**优势**: 增强模型的表示能力，能够捕获不同类型的关系和模式。

### 3. 加性/连接注意力 (Additive/Concat Attention)
```python
class AdditiveAttention(nn.Module):
    def __init__(self, query_dim, key_dim, attn_dim):
        super().__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=False)
        self.key_layer = nn.Linear(key_dim, attn_dim, bias=False)
        self.energy_layer = nn.Linear(attn_dim, 1, bias=False)
        
    def forward(self, query, keys, values, mask=None):
        # query: [batch, query_dim]
        # keys: [batch, seq_len, key_dim]
        # values: [batch, seq_len, value_dim]
        
        query = self.query_layer(query).unsqueeze(1)  # [batch, 1, attn_dim]
        keys = self.key_layer(keys)  # [batch, seq_len, attn_dim]
        
        # 加性注意力计算
        energy = torch.tanh(query + keys)
        energy = self.energy_layer(energy).squeeze(-1)  # [batch, seq_len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(energy, dim=-1).unsqueeze(1)  # [batch, 1, seq_len]
        output = torch.bmm(attn_weights, values)  # [batch, 1, value_dim]
        
        return output.squeeze(1), attn_weights.squeeze(1)
```

**工作原理**: 使用一个前馈网络计算注意力权重，通过将查询和键映射到相同的空间，然后使用一个非线性激活函数和一个线性层计算权重。

**优势**: 在某些情况下比点积注意力有更强的表达能力，特别是当查询和键的维度不同时。

### 4. 线性注意力 (Linear Attention)
```python
class LinearAttention(nn.Module):
    """线性复杂度的注意力，避免O(n²)计算"""
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # 特征映射为线性注意力
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # 线性注意力计算
        kv = torch.einsum('bnd,bne->bde', k, v)
        context = torch.einsum('bnd,bde->bne', q, kv)
        out = self.to_out(context)
        
        return out
```

**工作原理**: 通过特殊的特征变换，将二次方的计算复杂度降为线性，实现更高效的注意力计算。

**使用场景**: 长序列数据处理，如长文本、高分辨率图像或视频序列，内存和计算资源有限的情况。

### 5. 局部注意力 (Local Attention)
```python
class LocalAttention(nn.Module):
    def __init__(self, dim, window_size=7):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        # 只关注局部窗口
        pad_size = self.window_size // 2
        k_padded = F.pad(k, (0, 0, pad_size, pad_size))
        v_padded = F.pad(v, (0, 0, pad_size, pad_size))
        
        attn_sum = torch.zeros_like(q)
        attn_weight_sum = torch.zeros(b, n, 1, device=x.device)
        
        for i in range(self.window_size):
            k_shifted = k_padded[:, i:i+n]
            v_shifted = v_padded[:, i:i+n]
            
            attn_weights = (q * k_shifted).sum(dim=-1, keepdim=True)
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            attn_sum += attn_weights * v_shifted
            attn_weight_sum += attn_weights
            
        out = attn_sum / (attn_weight_sum + 1e-8)
        out = self.to_out(out)
        
        return out
```

**工作原理**: 每个位置只关注其周围固定大小窗口内的元素，而不是全局注意力，显著减少计算复杂度。

**使用场景**: 图像、语音等具有局部相关性的数据，或作为全局注意力的辅助机制。

## 注意力机制变体

### 1. 自注意力 (Self-Attention)
每个元素都从同一组元素中获取注意力权重，用于捕获序列内部的关系。

```python
# 简化的自注意力实现
def self_attention(x):
    # x: [batch_size, seq_len, d_model]
    q = k = v = x
    attn, _ = scaled_dot_product_attention(q, k, v)
    return attn
```

### 2. 交叉注意力 (Cross-Attention)
查询来自一个序列，键和值来自另一个序列，用于序列之间的信息交互。

```python
# 简化的交叉注意力实现
def cross_attention(q, kv):
    # q: [batch_size, q_len, d_model]
    # kv: [batch_size, kv_len, d_model]
    k = v = kv
    attn, _ = scaled_dot_product_attention(q, k, v)
    return attn
```

### 3. 因果注意力 (Causal Attention)
自回归模型中使用的变体，每个位置只能关注自身及之前的位置，防止信息泄露。

```python
def causal_attention(x):
    # x: [batch_size, seq_len, d_model]
    seq_len = x.size(1)
    q = k = v = x
    
    # 创建因果掩码 (下三角矩阵)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    
    attn, _ = scaled_dot_product_attention(q, k, v, mask=mask)
    return attn
```

## 注意力机制对比
| 类型 | 计算复杂度 | 内存消耗 | 优势 | 适用场景 |
|------|------------|----------|------|----------|
| 点积注意力 | O(n²) | 高 | 实现简单，易于并行 | 短序列，足够内存 |
| 多头注意力 | O(n²) | 高 | 多角度特征提取 | Transformer模型 |
| 加性注意力 | O(n²) | 中 | 表达能力强 | RNN/LSTM集成 |
| 线性注意力 | O(n) | 低 | 长序列高效 | 长序列处理，内存受限 |
| 局部注意力 | O(n*w) | 低 | 聚焦局部特征 | 图像，语音，有局部性任务 |
| 自注意力 | O(n²) | 高 | 捕获序列内关系 | 序列编码 |
| 交叉注意力 | O(n*m) | 中 | 序列间交互 | 机器翻译，多模态 |
| 因果注意力 | O(n²) | 高 | 自回归生成 | 文本生成，语言模型 |

## 注意力应用场景

### 自然语言处理
- **机器翻译**: 源语言和目标语言之间的对齐
- **文本摘要**: 识别文本中的关键信息
- **问答系统**: 将问题与上下文中的相关部分联系起来
- **情感分析**: 识别文本中表达情感的关键词/短语

### 计算机视觉
- **图像分类**: 聚焦关键区域提高分类准确度
- **目标检测**: 精确定位多个目标
- **图像生成**: 控制生成过程中的细节关注
- **视频理解**: 捕获时空依赖关系

### 多模态任务
- **图像描述**: 将图像区域与文本描述对齐
- **视觉问答**: 根据问题关注图像的相关部分
- **跨模态检索**: 学习不同模态数据间的关联

### 其他领域
- **推荐系统**: 用户-物品交互建模
- **图神经网络**: 节点间关系建模
- **强化学习**: 关注状态空间中的重要部分

## 实现注意事项

1. **计算效率**:
   - 注意力矩阵通常很大(O(n²))，考虑分块计算或稀疏化
   - 对于长序列，考虑线性复杂度的变体

2. **内存优化**:
   - 梯度检查点技术(gradient checkpointing)
   - 混合精度训练(FP16/BF16)
   - 注意力蒸馏(Attention Distillation)

3. **实践技巧**:
   - 多头注意力时，头的数量通常为8或16
   - 注意力后通常跟一个残差连接和层归一化
   - 对于生成任务，通常使用因果注意力掩码