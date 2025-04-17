# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'WanVAE',
]

CACHE_T = 2
# 2帧，缓存的时间长度；在encode和decode中都用到；在encode中，cache_x是上一个block的输出；在decode中，cache_x是当前block的输入；


class CausalConv3d(nn.Conv3d):#数据预处理
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        #【2】：左右填充相同宽度；【1】：上下填充相同高度；【0】：前填充2倍，后不填（保持大小，不看到后面的信息）
        self.padding = (0, 0, 0)
        #padding的分量：(width_left, width_right, height_top, height_bottom, depth_past, depth_future(0))

    def forward(self, x, cache_x=None):#x是输入，cache_x是上一个block的输出
        padding = list(self._padding)#按照上面的方法进行填充
        if cache_x is not None and self._padding[4] > 0:#是否已经缓存；是否需要在时间维度上填充
            cache_x = cache_x.to(x.device)#移动到当前设备
            x = torch.cat([cache_x, x], dim=2)#在时间维度（dim=2)上拼接
            #torch.cat:拼接两个tensor，dim=2表示在时间维度上拼接；cache_x是上一个block的输出，x是当前block的输入
            #cache_x.shape[2]表示上一个block的输出的时间维度大小；x.shape[2]表示当前block的输入的时间维度大小
            #为什么要拼接：将之前的帧加入输入中，实现因果卷积
            padding[4] -= cache_x.shape[2]#由于拼接了cache_x，所以需要减少填充的数量，避免重复填充
        #填充前x分量：（batch_size, in_channels, depth, height, width）
        x = F.pad(x, padding)#在时间维度上填充
        #填充后x:(batch_size, in_channels, depth+padding[4](时间过去）+padding[5](时间填充),
        #  height+padding[2]（高度顶部填充）+padding[3](高度底部填充), width+padding[0]（宽度左侧填充）+padding[1]（宽度右侧填充)
        #F.pad:对输入进行填充，padding是填充的大小；x是当前block的输入
        return super().forward(x)#调用父类的forward方法进行卷积操作；super()是父类
        #继承自nn.Conv3d的forward方法，进行卷积操作；x是当前block的输入；用的卷积核是self.weight，偏置是self.bias
        #nn.Conv3d:三维卷积，输入是一个五维tensor，输出是一个五维tensor；输入的shape是(batch_size, in_channels, depth, height, width)，输出的shape是(batch_size, out_channels, depth_out, height_out, width_out)


class RMS_norm(nn.Module):
    #归一化
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)#图像为二维，视频为三维
        # channel_first:是否为通道优先（输入tensor的通道维度（channle分量）在前）；false表示通道在最后
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)#保证后面正确归一化；dim=channle的数量
        self.channel_first = channel_first
        self.scale = dim**0.5#归一化的尺度（RMS只做缩放，乘以sqrt(dim)保持均值和方差不变）
        self.gamma = nn.Parameter(torch.ones(shape))#初始化为1：#gamma（缩放因子）是一个可学习的参数，shape是(1,1)或者(1,1,1)，表示每个通道的缩放因子；
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.#nn.Parameter:可学习的参数；
        #bias（偏移量）是一个可学习的参数，shape是(1,1)或者(1,1,1)，表示每个通道的偏置；如果bias为false，则bias为0；

    def forward(self, x):
        return F.normalize(#F.normalize:对输入进行归一化操作；x是当前block的输入
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
            #dim根据channel是否在前决定在哪个维度归一；*scale(尺度)*gamma（缩放因子）+bias（偏移量）

class Upsample(nn.Upsample):#继承自torch.nn.Upsample（上采样，解码，低分辨率->高分辨率）

    def forward(self, x):
        return super().forward(x.float()).type_as(x)#修复 bfloat16 数据类型在最近邻插值中的支持问题。
# 使用 bfloat16 可以减少内存占用和计算时间，从而加速模型的训练和推理。
# 但是，bfloat16 在某些操作中可能存在精度问题，例如最近邻插值。
# 需要先将 bfloat16 张量转换为 float32，进行插值操作，然后再转换回 bfloat16

class Resample(nn.Module):#特征图的重采样

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),#在高度和宽度上进行上采样，scale_factor=(2., 2.)表示上采样倍数为2（输入放大2倍）
                nn.Conv2d(dim, dim // 2, 3, padding=1))#2D卷积层调整通道数，‘//2’：通道减少一半， 卷积核大小为3，padding保持大小不变（填充）
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))#同上
            self.time_conv = CausalConv3d(#因果3D卷积
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))#通道数x2,加强特征图表达能力；
                #卷积核大小为(3时间, 1高度, 1宽度)，padding=(1, 0, 0)表示在时间维度上填充1个像素，在高度和宽度上不填充
                #填充原因：卷积到边缘时，卷积核会超出边界，导致输出大小不一致；不填充有些数据不参与卷积，丢失
        elif mode == 'downsample2d':
            #下采样：减小分辨率，有max_pooling,average_pooling, strided_conv等方法
            #可以减少计算量，提取高级特征，增加平移不变性，减少过拟合
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),#在高度和宽度上填充1个像素
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))#2D卷积，步长为2，stride=(2, 2)表示在高度和宽度上步长为2（下采样）
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))#转3D卷积，步长为2，stride=(2, 1, 1)表示在时间维度上步长为2（下采样）

        else:
            self.resample = nn.Identity()#不进行任何操作，直接返回输入；nn.Identity()是一个占位符，表示不进行任何操作；

    def forward(self, x, feat_cache=None, feat_idx=[0]):#x是当前block的输入，feat_cache是上一个block的输出，feat_idx是当前block的索引
        #x.shape: (batch_size, in_channels, depth, height, width)
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:#如果没有缓存
                    feat_cache[idx] = 'Rep'#用Rep标记
                    feat_idx[0] += 1#移到下一个block
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()#提取最后CACHE_T帧的特征图，复制到cache_x中（创建副本）
                    if cache_x.shape[2] < 2 and feat_cache[#
                            idx] is not None and feat_cache[idx] != 'Rep':#非Rep标记，即已经缓存了有效特征图，不是初始                       
                        cache_x = torch.cat([#从之前的块中缓存最后一帧特征图
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(#增加一个时间维度，cache_x.shape[2] = 1
                                cache_x.device), cache_x#将最后一帧特征图拼接到cache_x中
                        ],
                                            dim=2)#在时间维度上拼接
                    if cache_x.shape[2] < 2 and feat_cache[#
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),#零填充
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)#如果没有缓存，直接进行时间卷积：不使用缓存的特征图（首帧，无前帧可用）
                    else:
                        x = self.time_conv(x, feat_cache[idx])#使用缓存的特征图进行时间卷积
                    feat_cache[idx] = cache_x#缓存当前块的输出
                    feat_idx[0] += 1#移到下一个block

                    x = x.reshape(b, 2, c, t, h, w)#将每个块的输出reshape为2个时间块，2个时间块分别是当前块的输出和上一个块的输出
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),#torch.stack:堆叠张量，将两个时间块在时间维度上拼接
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)#完成采样，时间维度扩大2倍，通道数不变；
        t = x.shape[2]#获取时间维度大小
        x = rearrange(x, 'b c t h w -> (b t) c h w')#批次大小与时间维度合并，使卷积作用于每个时间步；
        x = self.resample(x)#进行上采样或下采样操作；
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)##将输出x的维度调整为(b, c, t, h, w)，恢复原来的维度；

        if self.mode == 'downsample3d':
            if feat_cache is not None:#整体是否采开启特征缓存
                idx = feat_idx[0]
                if feat_cache[idx] is None:#当前位置是否有缓存的特征图
                    feat_cache[idx] = x.clone()#缓存当前块的输出
                    feat_idx[0] += 1#移到下一个block
                else:

                    cache_x = x[:, :, -1:, :, :].clone()#提取最后一帧的特征图，复制到cache_x中（创建副本）
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(#
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))#提取最后一帧的特征图，拼接到当前块的输出上
                    feat_cache[idx] = cache_x#缓存当前块的输出
                    feat_idx[0] += 1#移到下一个block
        return x

    def init_weight(self, conv):#初始化卷积层的权重
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)#初始化为0
        c1, c2, t, h, w = conv_weight.size()#获取卷积核的大小
        one_matrix = torch.eye(c1, c2)#单位矩阵，大小为(c1, c2)；torch.eye:创建一个单位矩阵，主对角线为1，其余元素为0；
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)#初始化为0
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5;
        conv.weight.data.copy_(conv_weight)#将初始化的卷积核复制到conv_weight中
        nn.init.zeros_(conv.bias.data)#初始化偏置为0

    def init_weight2(self, conv):#与上面的区别：将单位矩阵设置在时间维度的最后一帧上，输出通道分为两部分，分别对应于输入通道的前后两部分；
        conv_weight = conv.weight.data#
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):#残差块，将输入特征图直接添加到经过一系列层处理后的输出特征图上；
#残差的作用：通过跳过连接（skip connection）来缓解深度网络中的梯度消失和爆炸问题；
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim#输入通道数
        self.out_dim = out_dim#输出通道数

        # layers
        self.residual = nn.Sequential(#定义残差函数
            RMS_norm(in_dim, images=False), nn.SiLU(),#（上面的）RMS归一化+ SiLU激活函数
            CausalConv3d(in_dim, out_dim, 3, padding=1),#卷积层，卷积核大小为3，padding=1保持大小不变
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),#归一化+激活函数+dropout（防止过拟合）
            CausalConv3d(out_dim, out_dim, 3, padding=1))#卷积层，卷积核大小为3，padding=1保持大小不变
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()#残差连接；如果输入通道数和输出通道数不相等，则使用1x1卷积进行匹配；否则使用恒等映射（nn.Identity()）

    def forward(self, x, feat_cache=None, feat_idx=[0]):#残差块的前向传播
        h = self.shortcut(x)#shortcut连接；x是当前块的输入
        for layer in self.residual:#残差块的每一层
            if isinstance(layer, CausalConv3d) and feat_cache is not None:#是CausalConv3d层，且开启了特征缓存
                idx = feat_idx[0]#获取当前块的索引
                cache_x = x[:, :, -CACHE_T:, :, :].clone()##提取最后CACHE_T帧的特征图，复制到cache_x中（创建副本）
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:#检查时间维度<2（头尾发生）即是否要缓存，且缓存不为空
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(#提取最后一帧，增加一个时间维度，cache_x.shape[2] = 1；unsqueeze(2)表示在时间维度上增加一个维度
                            cache_x.device), cache_x#移动到当前设备
                    ],
                                        dim=2)#把cache_x和feat_cache[idx]拼接在一起，dim=2表示在时间维度上拼接
                x = layer(x, feat_cache[idx])##使用缓存的特征图进行卷积操作；layer是CausalConv3d层
                feat_cache[idx] = cache_x#缓存当前块的输出
                feat_idx[0] += 1#移到下一个block
            else:
                x = layer(x)#进行卷积操作；layer是RMS_norm、SiLU、Dropout等层
        return x + h#残差连接；将输入x和经过一系列层处理后的输出x相加；
#与其让网络学习一个复杂的映射 H(x)，不如让网络学习残差 F(x) = H(x) - x，然后将残差加到输入 x 上，得到最终的输出 H(x) = F(x) + x。
#h是x与预期输出之间的差异，x是输入；通过残差连接，网络可以更容易地学习到输入和输出之间的关系；
#h更小容易学习；直接梯度传播防止消失；容易学到恒等映射


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x #保留输入副本，用于残差链接
        b, c, t, h, w = x.size()#获取输入形状
        x = rearrange(x, 'b c t h w -> (b t) c h w')#批次和时间合并，每个事件步作为单独样本处理
        #不跨时间，保持因果性，看不到未来帧
        x = self.norm(x)#归一化提高训练稳定性
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,#单头注意力
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],#渐进式通道增长模式，减小特征图空间尺寸同时增加通道数
                 num_res_blocks=2,#可配置的残差块数量，便于调整网络深度
                 attn_scales=[],#可选择性地在特定尺度添加注意力，减少计算量
                 temperal_downsample=[True, True, False],#灵活配置时间维度下采样策略
                 dropout=0.0):# 改进点：可考虑在训练时使用更强的正则化策略如stochastic depth
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)# 亮点：使用因果卷积，保持时序因果性

        # downsample blocks，多级下采样压缩
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout)#残差连接设计
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))# 亮点：选择性注意力机制，平衡计算效率和表达能力
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'# 亮点：灵活的下采样策略，可单独控制时间维度
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks，用于最低分辨率处保护注意力捕获的信息
        self.middle = nn.Sequential(#三明治结构：节省注意力计算量，残差确保梯度避免爆炸/消失
            ResidualBlock(out_dim, out_dim, dropout),#第一个残差块，预处理特征，为注意力机制准备更好的特征分布
            AttentionBlock(out_dim),# 瓶颈处使用注意力机制捕获全局依赖关系，允许信息在整个特征图流动
            ResidualBlock(out_dim, out_dim, dropout)) # 提炼特征，融合局部和全局信息，残差保证梯度流动和特征保留
        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))#映射，维持时序因果

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    """
    WanVAE核心模型类，实现了视频VAE的编码器和解码器功能。
    
    亮点：
    1. 采用3D卷积网络处理视频数据，保持时空连续性
    2. 使用因果卷积确保实时推理能力，避免时序泄露
    3. 高效的时序分块处理机制，支持任意长度视频处理
    4. 缓存机制减少冗余计算，提高推理速度
    
    改进点：
    1. 可考虑添加条件控制机制，增强对视频内容的控制能力
    2. 引入更强的时序一致性约束，提高生成视频的连贯性
    3. 优化内存使用，降低长视频处理的显存占用
    """

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        """
        初始化视频VAE模型。
        
        参数:
            dim: 基础特征维度
            z_dim: 潜在空间维度
            dim_mult: 不同层级的通道倍增系数
            num_res_blocks: 每个尺度的残差块数量
            attn_scales: 应用注意力机制的尺度
            temperal_downsample: 时间维度下采样策略
            dropout: Dropout比率，用于正则化
        """
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        """
        模型前向传播，完成编码-采样-解码过程。
        
        亮点：完整的VAE流程，包含重参数化采样环节
        参数:
            x: 输入视频张量 [B,C,T,H,W]
        返回:
            x_recon: 重建视频
            mu: 均值
            log_var: 对数方差
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        """
        视频编码器，将视频映射到潜在空间。
        
        亮点：
        1. 高效分块处理机制，支持任意长度视频
        2. 时间维度上首帧单独处理，后续帧分组处理(1,4,4,4...)，优化时序建模
        3. 特征缓存机制避免重复计算
        
        参数:
            x: 输入视频 [B,C,T,H,W]
            scale: 潜在空间归一化参数 [mean, std]
        返回:
            mu: 归一化后的潜在表示
        """
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                # 首帧单独处理，确保稳定的起始状态
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                # 后续帧分组处理，每组4帧，利用时序连续性
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        # 分离均值和对数方差
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        # 执行归一化操作，使潜在空间分布更加稳定
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        """
        从潜在表示解码重建视频。
        
        亮点：
        1. 逐帧解码策略，保持因果性，适合流式处理
        2. 高效特征缓存机制，避免重复计算
        3. 解码器与编码器对称设计，有助于信息保留
        
        改进点：
        1. 可引入条件信息指导解码过程
        2. 考虑多尺度融合提升细节质量
        
        参数:
            z: 潜在表示 [B,C,T,H,W]
            scale: 潜在空间归一化参数 [mean, std]
        返回:
            out: 重建视频
        """
        self.clear_cache()
        # 反归一化处理
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        # 获取时间长度
        iter_ = z.shape[2]
        # 初始卷积处理
        x = self.conv2(z)
        # 逐帧解码，保持时序因果性
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                # 首帧解码
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                # 后续帧解码，利用特征缓存
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                # 拼接结果
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        """
        VAE重参数化技巧，在保持可微分的情况下进行采样。
        
        亮点：通过乘加操作优化采样过程，提高训练稳定性
        
        参数:
            mu: 均值
            log_var: 对数方差
        返回:
            采样得到的潜在表示
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        """
        从输入图像采样潜在表示。
        
        亮点：支持确定性模式和随机采样模式
        
        参数:
            imgs: 输入图像
            deterministic: 是否确定性采样(不加噪声)
        返回:
            采样得到的潜在表示
        """
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        """
        清除特征缓存，为新的处理准备环境。
        
        亮点：高效的缓存管理机制，减少内存占用
        """
        # 解码器缓存
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # 编码器缓存
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    构建并加载预训练的视频VAE模型。
    
    亮点：
    1. 使用元设备(meta device)初始化模型，减少内存占用
    2. 灵活的配置参数，支持不同规模和性能需求的模型
    3. 预训练权重加载机制，加速部署
    
    参数:
        pretrained_path: 预训练权重路径
        z_dim: 潜在空间维度
        device: 计算设备
        **kwargs: 其他模型参数
    返回:
        加载好权重的VAE模型
    """
    # 默认配置参数
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    # 更新自定义参数
    cfg.update(**kwargs)

    # 在meta设备上初始化模型，节省内存
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # 加载预训练权重
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:
    """
    WanVAE对外接口类，封装了视频编码和解码功能。
    
    亮点：
    1. 简化的API设计，易于集成到其他系统
    2. 自动混合精度支持，提高性能
    3. 预定义的归一化参数，确保编解码一致性
    
    改进点：
    1. 可增加批处理支持，提高吞吐量
    2. 考虑渐进式编解码，支持超高分辨率视频
    3. 添加更多预处理和后处理选项
    """

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        """
        初始化WanVAE接口。
        
        参数:
            z_dim: 潜在空间维度
            vae_pth: 预训练模型路径
            dtype: 计算精度类型
            device: 计算设备
        """
        self.dtype = dtype
        self.device = device

        # 预定义的潜在空间归一化参数
        # 这些参数是在大规模数据集上预计算的，确保潜在空间分布稳定
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        # 转换为张量并移至指定设备
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # 初始化底层模型并设置为评估模式
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        编码视频到潜在空间。
        
        亮点：
        1. 支持列表输入，方便批处理
        2. 自动混合精度计算，平衡速度和精度
        
        参数:
            videos: 视频列表，每个视频形状为[C,T,H,W]
        返回:
            潜在表示列表
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        """
        从潜在表示解码重建视频。
        
        亮点：
        1. 支持列表输入，方便批处理
        2. 自动值域裁剪(-1,1)，确保输出合法
        
        参数:
            zs: 潜在表示列表
        返回:
            重建视频列表
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
