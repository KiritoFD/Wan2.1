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
                    if cache_x.shape[2] < 2 and feat_cache[#检查时间维度<2（头尾发生），且缓存不为空
                            idx] is not None and feat_cache[idx] != 'Rep':#非Rep标记，即已经缓存了有效特征图，不是初始                       
                        cache_x = torch.cat([#从之前的块中缓存最后一帧特征图
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(#增加一个时间维度，cache_x.shape[2] = 1
                                cache_x.device), cache_x#将最后一帧特征图拼接到cache_x中
                        ],
                                            dim=2)#在时间维度上拼接
                    if cache_x.shape[2] < 2 and feat_cache[#初始状态
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
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
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
        return x + h


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
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
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
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
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
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

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

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
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
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
