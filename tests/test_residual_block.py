import torch
import pytest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wan.modules.vae import ResidualBlock, CausalConv3d, RMS_norm


class TestResidualBlock:
    """测试 ResidualBlock 类的功能"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        # 设置随机种子以确保结果可重现
        torch.manual_seed(42)
        
        # 定义测试参数
        self.batch_size = 2
        self.in_channels = 16
        self.out_channels = 32
        self.time_frames = 4
        self.height = 8
        self.width = 8
        self.dropout_rate = 0.1
        
        # 创建一个 ResidualBlock 实例
        self.res_block = ResidualBlock(
            in_dim=self.in_channels, 
            out_dim=self.out_channels, 
            dropout=self.dropout_rate
        )
        
        # 创建一个输入张量
        self.input_tensor = torch.randn(
            self.batch_size, 
            self.in_channels, 
            self.time_frames, 
            self.height, 
            self.width
        )

    def test_init(self):
        """测试 ResidualBlock 的初始化"""
        # 验证属性是否正确设置
        assert self.res_block.in_dim == self.in_channels
        assert self.res_block.out_dim == self.out_channels
        
        # 验证残差序列的组成
        residual_layers = list(self.res_block.residual)
        assert len(residual_layers) == 6  # RMS_norm, SiLU, Conv3d, RMS_norm, SiLU, Dropout, Conv3d
        
        # 验证第一个归一化层
        assert isinstance(residual_layers[0], RMS_norm)
        assert residual_layers[0].gamma.shape[0] == self.in_channels
        
        # 验证第一个卷积层
        assert isinstance(residual_layers[2], CausalConv3d)
        assert residual_layers[2].in_channels == self.in_channels
        assert residual_layers[2].out_channels == self.out_channels
        
        # 验证第二个归一化层
        assert isinstance(residual_layers[3], RMS_norm)
        assert residual_layers[3].gamma.shape[0] == self.out_channels
        
        # 验证 dropout 层
        assert isinstance(residual_layers[5], torch.nn.Dropout)
        assert residual_layers[5].p == self.dropout_rate
        
        # 验证第二个卷积层
        assert isinstance(residual_layers[6], CausalConv3d)
        assert residual_layers[6].in_channels == self.out_channels
        assert residual_layers[6].out_channels == self.out_channels
        
        # 验证 shortcut 连接
        assert isinstance(self.res_block.shortcut, CausalConv3d)
        assert self.res_block.shortcut.in_channels == self.in_channels
        assert self.res_block.shortcut.out_channels == self.out_channels
        assert self.res_block.shortcut.kernel_size == (1, 1, 1)  # 1x1 卷积

    def test_forward_shape(self):
        """测试前向传播的输出形状"""
        output = self.res_block(self.input_tensor)
        
        # 验证输出形状
        expected_shape = (
            self.batch_size, 
            self.out_channels,  # 变为输出通道数
            self.time_frames, 
            self.height, 
            self.width
        )
        assert output.shape == expected_shape
        
        # 验证输出与输入的数据类型相同
        assert output.dtype == self.input_tensor.dtype
        
        # 验证输出与输入的设备相同
        assert output.device == self.input_tensor.device

    def test_identity_shortcut(self):
        """测试输入输出通道数相同时的恒等映射"""
        # 创建输入输出通道数相同的 ResidualBlock
        identity_res_block = ResidualBlock(
            in_dim=self.in_channels, 
            out_dim=self.in_channels,  # 相同的输入和输出通道
            dropout=0.0  # 禁用 dropout 以便比较
        )
        
        # 验证 shortcut 是否为 Identity
        assert isinstance(identity_res_block.shortcut, torch.nn.Identity)
        
        # 测试前向传播
        with torch.no_grad():
            # 初始化权重为接近零的值，使残差部分对输出的贡献很小
            for name, param in identity_res_block.residual.named_parameters():
                if 'weight' in name:
                    param.data.fill_(1e-6)
                if 'bias' in name:
                    param.data.fill_(0)
        
        output = identity_res_block(self.input_tensor)
        
        # 由于残差部分很小，输出应该非常接近输入
        # 我们允许较大的容差，因为 RMS 归一化和 SiLU 激活仍会引入变化
        assert torch.allclose(output, self.input_tensor, rtol=1e-1, atol=1e-1)

    def test_residual_connection(self):
        """测试残差连接的有效性"""
        # 将残差部分置零，以便测试 shortcut 是否正常工作
        with torch.no_grad():
            for layer in self.res_block.residual:
                if isinstance(layer, CausalConv3d):
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()
        
        # shortcut 依然有效
        output = self.res_block(self.input_tensor)
        
        # 验证输出形状
        assert output.shape == (
            self.batch_size, 
            self.out_channels, 
            self.time_frames, 
            self.height, 
            self.width
        )
        
        # 输出应该只来自 shortcut 连接
        shortcut_output = self.res_block.shortcut(self.input_tensor)
        assert torch.allclose(output, shortcut_output, rtol=1e-5, atol=1e-5)

    def test_caching_mechanism(self):
        """测试特征缓存机制"""
        # 创建特征缓存和索引
        feat_cache = [None] * 10  # 10 个 None 元素的列表
        feat_idx = [0]  # 初始索引为 0
        
        # 将时间帧减少到 1，以测试特征缓存
        small_input = torch.randn(
            self.batch_size,
            self.in_channels,
            1,  # 只有一帧
            self.height,
            self.width
        )
        
        # 第一次前向传播
        output1 = self.res_block(small_input, feat_cache, feat_idx)
        
        # 验证索引是否增加
        assert feat_idx[0] == 2  # 两个 CausalConv3d 层应该增加两次索引
        
        # 验证缓存是否被填充
        assert feat_cache[0] is not None
        assert feat_cache[1] is not None
        
        # 第二次前向传播，使用不同的输入
        small_input2 = torch.randn_like(small_input)
        output2 = self.res_block(small_input2, feat_cache, feat_idx)
        
        # 验证索引是否再次增加
        assert feat_idx[0] == 4
        
        # 验证新的缓存是否被填充
        assert feat_cache[2] is not None
        assert feat_cache[3] is not None
        
        # 重置索引并使用相同的缓存再次进行前向传播
        feat_idx[0] = 0
        output3 = self.res_block(small_input, feat_cache, feat_idx)
        
        # 验证输出是否与第一次不同（因为现在使用了不同的缓存）
        assert not torch.allclose(output1, output3, rtol=1e-5, atol=1e-5)

    def test_dropout_effect(self):
        """测试 dropout 的效果"""
        # 创建一个高 dropout 率的残差块，以增强效果
        high_dropout_res_block = ResidualBlock(
            in_dim=self.in_channels,
            out_dim=self.out_channels,
            dropout=0.9  # 高 dropout 率
        )
        
        # 设置为训练模式
        high_dropout_res_block.train()
        
        # 前向传播两次
        output1 = high_dropout_res_block(self.input_tensor)
        output2 = high_dropout_res_block(self.input_tensor)
        
        # 验证两次输出不同，因为 dropout 随机屏蔽不同的神经元
        assert not torch.allclose(output1, output2, rtol=1e-3, atol=1e-3)
        
        # 设置为评估模式
        high_dropout_res_block.eval()
        
        # 前向传播两次
        output3 = high_dropout_res_block(self.input_tensor)
        output4 = high_dropout_res_block(self.input_tensor)
        
        # 验证两次输出相同，因为评估模式下 dropout 被禁用
        assert torch.allclose(output3, output4, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_gpu_compatibility(self):
        """测试 GPU 兼容性"""
        device = torch.device("cuda")
        
        # 将模型和输入移动到 GPU
        res_block_gpu = self.res_block.to(device)
        input_tensor_gpu = self.input_tensor.to(device)
        
        # 前向传播
        output_gpu = res_block_gpu(input_tensor_gpu)
        
        # 验证输出是否在 GPU 上
        assert output_gpu.device.type == "cuda"
        assert output_gpu.shape == (
            self.batch_size,
            self.out_channels,
            self.time_frames, 
            self.height, 
            self.width
        )


if __name__ == "__main__":
    # 直接运行测试
    pytest.main(["-xvs", __file__])