import torch
import pytest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wan.modules.vae import CausalConv3d


class TestCausalConv3d:
    """测试 CausalConv3d 类的功能"""

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
        self.kernel_size = (3, 3, 3)  # (time, height, width)
        self.padding = (1, 1, 1)  # (time, height, width)
        
        # 创建一个 CausalConv3d 实例
        self.conv3d = CausalConv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding
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
        """测试 CausalConv3d 的初始化"""
        # 验证基本属性是否正确设置
        assert self.conv3d.in_channels == self.in_channels
        assert self.conv3d.out_channels == self.out_channels
        assert self.conv3d.kernel_size == self.kernel_size
        
        # 验证 padding 是否被重置为 (0, 0, 0)
        assert self.conv3d.padding == (0, 0, 0)
        
        # 验证 _padding 属性是否正确设置
        # _padding 格式为 (width_left, width_right, height_top, height_bottom, depth_past, depth_future)
        expected_padding = (
            self.padding[2],    # width_left
            self.padding[2],    # width_right
            self.padding[1],    # height_top
            self.padding[1],    # height_bottom
            2 * self.padding[0],  # depth_past - 注意这里是2倍
            0                    # depth_future - 因果卷积不看未来
        )
        assert self.conv3d._padding == expected_padding

    def test_forward_shape(self):
        """测试前向传播的输出形状"""
        output = self.conv3d(self.input_tensor)
        
        # 验证输出形状
        expected_shape = (
            self.batch_size, 
            self.out_channels,
            self.time_frames,  # 时间维度保持不变
            self.height,       # 高度保持不变
            self.width         # 宽度保持不变
        )
        assert output.shape == expected_shape
        
        # 验证输出与输入数据类型相同
        assert output.dtype == self.input_tensor.dtype
        
        # 验证输出与输入设备相同
        assert output.device == self.input_tensor.device

    def test_causality(self):
        """测试因果性：确保当前输出只依赖于过去和当前的输入"""
        # 创建两个相同的输入，但在最后一个时间步不同
        input1 = torch.randn(
            self.batch_size, 
            self.in_channels, 
            self.time_frames, 
            self.height, 
            self.width
        )
        
        input2 = input1.clone()
        input2[:, :, -1, :, :] = torch.randn_like(input2[:, :, -1, :, :])
        
        # 前向传播
        output1 = self.conv3d(input1)
        output2 = self.conv3d(input2)
        
        # 验证最后一个时间步的输出不同，但前面的时间步相同
        # 这证明了每个时间步的输出只依赖于当前和之前的输入
        assert torch.allclose(output1[:, :, :-1, :, :], output2[:, :, :-1, :, :], rtol=1e-5, atol=1e-5)
        assert not torch.allclose(output1[:, :, -1, :, :], output2[:, :, -1, :, :], rtol=1e-5, atol=1e-5)

    def test_caching_mechanism(self):
        """测试缓存机制的有效性"""
        # 测试无缓存情况下的输出
        output_no_cache = self.conv3d(self.input_tensor)
        
        # 测试有缓存但为第一次调用（缓存为 None）的情况
        # 将输入分成两部分
        input_part1 = self.input_tensor[:, :, :2, :, :]
        input_part2 = self.input_tensor[:, :, 2:, :, :]
        
        # 创建缓存
        cache = None
        
        # 第一次调用
        output_part1 = self.conv3d(input_part1, cache)
        
        # 更新缓存为第一部分的最后 CACHE_T 帧
        cache = input_part1[:, :, -2:, :, :].clone()
        
        # 第二次调用，使用缓存
        output_part2 = self.conv3d(input_part2, cache)
        
        # 合并结果
        output_with_cache = torch.cat([output_part1, output_part2], dim=2)
        
        # 验证有缓存和无缓存的输出在第二部分接近相同
        # 注意：由于边界效应，我们预计会有一些差异，尤其是在输出的前几帧
        # 所以我们使用较松的容差
        assert torch.allclose(
            output_no_cache[:, :, 2:, :, :], 
            output_with_cache[:, :, 2:, :, :], 
            rtol=1e-2, atol=1e-2
        )

    def test_padding_application(self):
        """测试填充是否正确应用"""
        # 创建 1x1 卷积（无填充）来测试内部填充功能
        conv_no_pad = CausalConv3d(
            self.in_channels, 
            self.out_channels, 
            kernel_size=1, 
            padding=0
        )
        
        # 如果内部填充正确应用，输出形状应与输入相同
        output = conv_no_pad(self.input_tensor)
        assert output.shape == (
            self.batch_size,
            self.out_channels,
            self.time_frames,
            self.height,
            self.width
        )
        
        # 测试较大内核大小的填充
        conv_large_kernel = CausalConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(5, 5, 5),
            padding=(2, 2, 2)
        )
        
        output_large = conv_large_kernel(self.input_tensor)
        assert output_large.shape == (
            self.batch_size,
            self.out_channels,
            self.time_frames,
            self.height,
            self.width
        )

    def test_cache_with_multiple_frames(self):
        """测试使用多帧缓存的情况"""
        # 创建一个短序列输入
        input_short = torch.randn(
            self.batch_size,
            self.in_channels,
            2,  # 只有两帧
            self.height,
            self.width
        )
        
        # 创建一个缓存，包含两帧
        cache = torch.randn(
            self.batch_size,
            self.in_channels,
            2,  # 两帧缓存
            self.height,
            self.width
        )
        
        # 使用缓存进行前向传播
        output_with_cache = self.conv3d(input_short, cache)
        
        # 将缓存和输入连接起来，进行常规前向传播
        combined_input = torch.cat([cache, input_short], dim=2)
        output_combined = self.conv3d(combined_input)
        
        # 验证使用缓存的输出与连接输入的输出在后两帧相同
        # 因为只有输入的部分是我们感兴趣的
        assert torch.allclose(
            output_with_cache, 
            output_combined[:, :, -2:, :, :],
            rtol=1e-5, atol=1e-5
        )

    def test_stride_dilation(self):
        """测试步长和膨胀参数的效果"""
        # 创建具有步长的卷积
        conv_stride = CausalConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            padding=1
        )
        
        output_stride = conv_stride(self.input_tensor)
        
        # 验证步长为 2 的情况下输出维度减半
        assert output_stride.shape == (
            self.batch_size,
            self.out_channels,
            self.time_frames // 2,
            self.height // 2,
            self.width // 2
        )
        
        # 创建具有膨胀的卷积
        conv_dilation = CausalConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        
        output_dilation = conv_dilation(self.input_tensor)
        
        # 验证尽管使用膨胀，输出维度仍保持不变（因为适当地增加了填充）
        assert output_dilation.shape == (
            self.batch_size,
            self.out_channels,
            self.time_frames,
            self.height,
            self.width
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_gpu_compatibility(self):
        """测试 GPU 兼容性"""
        device = torch.device("cuda")
        
        # 将模型和输入移动到 GPU
        conv3d_gpu = self.conv3d.to(device)
        input_tensor_gpu = self.input_tensor.to(device)
        
        # 前向传播
        output_gpu = conv3d_gpu(input_tensor_gpu)
        
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