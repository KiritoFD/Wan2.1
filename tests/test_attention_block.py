import torch
import pytest
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from wan.modules.vae import AttentionBlock, RMS_norm


class TestAttentionBlock:
    """测试 AttentionBlock 类的功能"""

    def setup_method(self):
        """每个测试方法执行前的设置"""
        # 设置随机种子以确保结果可重现
        torch.manual_seed(42)
        self.batch_size = 2
        self.channels = 16
        self.time_frames = 4
        self.height = 8
        self.width = 8
        
        # 创建一个 AttentionBlock 实例
        self.attn_block = AttentionBlock(dim=self.channels)
        
        # 创建测试输入
        self.input_tensor = torch.randn(
            self.batch_size, 
            self.channels, 
            self.time_frames, 
            self.height, 
            self.width
        )

    def test_init(self):
        """测试 AttentionBlock 的初始化"""
        # 验证 dim 属性
        assert self.attn_block.dim == self.channels
        
        # 验证各层是否正确初始化
        assert isinstance(self.attn_block.norm, RMS_norm)
        assert isinstance(self.attn_block.to_qkv, torch.nn.Conv2d)
        assert isinstance(self.attn_block.proj, torch.nn.Conv2d)
        
        # 验证 to_qkv 层的参数
        assert self.attn_block.to_qkv.in_channels == self.channels
        assert self.attn_block.to_qkv.out_channels == self.channels * 3
        assert self.attn_block.to_qkv.kernel_size == (1, 1)
        
        # 验证 proj 层的参数
        assert self.attn_block.proj.in_channels == self.channels
        assert self.attn_block.proj.out_channels == self.channels
        assert self.attn_block.proj.kernel_size == (1, 1)
        
        # 验证 proj 权重是否初始化为零
        assert torch.all(self.attn_block.proj.weight == 0)

    def test_forward_shape(self):
        """测试前向传播的输出形状"""
        output = self.attn_block(self.input_tensor)
        
        # 验证输出形状与输入形状相同
        assert output.shape == self.input_tensor.shape
        
        # 验证输出与输入数据类型相同
        assert output.dtype == self.input_tensor.dtype
        
        # 验证输出与输入设备相同
        assert output.device == self.input_tensor.device

    def test_forward_residual(self):
        """测试残差连接功能"""
        # 设置 proj 权重为 0，这样注意力的输出应该为 0
        # 只有残差连接的输出
        with torch.no_grad():
            self.attn_block.proj.weight.zero_()
            self.attn_block.proj.bias.zero_()
        
        output = self.attn_block(self.input_tensor)
        
        # 由于 proj 层被设置为 0，除了浮点误差外，输出应接近输入
        assert torch.allclose(output, self.input_tensor, rtol=1e-5, atol=1e-5)

    def test_forward_computation(self):
        """测试注意力机制的计算过程"""
        # 使用可预测的输入
        test_input = torch.ones(
            self.batch_size, 
            self.channels, 
            self.time_frames, 
            self.height, 
            self.width
        )
        
        # 设置 proj 层的权重和偏置为非零值
        with torch.no_grad():
            self.attn_block.proj.weight.fill_(0.01)
            self.attn_block.proj.bias.fill_(0.01)
        
        output = self.attn_block(test_input)
        
        # 验证输出不等于输入（因为投影层现在有非零权重）
        assert not torch.allclose(output, test_input)
        
        # 输出应该是输入加上注意力的结果
        # 由于注意力机制的复杂性，我们只检查输出的值是否有变化
        assert output.min() != test_input.min() or output.max() != test_input.max()

    def test_forward_batch_independence(self):
        """测试批次之间的独立性"""
        # 创建两个批次，第二个批次的数据与第一个不同
        input1 = torch.randn(1, self.channels, self.time_frames, self.height, self.width)
        input2 = torch.randn(1, self.channels, self.time_frames, self.height, self.width)
        
        # 分别计算输出
        output1 = self.attn_block(input1)
        output2 = self.attn_block(input2)
        
        # 合并输入并计算输出
        combined_input = torch.cat([input1, input2], dim=0)
        combined_output = self.attn_block(combined_input)
        
        # 验证合并后的计算结果与分别计算的结果相同
        assert torch.allclose(output1, combined_output[0:1], rtol=1e-5, atol=1e-5)
        assert torch.allclose(output2, combined_output[1:2], rtol=1e-5, atol=1e-5)

    def test_spatial_attention(self):
        """测试空间维度上的注意力机制"""
        # 创建一个特殊的输入，其中特定位置有明显特征
        input_tensor = torch.zeros(
            1, self.channels, 1, self.height, self.width
        )
        # 在中心位置设置高值
        center_h, center_w = self.height // 2, self.width // 2
        input_tensor[0, :, 0, center_h, center_w] = 10.0
        
        # 使用自定义的 AttentionBlock 处理
        with torch.no_grad():
            # 初始化投影层权重为非零值，以便观察注意力效果
            self.attn_block.to_qkv.weight.fill_(0.1)
            self.attn_block.proj.weight.fill_(0.1)
        
        output = self.attn_block(input_tensor)
        
        # 由于中心位置的特征很强，注意力机制应该会关注这个位置
        # 我们期望输出在中心位置附近的值比远离中心的位置更高
        # 注意：这个测试假设注意力机制正常工作，可能不够严格
        center_region = output[0, :, 0, 
                         center_h-1:center_h+2, 
                         center_w-1:center_w+2].mean()
        corner_region = output[0, :, 0, :2, :2].mean()
        
        assert center_region > corner_region

    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_gpu_compatibility(self):
        """测试在 GPU 上的兼容性"""
        # 跳过如果没有可用的 GPU
        device = torch.device("cuda")
        
        # 将模型和输入移到 GPU
        attn_block_gpu = self.attn_block.to(device)
        input_tensor_gpu = self.input_tensor.to(device)
        
        # 在 GPU 上执行前向传播
        output_gpu = attn_block_gpu(input_tensor_gpu)
        
        # 验证输出在 GPU 上
        assert output_gpu.device.type == "cuda"
        assert output_gpu.shape == input_tensor_gpu.shape


if __name__ == "__main__":
    # 直接运行测试
    pytest.main(["-xvs", __file__])