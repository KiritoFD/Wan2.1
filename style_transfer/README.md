# 风格转换模型

本模块实现了基于对抗性自编码器（AAE）的VAE潜在空间风格转换模型，用于在不同风格的VAE编码特征之间进行映射。

## 功能介绍

该模型能够学习两种不同风格的VAE编码特征之间的映射关系，使得一种风格的特征可以转换为另一种风格。主要应用场景包括：

- 图像风格转换
- 跨域迁移学习
- 无监督风格适配

## 数据格式

输入和输出数据均为通过`image_to_vae_latent.py`生成的VAE编码特征文件(.pt)，特征形状为`[N, 16, 1, 32, 32]`，其中：
- N: 样本数量
- 16: VAE潜在空间维度
- 1: 时间维度
- 32x32: 空间维度

## 使用指南

### 环境依赖

```bash
pip install torch torchvision tqdm matplotlib
```

### 训练风格转换模型

```bash
python style_transfer/train.py \
    --style_a /path/to/style_a.pt \
    --style_b /path/to/style_b.pt \
    --output_dir models/style_transfer \
    --batch_size 8 \
    --epochs 100 \
    --device cuda
```

参数说明：
- `--style_a`: 风格A的VAE编码特征文件
- `--style_b`: 风格B的VAE编码特征文件
- `--output_dir`: 模型输出目录
- `--batch_size`: 训练批次大小
- `--latent_dim`: 潜在空间维度，默认为128
- `--epochs`: 训练轮数
- `--valid_split`: 验证集比例，默认为0.1
- `--device`: 训练设备，'cuda'或'cpu'
- `--seed`: 随机种子，默认为42

### 使用模型进行风格转换

```bash
python style_transfer/inference.py \
    --model models/style_transfer/checkpoints/style_transfer_model.pth \
    --input /path/to/input.pt \
    --output /path/to/output.pt \
    --device cuda
```

参数说明：
- `--model`: 训练好的模型文件路径
- `--input`: 待转换的VAE编码特征文件
- `--output`: 输出结果文件路径
- `--batch_size`: 批次大小，默认为16
- `--device`: 推理设备，'cuda'或'cpu'
- `--no_squeeze_time`: 不去除时间维度（如果原始数据格式需要）

## 技术实现

### 模型结构

该风格转换模型基于对抗性自编码器（AAE），包含三个主要组件：

1. **编码器**：将VAE潜在向量映射到新的潜在空间
2. **解码器**：将新的潜在空间映射回VAE潜在向量
3. **判别器**：区分潜在空间中的不同风格分布

### 训练过程

模型训练包括三个步骤：
1. 重建训练：编码器和解码器学习重建输入数据
2. 对抗训练：判别器学习区分两种风格的潜在表示
3. 生成训练：编码器学习生成能够欺骗判别器的潜在表示

### 损失函数

模型使用以下损失函数：
- 重建损失：均方误差损失（MSE）
- 对抗损失：二元交叉熵损失（BCE）
