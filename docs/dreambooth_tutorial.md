# Wan2.1 T2V-1.3B DreamBooth 微调教程

本教程将指导您如何使用类似DreamBooth的方法对Wan2.1的T2V-1.3B模型进行个性化微调，使其能够学习特定的概念，比如特定人物、特定风格等。

## 准备工作

1. 安装Wan2.1及其依赖
2. 下载预训练的T2V-1.3B模型
3. 准备数据集：收集3-10个与您想要教会模型的概念相关的短视频

## 数据集准备

创建一个目录来存放您的训练视频。确保所有视频:
- 格式为MP4、AVI或MOV
- 每个视频长度为2-5秒
- 分辨率建议为480x832或其他支持的分辨率
- 内容清晰，背景简单

例如，如果要微调一个特定人物，那么准备这个人物在不同场景、不同角度的短视频。

```
training_videos/
  ├── video1.mp4
  ├── video2.mp4
  ├── video3.mp4
  └── ...
```

## 微调步骤

### 1. 运行微调脚本

使用以下命令开始微调：

```bash
python scripts/finetune_t2v_dreambooth.py \
  --ckpt_dir path/to/model/cache \
  --video_dir path/to/training_videos \
  --output_dir dreambooth_output \
  --prompt_template "a [unique_token] person" \
  --unique_token "sks" \
  --num_epochs 100 \
  --learning_rate 1e-5 \
  --test_prompt "a [unique_token] person walking on a beach"
```

参数说明：

- `--ckpt_dir`: 预训练模型的路径
- `--video_dir`: 训练视频目录
- `--output_dir`: 输出目录，保存微调后的模型
- `--prompt_template`: 训练用的提示模板，使用[unique_token]作为占位符
- `--unique_token`: 用于表示特定概念的唯一标记（例如"sks"）
- `--num_epochs`: 训练轮次
- `--learning_rate`: 学习率
- `--test_prompt`: 用于测试生成的提示词

### 2. 监控训练进度

微调过程中，脚本会显示每个epoch的损失值。训练过程将持续保存检查点，您可以随时停止并继续训练。

### 3. 测试微调后的模型

训练结束后，会在输出目录中生成最终模型`dreambooth_final.pt`和测试视频`test_generation.mp4`。

## 使用微调后的模型

您可以使用我们提供的Gradio界面来使用微调后的模型生成视频：

```bash
python gradio/t2v_1.3B_dreambooth.py \
  --ckpt_dir path/to/model/cache \
  --dreambooth_model dreambooth_output/dreambooth_final.pt
```

在界面中，您可以使用`[token]`在提示词中引用您微调的概念，例如："[token] is dancing in a field of flowers"，系统会自动将其替换为您的唯一标记。

## 最佳实践

1. **数据质量**：确保训练视频质量高、内容清晰
2. **训练时间**：通常需要50-100个epoch才能达到好的效果
3. **提示词选择**：对于人物，使用"a [unique_token] person"；对于风格，使用"a video in [unique_token] style"
4. **学习率**：从较小的学习率开始，如1e-5，视情况调整
5. **过拟合**：如果生成的视频过于接近训练集，可以减少训练轮次或增加正则化

## 示例案例

假设您想微调模型以生成特定人物的视频：

1. 收集该人物的5-10个短视频片段
2. 使用唯一标记"john_doe"
3. 运行微调脚本：

```bash
python scripts/finetune_t2v_dreambooth.py \
  --video_dir john_doe_videos \
  --prompt_template "a [unique_token] person" \
  --unique_token "john_doe" \
  --num_epochs 80
```

4. 微调后，使用如下提示词生成视频：
   - "a john_doe person skiing in the Alps"
   - "a close-up of john_doe smiling"

## 注意事项

- 模型微调可能需要较长时间，建议在GPU上运行
- 保存好微调后的模型，以便日后使用
- 尊重版权和肖像权，不要未经许可使用他人图像
- 微调模型可能会继承训练数据中的偏见和特性
