import torch
import argparse
import os
import numpy as np

def convert_pt_to_txt(pt_file_path, txt_file_path=None):
    """
    将PyTorch模型文件(.pt)转换为文本文件(.txt)
    
    参数:
        pt_file_path: PyTorch模型文件路径
        txt_file_path: 输出文本文件路径，如果为None则自动生成
    """
    # 检查输入文件是否存在
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"找不到文件: {pt_file_path}")
    
    # 如果没有指定输出路径，则自动生成
    if txt_file_path is None:
        txt_file_path = os.path.splitext(pt_file_path)[0] + ".txt"
    
    print(f"正在加载模型: {pt_file_path}")
    # 加载模型
    try:
        model_data = torch.load(pt_file_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return
    
    print(f"正在将模型写入文本文件: {txt_file_path}")
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        # 如果是 OrderedDict 或 dict (常见的模型状态格式)
        if isinstance(model_data, dict) or hasattr(model_data, 'items'):
            f.write("模型参数:\n\n")
            for key, tensor in model_data.items():
                if isinstance(tensor, torch.Tensor):
                    f.write(f"参数: {key}\n")
                    f.write(f"形状: {tensor.shape}\n")
                    f.write(f"类型: {tensor.dtype}\n")
                    # 将张量转换为numpy数组并保存
                    tensor_np = tensor.detach().cpu().numpy()
                    f.write(f"值: {np.array2string(tensor_np, threshold=20, max_line_width=120)}\n\n")
                else:
                    f.write(f"参数: {key} (非张量类型: {type(tensor)})\n")
                    f.write(f"值: {str(tensor)}\n\n")
        else:
            f.write(f"模型类型: {type(model_data)}\n")
            f.write(str(model_data))
    
    print(f"转换完成: {txt_file_path}")
    return txt_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将PyTorch模型(.pt)转换为文本文件(.txt)')
    parser.add_argument('pt_file', type=str, help='PyTorch模型文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出文本文件路径')
    
    args = parser.parse_args()
    convert_pt_to_txt(args.pt_file, args.output)
