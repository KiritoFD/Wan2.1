import subprocess
import os
from tqdm import tqdm

# 定义Python解释器路径和脚本路径
python_exe = "C:/Users/xy/miniconda3/envs/torch/python.exe"
script_path = "c:/GitHub/Wan2.1/scripts/image_to_vae_latent.py"

# 循环处理style_1到style_50
for i in tqdm(range(1, 51), desc="处理样式"):
    style_folder = f"./stylizations/style_{i}"
    
    # 构建命令
    cmd = [python_exe, script_path, "--image_path", style_folder]
    
    # 执行命令
    print(f"正在处理 {style_folder}...")
    subprocess.run(cmd, check=True)
    print(f"完成处理 {style_folder}")

print("所有样式都已处理完成！")
