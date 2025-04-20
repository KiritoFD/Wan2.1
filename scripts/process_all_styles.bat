@echo off
setlocal enabledelayedexpansion

set PYTHON_EXE=C:/Users/xy/miniconda3/envs/torch/python.exe
set SCRIPT_PATH=c:/GitHub/Wan2.1/scripts/image_to_vae_latent.py

for /l %%i in (1, 1, 50) do (
    echo 正在处理 style_%%i...
    %PYTHON_EXE% %SCRIPT_PATH% --image_path .\stylizations\style_%%i
    echo 完成处理 style_%%i
)

echo 所有样式都已处理完成！
pause
