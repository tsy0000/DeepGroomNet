@echo off
REM DeepGroomNet_v2 一键建环境（新建，不动原项目）
REM 用法：双击本文件，或 cmd 中执行
setlocal
set CONDA=D:\Anaconda3\Scripts\conda.exe
set ENVDIR=D:\DeepGroomNet_v2\env

if not exist "%CONDA%" (
  echo [ERROR] conda not found at %CONDA%. Please fix the path in setup_env.bat
  pause & exit /b 1
)

echo [1/3] Creating conda env at %ENVDIR% (python 3.10) ...
"%CONDA%" create -p "%ENVDIR%" python=3.10 -y
if errorlevel 1 ( echo [ERROR] env create failed & pause & exit /b 1 )

echo [2/3] Installing torch 2.1 cu118 ...
"%CONDA%" install -p "%ENVDIR%" pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
if errorlevel 1 ( echo [ERROR] torch install failed & pause & exit /b 1 )

echo [3/3] Installing pip requirements ...
"%ENVDIR%\python.exe" -m pip install -r D:\DeepGroomNet_v2\requirements.txt
if errorlevel 1 ( echo [ERROR] pip install failed & pause & exit /b 1 )

echo.
echo ============================================
echo   Done. Activate with:
echo     call "%ENVDIR%\Scripts\activate.bat"
echo   Then run:
echo     python scripts/01_build_metadata.py
echo     python scripts/02_gen_clips.py
echo     python scripts/03_train.py
echo     python scripts/04_eval.py
echo ============================================
pause
