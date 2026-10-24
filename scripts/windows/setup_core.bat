@echo off
rem Environment setup without pauses (called by setup_env.bat and run_everything.bat).
rem Leaves the "pi" environment activated in the calling window. No setlocal on purpose.
echo ---- GPU
nvidia-smi
call "%~dp0find_conda.bat"
if not defined CONDA_BAT ( echo ERROR: could not find conda & exit /b 1 )
echo ---- conda: %CONDA_BAT%
call "%CONDA_BAT%" env list | findstr /r /c:"^pi " >nul
if errorlevel 1 (
  echo ---- creating env pi
  call "%CONDA_BAT%" create -y -n pi --override-channels -c conda-forge python=3.13 || exit /b 1
) else (
  echo ---- env pi already exists
)
call "%CONDA_BAT%" activate pi || exit /b 1
python --version
python -m pip install --upgrade pip
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo ---- installing PyTorch with CUDA
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  if errorlevel 1 python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
) else (
  echo ---- PyTorch with CUDA already installed
)
echo ---- installing requirements
python -m pip install -r requirements.txt || exit /b 1
python -m ipykernel install --user --name pi --display-name "Python (pi)" || exit /b 1
echo ---- check
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU', '| transformers', transformers.__version__)"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || ( echo ERROR: PyTorch cannot see the GPU & exit /b 1 )
exit /b 0
