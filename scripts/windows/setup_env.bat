@echo off
rem One-time setup: conda env "pi", PyTorch with CUDA, requirements, Jupyter kernel. Log: logs\setup.log
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
set "PYTHONUTF8=1"
echo Setting up the environment (10-20 minutes). Log: %CD%\logs\setup.log
echo [%date% %time%] setup started > logs\setup.log
call "%~dp0setup_core.bat" >> logs\setup.log 2>&1
set RC=%errorlevel%
echo [%date% %time%] SETUP FINISHED, exit code %RC% >> logs\setup.log
echo Finished with exit code %RC%. You can close this window.
pause
