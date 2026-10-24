@echo off
rem Start the interactive demo (opens http://127.0.0.1:7860 in your browser). Close this window to stop it.
cd /d "%~dp0\..\.."
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
call "%~dp0find_conda.bat"
if not defined CONDA_BAT ( echo ERROR: could not find conda & pause & exit /b 1 )
call "%CONDA_BAT%" activate pi || ( echo ERROR: env pi missing, run setup_env.bat first & pause & exit /b 1 )
python -c "import gradio" 2>nul || python -m pip install gradio==6.30.0
echo Starting the demo, the browser opens when the models are loaded (about 20 seconds)...
python app.py
pause
