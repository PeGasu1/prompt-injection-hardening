@echo off
rem Run notebooks in order and save their outputs in place.
rem Usage: run_notebooks.bat [01 02 ...]   (no argument = all five)
rem Log: logs\run.log (summary) and logs\run_XX.log (details per notebook).
setlocal EnableDelayedExpansion
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=60"
call "%~dp0find_conda.bat"
if not defined CONDA_BAT ( echo ERROR: could not find conda >> logs\run.log & exit /b 1 )
call "%CONDA_BAT%" activate pi || ( echo ERROR: env pi missing, run setup_env.bat first >> logs\run.log & exit /b 1 )
set "STEPS=%*"
if "%STEPS%"=="" set "STEPS=01 02 03 04 05"
echo [%date% %time%] RUN START: %STEPS% >> logs\run.log
for %%N in (%STEPS%) do (
  for %%F in (notebooks\%%N_*.ipynb) do (
    echo Running %%F ...
    echo [!date! !time!] start %%F >> logs\run.log
    jupyter nbconvert --to notebook --execute --inplace "%%F" --ExecutePreprocessor.timeout=10800 --ExecutePreprocessor.kernel_name=pi > logs\run_%%N.log 2>&1
    if errorlevel 1 (
      echo [!date! !time!] FAILED %%F - see logs\run_%%N.log >> logs\run.log
      echo FAILED %%F - see logs\run_%%N.log
      goto :end
    )
    echo [!date! !time!] OK %%F >> logs\run.log
  )
)
:end
echo [%date% %time%] RUN END >> logs\run.log
echo Done. You can close this window.
pause
