@echo off
rem Fully automatic run: setup, then notebooks 01-05 on the GPU. Double-click and leave it.
rem * each notebook is retried once if it fails; 02-04 do not block each other
rem * a notebook whose logs\done_XX.txt says "ok" is skipped, so re-running only redoes what failed
rem * the PC is kept awake while this window runs
rem Summary: logs\pipeline.log   Details: logs\setup.log, logs\run_XX.log
setlocal EnableDelayedExpansion
cd /d "%~dp0\..\.."
if not exist logs mkdir logs
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_XET=1"
set "HF_HUB_DOWNLOAD_TIMEOUT=60"
echo running> logs\RUNNING
echo [%date% %time%] PIPELINE START >> logs\pipeline.log
echo Running everything. This takes about 2 hours. Progress: %CD%\logs\pipeline.log

echo [%date% %time%] setup started >> logs\setup.log
call "%~dp0setup_core.bat" >> logs\setup.log 2>&1
if errorlevel 1 (
  echo [%date% %time%] SETUP FAILED - see logs\setup.log >> logs\pipeline.log
  goto :end
)
echo [%date% %time%] setup OK >> logs\pipeline.log
start "" /b python "%~dp0keep_awake.py" logs\RUNNING
echo [%date% %time%] downloading models >> logs\pipeline.log
python "%~dp0..\prefetch_models.py" > logs\download.log 2>&1
if errorlevel 1 ( echo [%date% %time%] model download had failures - see logs\download.log >> logs\pipeline.log ) else ( echo [%date% %time%] models OK >> logs\pipeline.log )

call :run_nb 01
if errorlevel 1 goto :end
call :run_nb 02
call :run_nb 03
call :run_nb 04
call :run_nb 05

:end
del logs\RUNNING 2>nul
echo [%date% %time%] PIPELINE END >> logs\pipeline.log
echo Finished. See logs\pipeline.log
pause
exit /b

:run_nb
set "N=%~1"
set "NBF="
for %%F in (notebooks\%N%_*.ipynb) do set "NBF=%%F"
set "MARK="
if exist logs\done_%N%.txt set /p MARK=<logs\done_%N%.txt
rem only 01 and 02 are skipped by marker; 03-05 always run and skip finished work themselves
if /i "!MARK!"=="ok" if %N% LEQ 02 (
  echo [!date! !time!] skip !NBF! - already done >> logs\pipeline.log
  exit /b 0
)
for /l %%A in (1,1,2) do (
  echo [!date! !time!] start !NBF! attempt %%A >> logs\pipeline.log
  echo Running !NBF! attempt %%A ...
  jupyter nbconvert --to notebook --execute --inplace "!NBF!" --ExecutePreprocessor.timeout=10800 --ExecutePreprocessor.kernel_name=pi > logs\run_%N%.log 2>&1
  set "RC=!errorlevel!"
  set "WROTE="
  findstr /c:"Writing" logs\run_%N%.log >nul && set "WROTE=1"
  if "!RC!"=="0" if defined WROTE (
    echo ok> logs\done_%N%.txt
    echo [!date! !time!] OK !NBF! >> logs\pipeline.log
    exit /b 0
  )
  copy /y logs\run_%N%.log logs\run_%N%_attempt%%A.log >nul
)
echo failed> logs\done_%N%.txt
echo [!date! !time!] FAILED !NBF! - see logs\run_%N%.log >> logs\pipeline.log
exit /b 1
