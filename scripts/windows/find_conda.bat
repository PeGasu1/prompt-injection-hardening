@echo off
rem Sets CONDA_BAT to the base conda.bat of the user's Anaconda/Miniconda install.
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\anaconda3" "D:\anaconda3" "D:\Anaconda" "C:\Anaconda3") do (
  if not defined CONDA_BAT if exist "%%~P\condabin\conda.bat" set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT if exist "%USERPROFILE%\.conda\environments.txt" (
  for /f "usebackq delims=" %%L in ("%USERPROFILE%\.conda\environments.txt") do (
    if not defined CONDA_BAT if exist "%%L\condabin\conda.bat" set "CONDA_BAT=%%L\condabin\conda.bat"
  )
)
if not defined CONDA_BAT for /f "delims=" %%C in ('where conda.bat 2^>nul') do if not defined CONDA_BAT set "CONDA_BAT=%%C"
