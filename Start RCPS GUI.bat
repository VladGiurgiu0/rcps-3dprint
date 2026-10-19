@echo off
setlocal EnableExtensions
rem Double-click launcher for rcps-gui (Windows).
rem macOS twin: "Start RCPS GUI.command". Keep the two in sync.
rem   1. uses .\.venv if present, otherwise the conda env named below
rem   2. first run: offers to create the env and install rcps
rem   3. starts the GUI with this folder as the project and opens the browser
rem Close this window (or press Ctrl+C) to stop the GUI.

set "ENV_NAME=rcps"
set "PORT=8421"

cd /d "%~dp0"
title rcps-gui
set "URL=http://127.0.0.1:%PORT%/"

rem Already running? Just reopen the browser tab.
curl.exe -s -o NUL --max-time 1 "%URL%" >NUL 2>&1
if not errorlevel 1 (
  echo rcps-gui is already running: %URL%
  start "" "%URL%"
  exit /b 0
)

set "ENV_DESC="
call :find_conda

if exist ".venv\Scripts\python.exe" (
  call ".venv\Scripts\activate.bat"
  set "ENV_DESC=.venv in this folder"
  goto check_packages
)
if not defined CONDA_BASE goto first_run
call "%CONDA_BASE%\Scripts\activate.bat" "%CONDA_BASE%"
call conda activate %ENV_NAME% >NUL 2>&1
if /i not "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" goto first_run
set "ENV_DESC=conda env '%ENV_NAME%'"
goto check_packages

:first_run
echo No Python environment for rcps found (looked for .venv and conda env '%ENV_NAME%').
choice /c YN /m "Create one now and install rcps (needs internet)"
if errorlevel 2 goto fail
if not defined CONDA_BASE goto make_venv
call conda create -y -n %ENV_NAME% -c conda-forge --override-channels python=3.11
if errorlevel 1 goto fail
call conda activate %ENV_NAME%
if /i not "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" goto fail
set "ENV_DESC=conda env '%ENV_NAME%'"
goto install

:make_venv
set "PYLAUNCH="
where py >NUL 2>&1 && set "PYLAUNCH=py -3"
if not defined PYLAUNCH (
  echo Python not found. Install Miniforge from https://conda-forge.org/download/
  echo and double-click this file again.
  goto fail
)
%PYLAUNCH% -m venv .venv
if errorlevel 1 goto fail
call ".venv\Scripts\activate.bat"
set "ENV_DESC=.venv in this folder"
goto install

:check_packages
python -c "import numpy, scipy, trimesh, yaml, iso2mesh, pymeshfix, rcps_gui.server" >NUL 2>&1
if not errorlevel 1 goto run
echo Some rcps packages are missing in %ENV_DESC%.
choice /c YN /m "Install them now (needs internet)"
if errorlevel 2 goto fail

:install
echo Installing rcps and its dependencies (a few minutes) ...
python -m pip install -e ".[skimage]"
if errorlevel 1 goto fail

:run
echo.
echo Starting rcps-gui  [%ENV_DESC%]
echo   project folder: %CD%
echo   Keep this window open while you use the GUI; close it to stop.
echo.
python -m rcps_gui.server --project . --port %PORT%
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
echo rcps-gui did not start - see the messages above.
pause
exit /b 1

rem ---- locate a conda install (Miniforge / Miniconda / Anaconda) ----
:find_conda
set "CONDA_BASE="
if defined CONDA_EXE if exist "%CONDA_EXE%" for %%B in ("%CONDA_EXE%\..\..") do set "CONDA_BASE=%%~fB"
if not defined CONDA_BASE for /f "delims=" %%C in ('where conda 2^>NUL') do if not defined CONDA_BASE for %%B in ("%%~dpC..") do set "CONDA_BASE=%%~fB"
if defined CONDA_BASE if not exist "%CONDA_BASE%\Scripts\activate.bat" set "CONDA_BASE="
for %%D in ("%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniforge3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\miniforge3" "C:\miniconda3" "C:\anaconda3") do (
  if not defined CONDA_BASE if exist "%%~D\Scripts\activate.bat" set "CONDA_BASE=%%~D"
)
exit /b 0
