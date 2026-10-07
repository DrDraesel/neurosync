@echo off
setlocal
cd /d "%~dp0"
echo NeuroSync setup - creates the local .venv and installs requirements.
echo (One time per computer. Needs Python 3.11-3.12: https://www.python.org/downloads/)
echo.

if exist ".venv\.setup-complete" (
  echo Setup already completed here. Nothing to do - use "Start NeuroSync.cmd".
  pause
  exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul && py -3.11 -m venv .venv 2>nul
)
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul && py -3.12 -m venv .venv 2>nul
)
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul && py -3 -m venv .venv 2>nul
)
if not exist ".venv\Scripts\python.exe" (
  where python >nul 2>nul && python -m venv .venv 2>nul
)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Could not create the environment. Install Python 3.11 from
  echo https://www.python.org/downloads/  ^(check "Add python.exe to PATH"^) and run setup.cmd again.
  pause
  exit /b 1
)

echo Using:
".venv\Scripts\python.exe" --version
echo Environment ready. Installing requirements (this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency installation FAILED - see the messages above.
  pause
  exit /b 1
)
echo. > ".venv\.setup-complete"
echo.
echo Setup complete. Start the app with "Start NeuroSync.cmd".
pause
