@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run setup.cmd once first - it needs Python 3.11.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u main.py %*
if errorlevel 1 pause
