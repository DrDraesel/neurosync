@echo off
cd /d "%~dp0"
"C:\Users\GAMEPOWER\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" -u main.py
if errorlevel 1 pause
