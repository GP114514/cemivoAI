@echo off
chcp 65001 >nul
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0训练.ps1"
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%
