@echo off
chcp 65001 >nul
powershell.exe -NoProfile -Command "Set-Content -LiteralPath '%~dp0neural\checkpoints\STOP' -Value 'pause' -Encoding ASCII"
set "RC=%ERRORLEVEL%"
echo 已请求暂停，将在当前训练步结束后保存模型。
if not "%RC%"=="0" echo [警告] 写入 STOP 失败，退出码 %RC%
pause
exit /b %RC%
