@echo off
chcp 65001 >nul
cd /d "%~dp0"
"%~dp0.camera-runtime\python.exe" -X utf8 -u "%~dp0auto_table_upright.py" --execute %*
set "table_exit_code=%errorlevel%"
echo.
echo 运行结束，退出码：%table_exit_code%
pause
exit /b %table_exit_code%
