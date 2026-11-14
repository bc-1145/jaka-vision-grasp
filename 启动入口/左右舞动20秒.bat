@echo off
chcp 65001 >nul
call "%~dp0左右舞动1分钟.bat" %*
exit /b %errorlevel%
