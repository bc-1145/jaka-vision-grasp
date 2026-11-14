@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "CARD_PYTHON=python"
if exist ".camera-runtime\python.exe" set "CARD_PYTHON=%~dp0..\.camera-runtime\python.exe"

"%CARD_PYTHON%" -X utf8 -u gripper_command.py --open --settle-seconds 1 %*
if errorlevel 1 goto failed

"%CARD_PYTHON%" -X utf8 -u one_click_loading.py --pose card_loading_align --lift-first --descend-last %*
if errorlevel 1 goto failed

"%CARD_PYTHON%" -X utf8 -u gripper_command.py --mm 5.96 --settle-seconds 1 %*
if errorlevel 1 goto failed

call "%~dp0卡片定位.bat" %*
if errorlevel 1 goto failed

echo CARD LOADING SEQUENCE OK.
exit /b 0

:failed
echo CARD LOADING SEQUENCE FAILED. Later steps were cancelled.
pause
exit /b 1
