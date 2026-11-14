@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "CARD_PYTHON=python"
if exist ".camera-runtime\python.exe" set "CARD_PYTHON=%~dp0..\.camera-runtime\python.exe"
set "CARD_CHECK_ONLY="
for %%A in (%*) do (
  if /I "%%~A"=="--check" set "CARD_CHECK_ONLY=1"
  if /I "%%~A"=="--help" set "CARD_CHECK_ONLY=1"
  if /I "%%~A"=="-h" set "CARD_CHECK_ONLY=1"
)

"%CARD_PYTHON%" -X utf8 -u one_click_loading.py --pose card_unloading %*
if errorlevel 1 goto failed

"%CARD_PYTHON%" -X utf8 -u gripper_command.py --open --settle-seconds 2 %*
if errorlevel 1 goto failed

if defined CARD_CHECK_ONLY (
  echo CHECK ONLY. Gripper wait and Home movement skipped.
  echo Next step: call 一键回Home.bat after the 2-second gripper wait.
  exit /b 0
)

call "%~dp0一键回Home.bat"
if errorlevel 1 goto failed

echo CARD UNLOADING AND HOME OK.
exit /b 0

:failed
echo CARD UNLOADING FAILED. Later steps were cancelled.
pause
exit /b 1
