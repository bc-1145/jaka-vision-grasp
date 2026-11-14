@echo off
setlocal
cd /d "%~dp0.."
if exist ".camera-runtime\python.exe" (
  ".camera-runtime\python.exe" -X utf8 -u one_click_loading.py
) else (
  python -X utf8 -u one_click_loading.py
)
if errorlevel 1 (
  echo.
  echo LOADING FAILED. Check the message above.
  pause
  exit /b 1
)
echo.
echo LOADING OK.
timeout /t 3 >nul
