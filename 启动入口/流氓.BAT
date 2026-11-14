@echo off
setlocal
cd /d "%~dp0.."
if exist ".camera-runtime\python.exe" (
  ".camera-runtime\python.exe" -X utf8 -u one_click_loading.py --pose liumang %*
) else (
  python -X utf8 -u one_click_loading.py --pose liumang %*
)
if errorlevel 1 (
  echo.
  echo RETURN FAILED. Check the message above.
  pause
  exit /b 1
)
echo.
echo RETURN OK.
timeout /t 3 >nul
