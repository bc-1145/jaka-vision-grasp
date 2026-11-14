@echo off
cd /d "%~dp0"
python -u arm_control.py home
if errorlevel 1 (
  echo.
  echo HOME FAILED. Check the error above.
  pause
  exit /b 1
)
echo.
echo HOME OK.
timeout /t 3 >nul
