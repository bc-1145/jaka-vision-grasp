@echo off
cd /d "%~dp0"
echo One-click vision pick: raise first, observe, pick shaft, place at Home.
echo.
.camera-runtime\python.exe -X utf8 -u one_click_pick.py --manual-verify
if errorlevel 1 (
  echo.
  echo PICK FAILED. Robot has been stopped. Check the message above.
  pause
  exit /b 1
)
echo.
echo PICK OK.
timeout /t 3 >nul
