@echo off
cd /d "%~dp0.."
echo Home - loading/bin - color/depth vision - open - shaft grip 8.65mm - Home.
echo.
.camera-runtime\python.exe -X utf8 -u one_click_pick.py %*
if errorlevel 1 (
  echo.
  echo PICK FAILED. Robot has been stopped. Check the message above.
  pause
  exit /b 1
)
echo.
echo SEQUENCE COMPLETE. At Home with gripper closed; verify pickup in saved images.
timeout /t 3 >nul
