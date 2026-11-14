@echo off
setlocal
cd /d "%~dp0.."
if exist ".runtime\python.exe" (
  ".runtime\python.exe" -X utf8 -u gripper_command.py --mm 8.88 %*
) else (
  python -X utf8 -u gripper_command.py --mm 8.88 %*
)
if errorlevel 1 (
  echo GRIPPER 8.88 MM FAILED. Check the message above.
  pause
  exit /b 1
)
echo GRIPPER 8.88 MM COMMAND OK.
exit /b 0
