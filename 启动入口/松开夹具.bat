@echo off
setlocal
cd /d "%~dp0.."
if exist ".runtime\python.exe" (
  ".runtime\python.exe" -X utf8 -u gripper_command.py --open %*
) else (
  python -X utf8 -u gripper_command.py --open %*
)
if errorlevel 1 (
  echo GRIPPER OPEN FAILED. Check the message above.
  pause
  exit /b 1
)
echo GRIPPER OPEN COMMAND OK.
exit /b 0
