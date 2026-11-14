@echo off
setlocal
cd /d "%~dp0.."
if exist ".camera-runtime\python.exe" (
  ".camera-runtime\python.exe" -X utf8 -u one_click_loading.py --pose card_position %*
) else (
  python -X utf8 -u one_click_loading.py --pose card_position %*
)
if errorlevel 1 (
  echo CARD POSITION FAILED. Check the message above.
  pause
  exit /b 1
)
echo CARD POSITION OK.
exit /b 0
