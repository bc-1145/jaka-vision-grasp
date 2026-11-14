@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "SWAY_START_MODE="
for %%A in (%*) do (
  if /I "%%~A"=="--check" set "SWAY_START_MODE=--check"
  if /I "%%~A"=="--plan" set "SWAY_START_MODE=--check"
  if /I "%%~A"=="--help" set "SWAY_START_MODE=--check"
  if /I "%%~A"=="-h" set "SWAY_START_MODE=--check"
)
if exist ".camera-runtime\python.exe" (
  ".camera-runtime\python.exe" -X utf8 -u one_click_loading.py --pose sway_start %SWAY_START_MODE%
) else (
  python -X utf8 -u one_click_loading.py --pose sway_start %SWAY_START_MODE%
)
if errorlevel 1 (
  echo SWAY START POSITION FAILED. Sway was cancelled.
  exit /b 1
)
if exist ".camera-runtime\python.exe" (
  ".camera-runtime\python.exe" -X utf8 -u horizontal_sway.py %*
) else (
  python -X utf8 -u horizontal_sway.py %*
)
if errorlevel 1 (
  echo SWAY FAILED OR STOPPED. Check the message above.
  pause
  exit /b 1
)
echo SWAY OK.
exit /b 0
