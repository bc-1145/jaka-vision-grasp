@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."
set "ROUTE_CHECK_ONLY="
for %%A in (%*) do (
  if /I "%%~A"=="--check" set "ROUTE_CHECK_ONLY=1"
  if /I "%%~A"=="--help" set "ROUTE_CHECK_ONLY=1"
  if /I "%%~A"=="-h" set "ROUTE_CHECK_ONLY=1"
)
if exist ".camera-runtime\python.exe" (
  ".camera-runtime\python.exe" -X utf8 -u one_click_loading.py --pose loading --then liumang %*
) else (
  python -X utf8 -u one_click_loading.py --pose loading --then liumang %*
)
if errorlevel 1 (
  echo.
  echo COMBINED ROUTE FAILED. Check the message above.
  pause
  exit /b 1
)
echo.
echo COMBINED ROUTE OK.
if defined ROUTE_CHECK_ONLY (
  echo CHECK ONLY. Gripper release skipped.
  exit /b 0
)
call "%~dp0松开夹具.bat"
if errorlevel 1 (
  echo FINAL GRIPPER RELEASE FAILED.
  exit /b 1
)
echo COMBINED ROUTE AND GRIPPER RELEASE OK.
timeout /t 3 >nul
exit /b 0
