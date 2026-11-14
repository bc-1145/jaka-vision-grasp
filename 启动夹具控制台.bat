@echo off
cd /d "%~dp0"
if exist ".runtime\python.exe" (
    ".runtime\python.exe" gripper_app.py
    if errorlevel 1 pause
    exit /b
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 gripper_app.py
) else (
    python gripper_app.py
)
if errorlevel 1 pause
