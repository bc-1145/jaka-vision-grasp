@echo off
cd /d "%~dp0.."
"%~dp0..\.camera-runtime\python.exe" -u "%~dp0..\pick_place_3d.py" run --visualize
if errorlevel 1 pause
