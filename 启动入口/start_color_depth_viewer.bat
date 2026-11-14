@echo off
cd /d "%~dp0.."
"%~dp0..\.camera-runtime\python.exe" -u "%~dp0..\color_depth_viewer.py" %*
if errorlevel 1 pause
