@echo off
cd /d "%~dp0.."
"%~dp0..\.camera-runtime\python.exe" -X utf8 -u "%~dp0..\color_depth_viewer.py" --live-model %*
if errorlevel 1 pause
