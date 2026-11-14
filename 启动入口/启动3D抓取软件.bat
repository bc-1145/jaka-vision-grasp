@echo off
cd /d "%~dp0.."
start "" "%~dp0..\.runtime\pythonw.exe" "%~dp0..\pick_place_app.py"
