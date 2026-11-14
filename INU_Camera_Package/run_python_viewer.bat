@echo off
rem Requires 64-bit Python 3.14, numpy and opencv-python.
setlocal
set "PATH=%~dp0bin;%PATH%"
python "%~dp0color_depth_viewer.py" --sdk "%~dp0" %*
if errorlevel 1 pause
endlocal
