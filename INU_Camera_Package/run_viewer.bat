@echo off
rem Launch the INU viewer using the DLLs bundled with this package.
setlocal
if not exist "%~dp0bin\inu_viewer.exe" (
    echo [x] Missing bin\inu_viewer.exe
    pause
    exit /b 1
)
set "PATH=%~dp0bin;%PATH%"
start "" /D "%~dp0" "%~dp0bin\inu_viewer.exe"
endlocal
