@echo off
rem Build INU color/depth viewer (requires Visual Studio 2022 Build Tools)
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo [x] VS Build Tools not found & exit /b 1)

cl /nologo /std:c++17 /EHsc /O2 /MD /utf-8 ^
   /DUNICODE /D_UNICODE ^
   /I "C:\Program Files\Inuitive\InuDev\include" ^
   inu_viewer.cpp ^
   /link /SUBSYSTEM:WINDOWS ^
   /LIBPATH:"C:\Program Files\Inuitive\InuDev\lib" ^
   InuStreams.lib CommonUtilities.lib ^
   user32.lib gdi32.lib shell32.lib advapi32.lib oleaut32.lib uuid.lib ^
   /OUT:inu_viewer.exe
if errorlevel 1 (echo [x] build failed & exit /b 1)
echo [ok] inu_viewer.exe
endlocal
