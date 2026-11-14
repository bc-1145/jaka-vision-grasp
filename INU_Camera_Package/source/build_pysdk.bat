@echo off
rem Build the InuDev Python extension (_inudev) for the local Python.
rem Requires Visual Studio 2022 Build Tools and pybind11 (pip install pybind11).
setlocal

set "SDK=C:\Program Files\Inuitive\InuDev"
set "OUT=%~dp0..\pythonsdk"

for /f "delims=" %%i in ('python -c "import sysconfig;print(sysconfig.get_paths()['include'])"') do set "PYINC=%%i"
for /f "delims=" %%i in ('python -c "import sysconfig;print(sysconfig.get_config_var('prefix'))"') do set "PYPREFIX=%%i"
for /f "delims=" %%i in ('python -c "import sysconfig;print(sysconfig.get_config_var('LIBRARY'))"') do set "PYLIB=%%i"
rem Python 3.14 reports the DLL name here; we need the import library next to it
set "PYLIB=%PYLIB:.dll=.lib%"
for /f "delims=" %%i in ('python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"') do set "EXT=%%i"
for /f "delims=" %%i in ('python -c "import pybind11;print(pybind11.get_include())"') do set "PBINC=%%i"

if not exist "%OUT%" mkdir "%OUT%"

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (echo [x] VS Build Tools not found & exit /b 1)

cl /nologo /std:c++17 /EHsc /O2 /MD /utf-8 ^
   /I "%PYINC%" /I "%PBINC%" /I "%SDK%\include" ^
   inu_module.cpp ^
   /LD /link ^
   /LIBPATH:"%PYPREFIX%\libs" /LIBPATH:"%SDK%\lib" ^
   %PYLIB% InuStreams.lib CommonUtilities.lib ^
   /OUT:"%OUT%\_inudev%EXT%"
if errorlevel 1 (echo [x] build failed & exit /b 1)

echo [ok] %OUT%\_inudev%EXT%
endlocal
