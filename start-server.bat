@echo off
chcp 65001 >nul
title selp-asr - local speech recognition service
cd /d "%~dp0"

rem This project ships its own Python runtime in runtime\python.
rem If it is missing (or incomplete), install-runtime.ps1 downloads it -- PowerShell
rem comes with Windows, so no Python, conda or PATH setup is needed on the user's machine.

set "PY=%~dp0runtime\python\python.exe"
rem Written only after a complete extraction, so a half-unpacked runtime (killed
rem mid-install, disk full, ...) is treated as "not installed" and reinstalled.
set "MARK=%~dp0runtime\python\.selp-asr-runtime.json"
set "PYTHONUTF8=1"

rem Use the absolute path: PATH is not something we want to depend on.
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" set "PS=powershell"

rem Windows PowerShell 5.1 needs its own module path to find cmdlets such as
rem Get-FileHash. Inheriting a broken one (e.g. from a shell that ships PS 7)
rem makes those cmdlets disappear, so pin it to the default.
set "PSModulePath=%ProgramFiles%\WindowsPowerShell\Modules;%SystemRoot%\system32\WindowsPowerShell\v1.0\Modules"

if not exist "%PY%" goto :install
if not exist "%MARK%" goto :install
goto :run

:install
echo.
echo   First run - installing the bundled Python runtime...
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install-runtime.ps1"
if errorlevel 1 goto :fail
if not exist "%PY%" goto :fail
if not exist "%MARK%" goto :fail

:run
"%PY%" bootstrap.py %*
set "CODE=%ERRORLEVEL%"

rem Keep the window open only when something failed, so a clean exit
rem (or a background launch) does not leave a cmd waiting for a keypress.
if not "%CODE%"=="0" pause
exit /b %CODE%

:fail
echo.
echo   Could not prepare the Python runtime.
echo   Fix the problem shown above, then run this file again.
echo.
pause
exit /b 1
