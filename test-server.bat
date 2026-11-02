@echo off
chcp 65001 >nul
title selp-asr - speech test
cd /d "%~dp0"

rem Uses the project's own runtime. Run start-server.bat once first.

set "PY=%~dp0runtime\python\python.exe"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo.
    echo   Runtime not installed yet. Run start-server.bat first.
    echo.
    pause
    exit /b 1
)

"%PY%" -m asr_server.mic %*
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
