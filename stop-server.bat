@echo off
chcp 65001 >nul
title selp-asr - stop
cd /d "%~dp0"

rem Uses the project's own runtime. Stops the service via /v1/audio/shutdown.

set "PY=%~dp0runtime\python\python.exe"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo   Runtime not installed yet. Nothing to stop.
    pause
    exit /b 1
)

"%PY%" stop.py
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
