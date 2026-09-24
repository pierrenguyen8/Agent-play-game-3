@echo off
setlocal
cd /d "%~dp0"

rem Use a fresh PowerShell process with ExecutionPolicy bypass only for this setup run.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1" %*
set EXITCODE=%ERRORLEVEL%

if not "%EXITCODE%"=="0" (
    echo.
    echo Environment setup failed with exit code %EXITCODE%.
    exit /b %EXITCODE%
)

endlocal
