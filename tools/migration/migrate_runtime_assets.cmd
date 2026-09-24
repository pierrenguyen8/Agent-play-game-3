@echo off
setlocal
cd /d "%~dp0\..\.."

if "%~1"=="" (
    echo Usage: tools\migration\migrate_runtime_assets.cmd OLD_REPO_PATH
    echo Example: tools\migration\migrate_runtime_assets.cmd "F:\tailieu\Agent-play-game-2"
    exit /b 2
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0migrate_from_old_repo.ps1" -OldRepo "%~1"
set EXITCODE=%ERRORLEVEL%
exit /b %EXITCODE%
