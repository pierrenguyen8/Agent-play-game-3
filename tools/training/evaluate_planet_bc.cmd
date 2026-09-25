@echo off
setlocal
cd /d "%~dp0\..\.."
".venv\Scripts\python.exe" tools\training\evaluate_planet_bc.py %*
endlocal
