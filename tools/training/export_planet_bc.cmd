@echo off
setlocal
set "ROOT=%~dp0..\.."
"%ROOT%\.venv\Scripts\python.exe" "%~dp0export_planet_bc.py" %*
endlocal
