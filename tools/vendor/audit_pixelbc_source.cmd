@echo off
setlocal
set "ROOT=%~dp0..\.."
"%ROOT%\.venv\Scripts\python.exe" "%~dp0audit_pixelbc_source.py" %*
endlocal
