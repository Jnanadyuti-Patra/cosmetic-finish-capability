@echo off
REM One-time setup: create a virtual environment and install dependencies.
cd /d "%~dp0"

echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 goto :fail

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo Setup complete.
echo   Run the study     : python scripts\run_study.py
echo   Run the dashboard : run_local.bat
echo   Run the tests     : python -m pytest
goto :eof

:fail
echo.
echo Setup failed. Check that Python 3.10+ is on your PATH.
exit /b 1
