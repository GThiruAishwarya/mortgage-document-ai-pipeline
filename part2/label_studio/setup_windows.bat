@echo off
echo =============================================
echo  Label Studio Setup - Mortgage Review Tool
echo =============================================
echo.

echo Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Download Python 3.11+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Installing Label Studio...
pip install label-studio requests

echo.
echo =============================================
echo  Installation complete!
echo =============================================
echo.
echo To start Label Studio, run:
echo   label-studio start
echo.
echo Then open http://localhost:8080 in your browser.
echo.
echo If 'label-studio' is not found after install, add Python Scripts to PATH:
echo   - Open Control Panel -> System -> Advanced -> Environment Variables
echo   - Add the Python Scripts folder (e.g. C:\Users\YourName\AppData\Local\Programs\Python\Python311\Scripts)
echo.
pause
