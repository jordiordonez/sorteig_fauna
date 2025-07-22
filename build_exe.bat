@echo off
echo =====================================
echo  Building Streamlit App EXE for Windows
echo =====================================

REM Install PyInstaller if not already installed
pip show pyinstaller >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Remove old build/dist folders (optional)
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Create the EXE
pyinstaller --onefile run_app.py

echo.
echo =====================================
echo  Build Complete!
echo  Your EXE is located in the "dist" folder.
echo =====================================
pause
