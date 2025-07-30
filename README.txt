# README - How to Build and Run the Streamlit EXE (sorteig_fauna)

## 1. Prerequisites
- Install Python 3.10+ on Windows from https://www.python.org/downloads/windows/
- During installation, CHECK the option "Add Python to PATH".
- Open Command Prompt (CMD).

---

## 2. Prepare Environment
Navigate to your project folder:
    cd path\to\sorteig_fauna

Install required dependencies:
    pip install -r requirements.txt

Install PyInstaller:
    pip install pyinstaller

---

## 3. Create `run_app.py`
Ensure `run_app.py` (in sorteig_fauna folder) contains:

    import os
    import sys
    import subprocess

    # Get the absolute path to the directory where the executable/script is located
    base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_dir, "app", "Home.py")

    # Launch Streamlit with the correct path
    subprocess.call(["streamlit", "run", app_path, "--server.headless=false"])

---

## 4. Build the EXE
Run:
    pyinstaller --onedir --add-data "app;app" --collect-all streamlit run_app.py



This creates:
    dist/run_app.exe

---

## 5. Copy Required Files
To make sure the app works correctly, copy these files into the `dist` folder:
    - isard.csv
    - altres.csv
    - sorteig.csv
    - (Any other static resources your app needs)

---

## 6. Running the App
Double-click:
    dist/run_app.exe
Your browser will open automatically.

---

## 7. Optional - Desktop Shortcut
- Right-click `dist/run_app.exe` → **Send to → Desktop (create shortcut)**.
- You can rename the shortcut, e.g., "Sorteig App".

---

## 8. Optional - Custom Icon
If you want a custom icon:
1. Prepare `icon.ico`.
2. Rebuild:
       pyinstaller --onefile --icon=icon.ico run_app.py
3. Copy Files
	copy isard.csv dist\run_app\
	copy altres.csv dist\run_app\
	copy sorteig.csv dist\run_app\
---

## 9. Stopping the App
- When running from CMD, stop with `CTRL + C`.
- If run from `.exe`, just close the CMD window that opens.

---

## 10. Debugging
If the app fails to run, open CMD and execute:
    dist\run_app.exe
to see any error messages.
