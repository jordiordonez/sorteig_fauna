import os
import sys
import subprocess

# Get the absolute path to the directory where the executable/script is located
base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
app_path = os.path.join(base_dir, "app", "Home.py")

# Debug: Print the path (optional)
print(f"Running Streamlit app from: {app_path}")

# Launch Streamlit with the correct path
subprocess.call(["streamlit", "run", app_path, "--server.headless=false"])
