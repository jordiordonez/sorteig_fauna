import os
import sys
import traceback
import multiprocessing

def run_streamlit():
    """Run streamlit by setting environment variables and using direct import"""
    try:
        # Set environment variables BEFORE importing streamlit
        os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'false'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

        # Get the base directory
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))

        app_path = os.path.join(base_dir, "app", "Home.py")

        print(f"Running Streamlit app from: {app_path}")

        # Verify the file exists
        if not os.path.exists(app_path):
            print(f"ERROR: Cannot find Home.py at {app_path}")
            input("Press Enter to exit...")
            return 1

        # Now import streamlit AFTER setting environment variables
        from streamlit.web import cli as stcli

        # Set up sys.argv for streamlit (no port specification)
        sys.argv = ["streamlit", "run", app_path]

        # Run streamlit
        return stcli.main()

    except Exception as e:
        print(f"Error running Streamlit: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        exit_code = run_streamlit()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        input("\nPress Enter to close...")
        sys.exit(1)