# Sorteig Fauna

This repository contains a [Streamlit](https://streamlit.io) application used to manage wildlife draw lotteries for species such as *Isard*, *Cabirol*, and *Mufló*.  The app provides an interface to configure draws, upload participant CSV files and inspect results through a simple dashboard.

## Repository structure

- `app/` – Streamlit application modules. `Home.py` selects between draw and dashboard pages found under `app/pages/` and shared utilities live in `app/utils/`.
- `run_app.py` – Entry point used when running locally or when packaging with PyInstaller.
- `altres.csv`, `isard.csv`, `sorteig.csv` – Example CSV files used as templates or sample data.
- `build_exe.bat` – Convenience script for building a Windows executable.
- `requirements.txt` – Python dependencies for the application.

## Getting started

1. Install [Python 3.10+](https://www.python.org/downloads/).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app/Home.py
   ```
   or
   ```bash
   python run_app.py
   ```

## Building a Windows executable

To create the Windows app, install Python 3.10+, install the project dependencies and PyInstaller, run `pyinstaller --onedir --add-data "app;app" --collect-all streamlit run_app.py`, then copy `isard.csv`, `altres.csv` and `sorteig.csv` into the generated `dist` folder before launching `dist/run_app.exe`.

## License

This project is released under the [MIT License](LICENSE).

