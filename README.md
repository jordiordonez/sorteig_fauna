# Sorteig Fauna

This repository contains a [Streamlit](https://streamlit.io) application used to manage wildlife draw lotteries for species such as *Isard*, *Cabirol* and *Mufló*.  The app lets you **configure the draw zones**, upload participant CSV files and inspect the results through a dashboard.

## Repository structure

- `app/` – Streamlit application modules. `Home.py` selects between the draw and dashboard pages found under `app/pages/`; shared logic lives in `app/utils/`.
  - `app/utils/config.py` – the editable catalogue of species and zones (data model + preloaded defaults).
  - `app/utils/draw_logic.py` – the draw engine.
- `tests/test_sorteig.py` – test suite for the draw engine (run with `python tests/test_sorteig.py` or `pytest`). Tests that need real data skip automatically when it is absent.
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

## Configuring zones

Each **species** holds an ordered list of **management zones** (*unitats de gestió*). The catalogue ships preloaded with the current zones and their defaults (`app/utils/config.py`), so a typical year you only **review and adjust** instead of entering everything from scratch.

For every zone you can configure, from the **Sorteig** page:

| Field | Meaning |
|---|---|
| **Nom** | Zone code/name. Must match the `Codi_Sorteig` value in the entries CSV. |
| **Tipus** | `Vedat` (parish-priority draw) or `TCC` (ordinary priority draw). |
| **Modalitat A/B** | When enabled, the zone uses the *colla + individual* mechanism (`Modalitat`, `Colla_ID`). Today only the Isard **IS TCC** uses it. |
| **Efectiu mínim per colla** | *(Modalitat A/B only)* Minimum number of members a colla must have (default 6; art. 56.1.a). If any colla is below it, the draw is **blocked** and the entries CSV must be fixed. `0` disables the check. |
| **Ordre aleatori** | Whether capture types are drawn in random order within the zone. |
| **% màx. estrangers** | Ceiling on captures awarded to foreign non-residents, applied **per zone over the offered captures** (default 10%; art. 53.4). It is a cap, not a reserve: foreigners compete on equal terms and simply drop out of the draw once the cap is reached. |
| **% reservat als locals** | *(Vedat only)* Share of captures over which hunters censused in the vedat's parishes get **preference** (default 50%; art. 55.1). It is a *preference ceiling*, not a closed block — see below. |
| **Distribució parroquial** | *(Vedat only)* How that share splits across the vedat's parishes **by land area**, in percentages that add up to 100 (art. 55.1.a.i). |
| **Tipus de captura** | One or more capture types (e.g. `Mascle`, `Femella + Trofeu`) each with its quantity; add more with **Afegeix Tipus**. |

**Draw order** is the order of the zone list — move zones up/down with ⬆️/⬇️. Within a zone, the configured capture types are processed in order (or randomly if *Ordre aleatori* is on). A hunter who wins in an earlier zone has a lowered priority in the zones drawn afterwards.

**How a *Vedat* is drawn (art. 55.1).** Captures are awarded one by one under three **hierarchical** criteria:

1. **Individual priority** (art. 54) — always dominates.
2. At **equal priority**, a hunter censused in one of the vedat's parishes goes ahead of a non-resident.
3. At equal priority, the **drawn order of the parishes** (art. 55.1.b, reported with the results) breaks the tie.

The *% reservat als locals* (50%) is turned into an **integer ceiling per parish** by land area (largest-remainder rounding — e.g. 24 captures → 12 / 6 / 6 / 0). This ceiling limits the **preference**, not eligibility: while a parish is below its ceiling its residents get criterion 2; once it reaches the ceiling, the preference stops for that parish (its residents keep competing on priority alone). So the 50% behaves as a **ceiling, not a guaranteed floor** — residents may end up below it (if non-residents outrank them or there are too few local applicants) or above it (by winning on priority once the preference is spent). The parish order is drawn per zone and shown above the results table. *TCC* zones (and *Modalitat A/B*) have no parish preference: a single draw by priority.

You can **add or remove** zones, restore the preloaded defaults, create new species (type *Altres*, which behaves like Cabirol/Mufló — no colles), and **export/import** a species' configuration as JSON to reuse it across years.

## CSV formats

**Priorities CSV** — one row per hunter. Required columns depend on the configured zones:

| Column | Required when |
|---|---|
| `ID` | always |
| `Prioritat` | always (1 = highest) |
| `anys_sense_captura` | always |
| `Estranger` | always (`Sí`/`No`) |
| `Parroquia` | any zone is a **Vedat** (name or 1–7 code) |
| `Modalitat` (`A`/`B`) and `Colla_ID` | any zone uses **modalitat A/B** |

**Entries CSV** — columns `ID` and `Codi_Sorteig`, where `Codi_Sorteig` matches a configured **zone name**.

### Results

The results table gives, per `ID` and per zone, the position at which a capture was awarded; `s1`, `s2`, … for inscribed hunters left without a place; and blank when not inscribed. `Tipus_<zone>` columns show the assigned capture type **for awarded hunters only** (substitutes carry no type). For vedats, the drawn parish order and each parish's preference ceiling are shown above the results table. `Nou_Anys_sense_captura` and `Nova_prioritat` carry provisional next-season values (`4` for hunters awarded a capture, `2` otherwise); priorities `1` (female actually hunted, art. 54.1.a) and `3` (non-applicants) are set by the government technicians, since they depend on data the app does not hold.

## Building a Windows executable

To create the Windows app, install Python 3.10+, install the project dependencies and PyInstaller, run `pyinstaller --onedir --add-data "app;app" --collect-all streamlit run_app.py`, then copy `isard.csv`, `altres.csv` and `sorteig.csv` into the generated `dist` folder before launching `dist/run_app.exe`.  See `README.txt` for the detailed build steps.

## License

This project is released under the [MIT License](LICENSE).
