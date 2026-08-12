"""Manual del sorteig dins de l'aplicació.

Mostra `manual.html`, que és autocontingut (diagrames, tipografia i estils a
dins): funciona sense connexió, també des de l'executable de Windows.

El fitxer es genera amb `python tools/build_manual.py` a partir de `manual.md`.
"""
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def ruta_manual():
    """Localitza `manual.html` tant en execució normal com empaquetat."""
    candidats = [Path(__file__).resolve().parents[2] / "manual.html"]
    base = getattr(sys, "_MEIPASS", None)
    if base:
        candidats.insert(0, Path(base) / "manual.html")
    candidats.append(Path.cwd() / "manual.html")
    return next((c for c in candidats if c.is_file()), None)


ruta = ruta_manual()

if ruta is None:
    st.error(
        "No s'ha trobat `manual.html`. Es genera amb "
        "`python tools/build_manual.py` i ha d'anar al costat de l'aplicació."
    )
    st.stop()

html = ruta.read_text(encoding="utf-8")

cap, desc = st.columns([3, 1])
cap.caption(
    "Manual de referència: com es decideix cada cas, amb l'article del "
    "Reglament que el governa."
)
desc.download_button(
    "⬇️ Descarregar el manual",
    html.encode("utf-8"),
    file_name="manual_sorteig_fauna.html",
    mime="text/html",
    use_container_width=True,
    help="Un sol fitxer, es pot obrir en qualsevol navegador sense connexió.",
)

components.html(html, height=900, scrolling=True)
