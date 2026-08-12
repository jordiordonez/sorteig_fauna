"""Punt d'entrada i navegació de l'aplicació.

El menú lateral es declara aquí amb `st.navigation`, de manera que els títols i
les icones són els que hi posem i no els que Streamlit dedueix dels noms de
fitxer. La configuració de la pàgina també va aquí: amb `st.navigation` només
es pot cridar `st.set_page_config` una vegada, des d'aquest fitxer.
"""
import streamlit as st

st.set_page_config(
    page_title="Sorteig de fauna",
    page_icon="🎲",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

pagines = [
    st.Page("pages/sorteig.py", title="Sorteig", icon="🎲", default=True),
    st.Page("pages/dashboard.py", title="Resultats", icon="📊"),
    st.Page("pages/manual.py", title="Manual de referència", icon="📘"),
]

st.navigation(pagines).run()
