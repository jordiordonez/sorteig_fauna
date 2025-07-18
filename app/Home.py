import streamlit as st

st.set_page_config(page_title="App Sorteig", layout="wide")

page = st.sidebar.radio("Menú", ["Sorteig", "Dashboard"])

if page == "Sorteig":
    st.switch_page("pages/sorteig.py")
else:
    st.switch_page("pages/dashboard.py")
