import streamlit as st
from utils.constants import ESPECIE_SORTEIGS, TIPUS_OPTIONS
from utils import draw_logic


@st.cache_data
def load_csv(file):
    import pandas as pd
    return pd.read_csv(file, sep=";")




# Instruccions d'ús en català
with st.expander("Instruccions d'ús", expanded=False):
    st.markdown(
        """
1. **Seleccioneu l'espècie.**  
2. **Configureu els sortejos:** per a cada **tipus** indiqueu el nombre de captures i si els sortejos s’han de fer en l’ordre dels tipus definits.  
   - Feu clic a **“Afegeix Tipus”** per crear-ne de nous (podeu seleccionar diverses opcions per tipus).  
3. **Pugeu** el **CSV de prioritats** i el **CSV d’inscrits** al sorteig.  
4. *(Només per a Isard)* Els participants **sense modalitat** no participaran al **TCC**.  
5. *(Opcional)* Introduïu una **llavor** per reproduir exactament el mateix sorteig.  
6. Premeu **“Executar sorteig”** per obtenir i descarregar els resultats.
        """
    )

with st.expander("Cas `Isard`"):
    st.markdown(
        """
Per a l'espècie **isard**, el fitxer CSV de **prioritats** ha de tenir el següent format:

| Columna              | Descripció                                                                 |
|----------------------|------------------------------------------------------------------------------|
| `ID`                 | Identificador únic del caçador                                               |
| `Modalitat`          | `A` = colla, `B` = individual, `""` (buit) si **no** es vol participar al TCC |
| `Colla_ID`           | Identificador de la colla (només si `Modalitat = A`)                         |
| `Prioritat`          | Prioritat actual (1 = màxima)                                                |
| `anys_sense_captura` | Nombre d’anys consecutius sense captura                                      |
| `Parroquia`          | Nom o codi de la parròquia (obligatori si és un vedat) |
| `Estranger`          | **Sí/No** – indica si el caçador és estranger                                |
        """
    )

with st.expander("Altres espècies / unitats de gestió"):
    st.markdown(
        """
Per a la resta d’espècies o unitats de gestió, el CSV té el mateix format però **sense les columnes `Modalitat` i `Colla_ID`**.

| Columna              | Descripció                                           |
|----------------------|------------------------------------------------------|
| `ID`                 | Identificador únic del caçador                       |
| `Prioritat`          | Prioritat actual (1 = màxima)                        |
| `anys_sense_captura` | Nombre d’anys consecutius sense captura              |
| `Estranger`          | **Sí/No** – indica si el caçador és estranger        |
        """
    )

with st.expander("Nota sobre les quotes parroquials en vedats"):
    st.markdown(
        """
        Quan es defineixen diversos tipus de captura per a un mateix vedat (per exemple, “Femella” i “Mascle+Trofeu”), la reserva del 50% de captures per a les parròquies s'aplica sobre la suma total de captures definides per al sorteig. Aquest percentatge es reparteix entre les parròquies afectades segons el percentatge establert per vedat.

        ⚠️ Aquest 50% no és obligatòriament assolit. L'assignació de captures dins aquesta quota segueix les prioritats individuals dels caçadors. La condició per donar preferència a un caçador de la parròquia és:
        - Que tingui la mateixa prioritat individual que altres sol·licitants.
        - Que la seva parròquia no hagi assolit encara el percentatge corresponent dins del 50%.

        Un cop es compleixen aquestes dues condicions, el sistema prioritza els caçadors locals fins a exhaurir la quota. Un cop superada, totes les captures es reparteixen exclusivament per prioritat individual.
        """
    )

with st.expander("Parròquies"):
    st.markdown(
        """
        | Codi | Parròquia              |
        |------|------------------------|
        | 1    | Canillo                |
        | 2    | Encamp                 |
        | 3    | Ordino                 |
        | 4    | La Massana             |
        | 5    | Andorra la Vella       |
        | 6    | Sant Julià de Lòria    |
        | 7    | Escaldes-Engordany     |

        Si el nom està escrit de manera alternativa (majúscules, minúscules, abreviatures com `SJ`, `ESCALDES`, etc.), també serà reconegut automàticament, però **es recomana el format numèric** per garantir la màxima fiabilitat.
        """
    )

with st.expander("Columnes del fitxer de resultats"):
    st.markdown(
        """
        El CSV resultants inclou, per a cada `ID`:
        - Per a cada codi de sorteig, la posició on s'ha adjudicat la captura. Si el caçador estava inscrit i no ha obtingut plaça apareix un codi `s1`, `s2`, ... amb l'ordre dels no adjudicats; si no estava inscrit el valor és buit.
        - Les columnes `Tipus_<codi>` indiquen el tipus de captura assignat en cada sorteig.
        - `Nou_Anys_sense_captura` i `Nova_prioritat` amb els valors resultants després de tots els sortejos.
        """
    )

st.markdown("💡 Pots descarregar exemples de fitxers aquí:")

with open("isard.csv", "rb") as f1:
    st.download_button(
        label="📥 Exemple Isard (isard.csv)",
        data=f1,
        file_name="isard.csv",
        mime="text/csv",
    )

with open("altres.csv", "rb") as f2:
    st.download_button(
        label="📥 Altres espècies (altres.csv)",
        data=f2,
        file_name="altres.csv",
        mime="text/csv",
    )

with open("sorteig.csv", "rb") as f3:
    st.download_button(
        label="📥 Exemple Inscripcions Sortejos (sorteig.csv)",
        data=f3,
        file_name="sorteig.csv",
        mime="text/csv",
    )

especie = st.selectbox("Espècie", list(ESPECIE_SORTEIGS.keys()))

with st.expander("Configuració de captures per sorteig"):
    for sorteig in ESPECIE_SORTEIGS[especie]:
        st.markdown(f"### {sorteig}")
        key_prefix = sorteig.replace(" ", "_")
        if especie == "Isard" and sorteig == "IS TCC":
            st.number_input(
                "Quantitat Captures", min_value=0, step=1, key=f"total_{key_prefix}"
            )
            st.session_state.setdefault(f"configs_{key_prefix}", [])
        else:
            aleatori_key = f"aleatori_{key_prefix}"
            st.checkbox("Ordre aleatori", value=True, key=aleatori_key)

            cfg_key = f"configs_{key_prefix}"
            if cfg_key not in st.session_state:
                st.session_state[cfg_key] = [{"selections": [], "qty": 0}]

            if st.button("Afegeix Tipus", key=f"add_{key_prefix}"):
                st.session_state[cfg_key].append({"selections": [], "qty": 0})

            for idx in range(len(st.session_state[cfg_key])):
                conf = st.session_state[cfg_key][idx]
                st.subheader(f"Tipus {idx+1}")

                sel_key = f"{key_prefix}_sel_{idx}"
                if sel_key not in st.session_state:
                    st.session_state[sel_key] = conf["selections"]
                st.multiselect(
                    f"Valors Tipus {idx+1}",
                    TIPUS_OPTIONS,
                    key=sel_key,
                    on_change=draw_logic.sanitize_indeterminat,
                    args=(sel_key,),
                )
                sel = st.session_state.get(sel_key, [])

                qty_key = f"{key_prefix}_qty_{idx}"
                if qty_key not in st.session_state:
                    st.session_state[qty_key] = conf["qty"]
                qty = st.number_input("Quantitat", min_value=0, step=1, key=qty_key)

                st.session_state[cfg_key][idx] = {"selections": sel, "qty": qty}

csv1 = st.file_uploader("CSV de prioritats", type="csv", key="csv1")
csv2 = st.file_uploader("CSV d'inscrits", type="csv", key="csv2")

seed_input = st.number_input("Llavor opcional", value=0, step=1)
seed = int(seed_input) if seed_input else None

# Track whether the draw process should be executed across reruns
st.session_state.setdefault("run_draw", False)

# ─────────────────────  EXECUTAR SORTEIG  ────────────────────────────────
# When the user presses the button we set a session flag so that
# the computation can survive Streamlit reruns (e.g. when asking to
# confirm missing modalities).
if st.button("Executar sorteig"):
    st.session_state["run_draw"] = True

if st.session_state.get("run_draw"):

    # ------------------------------------------------------------------ #
    import pandas as pd
    # 1️⃣  Load & validate the CSVs                                       #
    # ------------------------------------------------------------------ #
    if not csv1 or not csv2:
        st.error("Cal carregar els dos CSV")
        st.stop()

    df1 = load_csv(csv1)
    df2 = load_csv(csv2)
    try:
        draw_logic.validar_csv2(df2)
        (draw_logic.validar_csv_isard if especie == "Isard" else draw_logic.validar_csv_altres)(df1)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # ------------------------------------------------------------------ #
    # 2️⃣  Warn about IDs present only in one CSV                        #
    # ------------------------------------------------------------------ #
    ids_prio = set(df1["ID"].astype(str))
    ids_ins = set(df2["ID"].astype(str))

    missing_in_ins = ids_prio - ids_ins
    if especie == "Isard" and "Modalitat" in df1.columns:
        ab_ids = set(df1[df1["Modalitat"].isin(["A", "B"])]["ID"].astype(str))
        missing_in_ins -= ab_ids
    missing_in_prio = ids_ins - ids_prio

    if (missing_in_ins or missing_in_prio) and not st.session_state.get(
        "confirm_missing_ids", False
    ):
        msg_parts = []
        if missing_in_ins:
            msg_parts.append(
                "IDs al CSV de prioritats però no al d'inscrits: "
                + ", ".join(sorted(missing_in_ins))
            )
        if missing_in_prio:
            msg_parts.append(
                "IDs al CSV d'inscrits però no al de prioritats: "
                + ", ".join(sorted(missing_in_prio))
            )
        st.warning(" ".join(msg_parts) + " S'ignoraran si continues.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Ignorar i continuar", key="confirm_missing_ids_btn"):
                st.session_state["confirm_missing_ids"] = True
                st.rerun()
        with col2:
            if st.button("Atura el procés", key="stop_missing_ids"):
                st.stop()
        st.stop()

    # ------------------------------------------------------------------ #
    # 3️⃣  IS‑TCC: detect hunters without Modalitat                       #
    # ------------------------------------------------------------------ #
    ids_to_skip = []  # ← will hold the IDs we really want to ignore

    if especie == "Isard":
        inscrits_tcc = df2[df2["Codi_Sorteig"] == "IS TCC"]
        missing_mod = inscrits_tcc.merge(df1[["ID", "Modalitat"]], on="ID", how="left")
        missing_mod = missing_mod[
            missing_mod["Modalitat"].isna()
            | (missing_mod["Modalitat"].astype(str).str.strip() == "")
        ]

        # -- Ask the user what to do ------------------------------------
        if not missing_mod.empty and not st.session_state.get(
            "confirm_missing_mod", False
        ):
            st.warning(
                "Els següents caçadors s'han inscrit al TCC però no tenen modalitat "
                "especificada i s'ignoraran si continues: "
                + ", ".join(missing_mod["ID"].astype(str))
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Ignorar i continuar", key="confirm_missing_mod_btn"):
                    st.session_state["confirm_missing_mod"] = True
                    st.session_state["ids_to_skip_tcc"] = missing_mod["ID"].tolist()
                    st.rerun()
            with col2:
                if st.button("Atura el procés", key="stop_missing_mod"):
                    st.stop()
            st.stop()  # wait until the user picks an option

        # -- User already confirmed on a previous run -------------------
        ids_to_skip = st.session_state.get("ids_to_skip_tcc", [])

    # ------------------------------------------------------------------ #
    # 4️⃣  Drop those IDs only from IS TCC                               #
    # ------------------------------------------------------------------ #
    if ids_to_skip:
        mask = (df2["Codi_Sorteig"] == "IS TCC") & (df2["ID"].isin(ids_to_skip))
        df2 = df2.loc[~mask].copy()

    # ------------------------------------------------------------------ #
    # 5️⃣  Build the configuration DataFrame from the UI inputs          #
    # ------------------------------------------------------------------ #
    config_rows = []
    for sorteig in ESPECIE_SORTEIGS[especie]:
        key_prefix = sorteig.replace(" ", "_")

        if especie == "Isard" and sorteig == "IS TCC":
            total = st.session_state.get(f"total_{key_prefix}", 0)
            config_rows.append(
                {
                    "Codi_Sorteig": sorteig,
                    "Tipus": "",
                    "Quantitat": total,
                    "Aleatori": True,
                }
            )
        else:
            aleatori = st.session_state.get(f"aleatori_{key_prefix}", True)
            for conf in st.session_state.get(f"configs_{key_prefix}", []):
                tip = "+".join(conf["selections"]) if conf["selections"] else ""
                config_rows.append(
                    {
                        "Codi_Sorteig": sorteig,
                        "Tipus": tip,
                        "Quantitat": conf["qty"],
                        "Aleatori": aleatori,
                    }
                )

    config_df = pd.DataFrame(config_rows)

    # ------------------------------------------------------------------ #
    # 6️⃣  Run the draw and show results                                 #
    # ------------------------------------------------------------------ #
    try:
        resultat, resums = draw_logic.processar_sorteigs(df1, df2, config_df, especie, seed)
    except Exception as exc:
        st.error(f"🚫 Error en el sorteig: {exc}")
        st.stop()
    st.session_state["resultat"] = resultat  # full table, ~ID × columns
    st.session_state["resums"] = resums  # list of per-draw summaries
    st.subheader("Resultats")
    capture_cols = [s.replace(" ", "_") for s in ESPECIE_SORTEIGS[especie]]
    for col in capture_cols:
        if col in resultat.columns:
            resultat[col] = resultat[col].astype(str).replace('nan', '')
            resultat[col] = resultat[col].astype(str).replace('<NA>', '')
    st.dataframe(resultat, use_container_width=True)

    st.download_button(
        "Descarregar CSV",
        resultat.to_csv(index=False).encode("utf-8"),
        file_name="resultats.csv",
    )

    # ------------------------------------------------------------------ #
    # 7️⃣  Clean‑up session flags so next run starts fresh                #
    # ------------------------------------------------------------------ #
    st.session_state.pop("confirm_missing_mod", None)
    st.session_state.pop("ids_to_skip_tcc", None)
    st.session_state.pop("confirm_missing_ids", None)
    st.session_state["run_draw"] = False
