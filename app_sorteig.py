import math  # ← NEW: for ceil, floor, isnan
import pandas as pd
import numpy as np
import streamlit as st
import unicodedata
from collections import OrderedDict
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
import plotly.express as px

st.set_page_config(
    page_title="App Sorteig Pla de Caça",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

# ── CONSTANTS ────────────────────────────────────────────────────────────────

ESPECIE_SORTEIGS = OrderedDict(
    {
        "Isard": ["IS TCC", "IS VCRS", "IS VCX", "IS VCE"],
        "Cabirol": ["CAB"],
        "Mufló": ["MUF UGEO", "MUF UGC", "MUF VTE-E", "MUF VCE", "MUF R"],
    }
)

# Order of vedats must be preserved for UI display
VEDAT_PARRÒQUIES = OrderedDict(
    [
        (
            "IS VCE",
            {
                "La Massana": 0.234,
                "Sant Julià de Lòria": 0.241,
                "Andorra la Vella": 0.522,
                "Escaldes-Engordany": 0.003,
            },
        ),
        ("IS VCRS", {"Canillo": 0.5, "Ordino": 0.5}),
        ("IS VCX", {"La Massana": 1.0}),
    ]
)

TIPUS_OPTIONS = [
    "Femella",
    "Mascle",
    "Adult",
    "Juvenil",
    "Trofeu",
    "Selectiu",
    "Indeterminat",
]


def sanitize_indeterminat(key: str) -> None:
    """Ensure multiselect keeps only 'Indeterminat' if chosen."""
    val = st.session_state.get(key, [])
    if "Indeterminat" in val and len(val) > 1:
        st.session_state[key] = ["Indeterminat"]


# ── UTILITIES ────────────────────────────────────────────────────────────────


def strip_accents(text: str) -> str:
    """Return the input string without diacritics."""
    text = str(text)
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalitza_parroquia(valor):
    CODI_PARROQUIES = {
        1: "Canillo",
        2: "Encamp",
        3: "Ordino",
        4: "La Massana",
        5: "Andorra la Vella",
        6: "Sant Julià de Lòria",
        7: "Escaldes-Engordany",
    }
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    txt = strip_accents(str(valor).strip()).lower()
    if txt.isdigit():
        return CODI_PARROQUIES.get(int(txt))
    txt = txt.replace("-", " ").replace("_", " ").replace("sj", "sant julia de loria")
    for name in CODI_PARROQUIES.values():
        canonical = strip_accents(name).lower().replace("-", " ")
        if canonical in txt:
            return name
    return None


def normalitza_estranger(valor) -> str:
    if isinstance(valor, str) and valor.strip().lower() in {
        "si",
        "sí",
        "s",
        "yes",
        "true",
        "1",
    }:
        return "si"
    return "no"


# ── CSV VALIDATION HELPERS ───────────────────────────────────────────────────


def validar_csv_isard(df):
    required = {
        "ID",
        "Modalitat",
        "Colla_ID",
        "Prioritat",
        "anys_sense_captura",
        "Parroquia",
        "Estranger",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Falten columnes: {', '.join(sorted(missing))}")


def validar_csv_altres(df):
    required = {"ID", "Prioritat", "anys_sense_captura", "Estranger"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Falten columnes: {', '.join(sorted(missing))}")


def validar_csv2(df):
    required = {"ID", "Codi_Sorteig"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Falten columnes: {', '.join(sorted(missing))}")


# ── HELPER: CHOOSE NEXT CANDIDATE ────────────────────────────────────────────


def tria_candidat(
    df,
    assigned,
    estr_cnt,
    assignats,
    vedat,
    assignats_parr,
    rng,
    estranger_limit,
):
    pool = df[~df["ID"].isin(assigned)].copy()
    if pool.empty:
        return None

    if estr_cnt >= estranger_limit:
        pool = pool[pool["Estranger"] == "no"]
    if pool.empty:
        return None

    pool["rand"] = rng.random(len(pool))

    if vedat and vedat in VEDAT_PARRÒQUIES:
        quotas = VEDAT_PARRÒQUIES[vedat]
        pool["quota_flag"] = pool["Parroquia"].apply(
            lambda p: 1 if quotas.get(p, 0) - assignats_parr.get(p, 0) > 0 else 0
        )
        order_cols, asc = ["Prioritat", "quota_flag", "anys_sense_captura", "rand"], [
            True,
            False,
            False,
            True,
        ]
    else:
        order_cols, asc = ["Prioritat", "anys_sense_captura", "rand"], [
            True,
            False,
            True,
        ]

    return pool.sort_values(order_cols, ascending=asc).index[0]


# ── HELPER: INDIVIDUAL DRAW (no colles) ──────────────────────────────────────


def sorteig_individual(df, tipus_quant, ordre_aleatori, vedat, rng):
    df = df.copy()
    df["Estranger"] = df["Estranger"].apply(normalitza_estranger)
    if "Parroquia" in df.columns:
        df["Parroquia"] = df["Parroquia"].apply(normalitza_parroquia)

    df["assigned"] = False
    df["ordre"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    df["tipus"] = pd.Series([pd.NA] * len(df), dtype="object")
    assignats_parr = {k: 0 for k in VEDAT_PARRÒQUIES.get(vedat, {})}

    captures_pool = [t for t, q in tipus_quant for _ in range(q)]
    total_caps = len(captures_pool)
    estranger_limit = math.floor(0.1 * total_caps)
    if not ordre_aleatori:
        captures_pool = captures_pool.copy()  # keep deterministic order

    ordre, estrangers, assignats = 1, 0, 0

    if ordre_aleatori:
        while captures_pool and not df.loc[~df["assigned"]].empty:
            idx = tria_candidat(
                df,
                set(df[df["assigned"]]["ID"]),
                estrangers,
                assignats,
                vedat,
                assignats_parr,
                rng,
                estranger_limit,
            )
            if idx is None:
                break
            tipus = rng.choice(captures_pool)
            captures_pool.remove(tipus)

            df.loc[idx, ["assigned", "ordre", "tipus"]] = [True, ordre, tipus]

            assignats += 1
            if df.at[idx, "Estranger"] == "si":
                estrangers += 1
            if vedat and df.at[idx, "Parroquia"] in assignats_parr:
                assignats_parr[df.at[idx, "Parroquia"]] += 1
            ordre += 1
    else:
        for tipus, q in tipus_quant:
            for _ in range(q):
                idx = tria_candidat(
                    df,
                    set(df[df["assigned"]]["ID"]),
                    estrangers,
                    assignats,
                    vedat,
                    assignats_parr,
                    rng,
                    estranger_limit,
                )
                if idx is None:
                    break
                df.loc[idx, ["assigned", "ordre", "tipus"]] = [True, ordre, tipus]

                assignats += 1
                if df.at[idx, "Estranger"] == "si":
                    estrangers += 1
                if vedat and df.at[idx, "Parroquia"] in assignats_parr:
                    assignats_parr[df.at[idx, "Parroquia"]] += 1
                ordre += 1

    cols = ["ID", "ordre", "tipus", "Estranger"]
    if "Parroquia" in df.columns:
        cols.append("Parroquia")
    out = df[cols].copy()
    out["ordre"] = out["ordre"].astype("Int64")
    return out


# ── HELPER: PARSE 'Tipus' FIELD ──────────────────────────────────────────────


def _parse_tipus(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [t.strip() for t in str(value).split(",") if t.strip()]


# ── MAIN: PROCESSAR SORTEIGS ────────────────────────────────────────────────


def processar_sorteigs(df1, df2, config, especie, seed):
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    ids_totals = df2["ID"].unique()
    if especie == "Isard":
        extra = df1.loc[df1["Modalitat"].notna() & ~df1["ID"].isin(ids_totals), "ID"]
        ids_totals = np.union1d(ids_totals, extra)
    base = df1[df1["ID"].isin(ids_totals)].drop_duplicates("ID")

    cols = ["ID"]
    if especie == "Isard":
        cols.extend(["Modalitat", "Colla_ID"])
    cols.extend(["Prioritat", "anys_sense_captura", "Estranger"])
    if "Parroquia" in base.columns:
        cols.append("Parroquia")

    resultat = base[cols].copy()

    captures_prev = {id_: 0 for id_ in resultat["ID"]}
    resum_sorteigs = []

    for sorteig in ESPECIE_SORTEIGS[especie]:
        conf_rows = config[config["Codi_Sorteig"] == sorteig].copy()
        conf_rows["Tipus"] = conf_rows["Tipus"].apply(_parse_tipus)

        col_base = sorteig.replace(" ", "_")
        resultat[col_base] = np.nan
        resultat[f"Tipus_{col_base}"] = np.nan

        subset = df2[df2["Codi_Sorteig"] == sorteig].copy()
        if especie == "Isard" and sorteig == "IS TCC":
            extra_ids = df1.loc[
                df1["Modalitat"].notna() & ~df1["ID"].isin(df2["ID"]), "ID"
            ]
            if not extra_ids.empty:
                subset = pd.concat(
                    [subset, pd.DataFrame({"ID": extra_ids, "Codi_Sorteig": sorteig})],
                    ignore_index=True,
                )
        if subset.empty or conf_rows.empty:
            continue
        if subset["ID"].duplicated().any():
            raise ValueError(f"ID duplicats al sorteig {sorteig}")

        sol_licituds_total = subset["ID"].nunique()
        part = subset.merge(df1, on="ID")
        part["Prioritat"] = part.apply(
            lambda r: (
                (5 + captures_prev.get(r["ID"], 0) - 1)
                if captures_prev.get(r["ID"], 0) > 0
                else r["Prioritat"]
            ),
            axis=1,
        )
        if especie == "Isard" and sorteig == "IS TCC":
            part = part[part["Modalitat"].isin(["A", "B"])].copy()
        subset_ids = set(part["ID"])

        if especie == "Isard" and sorteig == "IS TCC":
            total_cap = int(conf_rows["Quantitat"].sum())
            if total_cap <= 0:
                raise ValueError("Total de captures per IS TCC ha de ser > 0")
            asignats = assignar_isards_sorteig_csv(
                part, total_cap, seed=rng.randint(0, 2**32 - 1)
            )
            asignats["tipus"] = "+".join(conf_rows.iloc[0]["Tipus"])
        else:
            tipus_quant = []
            for _, r in conf_rows.iterrows():
                tipus = ["Indeterminat"] if "Indeterminat" in r["Tipus"] else r["Tipus"]
                tipus_quant.append(("+".join(tipus), int(r["Quantitat"])))
            vedat = sorteig if (especie == "Isard" and sorteig != "IS TCC") else None
            asignats = sorteig_individual(
                part,
                tipus_quant,
                bool(conf_rows.iloc[0].get("Aleatori", False)),
                vedat,
                np.random.RandomState(rng.randint(0, 2**32 - 1)),
            )

        # ── SAFE MERGE: unique column names per sorteig
        # 1️⃣  keep Estranger for the résumé, but don’t let it into the merge
        asignats = asignats.rename(
            columns={"ordre": f"ordre_{col_base}", "tipus": f"tipus_{col_base}"}
        )

        # 2️⃣  calculate 'estr' **before** we drop Estranger
        estr = asignats[asignats["Estranger"] == "si"][f"ordre_{col_base}"].count()

        # 3️⃣  we only need ID, ordre_*, tipus_* for the merge
        merge_cols = ["ID", f"ordre_{col_base}", f"tipus_{col_base}"]
        resultat = resultat.merge(asignats[merge_cols], on="ID", how="left")

        resultat[col_base] = resultat[f"ordre_{col_base}"]
        mask_no_cap = resultat["ID"].isin(subset_ids) & resultat[col_base].isna()
        resultat.loc[mask_no_cap, col_base] = 0
        resultat[f"Tipus_{col_base}"] = resultat[f"tipus_{col_base}"]

        winners = asignats.loc[asignats[f"ordre_{col_base}"].notna(), "ID"]
        for wid in winners:
            captures_prev[wid] = captures_prev.get(wid, 0) + 1

        resultat.drop(columns=[f"ordre_{col_base}", f"tipus_{col_base}"], inplace=True)
        resultat[col_base] = resultat[col_base].astype("Int64")

        asign_finals = asignats[f"ordre_{col_base}"].notna().sum()
        estr = asignats[asignats["Estranger"] == "si"][f"ordre_{col_base}"].count()
        tipus_counts = (
            asignats.loc[asignats[f"ordre_{col_base}"].notna(), f"tipus_{col_base}"]
            .value_counts()
            .to_dict()
        )

        parr_counts = {}
        if "Parroquia" in asignats.columns:
            parr = asignats.loc[asignats[f"ordre_{col_base}"].notna(), "Parroquia"]
            parr_counts = parr.value_counts().to_dict()

        previs_counts = {}
        for _, r in conf_rows.iterrows():
            tp = _parse_tipus(r["Tipus"])
            tip_label = "+".join(tp) if tp else "Indeterminat"
            previs_counts[tip_label] = previs_counts.get(tip_label, 0) + int(
                r["Quantitat"]
            )

        sol_licituds = sol_licituds_total
        resum_rows = []
        for t in set(previs_counts) | set(tipus_counts):
            resum_rows.append(
                {
                    "Sorteig": sorteig,
                    "Tipus": t,
                    "Assignacions_previstes": previs_counts.get(t, 0),
                    "Sol_licituds": sol_licituds,
                    "Assignacions_finals": tipus_counts.get(t, 0),
                    "% Estrangers": round(100 * estr / max(1, asign_finals), 1),
                }
            )
        if not resum_rows:
            resum_rows.append(
                {
                    "Sorteig": sorteig,
                    "Tipus": "Indeterminat",
                    "Assignacions_previstes": sum(previs_counts.values()),
                    "Sol_licituds": sol_licituds,
                    "Assignacions_finals": asign_finals,
                    "% Estrangers": round(100 * estr / max(1, asign_finals), 1),
                }
            )
        resum = pd.DataFrame(resum_rows)
        for p, v in parr_counts.items():
            resum[p] = v
        resum_sorteigs.append(resum)

    capture_cols = [s.replace(" ", "_") for s in ESPECIE_SORTEIGS[especie]]
    resultat["te_capture"] = resultat[capture_cols].apply(
        lambda r: r.fillna(0).gt(0).any(), axis=1
    )
    resultat["Nou_Anys_sense_captura"] = resultat.apply(
        lambda r: r["anys_sense_captura"] + 1 if not r["te_capture"] else 0, axis=1
    )
    resultat["Nova_prioritat"] = resultat.apply(
        lambda r: (
            4
            if any(str(r[f"Tipus_{c}"]).find("Mascle") >= 0 for c in capture_cols)
            else (4 if r["te_capture"] else 2)
        ),
        axis=1,
    )
    resultat.drop(columns=["te_capture"], inplace=True)
    return resultat, resum_sorteigs


# ── DRAW WITH COLLES (IS TCC) ────────────────────────────────────────────────


def assignar_isards_sorteig_csv(df, total_captures, seed=None):
    if total_captures <= 0:
        raise ValueError("total_captures ha de ser > 0 (reviseu 'Quantitat').")

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    required = {"ID", "Modalitat", "Prioritat", "Colla_ID", "anys_sense_captura"}
    if not required.issubset(df.columns):
        raise ValueError(f"Falten columnes: {required - set(df.columns)}")

    df = df.copy()
    df["adjudicats"] = 0
    df["ordre"] = np.nan
    df["Estranger"] = df["Estranger"].apply(normalitza_estranger)

    # Calcula límit global d’estrangers
    estranger_limit = math.floor(0.1 * total_captures)

    # Divideix per modalitat
    df_colla = df[df["Modalitat"] == "A"]
    df_indiv = df[df["Modalitat"] == "B"]

    total_applicants = len(df_colla) + len(df_indiv)
    if total_applicants == 0:
        raise ValueError("No hi ha cap caçador amb Modalitat A o B")

    estranger_limit_A = round(estranger_limit * len(df_colla) / total_applicants)
    estranger_limit_B = estranger_limit - estranger_limit_A

    ordre_counter = 1
    estrangers_A = 0
    estrangers_B = 0

    # Decideix quantes captures per modalitat
    ratio = math.ceil(total_applicants / total_captures)
    n_indiv = round(total_captures * len(df_indiv) / total_applicants)
    n_colla = total_captures - n_indiv

    # assignació colles (Modalitat A)
    colles = df_colla.groupby("Colla_ID").size().reset_index(name="caçadors")
    colles["assignats"] = (colles["caçadors"] // ratio).astype(int)
    leftover = n_colla - colles["assignats"].sum()
    for _ in range(leftover):
        colles["rati"] = colles["assignats"] / colles["caçadors"]
        cid = (
            colles.loc[np.isclose(colles["rati"], colles["rati"].min())]
            .sample(1, random_state=rng)["Colla_ID"]
            .iat[0]
        )
        colles.loc[colles["Colla_ID"] == cid, "assignats"] += 1

    for _, row in colles.iterrows():
        cid, to_assign = row["Colla_ID"], int(row["assignats"])
        while to_assign:
            sub = df[
                (df["Modalitat"] == "A")
                & (df["Colla_ID"] == cid)
                & (df["adjudicats"] == 0)
            ]
            if sub.empty:
                break
            group = sub.copy()
            group["rand"] = rng.random(len(group))
            idxs = group.sort_values(
                ["Prioritat", "anys_sense_captura", "rand"],
                ascending=[True, False, True],
            ).index[: min(to_assign, len(group))]

            for idx in idxs:
                if (
                    df.at[idx, "Estranger"] == "si"
                    and estrangers_A >= estranger_limit_A
                ):
                    continue  # límit d’estrangers colla assolit
                if df.at[idx, "adjudicats"] == 0:
                    df.at[idx, "ordre"] = ordre_counter
                    df.at[idx, "adjudicats"] = 1
                    ordre_counter += 1
                    if df.at[idx, "Estranger"] == "si":
                        estrangers_A += 1
                    to_assign -= 1

    # assignació individus (Modalitat B)
    rem = n_indiv
    while rem:
        sub = df[(df["Modalitat"] == "B") & (df["adjudicats"] == 0)]
        if sub.empty:
            break
        group = sub.copy()
        group["rand"] = rng.random(len(group))
        idxs = group.sort_values(
            ["Prioritat", "anys_sense_captura", "rand"], ascending=[True, False, True]
        ).index[: min(rem, len(group))]

        for idx in idxs:
            if df.at[idx, "Estranger"] == "si" and estrangers_B >= estranger_limit_B:
                continue  # límit d’estrangers individual assolit
            if df.at[idx, "adjudicats"] == 0:
                df.at[idx, "ordre"] = ordre_counter
                df.at[idx, "adjudicats"] = 1
                ordre_counter += 1
                if df.at[idx, "Estranger"] == "si":
                    estrangers_B += 1
                rem -= 1

    # Actualitzacions finals
    df["nova_prioritat"] = df.apply(
        lambda r: 5 + r["adjudicats"] - 1 if r["adjudicats"] > 0 else r["Prioritat"],
        axis=1,
    )
    df["nova_prioritat Any següent"] = df["adjudicats"].apply(
        lambda x: 4 if x > 0 else 2
    )
    df["nou_anys_sense_captura"] = df.apply(
        lambda r: 0 if r["adjudicats"] else r["anys_sense_captura"] + 1, axis=1
    )
    df["ordre"] = df["ordre"].astype("Int64")
    return df


# ── (Additional helper functions assignar_captura_csv & assignar_captura_parroquial_csv unchanged) ──
#    ↳ They are long but identical to what you pasted, no structural fix needed.


# ── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Hide Streamlit's default page navigation */
    header[data-testid="stHeader"] {display: none;}
    section[data-testid="stSidebarNav"],
    nav[data-testid="stSidebarNav"],
    ul[data-testid="stSidebarNavItems"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    section = option_menu(
        "Menú",
        ["Sorteig", "Dashboard"],
        icons=["dice-5", "bar-chart"],
        default_index=0,
    )
    st.session_state["section"] = section  # remember choice

if st.session_state.get("section") == "Dashboard":
    st.switch_page("pages/Dashboard.py")
st.title("App Sorteig Pla de Caça")

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
        - Per a cada codi de sorteig, la posició on s'ha adjudicat la captura. Si el caçador estava inscrit i no ha obtingut plaça apareix `0`; si no estava inscrit el valor és buit.
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
                    on_change=sanitize_indeterminat,
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
    # 1️⃣  Load & validate the CSVs                                       #
    # ------------------------------------------------------------------ #
    if not csv1 or not csv2:
        st.error("Cal carregar els dos CSV")
        st.stop()

    df1 = pd.read_csv(csv1, sep=";")
    df2 = pd.read_csv(csv2, sep=";")

    try:
        validar_csv2(df2)
        (validar_csv_isard if especie == "Isard" else validar_csv_altres)(df1)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # ------------------------------------------------------------------ #
    # 2️⃣  IS‑TCC: detect hunters without Modalitat                       #
    # ------------------------------------------------------------------ #
    ids_to_skip = []  # ← will hold the IDs we really want to ignore

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
    # 3️⃣  Drop those IDs only from IS TCC                               #
    # ------------------------------------------------------------------ #
    if ids_to_skip:
        mask = (df2["Codi_Sorteig"] == "IS TCC") & (df2["ID"].isin(ids_to_skip))
        df2 = df2.loc[~mask].copy()

    # ------------------------------------------------------------------ #
    # 4️⃣  Build the configuration DataFrame from the UI inputs          #
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
    # 5️⃣  Run the draw and show results                                 #
    # ------------------------------------------------------------------ #
    try:
        resultat, resums = processar_sorteigs(df1, df2, config_df, especie, seed)
    except Exception as exc:
        st.error(f"🚫 Error en el sorteig: {exc}")
        st.stop()
    st.session_state["resultat"] = resultat  # full table, ~ID × columns
    st.session_state["resums"] = resums  # list of per-draw summaries
    st.subheader("Resultats")
    st.dataframe(resultat, use_container_width=True)

    st.download_button(
        "Descarregar CSV",
        resultat.to_csv(index=False).encode("utf-8"),
        file_name="resultats.csv",
    )

    # ------------------------------------------------------------------ #
    # 6️⃣  Clean‑up session flags so next run starts fresh                #
    # ------------------------------------------------------------------ #
    st.session_state.pop("confirm_missing_mod", None)
    st.session_state.pop("ids_to_skip_tcc", None)
    st.session_state["run_draw"] = False
