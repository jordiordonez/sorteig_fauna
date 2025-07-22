import math
import numpy as np
import streamlit as st
import unicodedata
from .constants import ESPECIE_SORTEIGS, VEDAT_PARRÒQUIES


def sanitize_indeterminat(key: str) -> None:
    """Ensure multiselect keeps only 'Indeterminat' if chosen."""
    val = st.session_state.get(key, [])
    if "Indeterminat" in val and len(val) > 1:
        st.session_state[key] = ["Indeterminat"]


# ── UTILITIES ────────────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    """Remove diacritics for easier matching."""
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
    import pandas as pd
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
    import pandas as pd
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

@st.cache_data
def processar_sorteigs(df1, df2, config, especie, seed):
    import pandas as pd
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
                part, total_cap, seed=rng.randint(0, 2**31 - 1)
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
                np.random.RandomState(rng.randint(0, 2**31 - 1)),
            )

        asignats = asignats.rename(
            columns={"ordre": f"ordre_{col_base}", "tipus": f"tipus_{col_base}"}
        )

        estr = asignats[asignats["Estranger"] == "si"][f"ordre_{col_base}"].count()

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
            previs_counts[tip_label] = previs_counts.get(tip_label, 0) + int(r["Quantitat"])

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
    import pandas as pd
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
    total_non_strangers = (df["Estranger"] == "no").sum()
    estranger_limit = math.floor(0.1 * total_captures)

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

    ratio = math.ceil(total_applicants / total_captures)
    n_indiv = round(total_captures * len(df_indiv) / total_applicants)
    n_colla = total_captures - n_indiv

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
                    continue
                if df.at[idx, "adjudicats"] == 0:
                    df.at[idx, "ordre"] = ordre_counter
                    df.at[idx, "adjudicats"] = 1
                    ordre_counter += 1
                    if df.at[idx, "Estranger"] == "si":
                        estrangers_A += 1
                    to_assign -= 1

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
                continue
            if df.at[idx, "adjudicats"] == 0:
                df.at[idx, "ordre"] = ordre_counter
                df.at[idx, "adjudicats"] = 1
                ordre_counter += 1
                if df.at[idx, "Estranger"] == "si":
                    estrangers_B += 1
                rem -= 1

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

