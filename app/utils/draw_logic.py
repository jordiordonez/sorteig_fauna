import math
import numpy as np
import streamlit as st
import unicodedata


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

def validar_csv2(df):
    required = {"ID", "Codi_Sorteig"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Falten columnes: {', '.join(sorted(missing))}")


def colles_per_sota_minim(df1, df2, zones):
    """Colles amb menys membres que l'efectiu mínim exigit (art. 56.1.a).

    Retorna una llista de tuples ``(zona, colla_id, mida, minim)`` per a les
    colles per sota del mínim en zones amb modalitat A/B. Llista buida si no
    n'hi ha cap. Serveix per aturar el sorteig i demanar la correcció del CSV
    d'inscrits.
    """
    if not {"Modalitat", "Colla_ID"}.issubset(df1.columns):
        return []
    curtes = []
    for z in zones:
        if not z.get("modalitat"):
            continue
        minim = int(z.get("min_colla", 0) or 0)
        if minim <= 0:
            continue
        inscrits = df2[df2["Codi_Sorteig"] == z["nom"]]["ID"]
        colles = df1[
            df1["ID"].isin(inscrits) & (df1["Modalitat"] == "A")
        ].dropna(subset=["Colla_ID"])
        for cid, mida in colles.groupby("Colla_ID").size().items():
            if mida < minim:
                curtes.append((z["nom"], cid, int(mida), minim))
    return curtes


# ── HELPER: APPORTIONMENT (largest remainder) ────────────────────────────────

def _round_meitat(x):
    """Arrodoniment a l'enter més proper amb els mitjos cap amunt (4,5 → 5).
    S'usa per al total reservat d'un vedat (art. 55.1.a: «50% com a mínim»)."""
    return int(math.floor(x + 0.5))


def _reparteix_residu_mes_gran(total, pesos):
    """Reparteix `total` captures (enter) entre les claus de `pesos` pel mètode
    del residu més gran. La suma dels resultats és sempre exactament `total`."""
    suma = sum(pesos.values())
    if total <= 0 or suma <= 0:
        return {k: 0 for k in pesos}
    exacte = {k: total * p / suma for k, p in pesos.items()}
    quota = {k: int(math.floor(v)) for k, v in exacte.items()}
    falten = total - sum(quota.values())
    if falten > 0:
        ordenat = sorted(
            exacte, key=lambda k: exacte[k] - math.floor(exacte[k]), reverse=True
        )
        for k in ordenat[:falten]:
            quota[k] += 1
    return quota


# ── HELPER: ORDER A POOL ─────────────────────────────────────────────────────

def _ordena_pool(pool, extra_cols=None):
    """Índexs d'un pool ordenats per prioritat (asc), amb `extra_cols`
    intercalades tot seguit, i després anys sense captura (desc) i rand (asc)."""
    extra = extra_cols or []
    cols = ["Prioritat"] + extra + ["anys_sense_captura", "rand"]
    asc = [True] + [True] * len(extra) + [False, True]
    return pool.sort_values(cols, ascending=asc).index


# ── HELPER: CHOOSE NEXT CANDIDATE (vedats amb prioritat parroquial) ──────────

def tria_candidat(df, assignats_parr, cupos, parr_rank, estr_cnt, estranger_limit, rng):
    """Tria el següent adjudicatari aplicant els criteris JERÀRQUICS de l'art. 55.1:

    1r  Prioritat individual (art. 54) — mana sempre.
    2n  A igualtat de prioritat, els residents a una parròquia del vedat que
        encara no hagi arribat al seu cupo passen davant dels no residents.
    3r  A igualtat, l'ordre de les parròquies sortejat (art. 55.1.b).
    (i, com a últim desempat, anys sense captura i atzar.)

    El cupo parroquial actua com un **sostre de preferència**: quan una parròquia
    l'assoleix, els seus residents deixen de tenir preferència (però segueixen
    competint per prioritat). Retorna l'índex triat o None si no hi ha candidats.
    """
    pool = df[~df["assigned"]].copy()
    if pool.empty:
        return None
    if estr_cnt >= estranger_limit:
        pool = pool[pool["Estranger"] == "no"]
    if pool.empty:
        return None
    pool["rand"] = rng.random(len(pool))

    if cupos and "Parroquia" in pool.columns:
        pool["quota_flag"] = pool["Parroquia"].apply(
            lambda p: 1 if assignats_parr.get(p, 0) < cupos.get(p, 0) else 0
        )
        big = len(parr_rank) + 1
        pool["parr_rank"] = pool.apply(
            lambda r: parr_rank.get(r["Parroquia"], big) if r["quota_flag"] == 1 else big,
            axis=1,
        )
        order_cols = ["Prioritat", "quota_flag", "parr_rank", "anys_sense_captura", "rand"]
        asc = [True, False, True, False, True]
    else:
        order_cols = ["Prioritat", "anys_sense_captura", "rand"]
        asc = [True, False, True]

    return pool.sort_values(order_cols, ascending=asc).index[0]


# ── HELPER: INDIVIDUAL DRAW (no colles) ──────────────────────────────────────

def sorteig_individual(
    df, tipus_quant, ordre_aleatori, parroquies_frac, reserva_frac, estranger_pct, rng
):
    import pandas as pd
    df = df.copy()
    df["Estranger"] = df["Estranger"].apply(normalitza_estranger)
    if "Parroquia" in df.columns:
        df["Parroquia"] = df["Parroquia"].apply(normalitza_parroquia)
    df["assigned"] = False

    captures_pool = [t for t, q in tipus_quant for _ in range(int(q))]
    total_caps = len(captures_pool)
    estranger_limit = math.floor(estranger_pct / 100.0 * total_caps)

    es_vedat = bool(parroquies_frac) and reserva_frac > 0
    if es_vedat:
        # Cupo parroquial ENTER (sostre de preferència), pel residu més gran sobre
        # el 50% del total (mitjos amunt): p. ex. 24 -> ALV 12, SJL 6, LM 6, EE 0.
        reserva_total = _round_meitat(total_caps * reserva_frac)
        cupos = _reparteix_residu_mes_gran(reserva_total, parroquies_frac)
        parroquies = [str(p) for p in parroquies_frac.keys()]
        ordre_parr = (
            [str(p) for p in rng.permutation(parroquies)]
            if len(parroquies) > 1 else parroquies
        )
        parr_rank = {p: i for i, p in enumerate(ordre_parr)}
        traça = {"reserva_total": reserva_total, "ordre_parroquial": ordre_parr, "cupos": cupos}
    else:
        cupos, parr_rank, traça = {}, {}, None

    assignats_parr = {p: 0 for p in cupos}
    guanyadors = []
    estrangers = 0
    while len(guanyadors) < total_caps:
        idx = tria_candidat(
            df, assignats_parr, cupos, parr_rank, estrangers, estranger_limit, rng
        )
        if idx is None:
            break
        df.at[idx, "assigned"] = True
        guanyadors.append(idx)
        if df.at[idx, "Estranger"] == "si":
            estrangers += 1
        p = df.at[idx, "Parroquia"] if "Parroquia" in df.columns else None
        if p in assignats_parr:
            assignats_parr[p] += 1

    # Assignació de tipus als guanyadors, en l'ordre d'adjudicació (que és l'ordre
    # de prioritat). Si l'ordre de la zona és aleatori, es barreja el pool.
    if ordre_aleatori:
        pool_tipus = list(captures_pool)
        rng.shuffle(pool_tipus)
    else:
        pool_tipus = captures_pool

    df["ordre"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    df["tipus"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for pos, idx in enumerate(guanyadors):
        df.at[idx, "ordre"] = pos + 1
        if pos < len(pool_tipus):
            df.at[idx, "tipus"] = pool_tipus[pos]

    cols = ["ID", "ordre", "tipus", "Estranger"]
    if "Parroquia" in df.columns:
        cols.append("Parroquia")
    out = df[cols].copy()
    out["ordre"] = out["ordre"].astype("Int64")
    out.attrs["traça"] = traça  # cupo parroquial (sostre) i ordre sortejat
    return out


# ── MAIN: PROCESSAR SORTEIGS ────────────────────────────────────────────────

@st.cache_data
def processar_sorteigs(df1, df2, zones, especie, seed):
    import pandas as pd
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    # Una zona amb modalitat A/B (avui, l'IS TCC) admet participants amb
    # Modalitat encara que no constin al CSV d'inscrits.
    has_modalitat = any(z.get("modalitat") for z in zones)

    ids_totals = df2["ID"].unique()
    if has_modalitat and "Modalitat" in df1.columns:
        extra = df1.loc[df1["Modalitat"].notna() & ~df1["ID"].isin(ids_totals), "ID"]
        ids_totals = np.union1d(ids_totals, extra)
    base = df1[df1["ID"].isin(ids_totals)].drop_duplicates("ID")

    cols = ["ID"]
    if has_modalitat and {"Modalitat", "Colla_ID"}.issubset(df1.columns):
        cols.extend(["Modalitat", "Colla_ID"])
    cols.extend(["Prioritat", "anys_sense_captura", "Estranger"])
    if "Parroquia" in base.columns:
        cols.append("Parroquia")

    resultat = base[cols].copy()

    captures_prev = {id_: 0 for id_ in resultat["ID"]}
    resum_sorteigs = []
    traces_vedats = {}  # zona -> {ordre_parroquial, cupos, bloc_reservat}

    for zona in zones:
        sorteig = zona["nom"]
        is_modalitat = bool(zona.get("modalitat"))
        es_vedat = zona.get("tipus") == "Vedat"
        captures = zona.get("captures", [])
        estranger_pct = float(zona.get("estranger_pct", 10.0))
        reserva_frac = float(zona.get("reserva_pct") or 0) / 100.0
        parroquies_frac = (
            {p: float(v) / 100.0 for p, v in (zona.get("parroquies") or {}).items()}
            if es_vedat
            else {}
        )

        col_base = sorteig.replace(" ", "_")
        resultat[col_base] = np.nan
        resultat[f"Tipus_{col_base}"] = np.nan

        subset = df2[df2["Codi_Sorteig"] == sorteig].copy()
        if is_modalitat and "Modalitat" in df1.columns:
            extra_ids = df1.loc[
                df1["Modalitat"].notna() & ~df1["ID"].isin(df2["ID"]), "ID"
            ]
            if not extra_ids.empty:
                subset = pd.concat(
                    [subset, pd.DataFrame({"ID": extra_ids, "Codi_Sorteig": sorteig})],
                    ignore_index=True,
                )

        total_quant = sum(int(c.get("quantitat", 0)) for c in captures)
        if subset.empty or total_quant <= 0:
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
        if is_modalitat:
            part = part[part["Modalitat"].isin(["A", "B"])].copy()
        subset_ids = set(part["ID"])

        if is_modalitat:
            asignats = assignar_isards_sorteig_csv(
                part,
                total_quant,
                estranger_pct=estranger_pct,
                seed=rng.randint(0, 2**31 - 1),
            )
            # El tipus només s'assigna als ADJUDICATARIS (ordre no nul); els
            # substituts no han de portar cap tipus de captura.
            primer = captures[0].get("tipus") if captures else []
            tipus_label = "+".join(primer or [])
            asignats["tipus"] = np.nan
            asignats.loc[asignats["ordre"].notna(), "tipus"] = tipus_label
        else:
            tipus_quant = []
            for c in captures:
                tp = c.get("tipus") or []
                tp = ["Indeterminat"] if "Indeterminat" in tp else tp
                tipus_quant.append(("+".join(tp), int(c.get("quantitat", 0))))
            asignats = sorteig_individual(
                part,
                tipus_quant,
                bool(zona.get("aleatori", True)),
                parroquies_frac,
                reserva_frac,
                estranger_pct,
                np.random.RandomState(rng.randint(0, 2**31 - 1)),
            )
            traça_zona = asignats.attrs.get("traça")
            if traça_zona:
                traces_vedats[sorteig] = traça_zona

        asignats = asignats.rename(
            columns={"ordre": f"ordre_{col_base}", "tipus": f"tipus_{col_base}"}
        )

        estr = asignats[asignats["Estranger"] == "si"][f"ordre_{col_base}"].count()

        merge_cols = ["ID", f"ordre_{col_base}", f"tipus_{col_base}"]
        resultat = resultat.merge(asignats[merge_cols], on="ID", how="left")

        resultat[col_base] = resultat[f"ordre_{col_base}"]
        mask_no_cap = resultat["ID"].isin(subset_ids) & resultat[col_base].isna()
        if mask_no_cap.any():
            # Convert column to object dtype to allow mixed types (numbers and strings)
            resultat[col_base] = resultat[col_base].astype('object')
            
            pending_ids = resultat.loc[mask_no_cap, "ID"]
            tmp = part[part["ID"].isin(pending_ids)].copy()
            tmp["rand"] = rng.random(len(tmp))
            tmp.sort_values(
                ["Prioritat", "anys_sense_captura", "rand"],
                ascending=[True, False, True],
                inplace=True,
            )
            labels = [f"s{i+1}" for i in range(len(tmp))]
            
            # Assign substitute labels one by one to avoid dtype conflicts
            for i, id_val in enumerate(tmp["ID"]):
                resultat.loc[resultat["ID"] == id_val, col_base] = labels[i]
        resultat[f"Tipus_{col_base}"] = resultat[f"tipus_{col_base}"]

        winners = asignats.loc[asignats[f"ordre_{col_base}"].notna(), "ID"]
        for wid in winners:
            captures_prev[wid] = captures_prev.get(wid, 0) + 1

        resultat.drop(columns=[f"ordre_{col_base}", f"tipus_{col_base}"], inplace=True)

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
        for c in captures:
            tp = c.get("tipus") or []
            tip_label = "+".join(tp) if tp else "Indeterminat"
            previs_counts[tip_label] = previs_counts.get(tip_label, 0) + int(c.get("quantitat", 0))

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

    capture_cols = [z["nom"].replace(" ", "_") for z in zones]
    
    # Function to convert values: s1, s2, etc. -> 0, numeric values stay as numeric
    def convert_capture_value(val):
        if pd.isna(val):
            return 0
        val_str = str(val)
        # If it starts with 's', it's a substitute position -> convert to 0
        if val_str.startswith('s'):
            return 0
        # Otherwise try to convert to numeric, default to 0 if fails
        try:
            return pd.to_numeric(val, errors='coerce')
        except:
            return 0
    
    # Apply the conversion and check for captures > 0
    resultat["te_capture"] = resultat[capture_cols].apply(
        lambda r: r.apply(convert_capture_value).fillna(0).gt(0).any(),
        axis=1,
    )
    resultat["Nou_Anys_sense_captura"] = resultat.apply(
        lambda r: r["anys_sense_captura"] + 1 if not r["te_capture"] else 0, axis=1
    )

    # `Nova_prioritat` és una estimació de sortida: 4 als adjudicataris, 2 als
    # que s'han presentat i no han guanyat. Les prioritats 1 (femella caçada,
    # art. 54.1.a) i 3 (no presentats) les fixen els tècnics del Govern, perquè
    # depenen d'informació que l'app no té (captura efectuada, cens complet).
    resultat["Nova_prioritat"] = resultat["te_capture"].map(lambda x: 4 if x else 2)

    resultat.drop(columns=["te_capture"], inplace=True)
    resultat.attrs["traces_vedats"] = traces_vedats
    return resultat, resum_sorteigs


# ── DRAW WITH COLLES (IS TCC) ────────────────────────────────────────────────

def _round_sorteig(x, rng):
    """Arrodoneix al valor enter més proper. En cas d'empat exacte a .5, sorteja
    cap a quin costat cau (art. 56: elimina el romanent decimal)."""
    baix = math.floor(x)
    frac = x - baix
    if frac < 0.5:
        return int(baix)
    if frac > 0.5:
        return int(baix) + 1
    return int(baix) + (1 if rng.random() < 0.5 else 0)


def assignar_isards_sorteig_csv(df, total_captures, estranger_pct=10.0, seed=None):
    """Sorteig de l'isard en TCC amb modalitat de colla (A) i individual (B),
    segons els art. 56 i 57 del Reglament de caça del 4-6-2025.

    * Ràtio EXACTA = sol·licitants / captures (art. 56.2), sense arrodonir.
    * Captures de cada modalitat = arrodoniment al més proper de (n_mod / ràtio);
      en empat a .5, se sorteja el sentit. Així no queda romanent decimal.
    * Base de cada colla = part entera(mida / ràtio) (art. 56.4.a).
    * Captures restants de colla = se sortegen entre els no adjudicataris de la
      modalitat colla, per prioritat, amb un MÀXIM D'UNA per colla (art. 56.4.b).
    * Sostre d'estrangers ÚNIC per al sorteig (no partit per modalitat), sobre
      les captures ofertes (art. 53.4).
    """
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
    df["rand"] = rng.random(len(df))

    estranger_limit = math.floor(estranger_pct / 100.0 * total_captures)

    idx_A = df.index[df["Modalitat"] == "A"]
    idx_B = df.index[df["Modalitat"] == "B"]
    nA, nB = len(idx_A), len(idx_B)
    total_applicants = nA + nB
    if total_applicants == 0:
        raise ValueError("No hi ha cap caçador amb Modalitat A o B")

    ratio = total_applicants / total_captures            # exacta (art. 56.2)
    n_colla = _round_sorteig(nA / ratio, rng)            # al més proper
    n_colla = max(0, min(n_colla, total_captures))
    n_indiv = total_captures - n_colla                   # sense romanent decimal

    estat = {"ordre": 1, "estr": 0}

    def assigna(idx):
        """Adjudica una captura a `idx`. Retorna False si és estranger i ja s'ha
        arribat al sostre (llavors queda fora, sense penjar el sorteig)."""
        if df.at[idx, "Estranger"] == "si" and estat["estr"] >= estranger_limit:
            return False
        df.at[idx, "ordre"] = estat["ordre"]
        df.at[idx, "adjudicats"] = 1
        estat["ordre"] += 1
        if df.at[idx, "Estranger"] == "si":
            estat["estr"] += 1
        return True

    # ── Modalitat colla: base = part entera(mida / ràtio) ──
    colles = df.loc[idx_A].groupby("Colla_ID").size()
    assignats_colla = 0
    for cid, mida in colles.items():
        nbase = int(math.floor(mida / ratio))
        if nbase <= 0:
            continue
        membres = df.loc[idx_A]
        membres = membres[membres["Colla_ID"] == cid]
        fets = 0
        for idx in _ordena_pool(membres):
            if fets >= nbase:
                break
            if assigna(idx):
                fets += 1
        assignats_colla += fets

    # ── Captures restants de colla: sorteig entre no adjudicataris de la
    #    modalitat colla, per prioritat, màxim UNA per colla (art. 56.4.b) ──
    restants = n_colla - assignats_colla
    if restants > 0:
        pool = df.loc[idx_A]
        pool = pool[pool["adjudicats"] == 0]
        colles_amb_restant = set()
        fets = 0
        for idx in _ordena_pool(pool):
            if fets >= restants:
                break
            cid = df.at[idx, "Colla_ID"]
            if cid in colles_amb_restant:
                continue
            if assigna(idx):
                colles_amb_restant.add(cid)
                fets += 1

    # ── Modalitat individual ──
    pool = df.loc[idx_B]
    pool = pool[pool["adjudicats"] == 0]
    fets = 0
    for idx in _ordena_pool(pool):
        if fets >= n_indiv:
            break
        if assigna(idx):
            fets += 1

    df["ordre"] = df["ordre"].astype("Int64")
    return df
