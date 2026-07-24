"""Proves de la lògica de sorteig contra els fitxers de `proves/`.

Comproven el comportament EXIGIT pel Reglament de caça del 4-6-2025 després dels
afinaments de juliol del 2026 (vegeu `propostes.md`). Executables amb pytest o
directament: ``python tests/test_sorteig.py``.
"""
import os
import sys
import copy
import math
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import pandas as pd  # noqa: E402
from utils import draw_logic as d  # noqa: E402
from utils import config  # noqa: E402

PROVES = os.path.join(ROOT, "proves")
CAPS_IS = {"IS TCC": 175, "IS VCRS": 16, "IS VCX": 8, "IS VCE": 48}

# Els fitxers de `proves/` contenen dades reals de caçadors i NO es publiquen al
# repositori. Les proves que en depenen se salten soles si no hi són; les proves
# de lògica pura (apportionment, arrodoniment, cas parroquial sintètic) sempre
# corren.
_FITXERS = [os.path.join(PROVES, "Prioritats_IS(gpt4).csv"),
            os.path.join(PROVES, "inscrits_IS(gpt4).csv")]
DADES_DISPONIBLES = all(os.path.exists(f) for f in _FITXERS)


class _Skip(Exception):
    pass


def _requereix_dades():
    if not DADES_DISPONIBLES:
        try:
            import pytest
            pytest.skip("falten els fitxers de proves/ (dades reals, no publicades)")
        except ImportError:
            raise _Skip()


def _isard():
    _requereix_dades()
    p = pd.read_csv(os.path.join(PROVES, "Prioritats_IS(gpt4).csv"), sep=";", encoding="utf-8-sig")
    ins = pd.read_csv(os.path.join(PROVES, "inscrits_IS(gpt4).csv"), sep=";", encoding="utf-8-sig")
    zones = config.default_zones("Isard")
    for z in zones:
        z["captures"] = [{"tipus": ["Indeterminat"], "quantitat": CAPS_IS[z["nom"]]}]
    return p, ins, zones


def _run_isard(seed=42):
    p, ins, zones = _isard()
    return d.processar_sorteigs(p.copy(), ins.copy(), copy.deepcopy(zones), "Isard", seed=seed)


# ── HELPERS D'APPORTIONMENT ──────────────────────────────────────────────────

def test_residu_mes_gran():
    fr = {"ALV": 0.522, "SJ": 0.241, "LM": 0.234, "EE": 0.003}
    assert d._reparteix_residu_mes_gran(24, fr) == {"ALV": 12, "SJ": 6, "LM": 6, "EE": 0}


def test_cupos_redistribucio_iterativa():
    fr = {"ALV": 0.522, "SJ": 0.241, "LM": 0.234, "EE": 0.003}
    # La Massana només 4 censats: es refà el repartiment dels 20 restants.
    assert d._cupos_parroquials(24, fr, {"ALV": 99, "SJ": 99, "LM": 4, "EE": 0}) == \
        {"ALV": 14, "SJ": 6, "LM": 4, "EE": 0}
    # Pocs censats en total (20): el bloc no s'omple, la resta anirà al sorteig obert.
    assert sum(d._cupos_parroquials(24, fr, {"ALV": 10, "SJ": 6, "LM": 4, "EE": 0}).values()) == 20


def test_round_sorteig():
    class _R:
        def __init__(self, v): self.v = v
        def random(self): return self.v
    assert d._round_sorteig(84.6774, _R(0.9)) == 85
    assert d._round_sorteig(90.3226, _R(0.9)) == 90
    assert d._round_sorteig(3.5, _R(0.3)) == 4   # empat: r<0.5 -> puja
    assert d._round_sorteig(3.5, _R(0.7)) == 3   # empat: r>0.5 -> baixa


# ── B — TIPUS NOMÉS ALS ADJUDICATARIS ────────────────────────────────────────

def test_substituts_sense_tipus():
    r, _ = _run_isard()
    sub = r[r["IS_TCC"].astype(str).str.startswith("s")]
    assert len(sub) > 0
    assert sub["Tipus_IS_TCC"].dropna().empty
    adj = r[pd.to_numeric(r["IS_TCC"], errors="coerce").notna()]
    assert (adj["Tipus_IS_TCC"] == "Indeterminat").all()


# ── C — MODALITAT A/B (art. 56) ──────────────────────────────────────────────

def test_modalitat_ratio_i_restants():
    p, ins, _ = _isard()
    tcc = ins[ins["Codi_Sorteig"] == "IS TCC"].merge(p, on="ID")
    nA = (tcc["Modalitat"] == "A").sum()
    nB = (tcc["Modalitat"] == "B").sum()
    ratio = (nA + nB) / 175
    r, _ = _run_isard()
    adj = r[pd.to_numeric(r["IS_TCC"], errors="coerce").notna()]
    assert len(adj) == 175  # cap captura perduda en romanent
    wA = adj[adj["Modalitat"] == "A"]
    assert len(wA) == round(nA / ratio)
    sizes = tcc[tcc["Modalitat"] == "A"].groupby("Colla_ID").size()
    base = (sizes // ratio).astype(int)
    restants = round(nA / ratio) - base.sum()
    assert restants <= len(sizes)                 # màxim una per colla és possible
    per = wA.groupby("Colla_ID").size()
    assert ((per - base).dropna() <= 1).all()     # cap colla amb més d'1 restant


# ── D — VEDATS: BLOC RESERVAT NOMÉS PARROQUIANS (art. 55.1) ──────────────────

def _norm(x):
    return d.normalitza_parroquia(x)


def test_vedat_bloc_no_cobert_passa_al_general():
    """Si no hi ha prou censats concernits per omplir el 50% reservat, les
    captures no cobertes cauen al sorteig general (art. 55.3, cas parroquial)."""
    import numpy as np
    rows = [
        {"ID": 100 + i, "Prioritat": 2, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "La Massana"} for i in range(2)
    ] + [
        {"ID": 200 + i, "Prioritat": 2, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "Encamp"} for i in range(20)
    ]
    df = pd.DataFrame(rows)
    out = d.sorteig_individual(
        df, [("Indeterminat", 8)], True, {"La Massana": 1.0}, 0.5, 10.0,
        np.random.RandomState(1),
    )
    adj = out[out["ordre"].notna()]
    assert len(adj) == 8  # totes les captures s'adjudiquen
    locals_ = adj[adj["Parroquia"] == "La Massana"]
    assert len(locals_) == 2  # només els 2 censats que existeixen
    assert out.attrs["traça"]["cupos_per_tipus"]["Indeterminat"] == {"La Massana": 2}


def test_reserva_per_tipus():
    """El 50% reservat es reparteix per tipus de captura, sense inflar per
    arrodoniment: 2 trofeus + 10 selectius -> reserva 6 = 1 trofeu + 5 selectius;
    3+3+3 -> reserva 5 = 2+2+1 (no 6)."""
    import numpy as np
    # Censats amb prioritat 2 i no residents amb prioritat 1 (millor): així els no
    # residents s'enduen tota la fase oberta i els censats es queden EXACTAMENT
    # amb la reserva, cosa que permet comprovar-la aïllada.
    rows = [
        {"ID": 100 + i, "Prioritat": 2, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "La Massana"} for i in range(40)
    ] + [
        {"ID": 200 + i, "Prioritat": 1, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "Encamp"} for i in range(40)
    ]
    df = pd.DataFrame(rows)

    out = d.sorteig_individual(
        df, [("Trofeu", 2), ("Selectiu", 10)], True, {"La Massana": 1.0}, 0.5, 10.0,
        np.random.RandomState(1),
    )
    assert out.attrs["traça"]["reserva_total"] == 6
    assert out.attrs["traça"]["reserva_per_tipus"] == {"Trofeu": 1, "Selectiu": 5}
    resid = out[out["ordre"].notna() & (out["Parroquia"] == "La Massana")]
    assert (resid["tipus"] == "Trofeu").sum() == 1
    assert (resid["tipus"] == "Selectiu").sum() == 5

    out2 = d.sorteig_individual(
        df, [("A", 3), ("B", 3), ("C", 3)], True, {"La Massana": 1.0}, 0.5, 10.0,
        np.random.RandomState(2),
    )
    assert out2.attrs["traça"]["reserva_total"] == 5  # 4,5 -> 5, no 6
    assert out2.attrs["traça"]["reserva_per_tipus"] == {"A": 2, "B": 2, "C": 1}


def test_vedat_bloc_reservat_nomes_parroquians():
    r, _ = _run_isard()
    casos = {
        "IS VCRS": ["Canillo", "Ordino"],
        "IS VCX": ["La Massana"],
        "IS VCE": ["La Massana", "Sant Julià de Lòria", "Andorra la Vella", "Escaldes-Engordany"],
    }
    for zona, parrs in casos.items():
        col = zona.replace(" ", "_")
        v = r[pd.to_numeric(r[col], errors="coerce").notna()].copy()
        v["o"] = pd.to_numeric(v[col])
        v = v.sort_values("o")
        v["pn"] = v["Parroquia"].apply(_norm)
        bloc = round(CAPS_IS[zona] * 0.5)
        primers = v.head(bloc)
        assert primers["pn"].isin(parrs).all(), f"{zona}: no residents al bloc reservat"
        assert v["pn"].isin(parrs).sum() >= bloc, f"{zona}: censats per sota del 50%"


# ── E / H — SOSTRE D'ESTRANGERS ──────────────────────────────────────────────

def test_sostre_estrangers_per_zona():
    r, _ = _run_isard()
    for zona, caps in CAPS_IS.items():
        col = zona.replace(" ", "_")
        v = r[pd.to_numeric(r[col], errors="coerce").notna()]
        estr = v["Estranger"].astype(str).str.upper().isin(["SI", "SÍ"]).sum()
        assert estr <= math.floor(0.10 * caps)


def test_sostre_zero_no_penja():
    p, ins, zones = _isard()
    for z in zones:
        z["estranger_pct"] = 0.0
    r, _ = d.processar_sorteigs(p.copy(), ins.copy(), copy.deepcopy(zones), "Isard", seed=7)
    assert r is not None and len(r) > 0


# ── A — EFECTIU MÍNIM PER COLLA ──────────────────────────────────────────────

def test_colles_per_sota_minim():
    p, ins, zones = _isard()
    # Totes les colles de la prova compleixen el mínim de 6.
    assert d.colles_per_sota_minim(p, ins, zones) == []
    # Forcem una colla curta: deixem només 2 membres de la colla 1 inscrits a TCC.
    colla1 = p[(p["Modalitat"] == "A") & (p["Colla_ID"] == 1)]["ID"].tolist()
    treure = set(colla1[2:])  # deixa'n 2
    ins2 = ins[~((ins["Codi_Sorteig"] == "IS TCC") & (ins["ID"].isin(treure)))]
    curtes = d.colles_per_sota_minim(p, ins2, zones)
    assert any(z == "IS TCC" and cid == 1 and mida == 2 and minim == 6
               for (z, cid, mida, minim) in curtes)


# ── REPRODUCTIBILITAT ────────────────────────────────────────────────────────

def test_reproductible_amb_llavor():
    r1, _ = _run_isard(seed=123)
    r2, _ = _run_isard(seed=123)
    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = skips = 0
    for fn in funcs:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except _Skip:
            skips += 1
            print(f" SKIP  {fn.__name__} (falten dades a proves/)")
        except Exception as exc:
            fails += 1
            print(f" FAIL  {fn.__name__}: {exc}")
    resum = "TOTES OK" if not fails else f"{fails} FALLEN"
    if skips:
        resum += f" ({skips} saltades)"
    print("\n" + resum)
    sys.exit(1 if fails else 0)
