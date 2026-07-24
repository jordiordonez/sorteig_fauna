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

def test_cupo_parroquial_per_superficie():
    # Cupo enter (sostre de preferència) pel residu més gran: 24 -> 12/6/6/0.
    fr = {"ALV": 0.522, "SJ": 0.241, "LM": 0.234, "EE": 0.003}
    assert d._reparteix_residu_mes_gran(24, fr) == {"ALV": 12, "SJ": 6, "LM": 6, "EE": 0}


def test_round_meitat():
    assert d._round_meitat(24.0) == 24
    assert d._round_meitat(4.5) == 5   # 50% de 9, mitjos amunt
    assert d._round_meitat(4.4) == 4


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


# ── D — VEDATS: PREFERÈNCIA PARROQUIAL JERÀRQUICA (art. 55.1) ────────────────

def _norm(x):
    return d.normalitza_parroquia(x)


def test_vedat_prioritat_mana_sobre_residencia():
    """1r criteri: la prioritat individual. Un no resident amb millor prioritat
    guanya davant d'un resident amb pitjor prioritat."""
    import numpy as np
    rows = [
        {"ID": 100 + i, "Prioritat": 4, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "La Massana"} for i in range(20)   # residents, pitjor prioritat
    ] + [
        {"ID": 200 + i, "Prioritat": 2, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "Encamp"} for i in range(20)       # no residents, millor prioritat
    ]
    df = pd.DataFrame(rows)
    out = d.sorteig_individual(
        df, [("Indeterminat", 8)], True, {"La Massana": 1.0}, 0.5, 10.0,
        np.random.RandomState(1),
    )
    v = out[out["ordre"].notna()].sort_values("ordre")
    # Els no residents (prioritat 2) s'enduen les 8 captures; els residents (4) cap.
    assert (v["Parroquia"] == "Encamp").all()


def test_vedat_preferencia_com_a_sostre():
    """2n criteri: a IGUALTAT de prioritat, els residents passen davant, fins al
    seu cupo (sostre). Passat el sostre, deixen de tenir preferència."""
    import numpy as np
    # Tots prioritat 2. Residents amb anys=0, no residents amb anys=5 (millor
    # desempat final). Cupo La Massana = round(8*0.5)=4. Els 4 primers han de ser
    # residents (preferència); els 4 següents, no residents (ja sense preferència,
    # guanyen per anys).
    rows = [
        {"ID": 100 + i, "Prioritat": 2, "anys_sense_captura": 0,
         "Estranger": "no", "Parroquia": "La Massana"} for i in range(20)
    ] + [
        {"ID": 200 + i, "Prioritat": 2, "anys_sense_captura": 5,
         "Estranger": "no", "Parroquia": "Encamp"} for i in range(20)
    ]
    df = pd.DataFrame(rows)
    out = d.sorteig_individual(
        df, [("Indeterminat", 8)], True, {"La Massana": 1.0}, 0.5, 10.0,
        np.random.RandomState(1),
    )
    assert out.attrs["traça"]["cupos"] == {"La Massana": 4}
    v = out[out["ordre"].notna()].sort_values("ordre")
    assert (v.head(4)["Parroquia"] == "La Massana").all()   # sostre: 4 residents primer
    assert (v.tail(4)["Parroquia"] == "Encamp").all()       # passat el sostre, no residents
    resid = out[out["ordre"].notna() & (out["Parroquia"] == "La Massana")]
    assert len(resid) == 4  # exactament el cupo, ni més (no arriben per anys)


def test_vedat_prioritat_monotona_a_les_proves():
    """Sobre les dades reals: dins de cada vedat, la prioritat efectiva mai no
    disminueix a mesura que avança l'ordre d'adjudicació (la prioritat mana)."""
    r, _ = _run_isard()
    for zona in ["IS VCRS", "IS VCX", "IS VCE"]:
        col = zona.replace(" ", "_")
        v = r[pd.to_numeric(r[col], errors="coerce").notna()].copy()
        v["o"] = pd.to_numeric(v[col])
        v = v.sort_values("o")
        assert list(v["Prioritat"]) == sorted(v["Prioritat"]), f"{zona}: prioritat no monòtona"


def test_vedat_cupo_es_sostre_de_preferencia():
    """El cupo per parròquia limita la PREFERÈNCIA, no el total: els residents no
    reben preferència més enllà del cupo, però encara poden guanyar per prioritat."""
    r, _ = _run_isard()
    # A IS VCE el cupo de superfície és 12/6/6/0; comprovem que la traça el porta.
    t = r.attrs["traces_vedats"]["IS VCE"]
    assert t["cupos"] == {
        "Andorra la Vella": 12, "Sant Julià de Lòria": 6,
        "La Massana": 6, "Escaldes-Engordany": 0,
    }
    assert t["reserva_total"] == 24


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
