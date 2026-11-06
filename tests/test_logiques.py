"""Segona bateria de proves: totes les lògiques del sorteig amb dades SINTÈTIQUES.

No depèn de cap fitxer de `proves/` (dades reals). Es divideix en:

1. Proves dirigides, una per regla del Reglament de caça del 4-6-2025 (modificat el
   23-9-2026) tal com la descriu `manual.md`.
2. Invariants comprovats sobre molts sortejos aleatoris (sortejos «de propietat»):
   el que ha de ser cert sempre, sigui quina sigui la població i la configuració.
3. Els fitxers d'exemple que es descarreguen des de l'app funcionen tal com són.

Executables amb pytest o directament: ``python tests/test_logiques.py``.
"""
import ast
import copy
import math
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from utils import draw_logic as d  # noqa: E402
from utils import config  # noqa: E402

PARR = ["Canillo", "Encamp", "Ordino", "La Massana", "Andorra la Vella",
        "Sant Julià de Lòria", "Escaldes-Engordany"]


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _caps(n, tipus=("Indeterminat",)):
    return [{"tipus": list(tipus), "quantitat": n}]


def tcc(nom, n, **kw):
    z = {"nom": nom, "tipus": "TCC", "captures": _caps(n), "estranger_pct": 10.0}
    z.update(kw)
    return z


def vedat(nom, n, parroquies, **kw):
    z = {"nom": nom, "tipus": "Vedat", "reserva_pct": 50, "parroquies": parroquies,
         "captures": _caps(n), "estranger_pct": 10.0}
    z.update(kw)
    return z


def caçadors(rows):
    """rows: llista de dicts amb com a mínim ID; omple valors per defecte."""
    base = {"Prioritat": 2, "anys_sense_captura": 0, "Estranger": "No"}
    return pd.DataFrame([{**base, **r} for r in rows])


def inscrits(pairs):
    return pd.DataFrame(pairs, columns=["ID", "Codi_Sorteig"])


def run(p, ins, zones, especie="Altres", seed=1):
    return d.processar_sorteigs(p.copy(), ins.copy(), copy.deepcopy(zones), especie, seed=seed)


def col(z):
    return z["nom"].replace(" ", "_")


def num(v):
    x = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(x) else int(x)


def es_subst(v):
    return bool(re.fullmatch(r"s\d+( \(E\))?", str(v)))


def n_subst(v):
    return int(re.match(r"s(\d+)", str(v)).group(1))


def es_estr(v):
    return d.normalitza_estranger(v) == "si"


def guanyadors(r, z):
    """IDs adjudicataris d'una zona, en ordre d'adjudicació."""
    c = col(z)
    v = r[[c, "ID"]].copy()
    v["n"] = v[c].map(num)
    return v.dropna(subset=["n"]).sort_values("n")["ID"].tolist()


def substituts(r, z):
    c = col(z)
    v = r[r[c].map(es_subst)][["ID", c]].copy()
    v["n"] = v[c].map(n_subst)
    return v.sort_values("n")["ID"].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROVES DIRIGIDES
# ══════════════════════════════════════════════════════════════════════════════

# ── Prioritats (art. 54) ────────────────────────────────────────────────────

def test_prioritat_mana_en_tcc():
    p = caçadors([{"ID": f"P{pr}_{i}", "Prioritat": pr, "anys_sense_captura": 9 - pr}
                  for pr in (4, 3, 2, 1) for i in range(3)])
    z = tcc("Z", 5)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    g = guanyadors(r, z)
    prio = dict(zip(p.ID, p.Prioritat))
    assert [prio[i] for i in g] == [1, 1, 1, 2, 2]
    # I els substituts continuen per prioritat.
    assert [prio[i] for i in substituts(r, z)] == [2, 3, 3, 3, 4, 4, 4]


def test_desempat_anys_sense_captura():
    p = caçadors([{"ID": f"A{a}", "anys_sense_captura": a} for a in range(8)])
    z = tcc("Z", 3)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    assert guanyadors(r, z) == ["A7", "A6", "A5"]


def test_desempat_atzar_reproductible_i_variable():
    p = caçadors([{"ID": f"X{i}"} for i in range(20)])
    ins = inscrits([(i, "Z") for i in p.ID])
    z = tcc("Z", 3)
    a, _ = run(p, ins, [z], seed=5)
    b, _ = run(p, ins, [z], seed=5)
    assert guanyadors(a, z) == guanyadors(b, z)
    diferents = {tuple(guanyadors(run(p, ins, [z], seed=s)[0], z)) for s in range(8)}
    assert len(diferents) > 1, "amb empat total l'atzar ha de decidir"


def test_prioritat_4_mes_n_entre_zones():
    """Qui guanya a Z1 arriba a Z2 amb prioritat 5: perd davant d'una P4."""
    p = caçadors([{"ID": "A", "Prioritat": 1}, {"ID": "B", "Prioritat": 4}])
    zs = [tcc("Z1", 1), tcc("Z2", 1)]
    r, _ = run(p, inscrits([("A", "Z1"), ("A", "Z2"), ("B", "Z2")]), zs)
    assert guanyadors(r, zs[0]) == ["A"]
    assert guanyadors(r, zs[1]) == ["B"]


def test_prioritat_6_despres_de_dues_captures():
    """A guanya Z1 i Z2 (prioritat 6) i B només Z1 (prioritat 5): a Z3 guanya B,
    tot i que A té més anys sense captura."""
    p = caçadors([{"ID": "A", "Prioritat": 1, "anys_sense_captura": 9},
                  {"ID": "B", "Prioritat": 1, "anys_sense_captura": 0}])
    zs = [tcc("Z1", 2), tcc("Z2", 1), tcc("Z3", 1)]
    ins = inscrits([("A", "Z1"), ("B", "Z1"), ("A", "Z2"), ("A", "Z3"), ("B", "Z3")])
    r, _ = run(p, ins, zs)
    assert guanyadors(r, zs[2]) == ["B"]


def test_ordre_de_zones_importa():
    p = caçadors([{"ID": "A", "Prioritat": 1}, {"ID": "B", "Prioritat": 2}])
    ins = inscrits([("A", "Z1"), ("B", "Z1"), ("A", "Z2"), ("B", "Z2")])
    r1, _ = run(p, ins, [tcc("Z1", 1), tcc("Z2", 1)])
    r2, _ = run(p, ins, [tcc("Z2", 1), tcc("Z1", 1)])
    assert guanyadors(r1, tcc("Z1", 1)) == ["A"] and guanyadors(r1, tcc("Z2", 1)) == ["B"]
    assert guanyadors(r2, tcc("Z2", 1)) == ["A"] and guanyadors(r2, tcc("Z1", 1)) == ["B"]


# ── Vedats (art. 55.1) ──────────────────────────────────────────────────────

def test_vedat_resident_passa_davant_a_igualtat():
    p = caçadors([{"ID": f"R{i}", "Parroquia": "Ordino"} for i in range(4)]
                 + [{"ID": f"N{i}", "Parroquia": "Encamp", "anys_sense_captura": 5} for i in range(4)])
    z = vedat("V", 4, {"Ordino": 100})
    r, _ = run(p, inscrits([(i, "V") for i in p.ID]), [z])
    g = guanyadors(r, z)
    assert g[:2] == sorted(g[:2]) and all(i.startswith("R") for i in g[:2])  # cupo = 2
    assert all(i.startswith("N") for i in g[2:])  # passat el sostre, mana el desempat


def test_vedat_prioritat_mana_sobre_residencia():
    p = caçadors([{"ID": "R", "Prioritat": 3, "Parroquia": "Ordino"},
                  {"ID": "N", "Prioritat": 2, "Parroquia": "Encamp"}])
    z = vedat("V", 1, {"Ordino": 100})
    r, _ = run(p, inscrits([("R", "V"), ("N", "V")]), [z])
    assert guanyadors(r, z) == ["N"]


def test_parroquia_en_codi_numeric_i_variants():
    assert d.normalitza_parroquia(3) == "Ordino"
    assert d.normalitza_parroquia("6") == "Sant Julià de Lòria"
    assert d.normalitza_parroquia("sant julia de loria") == "Sant Julià de Lòria"
    assert d.normalitza_parroquia("Escaldes_Engordany") == "Escaldes-Engordany"
    assert d.normalitza_parroquia("SJ") == "Sant Julià de Lòria"
    assert d.normalitza_parroquia(None) is None
    assert d.normalitza_parroquia(float("nan")) is None
    assert d.normalitza_parroquia("Barcelona") is None


def test_parroquia_desconeguda_es_no_resident():
    p = caçadors([{"ID": "U", "Parroquia": "Barcelona", "anys_sense_captura": 9},
                  {"ID": "R", "Parroquia": "Ordino"}])
    z = vedat("V", 1, {"Ordino": 100})
    r, _ = run(p, inscrits([("U", "V"), ("R", "V")]), [z])
    assert guanyadors(r, z) == ["R"]  # la preferència guanya als anys sense captura


def test_ordre_parroquial_desempata_entre_residents():
    p = caçadors([{"ID": f"C{i}", "Parroquia": "Canillo"} for i in range(5)]
                 + [{"ID": f"O{i}", "Parroquia": "Ordino"} for i in range(5)])
    z = vedat("V", 4, {"Canillo": 50, "Ordino": 50})
    for seed in range(6):
        r, _ = run(p, inscrits([(i, "V") for i in p.ID]), [z], seed=seed)
        primera = r.attrs["traces_vedats"]["V"]["ordre_parroquial"][0]
        assert guanyadors(r, z)[0][0] == primera[0]


def test_cupo_zero_no_dona_preferencia():
    p = caçadors([{"ID": "E", "Parroquia": "Escaldes-Engordany"},
                  {"ID": "N", "Parroquia": "Encamp", "anys_sense_captura": 3}])
    z = vedat("V", 2, {"Andorra la Vella": 99.7, "Escaldes-Engordany": 0.3})
    r, _ = run(p, inscrits([("E", "V"), ("N", "V")]), [z])
    assert r.attrs["traces_vedats"]["V"]["cupos"]["Escaldes-Engordany"] == 0
    assert guanyadors(r, z) == ["N", "E"]


def test_vedat_sense_repartiment_es_com_un_tcc():
    p = caçadors([{"ID": "R", "Parroquia": "Ordino"},
                  {"ID": "N", "Parroquia": "Encamp", "anys_sense_captura": 2}])
    z = vedat("V", 1, {})
    r, _ = run(p, inscrits([("R", "V"), ("N", "V")]), [z])
    assert guanyadors(r, z) == ["N"]
    assert "V" not in r.attrs["traces_vedats"]


# ── Sostre d'estrangers (art. 53.4) ─────────────────────────────────────────

def test_truncament_del_sostre():
    for caps, esperat in [(6, 0), (9, 0), (10, 1), (19, 1), (20, 2), (30, 3)]:
        p = caçadors([{"ID": f"E{i}", "Prioritat": 1, "Estranger": "Sí"} for i in range(caps)]
                     + [{"ID": f"L{i}", "Prioritat": 4} for i in range(caps)])
        z = tcc("Z", caps)
        r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
        g = guanyadors(r, z)
        assert sum(i.startswith("E") for i in g) == esperat, caps
        assert len(g) == caps
        fila = r.attrs["resum_zones"][0]
        assert fila["Max_estrangers"] == esperat and fila["Estrangers_adjudicats"] == esperat


def test_estrangers_competeixen_igual_mentre_hi_ha_lloc():
    p = caçadors([{"ID": "E", "Prioritat": 1, "Estranger": "Sí"}]
                 + [{"ID": f"L{i}", "Prioritat": 2} for i in range(9)])
    z = tcc("Z", 10)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    assert guanyadors(r, z)[0] == "E"


def test_substituts_estrangers_per_prioritat_i_marcats():
    p = caçadors([{"ID": f"E{i}", "Prioritat": 1, "Estranger": "Sí", "anys_sense_captura": 5 - i}
                  for i in range(4)]
                 + [{"ID": f"L{i}", "Prioritat": 3} for i in range(12)])
    z = tcc("Z", 10)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    assert guanyadors(r, z)[0] == "E0"           # sostre 1, ple de seguida
    s = substituts(r, z)
    assert s[:3] == ["E1", "E2", "E3"]           # encapçalen la llista per prioritat
    etiqueta = dict(zip(r.ID, r[col(z)]))
    assert [etiqueta[i] for i in s[:4]] == ["s1 (E)", "s2 (E)", "s3 (E)", "s4"]


def test_substitucio_no_consumeix_el_sostre():
    """Amb el sostre sense omplir per les captures, tots els estrangers que queden
    surten igualment com a substituts (cap no queda fora)."""
    p = caçadors([{"ID": f"E{i}", "Prioritat": 4, "Estranger": "Sí"} for i in range(6)]
                 + [{"ID": f"L{i}", "Prioritat": 1} for i in range(10)])
    z = tcc("Z", 10)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    assert not any(i.startswith("E") for i in guanyadors(r, z))
    assert sorted(substituts(r, z)) == sorted(f"E{i}" for i in range(6))


def test_sostre_zero_tots_els_estrangers_substituts():
    p = caçadors([{"ID": f"E{i}", "Prioritat": 1, "Estranger": "Sí"} for i in range(3)]
                 + [{"ID": f"L{i}", "Prioritat": 4} for i in range(3)])
    z = tcc("Z", 3, estranger_pct=0.0)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    assert sorted(guanyadors(r, z)) == ["L0", "L1", "L2"]
    assert sorted(substituts(r, z)) == ["E0", "E1", "E2"]


def test_valors_d_estranger_normalitzats():
    for v in ["Sí", "SI", "si", "sí", "S", "yes", "true", "1", " Si "]:
        assert d.normalitza_estranger(v) == "si", v
    for v in ["No", "NO", "no", "", None, float("nan"), "0", 0]:
        assert d.normalitza_estranger(v) == "no", v


def test_sostre_estrangers_en_vedat():
    p = caçadors([{"ID": f"E{i}", "Prioritat": 1, "Estranger": "Si", "Parroquia": "Ordino"}
                  for i in range(5)]
                 + [{"ID": f"L{i}", "Prioritat": 3, "Parroquia": "Encamp"} for i in range(15)])
    z = vedat("V", 10, {"Ordino": 100})
    r, _ = run(p, inscrits([(i, "V") for i in p.ID]), [z])
    assert sum(i.startswith("E") for i in guanyadors(r, z)) == 1
    s = substituts(r, z)
    # Els 4 estrangers restants (prioritat 1) encapçalen la substitució; entre ells
    # l'ordre el decideix l'atzar, perquè empaten en tot.
    assert set(s[:4]) == {"E1", "E2", "E3", "E4"}


# ── Art. 55.1.g ─────────────────────────────────────────────────────────────

def _isard_zones(caps_tcc=2, caps_v=2, modalitat=False):
    return [
        tcc("IS TCC", caps_tcc, modalitat=modalitat, min_colla=0),
        vedat("IS VA", caps_v, {"Encamp": 100}),
        vedat("IS VB", caps_v, {"Canillo": 100}),
    ]


def test_55g_valors_del_flag_normalitzats():
    for v in ["Sí", "SI", "sí", "1", "yes"]:
        p = caçadors([{"ID": "A", "Parroquia": "Encamp", d.COL_ISARD_VEDAT: v},
                      {"ID": "B", "Parroquia": "Encamp", d.COL_ISARD_VEDAT: "No"}])
        zs = _isard_zones()
        r, _ = run(p, inscrits([("A", "IS VA"), ("B", "IS VA")]), zs, "Isard")
        assert r.set_index("ID").at["A", "IS_VA"] == d.ETIQUETA_EXCLOS_55G, v


def test_55g_cicle_de_tres_anys():
    """Simula tres temporades encadenant la sortida d'un any com a entrada del
    següent: guanya un vedat l'any N, queda exclòs l'N+1 i torna a optar-hi l'N+2."""
    n = 10
    p = caçadors([{"ID": f"H{i}", "Prioritat": 2, "anys_sense_captura": n - i,
                   "Parroquia": "Encamp"} for i in range(n)])
    ins = inscrits([(i, z) for z in ["IS TCC", "IS VA"] for i in p.ID])
    zs = [tcc("IS TCC", 1), vedat("IS VA", 2, {"Encamp": 100})]
    historia = []
    for any_ in range(3):
        r, _ = run(p, ins, zs, "Isard", seed=any_)
        historia.append(r.set_index("ID"))
        p = r[["ID", "Nova_prioritat", "Nou_Anys_sense_captura", "Estranger",
               "Parroquia", d.COL_ISARD_VEDAT]].rename(
            columns={"Nova_prioritat": "Prioritat", "Nou_Anys_sense_captura": "anys_sense_captura"})
    any0, any1, any2 = historia
    guanyadors_v0 = [i for i in any0.index if num(any0.at[i, "IS_VA"]) is not None]
    assert guanyadors_v0 and all(any0.at[i, d.COL_ISARD_VEDAT] == "Sí" for i in guanyadors_v0)
    for i in guanyadors_v0:
        assert any1.at[i, "IS_VA"] == d.ETIQUETA_EXCLOS_55G      # N+1: exclòs
        assert any1.at[i, d.COL_ISARD_VEDAT] == "No"             # ...i surt lliure
        assert any2.at[i, "IS_VA"] != d.ETIQUETA_EXCLOS_55G      # N+2: hi torna a optar
    # Ningú guanya un vedat dos anys seguits.
    for a, b in [(any0, any1), (any1, any2)]:
        for i in a.index:
            if a.at[i, d.COL_ISARD_VEDAT] == "Sí":
                assert b.at[i, d.COL_ISARD_VEDAT] == "No"


def test_55g_no_afecta_el_tcc_amb_modalitat():
    rows = [{"ID": f"A{i}", "Modalitat": "A", "Colla_ID": "C1", "Parroquia": "Encamp",
             "Prioritat": 1, d.COL_ISARD_VEDAT: "Sí"} for i in range(6)]
    rows += [{"ID": f"B{i}", "Modalitat": "B", "Colla_ID": None, "Parroquia": "Encamp",
              "Prioritat": 4, d.COL_ISARD_VEDAT: "No"} for i in range(6)]
    p = caçadors(rows)
    ins = inscrits([(i, "IS TCC") for i in p.ID] + [(i, "IS VA") for i in p.ID])
    zs = _isard_zones(caps_tcc=4, caps_v=2, modalitat=True)
    r, _ = run(p, ins, zs, "Isard")
    ri = r.set_index("ID")
    assert any(num(ri.at[f"A{i}", "IS_TCC"]) for i in range(6))           # guanyen al TCC
    assert all(ri.at[f"A{i}", "IS_VA"] == d.ETIQUETA_EXCLOS_55G for i in range(6))


def test_55g_exclos_compta_com_a_presentat():
    p = caçadors([{"ID": "X", "Parroquia": "Encamp", "anys_sense_captura": 3,
                   d.COL_ISARD_VEDAT: "Sí"},
                  {"ID": "Y", "Parroquia": "Encamp", d.COL_ISARD_VEDAT: "No"}])
    r, _ = run(p, inscrits([("X", "IS VA"), ("Y", "IS VA")]), _isard_zones(), "Isard")
    x = r.set_index("ID").loc["X"]
    assert x["IS_VA"] == d.ETIQUETA_EXCLOS_55G
    assert x["Nova_prioritat"] == 2 and x["Nou_Anys_sense_captura"] == 4


def test_55g_substituts_de_vedat_surten_amb_no():
    p = caçadors([{"ID": f"H{i}", "Parroquia": "Encamp", "anys_sense_captura": i}
                  for i in range(5)])
    r, _ = run(p, inscrits([(i, "IS VA") for i in p.ID]), _isard_zones(caps_v=2), "Isard")
    ri = r.set_index("ID")
    for i in ri.index:
        esperat = "Sí" if num(ri.at[i, "IS_VA"]) is not None else "No"
        assert ri.at[i, d.COL_ISARD_VEDAT] == esperat


def test_55g_nomes_isard():
    p = caçadors([{"ID": "A", "Parroquia": "Encamp", d.COL_ISARD_VEDAT: "Sí"}])
    zs = [vedat("MUF V", 1, {"Encamp": 100})]
    for esp in ["Mufló", "Cabirol", "Altres"]:
        r, _ = run(p, inscrits([("A", "MUF V")]), zs, esp)
        assert num(r.set_index("ID").at["A", "MUF_V"]) == 1
        assert d.COL_ISARD_VEDAT not in r.columns
    assert d.es_isard("isard") and d.es_isard(" ISARD ") and not d.es_isard("Isards")


# ── Modalitat de colla i individual (arts. 56 i 57) ─────────────────────────

def _poblacio_colles(mides, n_b, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for c, m in enumerate(mides):
        rows += [{"ID": f"C{c}_{k}", "Modalitat": "A", "Colla_ID": f"C{c}",
                  "Prioritat": int(rng.randint(1, 5)), "anys_sense_captura": int(rng.randint(0, 5))}
                 for k in range(m)]
    rows += [{"ID": f"B{k}", "Modalitat": "B", "Colla_ID": None,
              "Prioritat": int(rng.randint(1, 5)), "anys_sense_captura": int(rng.randint(0, 5))}
             for k in range(n_b)]
    return caçadors(rows)


def test_modalitat_repartiment_exacte():
    mides = [6, 7, 9, 12, 6, 8]
    for caps in [5, 9, 13, 20]:
        p = _poblacio_colles(mides, n_b=30, seed=caps)
        z = tcc("IS TCC", caps, modalitat=True, min_colla=6)
        r, _ = run(p, inscrits([(i, "IS TCC") for i in p.ID]), [z], "Isard")
        g = set(guanyadors(r, z))
        assert len(g) == caps                                    # cap romanent
        nA, nB = sum(mides), 30
        ratio = (nA + nB) / caps
        wA = p[p.ID.isin(g) & (p.Modalitat == "A")]
        assert abs(len(wA) - nA / ratio) <= 0.5 + 1e-9           # arrodonit al més proper
        for c, m in enumerate(mides):
            base = math.floor(m / ratio)
            got = (wA.Colla_ID == f"C{c}").sum()
            assert base <= got <= base + 1, (caps, c, base, got)  # màxim una addicional


def test_modalitat_dins_colla_per_prioritat():
    p = caçadors([{"ID": f"K{k}", "Modalitat": "A", "Colla_ID": "C", "Prioritat": 1 + (k % 4)}
                  for k in range(12)]
                 + [{"ID": f"B{k}", "Modalitat": "B", "Prioritat": 2} for k in range(12)])
    z = tcc("IS TCC", 4, modalitat=True, min_colla=6)
    r, _ = run(p, inscrits([(i, "IS TCC") for i in p.ID]), [z], "Isard")
    prio = dict(zip(p.ID, p.Prioritat))
    colla = [i for i in guanyadors(r, z) if i.startswith("K")]
    assert colla and all(prio[i] == 1 for i in colla)


def test_modalitat_sostre_estrangers_unic():
    p = _poblacio_colles([10, 10], n_b=20, seed=3)
    p["Estranger"] = "Sí"
    z = tcc("IS TCC", 20, modalitat=True, min_colla=6)
    r, _ = run(p, inscrits([(i, "IS TCC") for i in p.ID]), [z], "Isard")
    assert len(guanyadors(r, z)) == 2                          # 10% de 20, en total
    assert all(str(v).endswith("(E)") for v in r[col(z)] if es_subst(v))


def test_modalitat_substituts_per_prioritat():
    p = _poblacio_colles([8, 8, 8], n_b=30, seed=7)
    z = tcc("IS TCC", 10, modalitat=True, min_colla=6)
    r, _ = run(p, inscrits([(i, "IS TCC") for i in p.ID]), [z], "Isard")
    prio = dict(zip(p.ID, p.Prioritat))
    s = substituts(r, z)
    assert [prio[i] for i in s] == sorted(prio[i] for i in s)
    assert len(s) + len(guanyadors(r, z)) == len(p)


def test_colla_minima():
    p = _poblacio_colles([6, 5], n_b=4)
    ins = inscrits([(i, "IS TCC") for i in p.ID])
    z = tcc("IS TCC", 3, modalitat=True, min_colla=6)
    assert d.colles_per_sota_minim(p, ins, [z]) == [("IS TCC", "C1", 5, 6)]
    assert d.colles_per_sota_minim(p, ins, [{**z, "min_colla": 0}]) == []
    assert d.colles_per_sota_minim(p, ins, [{**z, "min_colla": 5}]) == []


# ── Tipus de captura ────────────────────────────────────────────────────────

def test_tipus_en_ordre_configurat():
    p = caçadors([{"ID": f"A{a}", "anys_sense_captura": a} for a in range(6)])
    z = {"nom": "Z", "tipus": "TCC", "aleatori": False, "estranger_pct": 10.0,
         "captures": [{"tipus": ["Mascle"], "quantitat": 2},
                      {"tipus": ["Femella", "Trofeu"], "quantitat": 1}]}
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    t = dict(zip(r.ID, r["Tipus_Z"]))
    assert [t[i] for i in guanyadors(r, z)] == ["Mascle", "Mascle", "Femella+Trofeu"]
    assert all(pd.isna(t[i]) for i in substituts(r, z))


def test_ordre_aleatori_no_canvia_qui_guanya():
    p = caçadors([{"ID": f"X{i}", "Prioritat": 1 + i % 3} for i in range(15)])
    ins = inscrits([(i, "Z") for i in p.ID])
    caps = [{"tipus": ["Mascle"], "quantitat": 3}, {"tipus": ["Femella"], "quantitat": 3}]
    base = {"nom": "Z", "tipus": "TCC", "estranger_pct": 10.0, "captures": caps}
    r1, _ = run(p, ins, [{**base, "aleatori": False}], seed=4)
    r2, _ = run(p, ins, [{**base, "aleatori": True}], seed=4)
    assert guanyadors(r1, base) == guanyadors(r2, base)
    tipus2 = r2.loc[r2["Z"].map(num).notna(), "Tipus_Z"]
    assert sorted(tipus2) == ["Femella"] * 3 + ["Mascle"] * 3


def test_indeterminat_absorbeix_altres_tipus():
    p = caçadors([{"ID": "A"}])
    z = {"nom": "Z", "tipus": "TCC", "captures": [{"tipus": ["Mascle", "Indeterminat"], "quantitat": 1}]}
    r, _ = run(p, inscrits([("A", "Z")]), [z])
    assert r["Tipus_Z"].iloc[0] == "Indeterminat"


# ── Resultats, prioritat de sortida i casos límit ───────────────────────────

def test_prioritat_de_sortida():
    p = caçadors([{"ID": "G", "Prioritat": 1, "anys_sense_captura": 4},
                  {"ID": "S", "Prioritat": 3, "anys_sense_captura": 2},
                  {"ID": "Q", "Prioritat": 1, "anys_sense_captura": 7}])
    zs = [tcc("Z1", 1), tcc("Z2", 1)]
    r, _ = run(p, inscrits([("G", "Z1"), ("S", "Z1"), ("Q", "Z2")]), zs)
    ri = r.set_index("ID")
    assert ri.at["G", "Nova_prioritat"] == 4 and ri.at["G", "Nou_Anys_sense_captura"] == 0
    assert ri.at["S", "Nova_prioritat"] == 2 and ri.at["S", "Nou_Anys_sense_captura"] == 3
    assert ri.at["Q", "Nova_prioritat"] == 4  # una captura a qualsevol zona compta


def test_cel_la_buida_vol_dir_no_inscrit():
    p = caçadors([{"ID": "A"}, {"ID": "B"}])
    zs = [tcc("Z1", 1), tcc("Z2", 1)]
    r, _ = run(p, inscrits([("A", "Z1"), ("B", "Z1"), ("B", "Z2")]), zs)
    assert pd.isna(r.set_index("ID").at["A", "Z2"])


def test_id_duplicat_atura():
    p = caçadors([{"ID": "A"}])
    try:
        run(p, inscrits([("A", "Z"), ("A", "Z")]), [tcc("Z", 1)])
    except ValueError as e:
        assert "duplicats" in str(e)
    else:
        raise AssertionError("un ID duplicat a la mateixa zona ha d'aturar el sorteig")


def test_zona_sense_captures_o_sense_inscrits():
    p = caçadors([{"ID": "A"}])
    r, _ = run(p, inscrits([("A", "Z0")]), [tcc("Z0", 0), tcc("Zbuida", 3)])
    assert r["Z0"].isna().all() and r["Zbuida"].isna().all()


def test_menys_sollicitants_que_captures():
    p = caçadors([{"ID": f"A{i}"} for i in range(3)])
    z = tcc("Z", 10)
    r, _ = run(p, inscrits([(i, "Z") for i in p.ID]), [z])
    assert len(guanyadors(r, z)) == 3 and substituts(r, z) == []


def test_dashboard_reconeix_les_etiquetes():
    src = open(os.path.join(ROOT, "app", "pages", "dashboard.py"), encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "is_inscrit")
    ns = {"pd": pd}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "dashboard", "exec"), ns)
    is_inscrit = ns["is_inscrit"]
    for v in ["1", "12", "s1", "s23", "s3 (E)", d.ETIQUETA_EXCLOS_55G]:
        assert is_inscrit(v), v
    for v in [None, float("nan"), "", "x", "s"]:
        assert not is_inscrit(v), v


# ══════════════════════════════════════════════════════════════════════════════
# 2. INVARIANTS SOBRE SORTEJOS ALEATORIS
# ══════════════════════════════════════════════════════════════════════════════

def _escenari(seed, modalitat=False):
    rng = np.random.RandomState(seed)
    n = int(rng.randint(15, 60))
    rows = []
    for k in range(n):
        estr = rng.rand() < 0.2
        rows.append({
            "ID": f"H{k:03d}",
            "Prioritat": int(rng.randint(1, 5)),
            "anys_sense_captura": int(rng.randint(0, 6)),
            "Estranger": rng.choice(["Sí", "SI", "si"]) if estr else rng.choice(["No", "NO", ""]),
            "Parroquia": rng.choice(PARR + [None, "3"]),
            d.COL_ISARD_VEDAT: "Sí" if rng.rand() < 0.25 else "No",
        })
    p = pd.DataFrame(rows)
    zones = []
    if modalitat:
        mides = []
        restants = n
        colla = 0
        while restants > 12 and colla < 4:
            m = int(rng.randint(6, 10)); mides.append(m); restants -= m; colla += 1
        mod, cid = [], []
        idx = 0
        for c, m in enumerate(mides):
            mod += ["A"] * m; cid += [f"C{c}"] * m; idx += m
        mod += ["B"] * (n - idx); cid += [None] * (n - idx)
        p["Modalitat"], p["Colla_ID"] = mod, cid
        zones.append(tcc("IS TCC", int(rng.randint(2, 12)), modalitat=True, min_colla=6,
                         estranger_pct=float(rng.choice([0, 10, 20]))))
    else:
        zones.append(tcc("IS TCC", int(rng.randint(1, 15)), estranger_pct=float(rng.choice([0, 10, 25]))))
    zones.append(vedat("IS VA", int(rng.randint(1, 12)), {"Canillo": 50, "Ordino": 50},
                       estranger_pct=float(rng.choice([0, 10, 30]))))
    zones.append(vedat("IS VB", int(rng.randint(1, 10)), {"La Massana": 100}))
    zones.append(vedat("IS VC", int(rng.randint(1, 20)),
                       {"La Massana": 23.4, "Sant Julià de Lòria": 24.1,
                        "Andorra la Vella": 52.2, "Escaldes-Engordany": 0.3},
                       aleatori=bool(rng.rand() < 0.5)))
    pairs = []
    for z in zones:
        if z.get("modalitat"):
            pairs += [(i, z["nom"]) for i in p.ID]
        else:
            pairs += [(i, z["nom"]) for i in p.ID if rng.rand() < 0.6]
    return p, inscrits(pairs), zones


def _comprova_invariants(p, ins, zones, seed):
    r, _ = run(p, ins, zones, "Isard", seed=seed)
    ri = r.set_index("ID")
    pi = p.set_index("ID")
    # Al resultat hi surt exactament qui s'ha inscrit a alguna zona (més, a les
    # zones amb modalitat, qui té Modalitat). Els no presentats no hi surten: la
    # seva prioritat 3 la fixen els tècnics.
    presentats = set(ins.ID)
    if any(z.get("modalitat") for z in zones):
        presentats |= set(p.loc[p.Modalitat.isin(["A", "B"]), "ID"])
    assert set(r.ID) == presentats, f"seed={seed}"
    wins = {i: 0 for i in p.ID}
    for z in zones:
        c = col(z)
        caps = sum(x["quantitat"] for x in z["captures"])
        limit = math.floor(z["estranger_pct"] / 100.0 * caps)
        es_v = z["tipus"] == "Vedat"
        inscr = set(ins.loc[ins.Codi_Sorteig == z["nom"], "ID"])
        exclosos = {i for i in inscr if es_v and es_estr(pi.at[i, d.COL_ISARD_VEDAT])}
        elegibles = inscr - exclosos
        if z.get("modalitat"):
            elegibles = {i for i in elegibles if pi.at[i, "Modalitat"] in ("A", "B")}
        g = guanyadors(r, z)
        s = substituts(r, z)
        ctx = f"seed={seed} zona={z['nom']}"
        # Captures: mai més de les ofertes; sostre d'estrangers respectat.
        assert len(g) <= caps, ctx
        n_estr = sum(es_estr(pi.at[i, "Estranger"]) for i in g)
        assert n_estr <= limit, ctx
        if not z.get("modalitat"):
            nf = sum(not es_estr(pi.at[i, "Estranger"]) for i in elegibles)
            ef = len(elegibles) - nf
            assert len(g) == min(caps, nf + min(limit, ef)), ctx
        # Etiquetes: cada elegible té exactament una posició; numeració contigua.
        assert set(g) | set(s) == elegibles and not set(g) & set(s), ctx
        assert sorted(ri.loc[g, c].map(num)) == list(range(1, len(g) + 1)), ctx
        assert sorted(ri.loc[s, c].map(n_subst)) == list(range(1, len(s) + 1)), ctx
        # (E) si i només si és estranger.
        for i in s:
            assert str(ri.at[i, c]).endswith(d.SUFIX_SUBST_ESTRANGER) == es_estr(pi.at[i, "Estranger"]), ctx
        # Art. 55.1.g: exclosos marcats i fora de captures i substitució.
        for i in exclosos:
            assert ri.at[i, c] == d.ETIQUETA_EXCLOS_55G, ctx
        # No inscrits: cel·la buida.
        for i in presentats - inscr:
            if not z.get("modalitat"):
                assert pd.isna(ri.at[i, c]) or ri.at[i, c] in ("", "nan"), ctx
        # Prioritat efectiva (art. 54 + 4+n) mai no millora al llarg de la llista.
        def prio(i):
            return 5 + wins[i] - 1 if wins[i] else pi.at[i, "Prioritat"]
        if not z.get("modalitat"):
            pg = [prio(i) for i in g]
            assert pg == sorted(pg), ctx
        ps = [prio(i) for i in s]
        assert ps == sorted(ps), ctx
        # Resum per zona coherent.
        fila = next(x for x in r.attrs["resum_zones"] if x["Zona"] == z["nom"])
        assert fila["Max_estrangers"] == limit and fila["Estrangers_adjudicats"] == n_estr, ctx
        assert fila["Exclosos_55_1_g"] == len(exclosos), ctx
        # Tipus: només als adjudicataris.
        tip = ri[f"Tipus_{c}"]
        assert tip.loc[g].notna().all() and tip.loc[s].isna().all(), ctx
        for i in g:
            wins[i] += 1
    # Prioritat de sortida i columna de l'art. 55.1.g.
    vcols = [col(z) for z in zones if z["tipus"] == "Vedat"]
    for i in presentats:
        te = wins[i] > 0
        assert ri.at[i, "Nova_prioritat"] == (4 if te else 2)
        assert ri.at[i, "Nou_Anys_sense_captura"] == (0 if te else pi.at[i, "anys_sense_captura"] + 1)
        guanya_v = any(num(ri.at[i, vc]) is not None for vc in vcols)
        assert ri.at[i, d.COL_ISARD_VEDAT] == ("Sí" if guanya_v else "No")
        if es_estr(pi.at[i, d.COL_ISARD_VEDAT]):
            assert not guanya_v, f"seed={seed}: {i} guanya un vedat dos anys seguits"
    # Reproductibilitat.
    r2, _ = run(p, ins, zones, "Isard", seed=seed)
    pd.testing.assert_frame_equal(r.reset_index(drop=True), r2.reset_index(drop=True))


def test_invariants_sortejos_aleatoris():
    for seed in range(40):
        p, ins, zones = _escenari(seed)
        _comprova_invariants(p, ins, zones, seed)


def test_invariants_sortejos_aleatoris_amb_modalitat():
    for seed in range(100, 120):
        p, ins, zones = _escenari(seed, modalitat=True)
        assert d.colles_per_sota_minim(p, ins, zones) == []
        _comprova_invariants(p, ins, zones, seed)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FITXERS D'EXEMPLE DE L'APP
# ══════════════════════════════════════════════════════════════════════════════

EXEMPLES = {"Isard": "isard.csv", "Cabirol": "cabirol.csv", "Mufló": "muflo.csv"}


def _exemple(especie):
    p = pd.read_csv(os.path.join(ROOT, EXEMPLES[especie]), sep=";", encoding="utf-8-sig")
    ins = pd.read_csv(os.path.join(ROOT, "sorteig.csv"), sep=";", encoding="utf-8-sig")
    zones = config.default_zones(especie)
    for k, z in enumerate(zones):
        z["captures"] = _caps(3 + 2 * k)
    return p, ins, zones


def test_exemples_passen_les_validacions_de_l_app():
    """Replica les comprovacions de `pages/sorteig.py`: columnes obligatòries,
    avís d'IDs presents en un sol CSV, inscrits sense modalitat i colles curtes."""
    for esp in EXEMPLES:
        p, ins, zones = _exemple(esp)
        d.validar_csv2(ins)
        req = {"ID", "Prioritat", "anys_sense_captura", "Estranger"}
        if any(z.get("modalitat") for z in zones):
            req |= {"Modalitat", "Colla_ID"}
        if any(z.get("tipus") == "Vedat" for z in zones):
            req |= {"Parroquia"}
        assert req <= set(p.columns), (esp, req - set(p.columns))
        if esp == "Isard":
            assert d.COL_ISARD_VEDAT in p.columns
        noms = {z["nom"] for z in zones}
        ids_ins = set(ins.loc[ins.Codi_Sorteig.isin(noms), "ID"].astype(str))
        ids_prio = set(p.ID.astype(str))
        sobren = ids_prio - ids_ins
        if "Modalitat" in p.columns:
            sobren -= set(p.loc[p.Modalitat.isin(["A", "B"]), "ID"].astype(str))
        assert not sobren and not (ids_ins - ids_prio), esp
        mod = [z["nom"] for z in zones if z.get("modalitat")]
        if mod:
            m = ins[ins.Codi_Sorteig.isin(mod)].merge(p[["ID", "Modalitat"]], on="ID", how="left")
            assert m.Modalitat.notna().all(), esp
        assert d.colles_per_sota_minim(p, ins, zones) == [], esp
        assert set(ins.Codi_Sorteig) <= {z["nom"] for e in EXEMPLES for z in config.default_zones(e)}


def test_exemples_es_sortegen_i_exerciten_les_regles():
    vist_e = vist_g = 0
    for esp in EXEMPLES:
        p, ins, zones = _exemple(esp)
        r, _ = run(p, ins, zones, esp, seed=1)
        for z in zones:
            assert guanyadors(r, z), (esp, z["nom"])
        cols = [col(z) for z in zones]
        vist_e += int(r[cols].astype(str).apply(lambda s: s.str.endswith("(E)")).sum().sum())
        vist_g += int((r[cols] == d.ETIQUETA_EXCLOS_55G).sum().sum())
    assert vist_e > 0, "els exemples haurien de tenir substituts estrangers"
    assert vist_g > 0, "l'exemple d'isard hauria d'exercitar l'art. 55.1.g"


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in funcs:
        try:
            fn()
            print(f"  OK   {fn.__name__}")
        except Exception as exc:
            fails += 1
            import traceback
            print(f" FAIL  {fn.__name__}: {exc!r}")
            traceback.print_exc(limit=2)
    print("\n" + ("TOTES OK" if not fails else f"{fails} FALLEN") + f" ({len(funcs)} proves)")
    sys.exit(1 if fails else 0)
