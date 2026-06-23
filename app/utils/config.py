"""
Configuració parametritzable d'espècies i zones per al sorteig (2026+).

Substitueix les antigues constants hardcoded ``ESPECIE_SORTEIGS`` i
``VEDAT_PARRÒQUIES`` per una estructura editable que es pot:

* **precarregar** amb les zones vigents i els seus valors per defecte,
* **modificar** des de la interfície (afegir/treure zones, canviar percentatges,
  tipus de captura, ordre del sorteig, etc.),
* **exportar/importar** com a JSON per reutilitzar-la entre anys.

Model de dades
--------------
Una *espècie* (Isard, Cabirol, Mufló o una de nova amb nom lliure) conté una
llista ordenada de *zones*. Cada zona és un diccionari::

    {
        "nom": "IS VCE",                # nom/codi de la zona
        "tipus": "Vedat",               # "Vedat" | "TCC"
        "modalitat": False,             # cert -> sorteig amb modalitat A/B (colles + individual)
        "aleatori": True,               # ordre aleatori dins la zona
        "estranger_pct": 10,            # sostre d'estrangers (% de captures)
        "reserva_pct": 50,              # % reservat als locals (només Vedat)
        "parroquies": {                 # repartiment del % reservat (només Vedat)
            "La Massana": 23.4,         #   en % que sumen ~100
            "Sant Julià de Lòria": 24.1,
            "Andorra la Vella": 52.2,
            "Escaldes-Engordany": 0.3,
        },
        "captures": [                   # tipus de captura dins la zona
            {"tipus": ["Mascle"], "quantitat": 5},
            {"tipus": ["Femella", "Trofeu"], "quantitat": 3},
        ],
    }

L'ordre del sorteig és l'ordre de la llista (es poden pujar/baixar zones).
Dins de cada zona, els tipus es processen en l'ordre definit (o aleatòriament
si ``aleatori`` és cert).

Hi ha dos tipus de zona: ``"Vedat"`` (amb prioritat parroquial) i ``"TCC"``
(prioritat ordinària). A més, una zona pot tenir ``"modalitat": True``, que
activa la mecànica de modalitat A/B (colles + individual amb ``Colla_ID``).
Avui només l'IS TCC de l'isard la fa servir.
"""

import copy
import json

# ── PARRÒQUIES (ordre protocol·lari) ─────────────────────────────────────────

PARROQUIES = [
    "Canillo",
    "Encamp",
    "Ordino",
    "La Massana",
    "Andorra la Vella",
    "Sant Julià de Lòria",
    "Escaldes-Engordany",
]

CODI_PARROQUIES = {i + 1: name for i, name in enumerate(PARROQUIES)}

# ── TIPUS ────────────────────────────────────────────────────────────────────

# Tipus de captura disponibles per als tipus dins d'una zona.
TIPUS_OPTIONS = [
    "Femella",
    "Mascle",
    "Adult",
    "Juvenil",
    "Trofeu",
    "Selectiu",
    "Indeterminat",
]

# Tipus de zona (mecànica del sorteig).
TIPUS_ZONA = ["Vedat", "TCC"]

# ── VALORS PER DEFECTE ───────────────────────────────────────────────────────

ESTRANGER_PCT_DEFAULT = 10.0  # sostre d'estrangers per zona, en %
RESERVA_PCT_DEFAULT = 50.0    # % de captures reservat als locals en un vedat

# Espècies "de fàbrica". Se'n poden afegir de noves (categoria "Altres").
ESPECIES_BASE = ["Isard", "Cabirol", "Mufló"]


def _captures(*pairs):
    """Constructor breu per a llistes de tipus de captura."""
    return [{"tipus": list(t), "quantitat": q} for t, q in pairs]


# ── CATÀLEG PRECARREGAT ──────────────────────────────────────────────────────
#
# Cada zona ja porta el seu tipus i, per als vedats, el repartiment parroquial
# vigent. Les quantitats de captura es deixen a 0 perquè es revisen cada any.
#
# ⚠️  Els vedats de MUFLÓ (temporal d'Escaldes, Enclar i Ransol) no tenien
#     percentatges parroquials definits al sistema anterior. Es deixen com a
#     Vedat amb el repartiment BUIT, pendent de confirmació de la federació.

DEFAULT_CONFIG = {
    "Isard": [
        {
            "nom": "IS TCC",
            "tipus": "TCC",
            "modalitat": True,  # sorteig amb modalitat A/B (colles + individual)
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": None,
            "parroquies": {},
            "captures": _captures(([], 0)),  # quantitat total de captures
        },
        {
            "nom": "IS VCRS",
            "tipus": "Vedat",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": RESERVA_PCT_DEFAULT,
            "parroquies": {"Canillo": 50.0, "Ordino": 50.0},
            "captures": _captures(([], 0)),
        },
        {
            "nom": "IS VCX",
            "tipus": "Vedat",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": RESERVA_PCT_DEFAULT,
            "parroquies": {"La Massana": 100.0},
            "captures": _captures(([], 0)),
        },
        {
            "nom": "IS VCE",
            "tipus": "Vedat",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": RESERVA_PCT_DEFAULT,
            "parroquies": {
                "La Massana": 23.4,
                "Sant Julià de Lòria": 24.1,
                "Andorra la Vella": 52.2,
                "Escaldes-Engordany": 0.3,
            },
            "captures": _captures(([], 0)),
        },
    ],
    "Cabirol": [
        {
            "nom": "CAB",
            "tipus": "TCC",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": None,
            "parroquies": {},
            "captures": _captures(([], 0)),
        },
    ],
    "Mufló": [
        {
            "nom": "MUF UGEO",  # unitat de gestió Est i Oest
            "tipus": "TCC",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": None,
            "parroquies": {},
            "captures": _captures(([], 0)),
        },
        {
            "nom": "MUF UGC",  # unitat de gestió Centre
            "tipus": "TCC",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": None,
            "parroquies": {},
            "captures": _captures(([], 0)),
        },
        {
            "nom": "MUF VTE-E",  # vedat temporal d'Escaldes-Engordany
            "tipus": "Vedat",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": RESERVA_PCT_DEFAULT,
            "parroquies": {},  # ⚠️ pendent de la federació
            "captures": _captures(([], 0)),
        },
        {
            "nom": "MUF VCE",  # vedat de caça d'Enclar
            "tipus": "Vedat",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": RESERVA_PCT_DEFAULT,
            "parroquies": {},  # ⚠️ pendent de la federació
            "captures": _captures(([], 0)),
        },
        {
            "nom": "MUF R",  # vedat de caça de la vall de Ransol
            "tipus": "Vedat",
            "aleatori": True,
            "estranger_pct": ESTRANGER_PCT_DEFAULT,
            "reserva_pct": RESERVA_PCT_DEFAULT,
            "parroquies": {},  # ⚠️ pendent de la federació
            "captures": _captures(([], 0)),
        },
    ],
}


# ── HELPERS ──────────────────────────────────────────────────────────────────

def default_zones(especie):
    """Còpia profunda de les zones precarregades d'una espècie (o llista buida)."""
    return copy.deepcopy(DEFAULT_CONFIG.get(especie, []))


def new_zone(nom="Nova zona", tipus="TCC"):
    """Crea una zona buida amb els valors per defecte."""
    return {
        "nom": nom,
        "tipus": tipus,
        "modalitat": False,
        "aleatori": True,
        "estranger_pct": ESTRANGER_PCT_DEFAULT,
        "reserva_pct": RESERVA_PCT_DEFAULT if tipus == "Vedat" else None,
        "parroquies": {},
        "captures": _captures(([], 0)),
    }


def zones_to_json(zones):
    """Serialitza una llista de zones a JSON (per desar/exportar)."""
    return json.dumps(zones, ensure_ascii=False)


def zones_from_json(text):
    """Reconstrueix una llista de zones des de JSON."""
    return json.loads(text)
