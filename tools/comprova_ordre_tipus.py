#!/usr/bin/env python3
"""Dedueix, a partir d'un full de resultats, si l'«Ordre aleatori» estava actiu.

No cal el fitxer de configuració: n'hi ha prou amb l'ordre en què apareixen els
tipus de captura entre els adjudicataris.

  · Desactivat → els tipus surten en BLOCS seguits, en l'ordre configurat, de
    manera que els millors de prioritat s'enduen el primer tipus de la llista.
  · Activat    → els tipus surten BARREJATS entre les posicions.

Amb k tipus diferents, «desactivat» dona exactament k trams; barrejat en dona
molts més. La probabilitat que un repartiment aleatori caigui per casualitat en
blocs perfectes es calcula i es mostra, per no confondre un cas petit.

Ús:  python tools/comprova_ordre_tipus.py resultats.csv [--sep ;]
"""
import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd


def trams(seq):
    """Nombre de trams consecutius de valors iguals."""
    return 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def prob_blocs(recompte):
    """Probabilitat que una barreja a l'atzar surti en blocs perfectes."""
    total = sum(recompte)
    if total == 0:
        return 1.0
    # ordenacions en blocs / ordenacions totals (multinomial)
    log_favorables = math.lgamma(len(recompte) + 1)
    log_totals = math.lgamma(total + 1) - sum(math.lgamma(n + 1) for n in recompte)
    return math.exp(log_favorables - log_totals)


def analitza(df, zona, col_tipus):
    pos = pd.to_numeric(df[zona], errors="coerce")
    adj = df.loc[pos.notna()].assign(_pos=pos[pos.notna()]).sort_values("_pos")
    tipus = [t for t in adj[col_tipus].astype(str) if t and t.lower() not in ("nan", "<na>")]
    if not tipus:
        return None
    distints = sorted(set(tipus))
    if len(distints) < 2:
        return (zona, len(tipus), distints, None, "un sol tipus: la casella no hi fa res")

    n_trams = trams(tipus)
    recompte = [tipus.count(t) for t in distints]
    p = prob_blocs(recompte)
    if n_trams == len(distints):
        estat = "DESACTIVADA (tipus en blocs, seguint la prioritat)"
        if p > 0.01:
            estat += f" — atenció: amb tan poques captures, l'atzar hi arriba amb probabilitat {p:.1%}"
    else:
        estat = f"ACTIVADA (tipus barrejats: {n_trams} trams en lloc de {len(distints)})"
    return (zona, len(tipus), distints, n_trams, estat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="full de resultats del sorteig")
    ap.add_argument("--sep", default=";")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=args.sep, dtype=str, encoding="utf-8-sig")
    zones = [c for c in df.columns if f"Tipus_{c}" in df.columns]
    if not zones:
        print("No s'ha trobat cap parella de columnes «zona» i «Tipus_zona».")
        print("Columnes llegides:", ", ".join(map(str, df.columns))[:300])
        return 1

    print(f"{Path(args.csv).name} · {len(df)} files\n")
    for zona in zones:
        r = analitza(df, zona, f"Tipus_{zona}")
        if r is None:
            print(f"  {zona}: cap adjudicatari")
            continue
        _, n, distints, _, estat = r
        print(f"  {zona}: {n} captures · tipus {distints}")
        print(f"    → Ordre aleatori: {estat}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
