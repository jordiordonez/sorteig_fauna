#!/usr/bin/env python3
"""Genera `manual.html` a partir de `manual.md`.

El resultat és un fitxer **autocontingut**: els diagrames van incrustats com a SVG
i no hi ha cap CSS, tipografia ni script extern. Així s'obre sense connexió des de
l'aplicació empaquetada, que és com el fa servir el Govern.

Ús:  python tools/build_manual.py

Requisits: mistune, beautifulsoup4 i `npx @mermaid-js/mermaid-cli` per als
diagrames. Si mermaid-cli no hi és, els diagrames es deixen com a bloc de codi i
la resta del manual es genera igualment.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mistune
from bs4 import BeautifulSoup

ARREL = Path(__file__).resolve().parent.parent
ORIGEN = ARREL / "manual.md"
DESTI = ARREL / "manual.html"

# ── Paleta «pissarra i avet» ─────────────────────────────────────────────────
CLAR = {
    "fons": "#F2F3F0", "superficie": "#FFFFFF", "tinta": "#232A2E",
    "apagat": "#5C6A70", "regle": "#D8DCD7", "accent": "#1D5C4F",
    "segell": "#8E2434", "cita": "#43607A", "tint": "#E8EDE9",
}
FOSC = {
    "fons": "#1A1F22", "superficie": "#222A2E", "tinta": "#E4E7E3",
    "apagat": "#9AA7AC", "regle": "#333D42", "accent": "#5FBFA3",
    "segell": "#E08494", "cita": "#8FB4D4", "tint": "#26312F",
}

SERIF = ('"Iowan Old Style","Palatino Linotype",Palatino,Constantia,'
         '"Source Serif 4",Georgia,serif')
MONO = ('ui-monospace,"Cascadia Mono","SF Mono",Menlo,Consolas,'
        '"Roboto Mono",monospace')
SANS = ('system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif')


def config_mermaid(p):
    """Configuració de mermaid amb la paleta del manual."""
    return {
        "theme": "base",
        "themeVariables": {
            "background": "transparent",
            "primaryColor": p["superficie"],
            "primaryTextColor": p["tinta"],
            "primaryBorderColor": p["accent"],
            "secondaryColor": p["tint"],
            "tertiaryColor": p["fons"],
            "lineColor": p["apagat"],
            "textColor": p["tinta"],
            "fontFamily": SERIF,
            "fontSize": "15px",
            "nodeBorder": p["accent"],
            "clusterBkg": p["tint"],
            "edgeLabelBackground": p["fons"],
        },
        "flowchart": {"curve": "basis", "padding": 14,
                      "nodeSpacing": 42, "rankSpacing": 48, "useMaxWidth": True},
    }


def renderitza_diagrames(blocs):
    """Renderitza cada bloc mermaid a SVG, en versió clara i fosca.

    Retorna {index: {"clar": svg, "fosc": svg}}. Si mermaid-cli no està
    disponible retorna un diccionari buit i el manual es genera sense diagrames.
    """
    if not shutil.which("npx"):
        print("⚠️  npx no disponible: es generen els diagrames com a codi.")
        return {}

    sortida = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for mode, paleta in (("clar", CLAR), ("fosc", FOSC)):
            cfg = tmp / f"cfg-{mode}.json"
            cfg.write_text(json.dumps(config_mermaid(paleta)), encoding="utf-8")
            for i, codi in enumerate(blocs):
                entrada = tmp / f"d{i}-{mode}.mmd"
                sortida_svg = tmp / f"d{i}-{mode}.svg"
                entrada.write_text(codi, encoding="utf-8")
                r = subprocess.run(
                    ["npx", "--yes", "@mermaid-js/mermaid-cli",
                     "-i", str(entrada), "-o", str(sortida_svg),
                     "-c", str(cfg), "-b", "transparent"],
                    capture_output=True, text=True,
                )
                if r.returncode != 0 or not sortida_svg.exists():
                    print(f"⚠️  diagrama {i} ({mode}) ha fallat:\n{r.stderr[:400]}")
                    continue
                sortida.setdefault(i, {})[mode] = neteja_svg(
                    sortida_svg.read_text(encoding="utf-8"), f"mmd{i}{mode}"
                )
                print(f"   diagrama {i} · {mode} ✓")
    return sortida


def neteja_svg(svg, ident):
    """Fa l'SVG apte per conviure amb d'altres a la mateixa pàgina.

    mermaid-cli fa servir sempre el mateix identificador («my-svg») per als
    selectors CSS interns; amb diversos diagrames a la pàgina es trepitgen. Aquí
    se substitueix per un d'únic i es treuen les mides fixes.
    """
    svg = svg.replace("my-svg", ident)
    svg = re.sub(r'<\?xml[^>]*\?>', '', svg)
    svg = re.sub(r'\s(width|height)="[^"]*"', '', svg, count=2)
    svg = svg.replace("<svg ", '<svg role="img" ', 1)
    return svg.strip()


def amplada_natural(svg):
    """Amplada en píxels que demana l'SVG, llegida del seu viewBox."""
    vb = (svg.get("viewbox") or svg.get("viewBox") or "").split()
    try:
        return float(vb[2])
    except (IndexError, ValueError):
        return 0.0


def classifica_citacio(text):
    """Extreu la citació d'article d'un títol: «… (art. 55.1)» -> «art. 55.1»."""
    m = re.search(r"\(((?:arts?\.)[^)]*)\)\s*$", text.strip())
    if not m:
        return text, None
    return text[: m.start()].strip(), m.group(1)


def construeix(html_cos, diagrames):
    sopa = BeautifulSoup(html_cos, "lxml")

    # ── Diagrames: substitueix els marcadors pels SVG ────────────────────────
    for marcador in sopa.find_all("p", string=re.compile(r"^@@DIAGRAMA-\d+@@$")):
        idx = int(re.search(r"\d+", marcador.string).group())
        parells = diagrames.get(idx)
        if not parells:
            marcador.decompose()
            continue
        fig = sopa.new_tag("figure", attrs={"class": "diagrama"})
        for mode in ("clar", "fosc"):
            if mode not in parells:
                continue
            svg = BeautifulSoup(parells[mode], "lxml").find("svg")
            # Un diagrama ample encongit fins a cabre a la columna de text queda
            # il·legible: se li deixa una amplada mínima i surt del corse.
            ample = amplada_natural(svg)
            if ample > 720:
                fig["class"] = ["diagrama", "diagrama--ample"]
                svg["style"] = f"min-width:{int(ample * 0.78)}px"
            caixa = sopa.new_tag("div", attrs={"class": f"svg svg--{mode}"})
            caixa.append(svg)
            fig.append(caixa)
        marcador.replace_with(fig)

    # ── Blockquotes: interpretació / exemple / nota ──────────────────────────
    for bq in sopa.find_all("blockquote"):
        text = bq.get_text(" ", strip=True)
        if "Interpretació adoptada" in text or "El Reglament" in text:
            mena, etiqueta = "interpretacio", "Interpretació adoptada"
        elif text.lstrip().startswith(("Exemple", "Cas real", "Atenció")):
            mena, etiqueta = "exemple", None
        else:
            mena, etiqueta = "nota", None
        bq["class"] = bq.get("class", []) + ["crida", f"crida--{mena}"]
        if etiqueta:
            cap = sopa.new_tag("p", attrs={"class": "crida__etiqueta"})
            cap.string = etiqueta
            bq.insert(0, cap)

    # ── Paràgrafs d'avís ─────────────────────────────────────────────────────
    for p in sopa.find_all("p"):
        if p.get_text().lstrip().startswith("⚠️"):
            p["class"] = p.get("class", []) + ["avis"]

    # ── Títols: citació al marge + àncores + índex ───────────────────────────
    index = []
    for h in sopa.find_all(["h2", "h3"]):
        brut = h.get_text()
        titol, citacio = classifica_citacio(brut)
        numero = None
        m = re.match(r"^(\d+(?:\.\d+)?)\.?\s+(.*)$", titol)
        if m:
            numero, titol = m.group(1), m.group(2)

        ident = "s-" + re.sub(r"[^a-z0-9]+", "-",
                              (numero or titol).lower()).strip("-")
        h["id"] = ident
        h.clear()
        if numero:
            marca = sopa.new_tag("span", attrs={"class": "num"})
            marca.string = numero
            h.append(marca)
        titol_tag = sopa.new_tag("span", attrs={"class": "titol"})
        titol_tag.string = titol
        h.append(titol_tag)
        if citacio:
            cita = sopa.new_tag("span", attrs={"class": "citacio"})
            cita.string = citacio
            h.append(cita)
        if h.name == "h2":
            index.append((ident, numero, titol, citacio))

    # ── Taules: contenidor amb desplaçament propi ────────────────────────────
    for taula in sopa.find_all("table"):
        # Una capçalera sense text (markdown «| | |») no ha de deixar una franja
        # buida: es treu i la taula passa a ser de dues columnes, sense títols.
        cap = taula.find("thead")
        if cap and not cap.get_text(strip=True):
            cap.decompose()
            taula["class"] = taula.get("class", []) + ["sense-cap"]
        embolcall = sopa.new_tag("div", attrs={"class": "taula"})
        taula.wrap(embolcall)

    return sopa, index


def full(cos_html, index):
    nav = "\n".join(
        f'<li><a href="#{i}"><span class="nav__num">{n or "·"}</span>'
        f'<span class="nav__t">{t}</span>'
        + (f'<span class="nav__c">{c}</span>' if c else "")
        + "</a></li>"
        for i, n, t, c in index
    )
    articles = ["53.4", "54", "55.1", "55.2", "56", "57"]
    fitxes = "".join(f"<li>art. {a}</li>" for a in articles)

    return f"""<!doctype html>
<html lang="ca" data-tema="auto">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Manual del sorteig de fauna</title>
<style>
:root {{
  --fons:{CLAR['fons']}; --sup:{CLAR['superficie']}; --tinta:{CLAR['tinta']};
  --apagat:{CLAR['apagat']}; --regle:{CLAR['regle']}; --accent:{CLAR['accent']};
  --segell:{CLAR['segell']}; --cita:{CLAR['cita']}; --tint:{CLAR['tint']};
  --serif:{SERIF}; --mono:{MONO}; --sans:{SANS};
  --ombra:0 1px 2px rgba(35,42,46,.05),0 8px 24px -16px rgba(35,42,46,.25);
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-tema="clar"]) {{
    --fons:{FOSC['fons']}; --sup:{FOSC['superficie']}; --tinta:{FOSC['tinta']};
    --apagat:{FOSC['apagat']}; --regle:{FOSC['regle']}; --accent:{FOSC['accent']};
    --segell:{FOSC['segell']}; --cita:{FOSC['cita']}; --tint:{FOSC['tint']};
    --ombra:0 1px 2px rgba(0,0,0,.3),0 8px 24px -16px rgba(0,0,0,.6);
  }}
}}
:root[data-tema="fosc"] {{
  --fons:{FOSC['fons']}; --sup:{FOSC['superficie']}; --tinta:{FOSC['tinta']};
  --apagat:{FOSC['apagat']}; --regle:{FOSC['regle']}; --accent:{FOSC['accent']};
  --segell:{FOSC['segell']}; --cita:{FOSC['cita']}; --tint:{FOSC['tint']};
  --ombra:0 1px 2px rgba(0,0,0,.3),0 8px 24px -16px rgba(0,0,0,.6);
}}

*,*::before,*::after {{ box-sizing:border-box; }}
html {{ scroll-behavior:smooth; scroll-padding-top:1.5rem; }}
body {{
  margin:0; background:var(--fons); color:var(--tinta);
  font-family:var(--serif); font-size:17px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
@media (prefers-reduced-motion:reduce) {{ html {{ scroll-behavior:auto; }} }}

/* ── Estructura ───────────────────────────────────────────────────────── */
.pagina {{ display:grid; grid-template-columns:clamp(230px,22vw,290px) 1fr; }}

.lateral {{
  position:sticky; top:0; height:100vh; overflow-y:auto;
  padding:2rem 1rem 3rem 1.75rem;
  border-right:1px solid var(--regle); background:var(--sup);
}}
.marca {{ font-family:var(--mono); font-size:.66rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin-bottom:1.6rem; }}
.marca b {{ display:block; font-weight:600; }}
.marca span {{ color:var(--apagat); font-weight:400; }}

.lateral ol {{ list-style:none; margin:0; padding:0; }}
.lateral a {{
  display:grid; grid-template-columns:1.9rem 1fr; align-items:baseline;
  gap:.1rem .5rem; padding:.34rem .5rem .34rem .4rem; margin-left:-.4rem;
  text-decoration:none; color:var(--tinta); border-radius:4px;
  font-family:var(--sans); font-size:.83rem; line-height:1.35;
}}
.lateral a:hover {{ background:var(--tint); }}
.lateral a:focus-visible {{ outline:2px solid var(--accent); outline-offset:1px; }}
.nav__num {{ font-family:var(--mono); font-size:.72rem; color:var(--apagat);
  text-align:right; }}
.nav__c {{ grid-column:2; font-family:var(--mono); font-size:.68rem;
  color:var(--apagat); }}
.lateral a[aria-current="true"] {{ background:var(--tint); }}
.lateral a[aria-current="true"] .nav__t {{ color:var(--accent); font-weight:600; }}
.lateral a[aria-current="true"] .nav__num {{ color:var(--accent); }}

.tema {{
  margin-top:2rem; padding-top:1.2rem; border-top:1px solid var(--regle);
  font-family:var(--sans); font-size:.78rem;
}}
.tema button {{
  font:inherit; color:var(--apagat); background:none; cursor:pointer;
  border:1px solid var(--regle); border-radius:5px; padding:.35rem .7rem;
}}
.tema button:hover {{ color:var(--tinta); border-color:var(--apagat); }}

/* ── Document ─────────────────────────────────────────────────────────── */
.doc {{ padding:0 clamp(1.2rem,4vw,3.5rem) 7rem; min-width:0; }}
.llenc {{ max-width:74ch; margin:0 auto; }}

/* Capçalera */
.masta {{ padding:clamp(3rem,7vw,5.5rem) 0 2.2rem; border-bottom:2px solid var(--tinta); }}
.masta__sobre {{ font-family:var(--mono); font-size:.7rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin:0 0 1.1rem; }}
.masta h1 {{ font-size:clamp(2.3rem,5.4vw,3.5rem); line-height:1.04; margin:0;
  font-weight:600; letter-spacing:-.022em; }}
.masta__sota {{ font-size:1.12rem; color:var(--apagat); margin:1rem 0 0;
  max-width:52ch; }}
.base {{ display:flex; flex-wrap:wrap; gap:.4rem; list-style:none;
  margin:1.9rem 0 0; padding:0; }}
.base li {{ font-family:var(--mono); font-size:.72rem; color:var(--accent);
  border:1px solid var(--regle); border-radius:20px; padding:.2rem .65rem;
  background:var(--sup); }}

/* Títols amb citació al marge */
h2,h3 {{ position:relative; font-weight:600; letter-spacing:-.012em; }}
h2 {{ font-size:1.72rem; line-height:1.2; margin:4.2rem 0 1.1rem;
  padding-top:1.6rem; border-top:1px solid var(--regle); }}
/* Sota la capçalera ja hi ha un filet: el primer apartat no en porta un altre */
.masta + h2 {{ margin-top:3rem; padding-top:0; border-top:none; }}
h3 {{ font-size:1.16rem; margin:2.6rem 0 .7rem; }}
h2 .num,h3 .num {{
  font-family:var(--mono); font-weight:400; color:var(--accent);
  font-size:.78em; margin-right:.55rem;
}}
h2 .citacio,h3 .citacio {{
  display:block; font-family:var(--mono); font-size:.7rem; font-weight:400;
  color:var(--apagat); letter-spacing:.04em; margin-top:.4rem;
}}
/* A partir d'aquí hi cap la marginalia: la citació salta al marge esquerre */
@media (min-width:1180px) {{
  h2 .citacio,h3 .citacio {{
    position:absolute; left:-9.5rem; top:.42em; width:8rem;
    text-align:right; margin:0;
  }}
  h2 .citacio {{ top:1.9rem; }}
}}

p {{ margin:0 0 1.05rem; }}
a {{ color:var(--accent); text-underline-offset:2px; }}
strong {{ font-weight:600; }}
hr {{ display:none; }}
code {{ font-family:var(--mono); font-size:.86em; background:var(--tint);
  padding:.1em .38em; border-radius:3px; }}

ul,ol {{ margin:0 0 1.05rem; padding-left:1.3rem; }}
li {{ margin:.3rem 0; }}
li::marker {{ color:var(--apagat); }}

/* Taules */
.taula {{ overflow-x:auto; margin:1.4rem 0 1.8rem;
  border:1px solid var(--regle); border-radius:7px; background:var(--sup); }}
table {{ border-collapse:collapse; width:100%; font-size:.9rem; }}
th,td {{ text-align:left; padding:.62rem .85rem; vertical-align:top;
  border-bottom:1px solid var(--regle); }}
thead th {{ font-family:var(--sans); font-size:.72rem; font-weight:600;
  letter-spacing:.07em; text-transform:uppercase; color:var(--apagat);
  background:var(--tint); white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:none; }}
table.sense-cap td:first-child {{ font-family:var(--sans); font-size:.84rem;
  width:11rem; background:var(--tint); }}
td code {{ background:none; padding:0; color:var(--cita); }}

/* Crides */
.crida {{ margin:1.6rem 0; padding:1.05rem 1.3rem; border-radius:0 7px 7px 0;
  border-left:3px solid var(--apagat); background:var(--sup); font-size:.95rem; }}
.crida > p:last-child {{ margin-bottom:0; }}
.crida--interpretacio {{ border-left-color:var(--segell);
  background:color-mix(in srgb,var(--segell) 5%,var(--sup)); }}
.crida__etiqueta {{ font-family:var(--mono); font-size:.66rem; font-weight:600;
  letter-spacing:.15em; text-transform:uppercase; color:var(--segell);
  margin:0 0 .6rem !important; }}
.crida--exemple {{ border-left-color:var(--cita);
  background:color-mix(in srgb,var(--cita) 5%,var(--sup)); }}
.crida .taula {{ margin:.9rem 0 .4rem; }}
.avis {{ border-left:3px solid var(--segell); padding:.15rem 0 .15rem .95rem;
  margin:1.4rem 0; }}

/* Bloc de procediment */
pre {{ font-family:var(--mono); font-size:.82rem; line-height:1.6;
  background:var(--sup); border:1px solid var(--regle); border-radius:7px;
  padding:1.05rem 1.2rem; overflow-x:auto; margin:1.4rem 0; }}
pre code {{ background:none; padding:0; font-size:1em; }}

/* Diagrames */
.diagrama {{ margin:2rem 0; padding:1.4rem .8rem; background:var(--sup);
  border:1px solid var(--regle); border-radius:8px; box-shadow:var(--ombra);
  overflow-x:auto; }}
.diagrama svg {{ display:block; max-width:100%; height:auto; margin:0 auto; }}
/* Els diagrames amples surten de la columna de text fins on el marge ho permet */
.diagrama--ample {{ margin-inline:calc(-1 * clamp(0rem,5vw,7rem)); }}
.svg--fosc {{ display:none; }}
@media (prefers-color-scheme:dark) {{
  :root:not([data-tema="clar"]) .svg--clar {{ display:none; }}
  :root:not([data-tema="clar"]) .svg--fosc {{ display:block; }}
}}
:root[data-tema="fosc"] .svg--clar {{ display:none; }}
:root[data-tema="fosc"] .svg--fosc {{ display:block; }}
:root[data-tema="clar"] .svg--clar {{ display:block; }}
:root[data-tema="clar"] .svg--fosc {{ display:none; }}

/* Peu */
.peu {{ margin-top:4.5rem; padding-top:1.4rem; border-top:1px solid var(--regle);
  font-size:.83rem; color:var(--apagat); font-family:var(--sans); }}

/* ── Mòbil ─────────────────────────────────────────────────────────────── */
@media (max-width:900px) {{
  .pagina {{ grid-template-columns:1fr; }}
  .lateral {{ position:static; height:auto; border-right:none;
    border-bottom:1px solid var(--regle); padding:1.5rem 1.25rem; }}
  .lateral ol {{ columns:2; column-gap:1.2rem; }}
  .lateral a {{ break-inside:avoid; }}
  body {{ font-size:16px; }}
}}
@media print {{
  .lateral,.tema {{ display:none; }}
  .pagina {{ display:block; }}
  .diagrama {{ break-inside:avoid; box-shadow:none; }}
  h2 {{ break-after:avoid; }}
}}
</style>
</head>
<body>
<div class="pagina">
  <nav class="lateral" aria-label="Índex del manual">
    <div class="marca"><b>Sorteig de fauna</b><span>Manual de referència</span></div>
    <ol>{nav}</ol>
    <div class="tema">
      <button type="button" id="btema" aria-live="polite">Canvia el tema</button>
    </div>
  </nav>

  <main class="doc">
    <div class="llenc">
      <header class="masta">
        <p class="masta__sobre">Reglament de caça · 4 de juny del 2025</p>
        <h1>Com es decideix<br>cada cas</h1>
        <p class="masta__sota">Manual de referència de l'aplicació de sortejos de
        llicències especials de caça. Tota casuística, amb l'article que la governa.</p>
        <ul class="base">{fitxes}</ul>
      </header>
      {cos_html}
      <p class="peu">Aplicació desenvolupada per a la Direcció de Medi Ambient ·
      Generat des de <code>manual.md</code></p>
    </div>
  </main>
</div>

<script>
(function () {{
  var arrel = document.documentElement, boto = document.getElementById('btema');
  var desat = null;
  try {{ desat = localStorage.getItem('tema-manual'); }} catch (e) {{}}
  if (desat) arrel.setAttribute('data-tema', desat);
  boto.addEventListener('click', function () {{
    var fosc = getComputedStyle(document.body).backgroundColor === 'rgb(26, 31, 34)';
    var nou = fosc ? 'clar' : 'fosc';
    arrel.setAttribute('data-tema', nou);
    try {{ localStorage.setItem('tema-manual', nou); }} catch (e) {{}}
  }});

  var enllacos = Array.prototype.slice.call(
    document.querySelectorAll('.lateral a'));
  var seccions = enllacos.map(function (a) {{
    return document.getElementById(a.getAttribute('href').slice(1));
  }});
  if (!('IntersectionObserver' in window)) return;
  var vistes = new Set();
  var obs = new IntersectionObserver(function (entrades) {{
    entrades.forEach(function (e) {{
      if (e.isIntersecting) vistes.add(e.target.id); else vistes.delete(e.target.id);
    }});
    for (var i = 0; i < seccions.length; i++) {{
      var actiu = seccions[i] && vistes.has(seccions[i].id);
      enllacos[i].setAttribute('aria-current', actiu ? 'true' : 'false');
      if (actiu) {{ for (var j = i + 1; j < enllacos.length; j++)
        enllacos[j].setAttribute('aria-current', 'false'); break; }}
    }}
  }}, {{ rootMargin: '0px 0px -72% 0px' }});
  seccions.forEach(function (s) {{ if (s) obs.observe(s); }});
}})();
</script>
</body>
</html>
"""


def main():
    md = ORIGEN.read_text(encoding="utf-8")

    # El títol i la introducció els refà la capçalera HTML, i l'índex el fa el
    # menú lateral: es retallen tots dos del cos.
    md = re.sub(r"\A#\s+.*?(?=\n## )", "", md, flags=re.S)
    md = re.sub(r"\n## Índex\n.*?(?=\n## )", "", md, flags=re.S)

    blocs = re.findall(r"```mermaid\n(.*?)```", md, re.S)
    print(f"→ {len(blocs)} diagrames")
    for i in range(len(blocs)):
        md = re.sub(r"```mermaid\n.*?```", f"@@DIAGRAMA-{i}@@", md, count=1, flags=re.S)

    diagrames = renderitza_diagrames(blocs)

    cos = mistune.create_markdown(plugins=["table", "strikethrough"])(md)
    sopa, index = construeix(cos, diagrames)
    intern = "".join(str(c) for c in sopa.body.children) if sopa.body else str(sopa)

    DESTI.write_text(full(intern, index), encoding="utf-8")
    mida = DESTI.stat().st_size / 1024
    print(f"✓ {DESTI.name} · {mida:.0f} kB · {len(index)} apartats")


if __name__ == "__main__":
    sys.exit(main())
