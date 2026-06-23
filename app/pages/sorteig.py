import streamlit as st
from utils import draw_logic, config


@st.cache_data
def load_csv(file):
    import pandas as pd
    return pd.read_csv(file, sep=";")


# ──────────────────────────────────────────────────────────────────────────────
#  ESTAT DE LA CONFIGURACIÓ DE ZONES
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_state():
    st.session_state.setdefault("zones_cfg", {})
    st.session_state.setdefault("species_list", list(config.ESPECIES_BASE))
    st.session_state.setdefault("_zid", 0)


def _new_id():
    st.session_state["_zid"] += 1
    return st.session_state["_zid"]


def _zones_for(especie):
    """Retorna (i precarrega si cal) la llista de zones d'una espècie."""
    cfg = st.session_state["zones_cfg"]
    if especie not in cfg:
        zs = config.default_zones(especie)
        for z in zs:
            z["_id"] = _new_id()
        cfg[especie] = zs
    return cfg[especie]


def _clean_zones(zones):
    """Zones sense claus internes (_id), llestes per al motor o per exportar."""
    return [{k: v for k, v in z.items() if k != "_id"} for z in zones]


# ──────────────────────────────────────────────────────────────────────────────
#  EDITOR DE ZONES
# ──────────────────────────────────────────────────────────────────────────────

def _render_captures(z, zid):
    st.markdown("**Tipus de captura**")
    caps = z.setdefault("captures", [])
    if st.button("Afegeix Tipus", key=f"addcap_{zid}"):
        caps.append({"tipus": [], "quantitat": 0})
        st.rerun()
    for k, cap in enumerate(caps):
        c1, c2, c3 = st.columns([5, 2, 1])
        sel = c1.multiselect(
            f"Valors Tipus {k+1}",
            config.TIPUS_OPTIONS,
            default=[t for t in cap.get("tipus", []) if t in config.TIPUS_OPTIONS],
            key=f"captip_{zid}_{k}",
        )
        # Si es tria "Indeterminat", és exclusiu.
        if "Indeterminat" in sel and len(sel) > 1:
            sel = ["Indeterminat"]
        cap["tipus"] = sel
        cap["quantitat"] = int(
            c2.number_input(
                f"Quantitat {k+1}",
                min_value=0,
                step=1,
                value=int(cap.get("quantitat", 0)),
                key=f"capqty_{zid}_{k}",
            )
        )
        if c3.button("🗑", key=f"capdel_{zid}_{k}", help="Elimina aquest tipus"):
            caps.pop(k)
            st.rerun()


def _render_zone(zones, i):
    z = zones[i]
    zid = z["_id"]
    titol = f"{i+1}. {z.get('nom','(sense nom)')}  ·  {z.get('tipus','TCC')}"
    if z.get("modalitat"):
        titol += " · modalitat A/B"
    with st.expander(titol, expanded=False):
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        z["nom"] = c1.text_input("Nom de la zona", value=z.get("nom", ""), key=f"nom_{zid}")
        tipus_idx = config.TIPUS_ZONA.index(z.get("tipus", "TCC")) if z.get("tipus") in config.TIPUS_ZONA else 1
        z["tipus"] = c2.selectbox("Tipus", config.TIPUS_ZONA, index=tipus_idx, key=f"tipus_{zid}")
        z["aleatori"] = c3.checkbox("Ordre aleatori", value=z.get("aleatori", True), key=f"alea_{zid}")
        z["estranger_pct"] = float(
            c4.number_input(
                "% màx. estrangers",
                min_value=0.0, max_value=100.0, step=1.0,
                value=float(z.get("estranger_pct", config.ESTRANGER_PCT_DEFAULT)),
                key=f"estr_{zid}",
            )
        )

        z["modalitat"] = st.checkbox(
            "Modalitat A/B (colles + individual, com l'IS TCC)",
            value=z.get("modalitat", False),
            key=f"mod_{zid}",
        )

        if z["tipus"] == "Vedat":
            z["reserva_pct"] = float(
                st.number_input(
                    "% de captures reservat als locals",
                    min_value=0.0, max_value=100.0, step=1.0,
                    value=float(z.get("reserva_pct") or config.RESERVA_PCT_DEFAULT),
                    key=f"res_{zid}",
                )
            )
            st.caption("Repartiment d'aquest percentatge per parròquia (ha de sumar 100):")
            parr = z.get("parroquies") or {}
            new_parr = {}
            pcols = st.columns(len(config.PARROQUIES))
            for j, p in enumerate(config.PARROQUIES):
                v = pcols[j].number_input(
                    p, min_value=0.0, max_value=100.0, step=1.0,
                    value=float(parr.get(p, 0.0)), key=f"parr_{zid}_{j}",
                )
                if v > 0:
                    new_parr[p] = float(v)
            z["parroquies"] = new_parr
            tot = sum(new_parr.values())
            if new_parr and abs(tot - 100.0) > 0.5:
                st.warning(f"⚠️ Els percentatges parroquials sumen {tot:.1f}, no 100.")
            elif not new_parr:
                st.warning("⚠️ Aquest vedat encara no té cap distribució parroquial definida.")
        else:
            z["reserva_pct"] = None
            z["parroquies"] = {}

        _render_captures(z, zid)

        st.divider()
        b1, b2, b3 = st.columns(3)
        if b1.button("⬆️ Amunt", key=f"up_{zid}", disabled=(i == 0)):
            zones[i - 1], zones[i] = zones[i], zones[i - 1]
            st.rerun()
        if b2.button("⬇️ Avall", key=f"down_{zid}", disabled=(i == len(zones) - 1)):
            zones[i + 1], zones[i] = zones[i], zones[i + 1]
            st.rerun()
        if b3.button("🗑 Elimina la zona", key=f"delz_{zid}"):
            zones.pop(i)
            st.rerun()


def render_zones(especie):
    zones = _zones_for(especie)

    top1, top2 = st.columns([2, 3])
    if top1.button("➕ Afegeix zona", key=f"addzone_{especie}"):
        nz = config.new_zone(nom=f"Zona {len(zones)+1}")
        nz["_id"] = _new_id()
        zones.append(nz)
        st.rerun()
    if top2.button("↺ Restaura les zones per defecte", key=f"reset_{especie}"):
        zs = config.default_zones(especie)
        for z in zs:
            z["_id"] = _new_id()
        st.session_state["zones_cfg"][especie] = zs
        st.rerun()

    if not zones:
        st.info("Aquesta espècie encara no té zones. Afegeix-ne una amb «➕ Afegeix zona».")

    st.caption("L'ordre de la llista és l'ordre del sorteig (usa ⬆️/⬇️ per canviar-lo).")
    for i in range(len(zones)):
        _render_zone(zones, i)


def render_import_export(especie):
    import json
    with st.expander("Desar / carregar configuració (JSON)"):
        clean = _clean_zones(_zones_for(especie))
        st.download_button(
            "📤 Exporta la configuració d'aquesta espècie",
            data=config.zones_to_json(clean).encode("utf-8"),
            file_name=f"config_{especie}.json",
            mime="application/json",
            key=f"exp_{especie}",
        )
        up = st.file_uploader("📥 Importa configuració (JSON)", type="json", key=f"imp_{especie}")
        if up is not None and st.button("Carrega aquest fitxer", key=f"impbtn_{especie}"):
            try:
                zs = json.loads(up.getvalue().decode("utf-8"))
                for z in zs:
                    z["_id"] = _new_id()
                st.session_state["zones_cfg"][especie] = zs
                st.success("Configuració carregada.")
                st.rerun()
            except Exception as exc:
                st.error(f"No s'ha pogut llegir el fitxer: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
#  PÀGINA
# ──────────────────────────────────────────────────────────────────────────────

_ensure_state()

with st.expander("Instruccions d'ús", expanded=False):
    st.markdown(
        """
1. **Seleccioneu l'espècie** (o afegiu-ne una de nova, tipus «Altres»).
2. **Configureu les zones**: per a cada zona indiqueu el tipus (Vedat o TCC), si té
   modalitat A/B, els tipus de captura i les quantitats. En els vedats, fixeu el
   percentatge reservat als locals i el repartiment per parròquies.
3. **Ordeneu les zones** amb ⬆️/⬇️: el sorteig segueix l'ordre de la llista.
4. **Pugeu** el **CSV de prioritats** i el **CSV d'inscrits**.
5. *(Opcional)* Introduïu una **llavor** per reproduir exactament el mateix sorteig.
6. Premeu **«Executar sorteig»** per obtenir i descarregar els resultats.
        """
    )

with st.expander("Format dels CSV"):
    st.markdown(
        """
**CSV de prioritats** — una fila per caçador:

| Columna | Quan cal |
|---|---|
| `ID` | sempre |
| `Prioritat` | sempre (1 = màxima) |
| `anys_sense_captura` | sempre |
| `Estranger` | sempre (Sí/No) |
| `Parroquia` | si alguna zona és **Vedat** |
| `Modalitat` (`A`/`B`) i `Colla_ID` | si alguna zona té **modalitat A/B** |

**CSV d'inscrits** — columnes `ID` i `Codi_Sorteig` (el codi ha de coincidir amb el **nom de la zona**).
        """
    )

st.markdown("💡 Pots descarregar exemples de fitxers aquí:")
ex1, ex2, ex3 = st.columns(3)
with open("isard.csv", "rb") as f1:
    ex1.download_button("📥 Exemple Isard", f1, file_name="isard.csv", mime="text/csv")
with open("altres.csv", "rb") as f2:
    ex2.download_button("📥 Altres espècies", f2, file_name="altres.csv", mime="text/csv")
with open("sorteig.csv", "rb") as f3:
    ex3.download_button("📥 Exemple Inscripcions", f3, file_name="sorteig.csv", mime="text/csv")

# ── Selecció d'espècie ────────────────────────────────────────────────────────
sp1, sp2 = st.columns([2, 2])
especie = sp1.selectbox("Espècie", st.session_state["species_list"], key="especie_sel")
with sp2.expander("➕ Nova espècie (tipus «Altres»)"):
    new_sp = st.text_input("Nom", key="new_sp_name")
    if st.button("Crea espècie"):
        nom = (new_sp or "").strip()
        if nom and nom not in st.session_state["species_list"]:
            st.session_state["species_list"].append(nom)
            st.session_state["zones_cfg"][nom] = []
            st.rerun()

st.subheader(f"Zones — {especie}")
render_zones(especie)
render_import_export(especie)

# ── CSV i llavor ──────────────────────────────────────────────────────────────
csv1 = st.file_uploader("CSV de prioritats", type="csv", key="csv1")
csv2 = st.file_uploader("CSV d'inscrits", type="csv", key="csv2")

seed_input = st.number_input("Llavor opcional", value=0, step=1)
seed = int(seed_input) if seed_input else None

st.session_state.setdefault("run_draw", False)

# ─────────────────────  EXECUTAR SORTEIG  ────────────────────────────────
if st.button("Executar sorteig"):
    st.session_state["run_draw"] = True

if st.session_state.get("run_draw"):
    import pandas as pd

    zones_clean = _clean_zones(_zones_for(especie))

    if not csv1 or not csv2:
        st.error("Cal carregar els dos CSV")
        st.stop()
    if not zones_clean:
        st.error("Cal configurar com a mínim una zona per a aquesta espècie.")
        st.stop()

    df1 = load_csv(csv1)
    df2 = load_csv(csv2)

    # ── Validació de columnes segons la configuració de zones ─────────────
    needs_modalitat = any(z.get("modalitat") for z in zones_clean)
    needs_parroquia = any(z.get("tipus") == "Vedat" for z in zones_clean)
    required = {"ID", "Prioritat", "anys_sense_captura", "Estranger"}
    if needs_modalitat:
        required |= {"Modalitat", "Colla_ID"}
    if needs_parroquia:
        required |= {"Parroquia"}
    try:
        draw_logic.validar_csv2(df2)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    missing = required - set(df1.columns)
    if missing:
        st.error(f"Falten columnes al CSV de prioritats: {', '.join(sorted(missing))}")
        st.stop()

    # ── Avís d'IDs presents només en un CSV ───────────────────────────────
    ids_prio = set(df1["ID"].astype(str))
    ids_ins = set(df2["ID"].astype(str))
    missing_in_ins = ids_prio - ids_ins
    if needs_modalitat and "Modalitat" in df1.columns:
        ab_ids = set(df1[df1["Modalitat"].isin(["A", "B"])]["ID"].astype(str))
        missing_in_ins -= ab_ids
    missing_in_prio = ids_ins - ids_prio

    if (missing_in_ins or missing_in_prio) and not st.session_state.get("confirm_missing_ids", False):
        parts = []
        if missing_in_ins:
            parts.append("IDs a prioritats però no a inscrits: " + ", ".join(sorted(missing_in_ins)))
        if missing_in_prio:
            parts.append("IDs a inscrits però no a prioritats: " + ", ".join(sorted(missing_in_prio)))
        st.warning(" ".join(parts) + " S'ignoraran si continues.")
        c1, c2 = st.columns(2)
        if c1.button("Ignorar i continuar", key="confirm_missing_ids_btn"):
            st.session_state["confirm_missing_ids"] = True
            st.rerun()
        if c2.button("Atura el procés", key="stop_missing_ids"):
            st.stop()
        st.stop()

    # ── Zones amb modalitat: inscrits sense Modalitat ─────────────────────
    ids_to_skip = []
    if needs_modalitat:
        mod_zone_names = [z["nom"] for z in zones_clean if z.get("modalitat")]
        inscrits_mod = df2[df2["Codi_Sorteig"].isin(mod_zone_names)]
        missing_mod = inscrits_mod.merge(df1[["ID", "Modalitat"]], on="ID", how="left")
        missing_mod = missing_mod[
            missing_mod["Modalitat"].isna()
            | (missing_mod["Modalitat"].astype(str).str.strip() == "")
        ]
        if not missing_mod.empty and not st.session_state.get("confirm_missing_mod", False):
            st.warning(
                "Aquests caçadors estan inscrits en una zona amb modalitat però no en "
                "tenen cap especificada i s'ignoraran si continues: "
                + ", ".join(missing_mod["ID"].astype(str).unique())
            )
            c1, c2 = st.columns(2)
            if c1.button("Ignorar i continuar", key="confirm_missing_mod_btn"):
                st.session_state["confirm_missing_mod"] = True
                st.session_state["ids_to_skip_mod"] = missing_mod["ID"].tolist()
                st.session_state["mod_zone_names"] = mod_zone_names
                st.rerun()
            if c2.button("Atura el procés", key="stop_missing_mod"):
                st.stop()
            st.stop()
        ids_to_skip = st.session_state.get("ids_to_skip_mod", [])
        mod_zone_names = st.session_state.get("mod_zone_names", mod_zone_names)
        if ids_to_skip:
            mask = df2["Codi_Sorteig"].isin(mod_zone_names) & df2["ID"].isin(ids_to_skip)
            df2 = df2.loc[~mask].copy()

    # ── Executar ──────────────────────────────────────────────────────────
    try:
        resultat, resums = draw_logic.processar_sorteigs(df1, df2, zones_clean, especie, seed)
    except Exception as exc:
        st.error(f"🚫 Error en el sorteig: {exc}")
        st.stop()

    st.session_state["resultat"] = resultat
    st.session_state["resums"] = resums
    st.subheader("Resultats")
    capture_cols = [z["nom"].replace(" ", "_") for z in zones_clean]
    for col in capture_cols:
        if col in resultat.columns:
            resultat[col] = resultat[col].astype(str).replace("nan", "").replace("<NA>", "")
    st.dataframe(resultat, use_container_width=True)

    st.download_button(
        "Descarregar CSV",
        resultat.to_csv(index=False).encode("utf-8"),
        file_name="resultats.csv",
    )

    # Neteja de flags
    st.session_state.pop("confirm_missing_mod", None)
    st.session_state.pop("ids_to_skip_mod", None)
    st.session_state.pop("mod_zone_names", None)
    st.session_state.pop("confirm_missing_ids", None)
    st.session_state["run_draw"] = False
