"""
Dashboard · Café La Corona: ¿a quién y cómo promocionar?
Proyecto final INC-2505 Análisis y exploración de datos

Cómo está organizado el código (útil para explicarlo):
  1. Funciones de datos  -> cargar, limpiar (U3), segmentar (U5)   [con @st.cache_data para no recalcular]
  2. Barra lateral       -> filtros interactivos (tipo de cliente y turno)
  3. Encabezado          -> los hallazgos clave (la "historia" en 4 números)
  4. Seis pestañas       -> cada una cuenta un paso de la historia y corresponde a una unidad
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Café La Corona · Segmentación de clientes", page_icon="☕", layout="wide")

CAFE = "#8B5A2B"
PALETA_SEG = {"Habituales de rutina": "#2F6B9A", "Leales premium": "#2E8B57",
              "Gourmet ocasionales": "#E0A030", "Nuevos curiosos": "#9B59B6"}
VARS_KM = ["visitas_mes", "gasto_promedio", "antiguedad_meses"]
ETIQ = {"visitas_mes": "Visitas al mes", "gasto_promedio": "Gasto por visita (MXN)",
        "tiempo_espera_min": "Espera en fila (min)", "antiguedad_meses": "Antigüedad (meses)",
        "satisfaccion": "Satisfacción (1-10)"}
NUM = list(ETIQ)


# ───────────────────────── 1. FUNCIONES DE DATOS ─────────────────────────
@st.cache_data
def cargar_crudo() -> pd.DataFrame:
    return pd.read_csv("data/clientes_cafeteria_raw.csv")


def limites_tukey(s: pd.Series):
    """Regla de Tukey: [Q1 - 1.5·IQR, Q3 + 1.5·IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


@st.cache_data
def limpiar(df_raw: pd.DataFrame) -> pd.DataFrame:
    """U3 · (1) imputación por mediana y (2) capping con límites de Tukey."""
    df = df_raw.copy()
    for c in ["gasto_promedio", "tiempo_espera_min", "satisfaccion"]:
        df[c] = df[c].fillna(df[c].median())
    for c in ["gasto_promedio", "tiempo_espera_min"]:
        lo, hi = limites_tukey(df[c])
        df[c] = df[c].clip(lower=max(lo, 0), upper=hi)
    return df


@st.cache_data
def segmentar(df: pd.DataFrame, k: int):
    """U5 · (3) estandarización z-score + K-means. Devuelve etiquetas y la silueta para cada k."""
    Z = StandardScaler().fit_transform(df[VARS_KM])
    siluetas = {kk: silhouette_score(Z, KMeans(kk, n_init=10, random_state=42).fit(Z).labels_)
                for kk in range(2, 9)}
    etiquetas = KMeans(k, n_init=10, random_state=42).fit(Z).labels_
    return etiquetas, siluetas


def nombrar_segmentos(df: pd.DataFrame) -> dict:
    """Pone nombre a cada cluster según su perfil (válido para k = 4)."""
    p = df.groupby("cluster")[VARS_KM].mean()
    nuevos = p["antiguedad_meses"].idxmin()
    resto = p.drop(nuevos)
    gourmet = resto["visitas_mes"].idxmin()
    resto = resto.drop(gourmet)
    return {nuevos: "Nuevos curiosos", gourmet: "Gourmet ocasionales",
            resto["gasto_promedio"].idxmax(): "Leales premium",
            resto["gasto_promedio"].idxmin(): "Habituales de rutina"}


crudo = cargar_crudo()
limpio = limpiar(crudo)

# ───────────────────────── 2. BARRA LATERAL (interactividad) ─────────────────────────
with st.sidebar:
    st.header("☕ Filtros")
    tipos = st.multiselect("Tipo de cliente", sorted(limpio.tipo_cliente.unique()),
                           default=sorted(limpio.tipo_cliente.unique()))
    turnos = st.multiselect("Turno preferido", sorted(limpio.turno_preferido.unique()),
                            default=sorted(limpio.turno_preferido.unique()))
    st.caption("Los filtros afectan las pestañas **Explorar**, **Limpieza** e **Hipótesis**. "
               "La segmentación usa siempre a todos los clientes.")
    st.divider()
    st.caption("Datos **simulados** (generados con `generar_datos.py`). "
               "Proyecto INC-2505 · Análisis y exploración de datos.")

mask = limpio.tipo_cliente.isin(tipos) & limpio.turno_preferido.isin(turnos)
df_f, raw_f = limpio[mask], crudo[mask]
if len(df_f) < 30:
    st.warning("Con esos filtros quedan muy pocos clientes (menos de 30). Amplía la selección.")
    st.stop()

# Resultados que alimentan el encabezado (calculados sobre todos los datos)
con = limpio.loc[limpio.tarjeta_lealtad == "Sí", "gasto_promedio"]
sin = limpio.loc[limpio.tarjeta_lealtad == "No", "gasto_promedio"]
reg_global = stats.linregress(limpio.tiempo_espera_min, limpio.visitas_mes)

# ───────────────────────── 3. ENCABEZADO ─────────────────────────
st.title("☕ Café La Corona: ¿a quién y cómo promocionar?")
st.markdown("**Pregunta del administrador:** *¿qué tipos de clientes tengo y qué acción distinta "
            "conviene con cada uno para aumentar visitas y gasto?*")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Clientes analizados", f"{len(limpio)}")
m2.metric("Tipos de cliente encontrados", "4")
m3.metric("Gasto extra con tarjeta", f"+{con.mean() - sin.mean():.1f} MXN", help="Diferencia de medias por visita")
m4.metric("Cada minuto de espera", f"{reg_global.slope:+.1f} visitas/mes", help="Pendiente de la regresión")

tabs = st.tabs(["1 · El problema", "2 · Explorar", "3 · Limpieza", "4 · Hipótesis", "5 · Segmentos", "6 · Recomendación"])

# ───────────────────────── TAB 1 · U1 ─────────────────────────
with tabs[0]:
    st.subheader("La situación")
    st.write("*Café La Corona* es una cafetería dentro de un campus universitario. Cada cliente se registra en una app de "
             "lealtad con un **ID anónimo**. Hoy se da el mismo descuento a todos, y el administrador sospecha que "
             "se está desperdiciando dinero en clientes que ya vienen seguido.")
    st.subheader("Los datos")
    st.write("**320 clientes**, un renglón por cliente. Dataset **propio y simulado** (`generar_datos.py`, semilla fija), "
             "con problemas de calidad intencionales: valores faltantes por fallas de registro y valores extremos por "
             "pedidos de catering y fallas en caja.")
    st.table(pd.DataFrame({
        "Variable": ["tipo_cliente", "tarjeta_lealtad", "turno_preferido", "visitas_mes", "gasto_promedio",
                     "tiempo_espera_min", "antiguedad_meses", "satisfaccion"],
        "Tipo": ["Categórica nominal", "Categórica binaria", "Categórica binaria", "Numérica discreta",
                 "Numérica continua", "Numérica continua", "Numérica discreta", "Ordinal (1-10)"],
        "Qué mide": ["Estudiante / Docente / Administrativo", "Tiene tarjeta de lealtad (Sí/No)", "Mañana / Tarde",
                     "Visitas promedio al mes", "Gasto promedio por visita (MXN)", "Minutos de espera en fila",
                     "Meses desde su registro en la app", "Calificación que da el cliente"]}))
    st.subheader("Consideración ética y privacidad")
    c1, c2 = st.columns(2)
    c1.error("**Riesgo:** cruzar tipo de cliente, turno y hábitos de consumo podría **re-identificar** a una persona "
             "(por ejemplo, un docente de la tarde con compras muy altas).")
    c2.success("**Protocolo:** solo ID anónimo · consentimiento al registrarse · acceso solo del administrador · "
               "resultados **siempre agregados por segmento** · los segmentos dan beneficios, **nunca** exclusiones · "
               "datos conservados solo el semestre necesario.")
    st.info("**¿Por qué este dataset sirve?** Tiene las tres dimensiones clásicas para segmentar clientes (frecuencia, "
            "gasto y antigüedad), un grupo de comparación natural (con/sin tarjeta) y una variable que el administrador "
            "sí puede mover (el tiempo de espera).")

# ───────────────────────── TAB 2 · U2 ─────────────────────────
with tabs[1]:
    st.subheader("Qué vimos en los datos tal como llegaron")
    var = st.selectbox("Variable a explorar", NUM, format_func=lambda v: ETIQ[v], key="var_explorar")
    s = raw_f[var].dropna()
    lo, hi = limites_tukey(s)
    atip = s[(s < lo) | (s > hi)]

    a, b, c, d, e = st.columns(5)
    a.metric("Media", f"{s.mean():.1f}")
    b.metric("Mediana", f"{s.median():.1f}")
    c.metric("Desv. estándar", f"{s.std():.1f}")
    d.metric("Rango", f"{s.max() - s.min():.1f}")
    e.metric("Atípicos (Tukey)", f"{len(atip)}", help=f"Fuera de [{lo:.1f}, {hi:.1f}]")

    g1, g2 = st.columns(2)
    fig = px.histogram(s, nbins=25, color_discrete_sequence=[CAFE], labels={"value": ETIQ[var]})
    fig.add_vline(x=s.mean(), line_dash="dash", line_color="#C0392B", annotation_text="media")
    fig.add_vline(x=s.median(), line_color="#2E8B57", annotation_text="mediana", annotation_position="bottom right")
    fig.update_layout(title="Histograma", showlegend=False, yaxis_title="Clientes", margin=dict(t=50, b=10))
    g1.plotly_chart(fig, width="stretch")
    fig = px.box(raw_f, y=var, points="outliers", color_discrete_sequence=[CAFE], labels={var: ETIQ[var]})
    fig.update_layout(title="Diagrama de caja", margin=dict(t=50, b=10))
    g2.plotly_chart(fig, width="stretch")

    if len(atip) > 0:
        st.warning(f"**{len(atip)} valores atípicos** en *{ETIQ[var]}* (fuera de {lo:.1f} a {hi:.1f}). "
                   f"Media ({s.mean():.1f}) vs. mediana ({s.median():.1f}): "
                   "cuando se separan mucho, la mediana describe mejor al cliente típico.")
    else:
        st.success("Sin atípicos según la regla de Tukey en esta selección.")

    st.markdown("**Anomalías propias del problema:** el gasto de **270–328 MXN** son pedidos de **catering de personal "
                "administrativo** registrados como consumo individual; las esperas de **38–52 min** parecen "
                "**fallas en caja o en la app**.")
    st.subheader("Comparación entre subgrupos")
    gx = st.radio("Comparar por", ["tipo_cliente", "tarjeta_lealtad", "turno_preferido"], horizontal=True)
    fig = px.box(raw_f, x=gx, y=var, color=gx, points=False, labels={var: ETIQ[var]})
    fig.update_layout(showlegend=False, margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    st.caption("Conclusión accionable: los clientes con tarjeta gastan más (se prueba en *Hipótesis*) y el personal "
               "administrativo distorsiona el gasto con catering, así que conviene registrarlo como categoría aparte.")

# ───────────────────────── TAB 3 · U3 ─────────────────────────
with tabs[2]:
    st.subheader("Tres técnicas de limpieza, en este orden")
    t1, t2, t3 = st.columns(3)
    t1.markdown("**① Imputación por mediana**  \nRellena los faltantes con la mediana. *No la media*: los atípicos "
                "la inflan. *No eliminar renglones*: perderíamos hasta 15 % de clientes.")
    t2.markdown("**② Capping (winsorización)**  \nLos valores fuera de los límites de Tukey se reemplazan por el límite. "
                "El cliente sigue en el análisis, pero su valor extremo ya no domina.")
    t3.markdown("**③ Estandarización z-score**  \n`z = (x − media) / desv. est.` Pone todas las variables en la misma "
                "escala para que K-means no sea dominado por el gasto (en pesos). Se aplica al segmentar.")

    falt = crudo.isna().sum()
    falt = falt[falt > 0].to_frame("Faltantes")
    falt["% del total"] = (falt["Faltantes"] / len(crudo) * 100).round(1)
    st.markdown("**Valores faltantes detectados**")
    st.dataframe(falt.rename(index=ETIQ), width="content")

    st.markdown("### Antes vs. después")
    v3 = st.selectbox("Variable", ["gasto_promedio", "tiempo_espera_min"], format_func=lambda v: ETIQ[v], key="v3")
    antes, despues = raw_f[v3].dropna(), df_f[v3]
    cmp = pd.DataFrame({"Antes": [antes.mean(), antes.median(), antes.std(), antes.max()],
                        "Después": [despues.mean(), despues.median(), despues.std(), despues.max()]},
                       index=["Media", "Mediana", "Desv. estándar", "Máximo"]).round(2)
    cc1, cc2 = st.columns([1, 2])
    cc1.dataframe(cmp, width="stretch")
    figc = go.Figure()
    figc.add_trace(go.Box(x=antes, name="Antes", marker_color="#C0392B", boxpoints="outliers"))
    figc.add_trace(go.Box(x=despues, name="Después", marker_color="#2E8B57", boxpoints="outliers"))
    figc.update_layout(showlegend=False, margin=dict(t=10, b=10), height=260, xaxis_title=ETIQ[v3])
    cc2.plotly_chart(figc, width="stretch")
    st.info("Sin esta limpieza, K-means habría formado un grupo de 'catering' con 5 personas en vez de segmentos "
            "útiles, y la regresión habría salido distorsionada por las esperas de 40-50 minutos.")

# ───────────────────────── TAB 4 · U4 ─────────────────────────
with tabs[3]:
    st.subheader("¿Los clientes con tarjeta gastan más por visita?  (t de Student)")
    st.markdown("**H₀:** el gasto medio es igual con y sin tarjeta · **H₁:** es distinto · **α = 0.05**")
    cf = df_f.loc[df_f.tarjeta_lealtad == "Sí", "gasto_promedio"]
    sf = df_f.loc[df_f.tarjeta_lealtad == "No", "gasto_promedio"]
    if len(cf) < 5 or len(sf) < 5:
        st.warning("Muy pocos clientes en algún grupo con los filtros actuales.")
    else:
        t, p = stats.ttest_ind(cf, sf, equal_var=True)
        n1, n2 = len(cf), len(sf)
        sp2 = ((n1 - 1) * cf.var() + (n2 - 1) * sf.var()) / (n1 + n2 - 2)
        ee = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        dif = cf.mean() - sf.mean()
        tc = stats.t.ppf(0.975, n1 + n2 - 2)
        ic = (dif - tc * ee, dif + tc * ee)
        lev = stats.levene(cf, sf).pvalue

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Con tarjeta", f"{cf.mean():.1f} MXN")
        k2.metric("Sin tarjeta", f"{sf.mean():.1f} MXN")
        k3.metric("Valor p", f"{p:.4f}")
        k4.metric("IC 95 % de la diferencia", f"[{ic[0]:.1f}, {ic[1]:.1f}]")

        fig = go.Figure(go.Scatter(x=[dif], y=["Diferencia"], mode="markers", marker=dict(size=14, color=CAFE),
                                   error_x=dict(type="data", symmetric=False, array=[ic[1] - dif],
                                                arrayminus=[dif - ic[0]], color=CAFE, thickness=3)))
        fig.add_vline(x=0, line_dash="dash", line_color="#C0392B", annotation_text="sin diferencia")
        fig.update_layout(height=170, margin=dict(t=20, b=30), xaxis_title="Gasto extra con tarjeta (MXN) e IC 95 %")
        st.plotly_chart(fig, width="stretch")
        if p < 0.05:
            st.success(f"**Se rechaza H₀** (p = {p:.4f}): hay evidencia de que el gasto difiere. Con 95 % de confianza, "
                       f"quienes tienen tarjeta gastan entre {ic[0]:.1f} y {ic[1]:.1f} MXN más por visita; "
                       "el intervalo no contiene el 0.")
        else:
            st.info(f"**No se rechaza H₀** (p = {p:.4f}) con esta selección: el intervalo incluye al 0.")
        st.caption(f"Supuestos: varianzas similares (Levene p = {lev:.2f}); con ~150 clientes por grupo el teorema central "
                   "del límite sostiene la prueba. **Limitación:** no prueba causalidad; quizá quienes ya gastaban "
                   "más fueron quienes se sacaron la tarjeta.")

    st.divider()
    st.subheader("¿La espera en fila se relaciona con cuántas veces vuelve el cliente?  (regresión y Pearson)")
    reg = stats.linregress(df_f.tiempo_espera_min, df_f.visitas_mes)
    xs = np.linspace(df_f.tiempo_espera_min.min(), df_f.tiempo_espera_min.max(), 50)
    fig = px.scatter(df_f, x="tiempo_espera_min", y="visitas_mes", opacity=.55, color_discrete_sequence=[CAFE],
                     labels={"tiempo_espera_min": ETIQ["tiempo_espera_min"], "visitas_mes": ETIQ["visitas_mes"]})
    fig.add_trace(go.Scatter(x=xs, y=reg.intercept + reg.slope * xs, mode="lines", name="Regresión",
                             line=dict(color="#C0392B", width=3)))
    fig.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    r1, r2, r3 = st.columns(3)
    r1.metric("Pendiente", f"{reg.slope:.2f} visitas/min")
    r2.metric("Pearson r", f"{reg.rvalue:.2f}")
    r3.metric("R²", f"{reg.rvalue ** 2:.2f}", help="Proporción de la variación de las visitas explicada por la espera")
    st.write(f"Cada minuto extra de espera se asocia con **{abs(reg.slope):.2f} visitas menos al mes** "
             f"(relación negativa moderada-baja, p = {reg.pvalue:.1e}). La espera explica solo "
             f"~{reg.rvalue ** 2 * 100:.0f} % de la variación: el resto depende de otros factores, como el perfil del cliente.")
    st.warning("**Correlación no implica causalidad:** las horas pico tienen más fila y quizá atraen a otro tipo de cliente. "
               "Hace falta una prueba piloto para confirmarlo.")

    st.markdown("#### 🎚️ Simulador: ¿y si reducimos la espera?")
    st.caption("Estimación con la recta de regresión (asociación observada, **no** efecto garantizado).")
    red = st.slider("Minutos que se reduce la espera promedio", 0.0, 5.0, 2.0, 0.5)
    ganancia = abs(reg_global.slope) * red
    prom_gasto = limpio.gasto_promedio.mean()
    st.metric("Visitas adicionales estimadas por cliente al mes", f"+{ganancia:.2f}",
              delta=f"≈ {ganancia * prom_gasto * len(limpio):,.0f} MXN extra al mes en toda la base")

# ───────────────────────── TAB 5 · U5 ─────────────────────────
with tabs[4]:
    st.subheader("K-means: los grupos de clientes que descubrimos")
    k = st.slider("Número de segmentos (k)", 2, 8, 4, help="El coeficiente de silueta recomienda k = 4")
    etiquetas, siluetas = segmentar(limpio, k)
    seg_df = limpio.copy()
    seg_df["cluster"] = etiquetas
    mejor_k = max(siluetas, key=siluetas.get)

    if k == 4:
        nombres = nombrar_segmentos(seg_df)
        seg_df["segmento"] = seg_df.cluster.map(nombres)
        orden_seg = list(PALETA_SEG)
    else:
        seg_df["segmento"] = "Grupo " + (seg_df.cluster + 1).astype(str)
        orden_seg = sorted(seg_df.segmento.unique())

    cA, cB = st.columns([1, 1.4])
    fig = go.Figure(go.Scatter(x=list(siluetas), y=list(siluetas.values()), mode="lines+markers",
                               line=dict(color=CAFE)))
    fig.add_trace(go.Scatter(x=[k], y=[siluetas[k]], mode="markers", marker=dict(size=16, color="#C0392B")))
    fig.update_layout(title="Selección de k · coeficiente de silueta", showlegend=False, xaxis_title="k",
                      yaxis_title="Silueta", margin=dict(t=50, b=10), height=360)
    cA.plotly_chart(fig, width="stretch")
    fig = px.scatter(seg_df, x="visitas_mes", y="gasto_promedio", color="segmento", opacity=.75,
                     color_discrete_map=PALETA_SEG, category_orders={"segmento": orden_seg},
                     hover_data=["antiguedad_meses", "tipo_cliente"],
                     labels={"visitas_mes": ETIQ["visitas_mes"], "gasto_promedio": ETIQ["gasto_promedio"],
                             "segmento": ""})
    fig.update_layout(title="Segmentos de clientes", margin=dict(t=50, b=10), height=360,
                      legend=dict(orientation="h", y=-0.25))
    cB.plotly_chart(fig, width="stretch")
    st.caption(f"**¿Por qué k = 4?** La silueta alcanza su máximo en k = {mejor_k} ({siluetas[mejor_k]:.3f}); "
               f"k = 5 queda prácticamente empatada ({siluetas[5]:.3f}), así que se prefiere k = 4 por parsimonia "
               "(menos grupos, cada uno con una acción distinta y fácil de explicar).")

    if k == 4:
        perfil = seg_df.groupby("segmento").agg(
            Clientes=("id_cliente", "count"), Visitas_mes=("visitas_mes", "mean"),
            Gasto_visita=("gasto_promedio", "mean"), Antigüedad_meses=("antiguedad_meses", "mean"),
            Con_tarjeta=("tarjeta_lealtad", lambda x: (x == "Sí").mean() * 100)).loc[orden_seg].round(1)
        perfil["Clientes"] = perfil["Clientes"].astype(int)
        perfil = perfil.rename(columns={"Visitas_mes": "Visitas/mes", "Gasto_visita": "Gasto/visita (MXN)",
                                        "Antigüedad_meses": "Antigüedad (meses)", "Con_tarjeta": "% con tarjeta"})
        st.dataframe(perfil, width="stretch")

        st.markdown("### Una acción distinta para cada segmento")
        acciones = {
            "Habituales de rutina": ("Vienen casi todos los días pero gastan poco (un café o un pan).",
                                     "**Combo prepago de desayuno** (café + pan a precio cerrado) para subir el ticket sin descuento general."),
            "Leales premium": ("Vienen seguido, gastan mucho y son los más antiguos; casi todos tienen tarjeta.",
                               "**No descontar.** Beneficios no monetarios: preventa de bebidas de temporada, acceso anticipado, puntos dobles."),
            "Gourmet ocasionales": ("Vienen poco pero, cuando vienen, gastan lo más alto.",
                                    "**Subir su frecuencia:** cupón 'tu próxima visita esta semana' y **pre-pedido por la app** para evitar la fila."),
            "Nuevos curiosos": ("Se registraron hace ~2 meses; vienen poco y gastan medio.",
                                "**Bienvenida y tarjeta:** 2×1 la primera semana e invitación a la tarjeta de lealtad, cuando se forma el hábito.")}
        cols = st.columns(4)
        for col, (nombre, (desc, acc)) in zip(cols, acciones.items()):
            n = int(perfil.loc[nombre, "Clientes"])
            col.markdown(f"<div style='border-left:6px solid {PALETA_SEG[nombre]};padding:6px 12px;"
                         f"background:#fff;border-radius:6px'><b>{nombre}</b><br><small>{n} clientes</small></div>",
                         unsafe_allow_html=True)
            col.write(desc)
            col.success(acc)
    else:
        st.info("Mueve el control a **k = 4** para ver los segmentos con nombre y sus acciones.")

# ───────────────────────── TAB 6 · RECOMENDACIÓN ─────────────────────────
with tabs[5]:
    st.subheader("Recomendación al administrador")
    st.markdown(f"""
**La historia en tres pasos**

1. **No todos los clientes son iguales.** Encontramos 4 tipos con hábitos muy distintos. Un descuento general se desperdicia en los *Leales premium* y los *Habituales*, que ya vienen.
2. **La tarjeta se asocia con más gasto** (≈ {con.mean() - sin.mean():.1f} MXN por visita, p < 0.05), aunque no sabemos si la tarjeta *causa* ese gasto.
3. **La espera cuesta visitas:** cada minuto en fila se asocia con ≈ {abs(reg_global.slope):.1f} visitas menos al mes.

**Qué hacer el próximo semestre**

| Prioridad | Acción | Por qué (evidencia) |
|---|---|---|
| 1 | Convertir a los **Nuevos curiosos** en clientes con tarjeta (2×1 de bienvenida) | Es el segmento donde se forma el hábito; la tarjeta se asocia con más gasto |
| 2 | **Reducir la espera** (segunda caja en horas pico + pre-pedido en la app) | Regresión: −{abs(reg_global.slope):.1f} visitas/mes por minuto de espera |
| 3 | **Combo prepago** para los *Habituales de rutina* | Alta frecuencia, ticket bajo: hay espacio para subir el gasto |
| 4 | **Beneficios sin descuento** para los *Leales premium* | Ya gastan y vienen: proteger el margen |
| 5 | **Cupón de frecuencia** para los *Gourmet ocasionales* | Gastan mucho cuando vienen, pero vienen poco |

**Siguiente paso:** una **prueba piloto** (abrir una segunda caja en el turno de mayor fila y ofrecer la tarjeta al azar) para confirmar con evidencia causal lo que aquí solo observamos como asociación.

**Calidad de datos:** registrar el catering como categoría separada y generar una alerta cuando la espera supere ~15 minutos.

*Limitaciones:* datos simulados de un solo semestre; la segmentación depende de las tres variables elegidas; los segmentos describen, no predicen.
""")
