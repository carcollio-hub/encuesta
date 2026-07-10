import re
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG & STYLE
# ============================================================================
st.set_page_config(page_title="SCD · Encuesta de socios", page_icon="🎵", layout="wide")

NAVY = "#1F4E5F"
GOLD = "#D98E04"
GREEN = "#3E8E6B"
RED = "#C0473A"
GREY = "#6B7280"
PALETTE = [NAVY, GOLD, GREEN, RED, "#7A6C9B", "#3E7CB1", "#C4A35A", "#8FA998"]

st.markdown(f"""
<style>
.hero-box {{
    background: linear-gradient(135deg, {NAVY} 0%, #123542 100%);
    color: white; padding: 22px 26px; border-radius: 14px; margin-bottom: 10px;
}}
.hero-box h1 {{ margin:0; font-size: 1.7rem; }}
.hero-box p {{ margin: 4px 0 0 0; opacity:.85; }}
.callout {{
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; color: white;
}}
.callout-red {{ background: linear-gradient(135deg, {RED}, #8f342a); }}
.callout-gold {{ background: linear-gradient(135deg, {GOLD}, #a86903); }}
.callout-green {{ background: linear-gradient(135deg, {GREEN}, #1f5c44); }}
.callout h3 {{ margin: 0 0 6px 0; font-size: 1.05rem; }}
.callout p {{ margin:0; font-size: 0.92rem; opacity:.95; }}
.quote-card {{
    border-left: 5px solid {GOLD}; background: #F7F4EE; border-radius: 6px;
    padding: 10px 14px; margin-bottom: 8px; font-style: italic; color: #333;
}}
.quote-card.neg {{ border-left-color: {RED}; background: #FBF0EE; }}
.quote-card.pos {{ border-left-color: {GREEN}; background: #EFF7F2; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# COLUMN NAMES
# ============================================================================
COL_GENERO_MUS = "¿Con qué género(s) musical(es) te identificas?"
COL_ENTERASTE = "¿Cómo te enteraste del SCD?"
COL_FACIL_INS = "¿Qué tan fácil fue inscribirte en SCD?"
COL_COMPLEJO = "¿Qué fue lo más complejo del proceso?"
COL_FACIL_OBRAS = "¿Qué tan fácil fue inscribir tus obras/canciones?"
COL_FRECUENCIA = "¿Con qué frecuencia sientes que recibes información del SCD?"
COL_ACOMPANAMIENTO = "¿El acompañamiento te parece suficiente?"
COL_ESPERARIAS = "¿Qué esperarías del SCD que hoy no estés recibiendo?"
COL_NPS = "¿Recomendarías a otro músico incorporarse a SCD a otro músico?"
COL_PALABRA = "Si tuvieras que describir tu relación con SCD en una palabra, ¿cuál sería?"
COL_GENERO = "Genero"

STOPWORDS_ES = set("""
de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o
este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos
durante todos uno les ni contra otros ese eso ante ellos e esto mí antes algunos qué unos yo
tanto esa estos mucho quienes nada muchos cual poco ella estar estas algunas algo nosotros mi
mis tú te ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías tuyo tuya tuyos tuyas
suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra vuestros vuestras esos
esas soy eres es somos sois son esté estás está estamos estáis están esa scd les fue ser tan
si asi así siento uno una han sido va ha he han cada solo sólo tal
""".split())

# ============================================================================
# SENTIMENT LEXICON (heuristic, tuned to this survey's vocabulary)
# ============================================================================
POSITIVE_WORDS = set("""
buena bueno excelente confianza amistad amor simbiosis cordial amigable interesante tranquila
tranquilo genial inspira gracias profesional nueva esperanza compañero alianza apoyo protección
compromiso adecuada gestion aprendiendo aprendizaje musical satisfecho satisfecha contento
contenta feliz agradecido agradecida honor increible increíble especial segura tranquilidad
respaldado respaldada valorada valorado orgullo orgullosa positiva positivo fluido facil fácil
rapido rápida rápido claro clara amena cercania cercanía cercano cercana
""".split())

NEGATIVE_WORDS = set("""
distante lejana lejano confusa confuso fria frío escasa escaso desconocimiento burocracia odio
pasiva complicada complicado dificil difícil demora demorado lento lenta espera esperar tardanza
falta faltan problema problemas confusion confusión desorden desorganizado desorganizada mal
malo mala pesimo pésimo pesima pésima frustracion frustración frustrante enojo enojada enojado
injusta injusto abandono abandonado indiferencia indiferente ausente ausencia nula nulo
""".split())

FRICTION_THEMES = {
    "📄 Formularios y trámite presencial": [
        "formulario", "formularios", "presencial", "papel", "papeles", "manual", "manuales",
        "llenar", "rellenar", "escanear", "escaneado", "fisico", "físico", "sucursal", "imprimir",
    ],
    "⏳ Tiempos de espera / respuesta": [
        "espera", "esperar", "demora", "demorado", "tardanza", "lento", "lenta", "dias", "días",
        "meses", "tiempo", "rapidez", "rapidas", "rápidas",
    ],
    "❓ Falta de orientación / claridad": [
        "orientacion", "orientación", "instruccion", "instrucción", "claro", "clara", "claridad",
        "entender", "confuso", "confusa", "dudas", "duda", "desconocimiento", "explicacion",
    ],
    "☎️ Atención y soporte": [
        "telefono", "teléfono", "llamar", "llamada", "contestar", "atencion", "atención",
        "encargada", "encargado", "correos", "responder",
    ],
    "🎼 Documentación técnica (partituras/derechos)": [
        "partitura", "partituras", "documentacion", "documentación", "documentos", "derechos",
        "porcentajes", "melodica", "melódica", "fonografico", "fonográfico",
    ],
    "✅ Criterios de aceptación de obras": [
        "aceptar", "aceptado", "aceptaran", "composiciones", "requisitos", "criterios", "categoria", "categoría",
    ],
}

OPPORTUNITY_THEMES = {
    "🌟 Apoyo a artistas emergentes": [
        "emergentes", "emergente", "nuevos", "nuevo", "nueva", "involucrar", "oportunidad", "oportunidades",
    ],
    "📢 Difusión y visibilidad": [
        "difusion", "difusión", "radio", "radios", "redes", "visibilidad", "promocion", "promoción",
        "reconocimiento",
    ],
    "ℹ️ Información y transparencia": [
        "informacion", "información", "claridad", "boletines", "transparencia", "monetizar",
        "monetizacion", "ganancias", "gananciales", "actualizaciones", "estado", "reportes",
        "planilla", "escuchadas",
    ],
    "📍 Presencia en regiones": [
        "region", "región", "regiones", "cercania", "cercanía", "concepcion", "concepción", "provincia",
    ],
    "🎤 Eventos, redes y escenarios": [
        "eventos", "invitaciones", "conciertos", "premiaciones", "presentaciones", "vivo", "salas",
        "festivales", "concursos", "tocar",
    ],
    "🎁 Beneficios y asesoría": [
        "beneficios", "asesoria", "asesoría", "capacitaciones", "gestionar", "gestiones", "dinero",
        "recursos", "proyectos", "economico", "económico",
    ],
}

# ============================================================================
# HELPERS
# ============================================================================
@st.cache_data
def load_data(file) -> pd.DataFrame:
    return pd.read_excel(file)


def protect_commas(s: str) -> str:
    return re.sub(r"\([^)]*\)", lambda m: m.group(0).replace(",", "§"), s)


def split_multiselect(series: pd.Series) -> Counter:
    items = []
    for v in series.dropna():
        v2 = protect_commas(str(v))
        for g in v2.split(","):
            g = g.strip().replace("§", ",")
            if g:
                items.append(g)
    return Counter(items)


def tokenize(text: str, min_len=3):
    text = str(text).lower()
    words = re.findall(r"[a-záéíóúñü]+", text)
    return [w for w in words if len(w) >= min_len]


def clean_words(series: pd.Series, extra_stop=None, min_len=3, max_words_filter=None) -> Counter:
    stop = STOPWORDS_ES | (extra_stop or set())
    s = series.dropna().astype(str)
    if max_words_filter:
        s = s[s.str.split().str.len() <= max_words_filter]
    words = []
    for t in s:
        words.extend(w for w in tokenize(t, min_len) if w not in stop)
    return Counter(words)


def sentiment_score(text: str) -> int:
    """Returns +1 per positive word hit, -1 per negative word hit -> net score."""
    words = tokenize(text, min_len=3)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return pos - neg


def sentiment_label(score: int) -> str:
    if score > 0:
        return "Positivo"
    if score < 0:
        return "Negativo"
    return "Neutral"


def classify_themes(series: pd.Series, theme_dict: dict) -> pd.DataFrame:
    """Returns long dataframe: original_text, theme (a text can match >1 theme)."""
    rows = []
    for txt in series.dropna():
        words = set(tokenize(str(txt), min_len=3))
        matched = False
        for theme, keywords in theme_dict.items():
            if words & set(keywords):
                rows.append({"texto": txt, "tema": theme})
                matched = True
        if not matched:
            rows.append({"texto": txt, "tema": "🔹 Otros / sin clasificar"})
    return pd.DataFrame(rows)


@st.cache_data
def make_wordcloud_png(freqs: dict, colormap: str = None, use_sentiment_colors: bool = False) -> bytes:
    if not freqs:
        return None
    kwargs = dict(width=1600, height=900, background_color="white", max_words=45,
                  prefer_horizontal=0.95, collocations=False, relative_scaling=0.55,
                  min_font_size=12, max_font_size=220, margin=6)
    if use_sentiment_colors:
        def color_func(word, **kw):
            wl = word.lower()
            if wl in POSITIVE_WORDS:
                return GREEN
            if wl in NEGATIVE_WORDS:
                return RED
            return GREY
        kwargs["color_func"] = color_func
    else:
        import matplotlib as mpl
        try:
            cmap = mpl.colormaps[colormap or "cividis"]
        except (AttributeError, KeyError):
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap or "cividis")
        max_freq = max(freqs.values()) if freqs else 1

        def color_func(word, font_size=None, **kw):
            # Map word importance to the darker/more saturated 55%-100% band of the
            # colormap so every word stays legible on a white background.
            freq = freqs.get(word, 0)
            t = 0.55 + 0.45 * (freq / max_freq if max_freq else 0)
            r, g, b, _ = cmap(t)
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        kwargs["color_func"] = color_func
    wc = WordCloud(**kwargs).generate_from_frequencies(freqs)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.imshow(wc, interpolation="lanczos")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def quote_card(text, kind="neutral"):
    cls = {"pos": "pos", "neg": "neg"}.get(kind, "")
    st.markdown(f'<div class="quote-card {cls}">"{text.strip()}"</div>', unsafe_allow_html=True)


def callout(title, text, kind="gold"):
    st.markdown(f"""
    <div class="callout callout-{kind}">
        <h3>{title}</h3>
        <p>{text}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# SIDEBAR — DATA SOURCE & FILTERS
# ============================================================================
st.sidebar.title("🎵 SCD — Encuesta de socios")
uploaded = st.sidebar.file_uploader("Subir archivo de respuestas (.xlsx)", type=["xlsx"])

if uploaded is not None:
    df_raw = load_data(uploaded)
else:
    df_raw = load_data("Encuesta.xlsx")
    st.sidebar.caption("Usando datos de ejemplo incluidos (Encuesta.xlsx). Sube tu propio archivo para reemplazarlos.")

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")
generos_disponibles = sorted(df_raw[COL_GENERO].dropna().unique().tolist())
sel_genero = st.sidebar.multiselect("Género", generos_disponibles, default=generos_disponibles)
df = df_raw[df_raw[COL_GENERO].isin(sel_genero)] if sel_genero else df_raw.copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"Mostrando **{len(df)}** de {len(df_raw)} respuestas totales.")

# ============================================================================
# PRE-COMPUTE SHARED METRICS
# ============================================================================
n = len(df)
nps_series = df[COL_NPS].dropna()
promoters = int((nps_series >= 9).sum())
detractors = int((nps_series <= 6).sum())
passives = len(nps_series) - promoters - detractors
nps_score = round((promoters - detractors) / len(nps_series) * 100, 1) if len(nps_series) else 0
acomp_si_pct = round((df[COL_ACOMPANAMIENTO] == "Sí").mean() * 100, 1) if n else 0
facil_prom = round(df[COL_FACIL_INS].mean(), 2) if n else 0

# sentiment on "en una palabra"
serie_palabra = df[COL_PALABRA].dropna().astype(str)
short_palabra = serie_palabra[serie_palabra.str.split().str.len() <= 3]
sent_scores = short_palabra.apply(sentiment_score)
sent_labels = sent_scores.apply(sentiment_label)
sent_counts = sent_labels.value_counts()
pos_pct = round(sent_counts.get("Positivo", 0) / len(sent_labels) * 100, 1) if len(sent_labels) else 0
neg_pct = round(sent_counts.get("Negativo", 0) / len(sent_labels) * 100, 1) if len(sent_labels) else 0
neu_pct = round(sent_counts.get("Neutral", 0) / len(sent_labels) * 100, 1) if len(sent_labels) else 0

# friction themes
friction_df = classify_themes(df[COL_COMPLEJO], FRICTION_THEMES)
friction_counts = friction_df[friction_df["tema"] != "🔹 Otros / sin clasificar"]["tema"].value_counts()
top_friction = friction_counts.index[0] if len(friction_counts) else "N/A"
top_friction_n = int(friction_counts.iloc[0]) if len(friction_counts) else 0

# opportunity themes
opp_df = classify_themes(df[COL_ESPERARIAS], OPPORTUNITY_THEMES)
opp_counts = opp_df[opp_df["tema"] != "🔹 Otros / sin clasificar"]["tema"].value_counts()
top_opp = opp_counts.index[0] if len(opp_counts) else "N/A"
top_opp_n = int(opp_counts.iloc[0]) if len(opp_counts) else 0

# ============================================================================
# HERO HEADER
# ============================================================================
st.markdown(f"""
<div class="hero-box">
    <h1>🎵 Encuesta de relación y satisfacción de socios/as — SCD</h1>
    <p>{n} respuestas analizadas · NPS {nps_score:+.1f} · Sentimiento positivo {pos_pct}% en la relación con SCD</p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Respuestas", f"{n}")
k2.metric("NPS (recomendación)", f"{nps_score:+.1f}")
k3.metric("Sentimiento positivo", f"{pos_pct}%", delta=f"-{neg_pct}% negativo", delta_color="inverse")
k4.metric("Acompañamiento suficiente", f"{acomp_si_pct}%")
k5.metric("Facilidad inscripción (1-5)", f"{facil_prom}")

c1, c2, c3 = st.columns(3)
with c1:
    callout("🔴 Mayor punto de fricción", f"<b>{top_friction}</b> — mencionado en {top_friction_n} respuestas sobre lo más complejo del proceso.", "red")
with c2:
    callout("🟡 Mayor oportunidad de mejora", f"<b>{top_opp}</b> — la expectativa no cubierta más repetida, con {top_opp_n} menciones.", "gold")
with c3:
    callout("🟢 Mayor fortaleza", f"<b>{promoters} socios ({promoters/len(nps_series)*100:.0f}%)</b> son promotores activos (NPS 9-10) y recomendarían SCD a otro músico.", "green")

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================
tab_sent, tab_fric, tab_opp, tab_perfil, tab_nps, tab_datos, tab_conclusiones = st.tabs(
    ["😊 Sentimiento", "⚠️ Puntos de fricción", "💡 Oportunidades de mejora", "👤 Perfil", "⭐ NPS", "📋 Datos", "🧭 Conclusiones"]
)

# ---------------------------------------------------------------- SENTIMIENTO
with tab_sent:
    st.subheader("¿Cómo se sienten los socios respecto a SCD?")
    st.caption("Basado en la pregunta \"Si tuvieras que describir tu relación con SCD en una palabra, ¿cuál sería?\", clasificada automáticamente en positiva / neutral / negativa según el tono de las palabras usadas.")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        order = ["Positivo", "Neutral", "Negativo"]
        vc = sent_counts.reindex([o for o in order if o in sent_counts.index], fill_value=0)
        colors_map = {"Positivo": GREEN, "Neutral": GREY, "Negativo": RED}
        fig = px.pie(values=vc.values, names=vc.index, hole=0.45,
                     color=vc.index, color_discrete_map=colors_map,
                     title="Distribución del sentimiento")
        fig.update_traces(textinfo="percent+value")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        freqs = dict(clean_words(short_palabra, min_len=3).most_common(60))
        png = make_wordcloud_png(freqs, use_sentiment_colors=True)
        st.markdown("**Nube de palabras coloreada por sentimiento**")
        if png:
            st.image(png, use_container_width=True)
        st.caption("🟢 Verde = tono positivo · 🔴 Rojo = tono negativo · ⚪ Gris = neutral / descriptivo")

    st.markdown("#### Voces de los socios")

    def dedup_preserve_order(texts):
        """Deduplicate quotes that are the same word/phrase (case-insensitive),
        keeping the first occurrence found."""
        seen = set()
        unique = []
        for t in texts:
            key = str(t).strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(t)
        return unique

    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**😊 Respuestas de tono positivo**")
        pos_texts = dedup_preserve_order(short_palabra[sent_scores > 0].tolist())
        for t in pos_texts[:6]:
            quote_card(t, "pos")
    with q2:
        st.markdown("**😟 Respuestas de tono negativo**")
        neg_texts = dedup_preserve_order(short_palabra[sent_scores < 0].tolist())
        for t in neg_texts[:6]:
            quote_card(t, "neg")

    st.markdown("#### Relación entre sentimiento y recomendación (NPS)")
    tmp = df.loc[short_palabra.index].copy()
    tmp["sentimiento"] = sent_labels.values
    tmp["nps_val"] = tmp[COL_NPS]
    cross = tmp.groupby("sentimiento")["nps_val"].mean().reindex(["Positivo", "Neutral", "Negativo"])
    fig = px.bar(x=cross.index, y=cross.values, color=cross.index,
                 color_discrete_map=colors_map,
                 title="Puntaje promedio de recomendación (0-10) según el sentimiento expresado",
                 text=[f"{v:.1f}" for v in cross.values])
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Puntaje NPS promedio", yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Confirma lo esperable: quienes describen la relación en tono negativo también recomiendan menos a SCD, y viceversa.")

# ---------------------------------------------------------------- FRICCIONES
with tab_fric:
    st.subheader("¿Dónde están las principales fricciones del proceso?")
    st.caption("Clasificación automática por temas de la pregunta \"¿Qué fue lo más complejo del proceso?\" (una respuesta puede tocar más de un tema).")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fc = friction_counts.sort_values()
        fig = px.bar(x=fc.values, y=fc.index, orientation="h", color_discrete_sequence=[RED],
                     title="Ranking de puntos de fricción mencionados", text=fc.values)
        fig.update_layout(xaxis_title="N° de menciones", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        freqs = dict(clean_words(df[COL_COMPLEJO], extra_stop={"proceso", "nada", "complejo", "complejidad"}).most_common(60))
        png = make_wordcloud_png(freqs, colormap="OrRd")
        st.markdown("**Nube de palabras — lo más complejo**")
        if png:
            st.image(png, use_container_width=True)

    st.markdown("#### Facilidad percibida de los procesos (cuantitativo)")

    def likert_diverging(col_map, title):
        """Builds a horizontal 100%-stacked diverging bar: difícil (1-2) / neutral (3) / fácil (4-5)."""
        rows = []
        for label, col in col_map.items():
            s = df[col].dropna()
            total = len(s)
            dificil = (s <= 2).sum() / total * 100 if total else 0
            neutral = (s == 3).sum() / total * 100 if total else 0
            facil = (s >= 4).sum() / total * 100 if total else 0
            rows.append({"proceso": label, "Difícil (1-2)": dificil, "Neutral (3)": neutral, "Fácil (4-5)": facil})
        d = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(y=d["proceso"], x=d["Difícil (1-2)"], name="Difícil (1-2)", orientation="h",
                    marker_color=RED, text=[f"{v:.0f}%" for v in d["Difícil (1-2)"]], textposition="inside")
        fig.add_bar(y=d["proceso"], x=d["Neutral (3)"], name="Neutral (3)", orientation="h",
                    marker_color="#D9D2C4", text=[f"{v:.0f}%" for v in d["Neutral (3)"]], textposition="inside")
        fig.add_bar(y=d["proceso"], x=d["Fácil (4-5)"], name="Fácil (4-5)", orientation="h",
                    marker_color=GREEN, text=[f"{v:.0f}%" for v in d["Fácil (4-5)"]], textposition="inside")
        fig.update_layout(barmode="stack", title=title, xaxis_title="% de respuestas", yaxis_title="",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        fig.update_xaxes(ticksuffix="%")
        return fig

    fig = likert_diverging(
        {"Inscripción en SCD (socio)": COL_FACIL_INS, "Inscripción de obras/canciones": COL_FACIL_OBRAS},
        "¿Qué tan fácil fue el trámite? (% de respuestas por categoría)"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada barra suma 100%. Rojo = calificó el trámite como difícil o muy difícil (1-2) · Gris = neutral (3) · Verde = fácil o muy fácil (4-5).")

    cc2 = st.container()
    with cc2:
        order2 = ["Sí", "Me gustaría más", "No"]
        vc = df[COL_ACOMPANAMIENTO].value_counts().reindex([o for o in order2 if o in df[COL_ACOMPANAMIENTO].unique()])
        colors_map2 = {"Sí": GREEN, "Me gustaría más": GOLD, "No": RED}
        fig = px.bar(x=vc.index, y=vc.values, title="¿El acompañamiento te parece suficiente?",
                     color=vc.index, color_discrete_map=colors_map2)
        fig.update_layout(xaxis_title="", yaxis_title="N° de respuestas", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Explora las citas por tema")
    tema_sel = st.selectbox("Elige un punto de fricción para ver ejemplos textuales:", friction_counts.index.tolist())
    ejemplos = friction_df[friction_df["tema"] == tema_sel]["texto"].dropna().unique().tolist()
    for t in ejemplos[:8]:
        quote_card(t, "neg")

# ---------------------------------------------------------------- OPORTUNIDADES
with tab_opp:
    st.subheader("¿Dónde están las oportunidades de mejora?")
    st.caption("Clasificación automática por temas de la pregunta \"¿Qué esperarías del SCD que hoy no estés recibiendo?\"")

    c1, c2 = st.columns([1.3, 1])
    with c1:
        oc = opp_counts.sort_values()
        fig = px.bar(x=oc.values, y=oc.index, orientation="h", color_discrete_sequence=[GOLD],
                     title="Ranking de oportunidades / expectativas no cubiertas", text=oc.values)
        fig.update_layout(xaxis_title="N° de menciones", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        freqs = dict(clean_words(df[COL_ESPERARIAS]).most_common(60))
        png = make_wordcloud_png(freqs, colormap="Blues")
        st.markdown("**Nube de palabras — expectativas**")
        if png:
            st.image(png, use_container_width=True)

    st.markdown("#### Frecuencia de información recibida (contexto)")
    order = ["Nunca", "Rara vez", "Mensual", "Semanal", "Demasiado seguido"]
    vc = df[COL_FRECUENCIA].value_counts().reindex([o for o in order if o in df[COL_FRECUENCIA].unique()])
    fig = px.bar(x=vc.index, y=vc.values, title="Frecuencia percibida de información recibida",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(xaxis_title="", yaxis_title="N° de respuestas")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("La comunicación llega, pero el contenido no siempre responde a lo que los socios más valoran (ver ranking de oportunidades arriba).")

    st.markdown("#### Explora las citas por tema")
    tema_sel2 = st.selectbox("Elige una oportunidad de mejora para ver ejemplos textuales:", opp_counts.index.tolist())
    ejemplos2 = opp_df[opp_df["tema"] == tema_sel2]["texto"].dropna().unique().tolist()
    for t in ejemplos2[:8]:
        quote_card(t, "pos")

# ---------------------------------------------------------------- PERFIL
with tab_perfil:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        vc = df[COL_GENERO].value_counts()
        fig = px.pie(values=vc.values, names=vc.index, color=vc.index,
                     color_discrete_map={"Hombre": NAVY, "Mujer": GOLD},
                     title="Distribución por género", hole=0.35)
        fig.update_traces(textinfo="percent+value")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        vc = df[COL_ENTERASTE].value_counts().sort_values()
        fig = px.bar(x=vc.values, y=vc.index, orientation="h",
                     title="¿Cómo te enteraste del SCD?", color_discrete_sequence=[NAVY])
        fig.update_layout(xaxis_title="N° de respuestas", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    gc = split_multiselect(df[COL_GENERO_MUS])
    top_gen = pd.Series(dict(gc.most_common(12))).sort_values()
    fig = px.bar(x=top_gen.values, y=top_gen.index, orientation="h",
                 title="Géneros musicales más representados (respuesta múltiple)",
                 color_discrete_sequence=[GOLD])
    fig.update_layout(xaxis_title="N° de menciones", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- NPS
with tab_nps:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        vc = df[COL_NPS].value_counts().reindex(range(0, 11), fill_value=0)
        colors = [RED if i <= 6 else (GOLD if i <= 8 else GREEN) for i in range(0, 11)]
        pcts = (vc / vc.sum() * 100) if vc.sum() else vc
        fig = go.Figure(go.Bar(
            x=list(range(0, 11)), y=vc.values, marker_color=colors,
            text=[f"{v}<br>({p:.0f}%)" if v > 0 else "" for v, p in zip(vc.values, pcts)],
            textposition="outside",
        ))
        # Shaded background zones so the three groups are visually obvious
        fig.add_vrect(x0=-0.5, x1=6.5, fillcolor=RED, opacity=0.07, line_width=0,
                      annotation_text="Detractores", annotation_position="top left",
                      annotation_font_color=RED, annotation_font_size=11)
        fig.add_vrect(x0=6.5, x1=8.5, fillcolor=GOLD, opacity=0.09, line_width=0,
                      annotation_text="Pasivos", annotation_position="top",
                      annotation_font_color=GOLD, annotation_font_size=11)
        fig.add_vrect(x0=8.5, x1=10.5, fillcolor=GREEN, opacity=0.08, line_width=0,
                      annotation_text="Promotores", annotation_position="top right",
                      annotation_font_color=GREEN, annotation_font_size=11)
        fig.update_layout(
            title="¿Qué tan probable es que recomiendes SCD a otro músico? (0 a 10)",
            xaxis_title="Puntaje otorgado", yaxis_title="N° de respuestas",
            xaxis=dict(tickmode="linear", dtick=1), yaxis=dict(range=[0, max(vc.values) * 1.28]),
            margin=dict(t=70),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cada barra muestra el número de respuestas y, entre paréntesis, el porcentaje sobre el total. Las franjas de color agrupan los puntajes según la metodología NPS.")
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=nps_score, title={"text": "NPS"},
            gauge={"axis": {"range": [-100, 100]}, "bar": {"color": NAVY},
                   "steps": [{"range": [-100, 0], "color": "#F4CFC4"},
                             {"range": [0, 50], "color": "#FBE6BE"},
                             {"range": [50, 100], "color": "#CFE3D8"}]}))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Promotores (9-10):** {promoters} ({promoters/len(nps_series)*100:.0f}%)")
        st.write(f"**Pasivos (7-8):** {passives} ({passives/len(nps_series)*100:.0f}%)")
        st.write(f"**Detractores (0-6):** {detractors} ({detractors/len(nps_series)*100:.0f}%)")

# ---------------------------------------------------------------- DATOS
with tab_datos:
    st.markdown("#### Respuestas completas por pregunta abierta")
    with st.expander("¿Qué fue lo más complejo del proceso?"):
        st.dataframe(df[[COL_COMPLEJO]].dropna().reset_index(drop=True), use_container_width=True)
    with st.expander("¿Qué esperarías del SCD que hoy no estés recibiendo?"):
        st.dataframe(df[[COL_ESPERARIAS]].dropna().reset_index(drop=True), use_container_width=True)
    with st.expander("Relación con SCD en una palabra"):
        st.dataframe(df[[COL_PALABRA]].dropna().reset_index(drop=True), use_container_width=True)
    with st.expander("Base completa"):
        st.dataframe(df.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------------------------- CONCLUSIONES
with tab_conclusiones:
    st.markdown("---")
    st.header("🧭 Conclusiones: análisis en profundidad")
    st.caption("Lectura integrada de los datos cuantitativos y del contenido textual, en el contexto del rol de SCD para los músicos en Chile.")

    facil_ins_pct = round((df[COL_FACIL_INS] >= 4).mean() * 100, 1) if n else 0
    facil_obras_pct = round((df[COL_FACIL_OBRAS] >= 4).mean() * 100, 1) if n else 0
    acomp_masq_pct = round((df[COL_ACOMPANAMIENTO] == "Me gustaría más").mean() * 100, 1) if n else 0
    top3_friction = friction_counts.head(3)
    top3_opp = opp_counts.head(3)

    st.markdown(f"""
    #### 1. El diagnóstico de fondo: una institución legitimada, pero poco cercana

    Los números cuentan una historia consistente y, a la vez, un poco incómoda. El NPS de **{nps_score:+.1f}**
    y un **{pos_pct}%** de tono positivo en la palabra que la gente usa para describir su relación con SCD
    muestran que, en lo sustantivo, la institución goza de aprobación: la mayoría de los socios reconoce su
    valor y la recomendaría. Pero esa aprobación convive con una fricción persistente — solo **{acomp_si_pct}%**
    siente que el acompañamiento es suficiente, mientras un **{acomp_masq_pct}%** dice explícitamente
    *"me gustaría más"*. Esta no es una contradicción: es el patrón típico de una organización que ha resuelto
    bien su función legal y económica central (recaudar y distribuir derechos de autor) pero que todavía no ha
    resuelto, al mismo nivel, la experiencia relacional de sus socios.

    #### 2. La paradoja "necesaria pero distante"

    La nube de palabras de la Sección de Sentimiento expone esta tensión de forma casi literal: las palabras más
    repetidas — *"buena"*, *"necesaria"*, *"excelente"* — conviven con *"distante"* y *"lejana"* entre las más
    mencionadas. Esto no es ruido estadístico: es una misma persona pudiendo sostener ambas ideas a la vez.
    SCD es percibida como *necesaria* — no es opcional para un músico que quiere vivir de su obra en Chile,
    dado que es la entidad que canaliza la recaudación de derechos — pero esa necesidad institucional no se
    traduce automáticamente en una sensación de cercanía humana. Es la diferencia entre confiar en una
    institución y sentirse acompañado por ella. La segunda dimensión es la que hoy está más débil.

    #### 3. Qué revela la fricción operativa sobre el diseño del proceso

    Los tres focos de fricción más mencionados en "¿Qué fue lo más complejo del proceso?" son:
    """)

    for i, (tema, count) in enumerate(top3_friction.items(), 1):
        st.markdown(f"- **{i}. {tema}** — {count} menciones")

    st.markdown(f"""
    Leídos en conjunto, no describen un solo problema puntual, sino un **patrón de diseño**: un proceso de
    inscripción que todavía depende de trámites físicos, formularios manuales y ciclos de corrección de
    documentos, con poca instrumentación digital de punta a punta. Esto tiene un costo doble. Primero, un costo
    directo de tiempo y esfuerzo, más alto para quienes no viven cerca de una sucursal o no manejan bien lo
    administrativo. Segundo, un costo simbólico: cada corrección de formulario o cada llamada sin respuesta es,
    para el socio, una señal (probablemente no intencionada) de que la institución no está diseñada pensando en
    su experiencia, sino en su propio proceso interno. Es coherente que solo **{facil_ins_pct}%** califique
    como fácil o muy fácil la inscripción como socio y **{facil_obras_pct}%** la inscripción de obras: son
    mayorías, pero no son los niveles que uno esperaría de un trámite bien resuelto digitalmente.

    #### 4. La brecha de acompañamiento es un síntoma, no la causa

    Es tentador leer el bajo puntaje de acompañamiento como un problema de "falta de cercanía humana" que se
    resolvería con más gente atendiendo teléfonos. Los datos sugieren algo más específico: **la frecuencia de
    comunicación no es el problema** — la gran mayoría recibe información semanal o mensualmente — sino su
    **relevancia percibida**. Cuando se pregunta qué esperarían y no reciben hoy, las respuestas no piden más
    comunicación en general, piden comunicación *útil*: información sobre cómo postular a festivales o
    concursos, claridad sobre monetización y regalías, orientación paso a paso cuando algo no calza en un
    formulario. El acompañamiento débil, entonces, no es un problema de volumen de contacto, sino de que ese
    contacto no está resolviendo las dudas concretas que un músico necesita resolver para avanzar en su carrera.

    #### 5. La dimensión de equidad: centro y periferia, consolidado y emergente

    Un hallazgo que merece atención especial, porque no es solo operativo sino también de **misión institucional**,
    es la demanda repetida de más apoyo para artistas emergentes y para músicos de regiones. Las oportunidades
    de mejora más mencionadas son:
    """)

    for i, (tema, count) in enumerate(top3_opp.items(), 1):
        st.markdown(f"- **{i}. {tema}** — {count} menciones")

    st.markdown(f"""
    Esto importa porque SCD, como entidad de gestión colectiva, cumple una función que trasciende lo administrativo:
    es, para muchos músicos independientes en Chile, la única vía formal de acceder a ingresos por derechos de
    autor. Cuando quienes recién empiezan — o quienes están fuera de Santiago — sienten que la institución les
    llega con menos fuerza, la conversación deja de ser solo sobre "buena atención al cliente" y pasa a ser sobre
    **qué tan equitativo es el acceso real** a la infraestructura de derechos de autor en Chile. Las respuestas
    textuales lo dicen sin rodeos: se percibe que "llegar a la radio o a un festival es casi un sueño con pocas
    probabilidades" para un músico emergente, y que la información sobre cómo participar de esos circuitos no
    siempre llega a quien más la necesita.

    #### 6. Qué dice esto sobre el rol de SCD en el ecosistema musical chileno

    SCD ocupa un lugar particular: no es una empresa que compite por retener clientes en un mercado abierto, es
    una entidad de gestión colectiva cuya relación con sus socios es, en la práctica, de largo plazo y con pocas
    alternativas reales. Esa posición trae una responsabilidad distinta a la de cualquier otro servicio: cuando
    no hay competencia que discipline la experiencia de socio, la calidad del acompañamiento depende casi
    enteramente de una decisión interna de priorizarlo. Los resultados de esta encuesta muestran que la base de
    socios de SCD, en general, valora lo que la institución representa — protección legal, ingresos, pertenencia
    a un gremio — pero le está pidiendo, con bastante claridad, que la experiencia de ser socio se sienta tan
    profesional y cuidada como la función que cumple en el papel. Eso implica invertir en dos frentes muy
    concretos y medibles con esta misma encuesta a futuro: **digitalizar el trámite de principio a fin** (para
    bajar la fricción de "{top_friction.split(' ', 1)[-1]}")
    y **rediseñar el acompañamiento como una función de orientación de carrera**, no solo de soporte
    administrativo — especialmente para los socios más nuevos, más jóvenes o más alejados de Santiago.

    #### 7. Síntesis

    En una frase: **SCD no tiene un problema de confianza, tiene un problema de cercanía.** La institución es
    vista como legítima y valiosa por la gran mayoría de sus socios (NPS {nps_score:+.1f}), pero el vínculo
    cotidiano — el trámite, el acompañamiento, la información oportuna — todavía se siente, para una parte
    importante de la base de socios, como algo que se recibe de forma distante más que como una relación activa
    y cercana. Cerrar esa brecha no requiere reconstruir la relación desde cero: requiere traducir la legitimidad
    institucional que SCD ya tiene en una experiencia de socio que se sienta igual de sólida en el día a día.
    """)

st.markdown("---")
st.caption("Panel generado a partir de las respuestas del formulario SCD. El sentimiento y los temas de fricción/oportunidad se clasifican con reglas heurísticas de palabras clave — útil para priorizar, no reemplaza la lectura cualitativa completa.")
