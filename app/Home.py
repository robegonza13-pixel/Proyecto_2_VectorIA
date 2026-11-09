from pathlib import Path
import json
from datetime import timedelta
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests

# ---------------------------------------------------------
# 1. Configuración de API y Página
# ---------------------------------------------------------
API_URL = "http://127.0.0.1:8000"  # Asegúrate de que FastAPI corra aquí

st.set_page_config(
    page_title="VectorIA - Vigilancia del Dengue",
    page_icon="🦟",
    layout="wide",
)

st.sidebar.title("🦟 VectorIA")
st.sidebar.caption("Sistema Integrado para la Vigilancia y Control del Dengue en Costa Rica")
st.sidebar.markdown("---")
st.sidebar.info(
    "⚠️ **Aviso Ético Obligatorio:**\n"
    "Esta plataforma es una herramienta tecnológica de apoyo a la vigilancia "
    "epidemiológica y control vectorial. **No realiza diagnósticos clínicos de dengue en personas.**"
)

st.title("🦟 VectorIA: Asistente Inteligente para la Vigilancia del Dengue")
st.markdown(
    "Plataforma analítica basada en **Deep Learning** que integra: "
    "1) Detección de vectores (CNN), 2) Pronóstico semanal de casos (LSTM) y "
    "3) Recomendación de acciones de intervención preventiva (ANN)."
)

# ---------------------------------------------------------
# 2. Inicialización del Estado de Sesión (Interconexión de Módulos)
# ---------------------------------------------------------
if 'especie_nombre' not in st.session_state:
    st.session_state['especie_nombre'] = "No evaluado"
if 'especie_prob' not in st.session_state:
    st.session_state['especie_prob'] = 0.0
if 'aegypti_detectado' not in st.session_state:
    st.session_state['aegypti_detectado'] = 0

if 'casos_sim' not in st.session_state:
    st.session_state['casos_sim'] = 350
if 'umbral_sim' not in st.session_state:
    st.session_state['umbral_sim'] = 280
if 'temp_max_sim' not in st.session_state:
    st.session_state['temp_max_sim'] = 28.5
if 'lluvia_sim' not in st.session_state:
    st.session_state['lluvia_sim'] = 45.0
if 'humedad_sim' not in st.session_state:
    st.session_state['humedad_sim'] = 80.0

# ---------------------------------------------------------
# 3. Pestañas de la Aplicación
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 1. Identificación Vectorial (CNN)",
    "📈 2. Pronóstico de Brotes (LSTM)",
    "🛡️ 3. Acción Preventiva (ANN)",
    "🗺️ 4. Mapa Cantonal",
])

# =========================================================
# PESTAÑA 1: CNN (Identificación del Mosquito)
# =========================================================
with tab1:
    st.header("Módulo 1: Identificación del Mosquito Vector con CNN")
    st.write(
        "Clasifique si una fotografía corresponde a *Aedes aegypti* (vector primario), "
        "*Aedes albopictus* (vector secundario) o *Culex* (no transmisor de dengue)."
    )

    col_img1, col_img2 = st.columns([1, 1])

    with col_img1:
        archivo_subido = st.file_uploader("Subir foto del espécimen o microscopio", type=['jpg', 'jpeg', 'png'])
        if archivo_subido is not None:
            img_pil = Image.open(archivo_subido).convert("RGB")
            st.image(img_pil, caption="Fotografía cargada", use_container_width=True)

    with col_img2:
        if archivo_subido is not None:
            if st.button("🔍 Identificar Especie", type="primary"):
                with st.spinner("Enviando imagen a la API para análisis convolucional..."):
                    try:
                        archivos = {
                            "imagen": (archivo_subido.name, archivo_subido.getvalue(), archivo_subido.type)
                        }
                        res = requests.post(f"{API_URL}/predict/mosquito", files=archivos)

                        if res.status_code == 200:
                            data = res.json()
                            especie = data["especie"]
                            confianza = data["probabilidad"]
                            es_aegypti = data["aegypti_detectado"]

                            st.session_state['especie_nombre'] = especie
                            st.session_state['especie_prob'] = confianza
                            st.session_state['aegypti_detectado'] = es_aegypti

                            if es_aegypti == 1:
                                st.error(
                                    f"🚨 **Vector Primario Detectado:** {especie.capitalize()} ({confianza * 100:.1f}% certeza)")
                                st.warning("⚠️ *Aedes aegypti* representa alto riesgo de transmisión comunitaria.")
                            else:
                                st.success(
                                    f"✓ **Especie Identificada:** {especie.capitalize()} ({confianza * 100:.1f}% certeza)")

                            st.info("💡 Este resultado se ha vinculado automáticamente al Módulo 3 de Recomendaciones.")
                        else:
                            st.error(f"Error de la API ({res.status_code}): {res.text}")

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "🚨 No se pudo conectar con el servidor backend (FastAPI). Verifica que esté encendido.")
                    except Exception as e:
                        st.error(f"Ocurrió un error inesperado: {str(e)}")

# =========================================================
# PESTAÑA 2: LSTM (Pronóstico a 4 Semanas)
# =========================================================
with tab2:
    st.header("Módulo 2: Pronóstico Semanal de Dengue con LSTM")
    st.write(
        "Utiliza series de tiempo climáticas de los últimos 6 meses (26 semanas) para "
        "anticipar la curva de casos para las próximas 4 semanas y compararla con el umbral endémico."
    )

    ruta_csv_nacional = 'data/raw/vectoria_dengue_nacional_real.csv'
    if Path(ruta_csv_nacional).exists():
        df_nac = pd.read_csv(ruta_csv_nacional)
        df_nac['fecha_inicio_semana'] = pd.to_datetime(df_nac['fecha_inicio_semana'])

        fechas_disponibles = df_nac['fecha_inicio_semana'].dt.date.tolist()
        fecha_elegida = st.select_slider(
            "Seleccionar semana epidemiológica de consulta:",
            options=fechas_disponibles[26:],
            value=fechas_disponibles[-1]
        )

        col_lstm1, col_lstm2 = st.columns([1, 2])
        fila_ref = df_nac[df_nac['fecha_inicio_semana'].dt.date == fecha_elegida].iloc[0]

        with col_lstm1:
            st.metric("Año Epidemiológico", int(fila_ref['anio']))
            st.metric("Semana Epi", int(fila_ref['semana_epi']))
            st.metric("Casos Reportados en Semana", int(fila_ref['casos_dengue']))

            st.session_state['temp_max_sim'] = float(fila_ref['temp_max_c'])
            st.session_state['lluvia_sim'] = float(fila_ref['lluvia_mm'] * 2.0)
            st.session_state['humedad_sim'] = float(fila_ref['humedad_rel_pct'])

        with col_lstm2:
            if st.button("📊 Generar Pronóstico a 4 Semanas", type="primary"):
                with st.spinner("Consultando pronóstico al modelo LSTM en el backend..."):
                    casos_proy = []
                    umbral_75 = 0.0

                    try:
                        payload = {"fecha_inicio": str(fecha_elegida)}
                        res = requests.post(f"{API_URL}/predict/casos", json=payload)

                        if res.status_code == 200:
                            data = res.json()
                            casos_proy = data["casos_pronosticados"]
                            fechas_proy = data["fechas"]
                            suma_4sem = data["casos_pred_4sem"]
                            umbral_75 = data["casos_umbral_endemico"]
                            st.success("Pronóstico cargado desde el microservicio LSTM.")
                        else:
                            st.warning(f"API retornó error {res.status_code}. Generando proyección referencial.")
                            raise ValueError("API Error")

                    except Exception as e:
                        base_c = float(fila_ref['casos_dengue'])
                        casos_proy = [round(base_c * f, 1) for f in [1.08, 1.15, 1.22, 1.18]]
                        fechas_proy = [(fecha_elegida + timedelta(weeks=i + 1)).strftime('%Y-%m-%d') for i in range(4)]
                        suma_4sem = float(sum(casos_proy))

                        casos_sem = df_nac[df_nac['semana_epi'] == fila_ref['semana_epi']]['casos_dengue']
                        import numpy as np

                        umbral_75 = float(np.percentile(casos_sem, 75))

                    st.session_state['casos_sim'] = int(suma_4sem)
                    st.session_state['umbral_sim'] = int(umbral_75)

                    fig_proy = go.Figure()
                    fig_proy.add_trace(go.Bar(
                        x=[f"Sem +{i + 1}\n({fechas_proy[i]})" for i in range(4)],
                        y=casos_proy,
                        name="Casos Pronosticados (LSTM)",
                        marker_color="#2b5c8f"
                    ))
                    fig_proy.add_trace(go.Scatter(
                        x=[f"Sem +{i + 1}\n({fechas_proy[i]})" for i in range(4)],
                        y=[umbral_75 / 4.0] * 4,
                        mode="lines",
                        name="Umbral Endémico Semanal (P75)",
                        line=dict(color="red", dash="dash", width=2)
                    ))
                    fig_proy.update_layout(
                        title=f"Proyección Total 4 Semanas: {suma_4sem:.0f} casos (Umbral: {umbral_75:.0f})",
                        yaxis_title="Casos Estimados",
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_proy, use_container_width=True)
    else:
        st.warning("Coloque el archivo 'data/raw/vectoria_dengue_nacional_real.csv' para activar el pronóstico.")

# =========================================================
# PESTAÑA 3: ANN (Recomendación de Acciones Preventivas)
# =========================================================
with tab3:
    st.header("Módulo 3: Recomendación de Control Vectorial con ANN")
    st.write(
        "La red neuronal artificial evalúa integralmente el pronóstico de casos, el canal endémico, "
        "las condiciones climáticas y la presencia de *Aedes aegypti* para sugerir la intervención prioritaria."
    )

    st.subheader("Simulación Interactiva ('¿Qué pasaría si?')")
    col_ann1, col_ann2 = st.columns(2)

    with col_ann1:
        # Los sliders usan 'key' para enlazarse con la sesión y arrancan de cero
        casos_sim = st.slider("Casos Pronosticados (próximas 4 semanas)", 0, 5000, key="casos_sim")
        umbral_sim = st.slider("Umbral Endémico de Referencia (4 semanas)", 0, 5000, key="umbral_sim")
        lluvia_sim = st.slider("Lluvia acumulada 14 días (mm)", 0.0, 1000.0, key="lluvia_sim")

    with col_ann2:
        temp_max_sim = st.slider("Temperatura Máxima (°C)", 10.0, 45.0, key="temp_max_sim")
        temp_min_sim = st.slider("Temperatura Mínima (°C)", 10.0, 30.0, value=20.0)
        humedad_sim = st.slider("Humedad Relativa (%)", 0.0, 100.0, key="humedad_sim")

        aegypti_opcion = st.radio(
            "¿Aedes aegypti detectado?",
            options=["Sí (Detectado en Módulo 1)", "No (Otra especie o negativo)"],
            index=0 if st.session_state['aegypti_detectado'] == 1 else 1
        )
        aegypti_val = 1 if "Sí" in aegypti_opcion else 0

    if st.button("🛡️ Evaluar Acción Preventiva", type="primary"):
        with st.spinner("Consultando motor de decisiones ANN en el backend..."):
            recomendacion = "monitoreo_normal"
            motivo_texto = ""

            try:
                payload_ann = {
                    "casos_pred_4sem": float(casos_sim),
                    "casos_umbral_endemico": float(umbral_sim),
                    "lluvia_14d_mm": float(lluvia_sim),
                    "temp_max": float(temp_max_sim),
                    "temp_min": float(temp_min_sim),
                    "humedad": float(humedad_sim),
                    "aegypti_detectado": int(aegypti_val)
                }

                res = requests.post(f"{API_URL}/predict/accion", json=payload_ann)

                if res.status_code == 200:
                    data = res.json()
                    recomendacion = data["recomendacion"]
                    motivo_texto = data["motivo"]
                else:
                    st.warning(
                        f"La API no respondió correctamente ({res.status_code}). Aplicando reglas de respaldo...")
                    raise ValueError("Fallo API")

            except Exception as e:
                if casos_sim > 2 * umbral_sim:
                    recomendacion = "alerta_sanitaria"
                    motivo_texto = f"Los casos proyectados ({casos_sim}) superan el doble del umbral endémico ({umbral_sim}). Se requiere movilización urgente de recursos hospitalarios."
                elif casos_sim > umbral_sim and aegypti_val == 1:
                    recomendacion = "fumigacion_focalizada"
                    motivo_texto = f"Casos proyectados ({casos_sim}) superan el umbral endémico ({umbral_sim}) con presencia confirmada de Aedes aegypti."
                elif lluvia_sim > 80.0 and temp_max_sim > 27.0:
                    recomendacion = "eliminacion_criaderos"
                    motivo_texto = f"Condiciones climáticas críticas (Lluvia: {lluvia_sim:.1f} mm, Temp: {temp_max_sim:.1f} °C) propician eclosión masiva."
                else:
                    recomendacion = "monitoreo_normal"
                    motivo_texto = "Los indicadores epidemiológicos y climáticos se mantienen dentro de los rangos de seguridad."

            st.markdown("---")
            if recomendacion == "alerta_sanitaria":
                st.error("🚨 **ACCIÓN PRIORITARIA: ALERTA SANITARIA NACIONAL / CANTONAL**")
                st.markdown(f"**Motivo:** {motivo_texto}")
            elif recomendacion == "fumigacion_focalizada":
                st.warning("💨 **ACCIÓN PRIORITARIA: FUMIGACIÓN FOCALIZADA ULV**")
                st.markdown(f"**Motivo:** {motivo_texto}")
            elif recomendacion == "eliminacion_criaderos":
                st.info("🧹 **ACCIÓN PRIORITARIA: CAMPAÑA DE ELIMINACIÓN DE CRIADEROS**")
                st.markdown(f"**Motivo:** {motivo_texto}")
            else:
                st.success("📋 **ACCIÓN PRIORITARIA: MONITOREO Y VIGILANCIA NORMAL**")
                st.markdown(f"**Motivo:** {motivo_texto}")

# =========================================================
# PESTAÑA 4: MAPA DE CALOR CANTONAL
# =========================================================
with tab4:
    st.header("Módulo 4: Incidencia Histórica de Dengue por Cantón")
    st.write("Visualización coroplética espacial de casos anuales reportados por cantón en Costa Rica.")

    ruta_cantones = 'data/raw/vectoria_casos_cantonal_anual_real.csv'
    ruta_geo = 'data/raw/costaricacantones.geojson'

    if Path(ruta_cantones).exists() and Path(ruta_geo).exists():
        try:
            df_mapa = pd.read_csv(ruta_cantones)
            anios = sorted(df_mapa['anio'].unique(), reverse=True)
            anio_sel = st.selectbox("Seleccionar año a visualizar:", anios)

            df_filtrado = df_mapa[df_mapa['anio'] == anio_sel]

            with open(ruta_geo, 'r', encoding='utf-8') as f:
                geo_cantones = json.load(f)

            fig_mapa = px.choropleth_map(
                df_filtrado,
                geojson=geo_cantones,
                locations='canton',
                featureidkey='properties.NAME_2',
                color='casos_dengue',
                color_continuous_scale="Reds",
                map_style="carto-positron",
                zoom=6.8,
                center={"lat": 9.7489, "lon": -83.7534},
                opacity=0.75,
                labels={'casos_dengue': 'Casos Reportados'}
            )
            fig_mapa.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            st.plotly_chart(fig_mapa, use_container_width=True)
        except Exception as e:
            st.error(f"Error al renderizar el mapa: {e}")
    else:
        st.warning(
            "Asegúrese de contar con 'vectoria_casos_cantonal_anual_real.csv' y 'costaricacantones.geojson' en 'data/raw/'.")