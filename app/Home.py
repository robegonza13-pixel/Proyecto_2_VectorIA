import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import json
# Importar aquí joblib, tensorflow y otras librerías cuando integren los modelos

# 1. Configuración de la página y Aviso Ético Obligatorio
st.set_page_config(page_title="VectorIA", layout="wide")

st.sidebar.title("VectorIA")
st.sidebar.info(
    "⚠️ **Aviso ético:** Esta aplicación es una herramienta de apoyo a la "
    "vigilancia epidemiológica. No diagnostica dengue en personas."
)

st.title("Asistente Inteligente para la Vigilancia del Dengue")

# 2. Inicializar variables de sesión para compartir datos entre pestañas
if 'especie_detectada' not in st.session_state:
    st.session_state['especie_detectada'] = 0 # 1 si es aegypti, 0 si no
if 'casos_predichos' not in st.session_state:
    st.session_state['casos_predichos'] = 0.0

# 3. Creación de las pestañas
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Detección (CNN)", 
    "2. Predicción (LSTM)", 
    "3. Recomendación (ANN)", 
    "4. Mapa"
])

# --- PESTAÑA 1: MÓDULO 1 (CNN) ---
with tab1:
    st.header("Identificación del Mosquito Vector")
    st.write("Sube una imagen para que el modelo CNN clasifique la especie.")
    
    archivo_subido = st.file_uploader("Cargar foto del mosquito", type=['jpg', 'jpeg', 'png'])
    
    if archivo_subido is not None:
        st.image(archivo_subido, caption="Imagen cargada", width=300)
        if st.button("Analizar Imagen"):
            # AQUÍ VA EL CÓDIGO DEL COMPAÑERO DE LA CNN
            # Simulamos el resultado:
            st.success("Especie detectada: Aedes aegypti (92% de confianza)")
            # Guardamos el resultado en la sesión para que el Módulo 3 lo pueda usar
            st.session_state['especie_detectada'] = 1 

# --- PESTAÑA 2: MÓDULO 2 (LSTM) ---
with tab2:
    st.header("Pronóstico de Casos (4 Semanas)")
    st.write("Selecciona una semana epidemiológica para predecir los casos.")
    
    semana_seleccionada = st.date_input("Seleccionar fecha")
    
    if st.button("Generar Predicción"):
        # AQUÍ VA EL CÓDIGO DEL COMPAÑERO DE LA LSTM
        # Simulamos el resultado:
        st.info("Generando gráfico de predicción vs umbral endémico...")
        st.session_state['casos_predichos'] = 350.0 # Se guarda para el Módulo 3

# --- PESTAÑA 3: MÓDULO 3 (ANN) ---
with tab3:
    st.header("Recomendación de Acción Preventiva")
    st.write("El modelo evalúa la predicción de casos, el clima y el vector detectado.")
    
    # Sliders de "¿Qué pasaría si?" para la demostración interactiva
    st.subheader("Simulación Interactiva")
    col1, col2 = st.columns(2)
    
    with col1:
        # Toma el valor de la LSTM por defecto, pero permite alterarlo
        casos_simulados = st.slider("Casos predichos", 0, 2000, int(st.session_state['casos_predichos']))
        lluvia_simulada = st.slider("Lluvia acumulada 14 días (mm)", 0.0, 300.0, 15.0)
    with col2:
        temp_max_simulada = st.slider("Temperatura Máxima (°C)", 20.0, 40.0, 28.0)
        # Toma la detección de la CNN automáticamente
        aegypti_estado = "Sí" if st.session_state['especie_detectada'] == 1 else "No"
        st.text_input("Aedes aegypti detectado en la Pestaña 1", value=aegypti_estado, disabled=True)
    
    if st.button("Obtener Recomendación"):
        # AQUÍ INTEGRAS EL CÓDIGO DEL MÓDULO 3 (Escalado y model.predict)
        # Utilizando los valores de los sliders como entrada
        st.success("Acción preventiva sugerida: Fumigación focalizada")

# --- PESTAÑA 4: MAPA DE CALOR ---
with tab4:
    st.header("Mapa de Casos de Dengue por Cantón")
    st.write("Visualización coroplética de la incidencia histórica del dengue.")
    
    try:
        # 1. Cargar el dataset con la ruta exacta proporcionada
        df_mapa = pd.read_csv('data/raw/vectoria_casos_cantonal_anual_real.csv')
        
        # Filtro interactivo por año en la interfaz
        anio_seleccionado = st.selectbox("Seleccione el año a visualizar", sorted(df_mapa['anio'].unique(), reverse=True))
        df_filtrado = df_mapa[df_mapa['anio'] == anio_seleccionado]
        
        # 2. Cargar el archivo GeoJSON con los límites de los cantones
        # (El equipo debe descargar y colocar este archivo en la misma ruta)
        with open('data/raw/costaricacantones.geojson', 'r', encoding='utf-8') as f:
            geo_cantones = json.load(f)
            
        # 3. Generar el mapa coroplético
        fig = px.choropleth_map(
            df_filtrado,
            geojson=geo_cantones,
            locations='canton',               
            featureidkey='properties.NAME_2', 
            color='casos_dengue',             
            color_continuous_scale="Reds",    
            map_style="carto-positron",   
            zoom=6.5,
            center={"lat": 9.7489, "lon": -83.7534},
            opacity=0.7,
            labels={'casos_dengue': 'Casos Reportados'}
        )
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError as e:
        st.error(f"Error al cargar los archivos: {e}")