import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Configuración básica de la página
st.title("Módulo 3: Recomendación de Acción Preventiva (ANN)")

# 1. Cargar el modelo y el escalador en memoria caché para no recargarlos en cada interacción
@st.cache_resource
def cargar_recursos():
    # Se asume que los archivos están en la carpeta /models
    modelo = tf.keras.models.load_model('models/ann.h5')
    escalador = joblib.load('models/scaler_ann.pkl')
    return modelo, escalador

modelo_ann, scaler = cargar_recursos()

# Nombres de las clases en el orden exacto que generó get_dummies durante el entrenamiento
clases = ['alerta_sanitaria', 'eliminacion_criaderos', 'fumigacion_focalizada', 'monitoreo_normal']

# 2. Interfaz para el ingreso manual de variables (simulación para el Módulo 3)
st.header("Ingreso de Variables Epidemiológicas y Climáticas")

col1, col2 = st.columns(2)

with col1:
    casos_pred_4sem = st.number_input("Casos predichos (Próximas 4 semanas)", min_value=0.0, value=300.0)
    casos_umbral = st.number_input("Casos umbral endémico", min_value=0.0, value=280.0)
    lluvia = st.number_input("Lluvia acumulada 14 días (mm)", min_value=0.0, value=15.0)
    humedad = st.number_input("Humedad (%)", min_value=0.0, max_value=100.0, value=82.0)

with col2:
    temp_max = st.number_input("Temperatura Máxima (°C)", min_value=0.0, value=28.0)
    temp_min = st.number_input("Temperatura Mínima (°C)", min_value=0.0, value=20.0)
    # Variable binaria para la detección de la CNN
    aegypti = st.selectbox("¿Aedes aegypti detectado?", options=[1, 0], format_func=lambda x: "Sí" if x == 1 else "No")

# 3. Lógica de predicción
if st.button("Generar Recomendación"):
    # Estructurar los datos de entrada en una matriz 2D con el orden exacto de X_train
    datos_entrada = np.array([[
        casos_pred_4sem, 
        casos_umbral, 
        lluvia, 
        temp_max, 
        temp_min, 
        humedad, 
        aegypti
    ]])
    
    # Aplicar el MinMaxScaler a los datos crudos
    datos_escalados = scaler.transform(datos_entrada)
    
    # Realizar la predicción con la ANN
    prediccion_probabilidades = modelo_ann.predict(datos_escalados)
    
    # Obtener el índice de la probabilidad más alta
    clase_predicha_idx = np.argmax(prediccion_probabilidades, axis=-1)[0]
    recomendacion_final = clases[clase_predicha_idx]
    
    # Formatear el texto de salida
    texto_recomendacion = recomendacion_final.replace('_', ' ').title()
    probabilidad = prediccion_probabilidades[0][clase_predicha_idx] * 100
    
    # Mostrar resultados en la interfaz
    st.subheader("Resultado de la Evaluación")
    st.success(f"Acción preventiva sugerida: **{texto_recomendacion}**")
    st.info(f"Nivel de confianza de la red neuronal: {probabilidad:.2f}%")
