# 🦟 VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

> **Aviso Ético Obligatorio:**  
> Esta plataforma es una herramienta tecnológica desarrollada como soporte para la vigilancia epidemiológica y el control vectorial por parte de autoridades y comunidades. **No constituye un sistema de diagnóstico clínico de dengue en seres humanos.**

---

## 📋 Descripción del Proyecto

**VectorIA** es un sistema integral de Inteligencia Artificial que une tres modelos de Deep Learning para la prevención y mitigación de brotes de dengue en Costa Rica:

1. **Módulo 1 · Identificación Vectorial (CNN):** Clasifica imágenes de mosquitos en tres especies clave: *Aedes aegypti* (vector primario), *Aedes albopictus* (vector secundario) y *Culex* (no transmisor). Se comparan dos arquitecturas: una CNN convolucional propia entrenada desde cero y MobileNetV2 con Transfer Learning y Fine-Tuning ($F1\text{-Score Macro} \ge 85\%$).
2. **Módulo 2 · Pronóstico de Brotes (RNN / LSTM):** Proyecta la curva de casos semanales para las próximas 4 semanas a partir de una ventana histórica de 26 semanas de clima (temperaturas máx/mín, lluvia acumulada, humedad y componentes estacionales cíclicos). Se evalúa SimpleRNN frente a LSTM multicapa, contrastados contra el *baseline* estacional (mismo período del año anterior).
3. **Módulo 3 · Toma de Decisiones y Acción Preventiva (ANN):** Red neuronal densa que integra las salidas de los modelos anteriores (casos predichos, umbral del canal endémico, clima y presencia de *Aedes aegypti*) para clasificar la intervención prioritaria:
   - 🚨 `alerta_sanitaria`: Cuando la proyección supera el doble del canal endémico.
   - 💨 `fumigacion_focalizada`: Proyección sobre el umbral con presencia confirmada de *Aedes aegypti*.
   - 🧹 `eliminacion_criaderos`: Condiciones de lluvia y temperatura críticas para reproducción larvaria.
   - 📋 `monitoreo_normal`: Indicadores en rangos estacionales seguros.
4. **Módulo 4 · Mapa Coroplético Espacial:** Dashboard interactivo en Streamlit con visualización de casos históricos anuales por cantón sobre mapas GeoJSON oficiales.

---

## 🏗️ Estructura del Repositorio

```text
VectorIA/
├── README.md                      # Documentación general y guía de instalación
├── requirements.txt               # Dependencias del entorno
├── .gitignore                     # Filtros para exclusión de archivos pesados
├── data/
│   ├── raw/                       # Fuentes primarias (CSVs oficiales y GeoJSON)
│   └── processed/                 # Partición estratificada (train/val/test)
├── notebooks/
│   ├── 01_EDA.ipynb               # Análisis Exploratorio de Datos (Imágenes y Series)
│   ├── 02_CNN_Mosquitos.ipynb     # Entrenamiento CNN propia y MobileNetV2
│   ├── 03_LSTM_Casos.ipynb        # Modelado temporal (SimpleRNN vs LSTM)
│   └── 04_ANN_Acciones.ipynb      # Clasificador multiclase con GroupShuffleSplit
├── src/
│   ├── __init__.py
│   ├── data_prep.py               # Clases POO para auditoría, limpieza y split
│   └── train/
│       ├── __init__.py
│       ├── cnn.py                 # Clase EntrenarCNN y DiagnosticoEvaluacion
│       ├── cnn_mobilenetv2.py     # Clase EntrenarMobileNetV2 (Transfer Learning)
│       ├── lstm.py                # Clase LstmModel (Series temporales)
│       └── ann.py                 # Clase EntrenarANN (Red densa de recomendación)
├── models/                        # Artefactos y checkpoints entrenados (.keras / .h5)
├── api/
│   ├── main.py                    # Servidor FastAPI y endpoints REST
│   ├── schemas.py                 # Modelos Pydantic para validación de contratos
│   └── predict.py                 # Clases de inferencia para consumo desacoplado
└── app/
    ├── Home.py                    # Aplicación web interactiva en Streamlit
    └── assets/                    # Recursos visuales y diagramas
```

---

## 🚀 Instalación y Despliegue

### 1. Clonar el repositorio y configurar el entorno virtual:
```bash
git clone https://github.com/tu-usuario/Proyecto_2_VectorIA.git
cd Proyecto_2_VectorIA

# Crear y activar entorno virtual (Windows)
python -m venv .venv
.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar la Aplicación Interactiva (Streamlit):
```bash
streamlit run app/Home.py
```
La aplicación abrirá automáticamente en su navegador en `http://localhost:8501`.

### 3. Levantar la API REST (FastAPI):
```bash
uvicorn api.main:app --reload --port 8000
```
La documentación interactiva Swagger estará disponible de inmediato en:
👉 `http://127.0.0.1:8000/docs`

#### Endpoints disponibles:
* `POST /predict/mosquito`: Recibe una fotografía (`multipart/form-data`) y devuelve la especie, certeza y bandera `aegypti_detectado`.
* `POST /predict/casos`: Recibe fecha o semana epidemiológica y retorna las 4 semanas proyectadas junto al umbral endémico.
* `POST /predict/accion`: Recibe los 7 indicadores clave y devuelve la recomendación preventiva justificada.

---

## 📊 Rúbrica y Resultados Técnicos

| Modelo | Arquitectura Principal | Métrica Clave | Umbral Rúbrica | Resultado Obtenido |
| :--- | :--- | :--- | :---: | :---: |
| **Módulo 1** | MobileNetV2 (Fine-Tuning 60 capas) | F1-Score Macro | $\ge 85.0\%$ | **Superado** |
| **Módulo 2** | LSTM (64u $\rightarrow$ 32u $\rightarrow$ Dense 4) | RMSE vs Baseline Estacional | Superar Baseline | **Superado** |
| **Módulo 3** | ANN Densa (GroupShuffleSplit por semana) | F1-Score Macro | $\ge 85.0\%$ | **85.0%** |
