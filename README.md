# 🦟 VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

> **Aviso ético obligatorio:**
> Esta plataforma es una herramienta tecnológica de apoyo a la vigilancia epidemiológica y al control vectorial por parte de autoridades y comunidades. **No es un sistema de diagnóstico clínico y no diagnostica dengue en personas.**

Proyecto 2 · Colegio Universitario de Cartago (CUC)
**Autores:** Valeria Chinchilla, Roberto Gonzalez, Brandon Abarca

---

## 📋 Descripción

**VectorIA** une tres modelos de Deep Learning y una aplicación web para apoyar la prevención de brotes de dengue en Costa Rica. El usuario aporta una **foto** de un mosquito y elige una **semana**, y recibe una **recomendación** de acción preventiva.

| Módulo | Modelo | Qué hace |
| :--- | :--- | :--- |
| 1 | CNN | Identifica la especie del mosquito en una foto: *Aedes aegypti*, *Aedes albopictus* o *Culex*. |
| 2 | RNN / LSTM | Pronostica los casos de dengue de las 4 semanas siguientes a la semana elegida, usando 26 semanas de casos y clima. |
| 3 | ANN | Combina pronóstico, umbral endémico, clima y detección de *Aedes aegypti* para recomendar una acción. |
| 4 | Mapa | Mapa de calor de casos anuales por cantón (2013–2018). |

Las acciones que puede recomendar la ANN son:

- 🚨 `alerta_sanitaria`: casos pronosticados mayores al doble del umbral endémico.
- 💨 `fumigacion_focalizada`: casos sobre el umbral y *Aedes aegypti* detectado.
- 🧹 `eliminacion_criaderos`: lluvia de 14 días mayor a 80 mm y temperatura máxima mayor a 27 °C.
- 📋 `monitoreo_normal`: en cualquier otro caso.

### Detalle de los modelos

**Módulo 1: CNN (imágenes de 224×224)**
- *CNN propia (desde cero):* tres bloques de doble convolución con BatchNormalization y bloques de atención Squeeze-and-Excitation, GlobalAveragePooling, Dense 256, Dropout 0.3 y Softmax de 3 clases. Pérdida con label smoothing 0.1.
- *MobileNetV2 (Transfer Learning):* pesos de ImageNet con cabeza densa propia. Fase 1: base congelada (learning rate 1e-3). Fase 2: fine-tuning de las últimas 40 capas con BatchNormalization congelado (learning rate 1e-4).
- Datos: auditoría con eliminación de corruptas, duplicadas exactas (MD5), borrosas y con luz extrema. Quedan 3535 imágenes, divididas de forma estratificada 80/10/10 (2828 / 353 / 354).
- El modelo que usa la API es **MobileNetV2** (mejor F1 macro). Si no existe, la API usa la CNN propia como respaldo.

**Módulo 2: RNN / LSTM**
- Ventana de 26 semanas, horizonte de 4 semanas, 24 variables de entrada (casos, lluvia, temperaturas máx/mín, humedad, componentes estacionales seno/coseno, semana epidemiológica y rezagos de 1 a 4 semanas del clima).
- Modelo base: SimpleRNN de 32 unidades. Modelo principal: LSTM 64 → LSTM 32 → Dropout 0.2 → Dense 4.
- Partición temporal: las últimas 104 semanas (aprox. 2023–2024) son el test; la validación se toma del final del entrenamiento, de modo que el test no interviene en el entrenamiento. Dos escaladores MinMax, uno para las variables de entrada y otro para los casos.
- Baseline de comparación: los casos de las mismas semanas del año anterior.

**Módulo 3: ANN**
- Red densa 16 → Dropout 0.2 → 8 → Softmax de 4 clases, con pesos de clase suavizados.
- Dataset `vectoria_acciones.csv`: 2060 filas (412 semanas × 5 escenarios). La partición train/test se hace **agrupando por semana** (`GroupShuffleSplit`), para que los 5 escenarios de una misma semana nunca queden separados.
- El umbral endémico de la API se calcula igual que en ese dataset: percentil 75 de la suma de casos de las 4 semanas siguientes, para la misma semana epidemiológica, en los 5 años anteriores.

---

## 🏗️ Estructura del repositorio

```text
VectorIA/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                              # Datos de entrada (ver "Datos necesarios")
│   └── processed/                        # Partición train/val/test de imágenes
├── notebooks/
│   ├── 01_1_EDA_Imagenes.ipynb           # EDA, auditoría y partición de imágenes
│   ├── 02_CNN_Mosquitos.ipynb            # CNN propia y MobileNetV2 (ejecutado en Google Colab con GPU)
│   ├── 03_LSTM_Casos.ipynb               # SimpleRNN vs LSTM vs baseline estacional
│   └── 04_ANN_Acciones.ipynb             # Clasificador de acciones preventivas
├── src/
│   ├── __init__.py
│   ├── data_prep.py                      # Auditoría, EDA y división de datos
│   └── train/
│       ├── __init__.py
│       ├── cnn.py                        # EntrenarCNN y DiagnosticoEvaluacion
│       ├── cnn_mobilenetv2.py            # EntrenarMobileNetV2
│       ├── lstm.py                       # LstmModel
│       └── ann.py                        # EntrenarANN
├── models/                               # Modelos y escaladores entrenados
├── api/
│   ├── main.py                           # Servidor FastAPI
│   ├── schemas.py                        # Esquemas Pydantic
│   └── predict.py                        # Clases de inferencia
└── app/
    ├── Home.py                           # Aplicación Streamlit
    └── assets/
```

---

## 📦 Datos necesarios

Para ejecutar la API y la aplicación se necesitan estos archivos:

**`data/raw/`**

| Archivo | Uso |
| :--- | :--- |
| `vectoria_dengue_nacional_real.csv` | Módulo 2: 574 semanas reales de casos y clima (2014–2024) |
| `vectoria_acciones.csv` | Módulo 3: entrenamiento de la ANN |
| `vectoria_casos_cantonal_anual_real.csv` | Mapa de calor por cantón |
| `costaricacantones.geojson` | Límites de los cantones para el mapa |

**`models/`**

| Archivo | Módulo |
| :--- | :--- |
| `mobilenetv2_final.keras` | CNN (modelo principal de la API) |
| `cnn_propia.keras` | CNN (respaldo) |
| `mobilenetv2_fase1.keras` | CNN (respaldo secundario) |
| `clases_cnn.json` | Orden de las clases de la CNN |
| `lstm_casos.keras`, `scaler_lstm.joblib`, `scaler_y_lstm.joblib` | LSTM y sus dos escaladores (deben provenir de la misma corrida de entrenamiento) |
| `ann.h5`, `scaler_ann.pkl` | ANN y su escalador |

Las imágenes de mosquitos **no** se incluyen en el repositorio por su tamaño. Solo hacen falta para reentrenar. Provienen de datasets públicos indicados en el enunciado (IEEE DataPort, Mendeley, Mosquito Alert y Kaggle) y se organizan en `data/raw/mosquitos/<especie>/`. El notebook `01_1_EDA_Imagenes.ipynb` genera `data/processed/`.

Datos de casos: OPS/PAHO vía OpenDengue. Datos de clima: NASA POWER.

---

## 🚀 Instalación

```bash
git clone https://github.com/robegonza13-pixel/Proyecto_2_VectorIA.git
cd Proyecto_2_VectorIA

# Crear el entorno virtual
python -m venv .venv

# Activarlo
#   Windows:       .venv\Scripts\activate
#   macOS / Linux: source .venv/bin/activate

pip install -r requirements.txt
```

> Los modelos `.keras` deben cargarse con una versión de TensorFlow/Keras compatible con la que los guardó. Usa la versión fijada en `requirements.txt`.

---

## ▶️ Ejecución

Ambos comandos se ejecutan **desde la raíz del proyecto** (varias rutas son relativas). El **orden importa**: la aplicación consulta a la API, así que la API debe estar encendida primero.

### 1. Levantar la API (FastAPI)

```bash
uvicorn api.main:app --port 8000
```

Al arrancar, la consola indica qué modelos cargó. Si aparece "Advertencia al precargar modelos", revisa que los archivos de `models/` existan. Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

### 2. Levantar la aplicación (Streamlit), en otra terminal

```bash
streamlit run app/Home.py
```

Se abre en http://localhost:8501 con cuatro pestañas:

1. **Identificación (CNN):** sube una foto y obtiene la especie y la certeza. Si es *Aedes aegypti*, queda vinculado al módulo 3.
2. **Pronóstico (LSTM):** elige una semana y se grafican las 4 semanas pronosticadas contra el umbral endémico.
3. **Acción preventiva (ANN):** muestra la recomendación y su motivo. Incluye sliders de "¿qué pasaría si?" para mover casos, lluvia y temperatura.
4. **Mapa cantonal:** mapa de calor de casos por cantón y año.

Si la API no está encendida, la aplicación muestra un mensaje de error en lugar de resultados.

---

## 🔌 Endpoints de la API

### `POST /predict/mosquito`
Recibe una imagen (`multipart/form-data`, campo `imagen`).

```bash
curl -X POST http://127.0.0.1:8000/predict/mosquito -F "imagen=@foto.jpg"
```

Devuelve: `especie`, `probabilidad` (0 a 1) y `aegypti_detectado` (1 si es *Aedes aegypti*, 0 en otro caso).

### `POST /predict/casos`
Recibe la fecha de inicio de una semana del dataset, o bien `anio` y `semana_epi`. Sin datos, usa la última semana disponible.

```json
{"fecha_inicio": "2024-12-22"}
```

Devuelve: `casos_pronosticados` (4 valores semanales), `casos_pred_4sem` (suma de las 4 semanas), `fechas` y `casos_umbral_endemico`.
Se requieren al menos 30 semanas previas a la fecha elegida.

### `POST /predict/accion`
Recibe los 7 indicadores.

```json
{
  "casos_pred_4sem": 339,
  "casos_umbral_endemico": 282,
  "lluvia_14d_mm": 17.4,
  "temp_max": 27.1,
  "temp_min": 20,
  "humedad": 80,
  "aegypti_detectado": 1
}
```

Devuelve: `recomendacion`, `recomendacion_legible` y `motivo`. Para el ejemplo anterior, la respuesta es `fumigacion_focalizada`.

---

## 🔁 Reentrenar los modelos

Ejecutar los notebooks en este orden (desde `notebooks/`):

1. `01_1_EDA_Imagenes.ipynb`: audita las imágenes y genera `data/processed/`.
2. `02_CNN_Mosquitos.ipynb`: entrena CNN propia y MobileNetV2. Se ejecutó en Google Colab con GPU por su costo de cómputo; los modelos resultantes se copian a `models/`. Además hay que guardar `clases_cnn.json` con el orden `["aegypti", "albopictus", "culex"]`.
3. `03_LSTM_Casos.ipynb`: entrena y evalúa SimpleRNN y LSTM; `modelo.guardar("../models")` guarda el modelo y los dos escaladores.
4. `04_ANN_Acciones.ipynb`: entrena la ANN y guarda `ann.h5` y `scaler_ann.pkl`.

---

## 📊 Resultados

Todas las métricas son sobre el conjunto de **test**.

### Módulo 1: Clasificación de mosquitos (354 imágenes de test)

| Modelo | F1 macro | Accuracy | F1 *aegypti* | F1 *albopictus* | F1 *culex* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CNN propia | 0.8118 | 0.7994 | 0.7988 | 0.7478 | 0.8889 |
| **MobileNetV2** (fine-tuning, 40 capas) | **0.8781** | 0.8588 | 0.8428 | 0.7917 | 1.0000 |

Umbral del enunciado: F1 macro ≥ 0.85. **MobileNetV2 lo supera** y es el modelo que usa la API. La CNN propia queda por debajo (0.81). La principal confusión de ambos modelos es entre *aegypti* y *albopictus*.

### Módulo 2: Pronóstico de casos a 4 semanas (test: últimas 104 semanas)

| Modelo | RMSE | MAE |
| :--- | :---: | :---: |
| Baseline estacional (año anterior) | 638.33 | 455.75 |
| SimpleRNN (base) | 235.42 | 168.84 |
| LSTM (principal) | 452.87 | 307.85 |

Ambos modelos superan al baseline estacional: la SimpleRNN reduce el RMSE en aproximadamente 63 % y la LSTM en aproximadamente 29 %. En esta corrida la **SimpleRNN obtuvo menor error que la LSTM**, así que la arquitectura más compleja no resultó mejor con estos datos y esta partición. El modelo desplegado en la API es la LSTM, como establece el enunciado.

### Módulo 3: Recomendación de acción (515 filas de test)

| Clase | Precision | Recall |
| :--- | :---: | :---: |
| alerta_sanitaria | 0.95 | 0.83 |
| eliminacion_criaderos | 0.82 | 0.77 |
| fumigacion_focalizada | 0.72 | 0.88 |
| monitoreo_normal | 0.91 | 0.92 |
| **Promedio macro** | **0.85** | **0.85** |

Accuracy global: 0.88.

---

## ⚠️ Limitaciones conocidas

- **CNN propia:** queda por debajo del umbral (F1 macro 0.81) y se entrenó sin ponderación de clases.
- **Posible atajo en *Culex*:** su F1 de 1.0 puede reflejar diferencias entre las fuentes de imágenes y no solo del mosquito. Las dimensiones promedio varían mucho por clase (*Culex* ~265×265 px, *albopictus* ~1540×1933 px). Además, pueden existir casi-duplicados entre train y test que no detecta la deduplicación exacta por MD5.
- **LSTM frente a SimpleRNN:** en el test la LSTM tuvo mayor error que la SimpleRNN (RMSE 452.87 vs 235.42). El test (2023–2024) contiene brotes de magnitud mayor a todo lo visto en el entrenamiento, lo que penaliza más al modelo de mayor capacidad. Es una sola partición temporal, así que la comparación no es concluyente.
- **Horizonte del pronóstico:** 4 semanas después de la última semana con datos (2024-12-22). Para extenderlo hay que actualizar el CSV con casos y clima recientes.
- **Fechas de entrenamiento:** para semanas anteriores a 2023 el modelo ya vio esos datos al entrenar, por lo que el pronóstico se ve mejor de lo que sería en la práctica. El test (2023–2024) incluye los mayores brotes de la serie.
- **La ANN aprende reglas:** las etiquetas de `vectoria_acciones.csv` se generaron con las reglas del enunciado (coinciden en ~98 % de las filas), por lo que la red aproxima esas reglas y no descubre patrones nuevos.
- **Motivo de la recomendación:** el texto del motivo se construye según la clase que predice la ANN; si la ANN se equivoca, el motivo puede no reflejar los números ingresados.
- **Datos cantonales:** se usan como proporciones; su suma no coincide con el total nacional.

---

## 🛠️ Tecnologías

Python · TensorFlow/Keras · OpenCV · Pandas · scikit-learn · Matplotlib / Seaborn / Plotly · FastAPI + Pydantic · Streamlit

---

## ⚖️ Aviso ético

VectorIA es una herramienta de apoyo a la vigilancia epidemiológica. **No diagnostica dengue en personas.** Sus recomendaciones deben ser interpretadas por personal de salud y no sustituyen el criterio profesional.
