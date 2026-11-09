# 🤖 Declaración de Uso de Inteligencia Artificial en el Proyecto VectorIA

**Curso:** Inteligencia Artificial Aplicada  
**Proyecto:** Proyecto 2 · VectorIA: Asistente Inteligente para la Vigilancia del Dengue en Costa Rica  
**Fecha:** Noviembre, 2026  

---

## 1. Herramientas de IA Utilizadas y Tareas Específicas

Durante la concepción, depuración e integración del proyecto VectorIA, se emplearon herramientas de Inteligencia Artificial Generativa bajo un esquema de **pair programming** y mentoría técnica interactiva:

| Herramienta | Versión / Modelo | Tareas Específicas Desarrolladas |
| :--- | :--- | :--- |
| **Antigravity AI (Google DeepMind)** | Gemini 3.8 / Claude 3.5 Sonnet | • Arquitectura de clases en Programación Orientada a Objetos (POO).<br>• Diagnóstico matemático del estancamiento del F1-Score en la clasificación de mosquitos (*Aedes aegypti* vs *Aedes albopictus*).<br>• Diseño del pipeline de Fine-Tuning en MobileNetV2 (congelación de BatchNorm y descongelamiento de capas profundas).<br>• Estructuración de los contratos de datos y endpoints con **FastAPI** (`schemas.py`, `predict.py`).<br>• Integración de componentes visuales en **Streamlit** (coropletas con Plotly y GeoJSON cantonal). |
| **ChatGPT / Claude** | GPT-4o / Claude 3.5 Sonnet | • Redacción de docstrings y comentarios explicativos de código.<br>• Consultas sobre la prevención de fuga de datos (*data leakage*) mediante `GroupShuffleSplit`.<br>• Optimización de visualizaciones y matrices de confusión en Matplotlib/Seaborn. |

---

## 2. Ejemplos Representativos de Prompts Utilizados

A continuación se presentan 4 prompts reales representativos utilizados durante el desarrollo del proyecto:

### Prompt 1: Diagnóstico de cuello de botella en métricas (Módulo 1 - CNN)
> *"Estoy teniendo problemas con las métricas de la CNN: he probado aumentando dimensiones, bajando el dropout y agregando una capa de 256 filtros pero no pasa del 65%-72% de val_accuracy. Luego implementamos MobileNetV2 y se queda estancado en 84%, justo 1% por debajo del 85% que pide la rúbrica. ¿Es un problema de los datos o se puede ajustar algo a nivel de arquitectura y función de pérdida?"*
* **Propósito:** Identificar la causa del estancamiento en la distinción biológica de especies y evaluar el impacto de *Data Augmentation*, *Label Smoothing* y descongelamiento de capas en Transfer Learning.

### Prompt 2: Prevención de fuga de datos en series temporales con escenarios (Módulo 3 - ANN)
> *"En el archivo vectoria_acciones.csv cada semana epidemiológica se repite en 5 escenarios simulados. Si hago un train_test_split aleatorio convencional, ¿se produce data leakage? ¿Cómo implemento un GroupShuffleSplit agrupando por fecha_inicio_semana y aplicando MinMaxScaler únicamente con el conjunto de entrenamiento?"*
* **Propósito:** Validar la metodología estadística para evitar sobreajuste artificial y asegurar que semanas idénticas no queden simultáneamente en train y test.

### Prompt 3: Diseño de API REST modular y contratos de datos (FastAPI)
> *"Necesito conectar los 3 modelos (CNN, LSTM y ANN) en una API con FastAPI. ¿Cómo diseño los esquemas con Pydantic para que la salida de la CNN (aegypti detectado) y el acumulado de 4 semanas de la LSTM pasen directamente como entrada al clasificador de la ANN, incluyendo las 4 recomendaciones sanitarias?"*
* **Propósito:** Construir contratos tipados y endpoints desacoplados (`/predict/mosquito`, `/predict/casos`, `/predict/accion`) alineados con la rúbrica.

### Prompt 4: Integración del Mapa Cantonal Coroplético (Streamlit)
> *"Tengo un archivo vectoria_casos_cantonal_anual_real.csv con casos por cantón y año, y un archivo costaricacantones.geojson. ¿Cómo genero en Streamlit un mapa coroplético interactivo con Plotly Express donde el usuario filtre por año y la escala de color refleje los casos reportados?"*
* **Propósito:** Implementar la visualización geoespacial interactiva de la Pestaña 4 de la aplicación web.

---

## 3. Modificaciones y Adaptaciones al Código Asistido por IA

El código sugerido por las herramientas de IA nunca fue adoptado a ciegas; requirió una serie de adaptaciones críticas por parte del equipo para amoldarse a los datos reales y restricciones del proyecto:

1. **Corrección del flujo funcional en MobileNetV2:**
   - La IA inicialmente omitió la conexión explícita de `x = self.base_model(x, training=False)` dentro de la API Funcional de Keras, lo que causaba que la cabeza densa recibiera los píxeles directos sin pasar por la red convolucional. Identificamos el error al revisar la arquitectura e insertamos la llamada manteniendo `training=False` para proteger las estadísticas de `BatchNormalization`.
2. **Sincronización de dimensiones y canales de entrada:**
   - Se ajustó el tamaño de entrada a un estándar único de **$224 \times 224$ píxeles** tanto para la CNN propia como para MobileNetV2, descartando resoluciones menores ($128 \times 128$) que borraban las escamas blancas del tórax de los mosquitos.
3. **Consistencia de variables en la ANN (7 vs 8 features):**
   - La IA planteó 8 variables de entrada en el endpoint; no obstante, al entrenar el modelo definitivo en el notebook se demostró que prescindir de `semana_epi` mejoraba la capacidad de generalización. Ajustamos `COLUMNAS_ANN` en `predict.py` y en el script de Streamlit para que coincidiera exactamente con las 7 columnas del escalador `scaler_ann.pkl`.
4. **Manejo de rutas relativas y carpetas:**
   - Se blindaron los métodos de guardado (`.save()`, `joblib.dump()`) agregando `Path('models').mkdir(parents=True, exist_ok=True)` para garantizar compatibilidad tanto en entornos locales de Windows como en Google Colab y contenedores Docker.

---

## 4. Reflexión sobre el Aprendizaje con Inteligencia Artificial

Trabajar con herramientas de Inteligencia Artificial en este proyecto transformó la dinámica de desarrollo: pasó de ser una búsqueda dispersa en foros y documentación a un proceso de **programación guiada y razonamiento crítico continuo**. La IA actuó como un tutor técnico que nos permitió contrastar hipótesis con rapidez (por ejemplo, entender que el umbral del 84% no se rompía simplemente apilando capas convolucionales, sino atacando la confusión biológica de las escamas del tórax y aplicando regularizaciones avanzadas como *Label Smoothing* y *Fine-Tuning* controlado).

Sin embargo, el mayor aprendizaje radicó en comprender que la IA carece de contexto de dominio si el desarrollador no la audita: los modelos predictivos generados habrían fallado silenciosamente de no haber revisado la concordancia de nombres entre datasets, la preservación de las capas de normalización o la prevención de fuga de datos en las particiones de series temporales. La IA aceleró la escritura de código repetitivo y la estructuración del proyecto, pero el rigor analítico, la curaduría de los datos y las decisiones epidemiológicas finales fueron producto directo de nuestro criterio técnico.
