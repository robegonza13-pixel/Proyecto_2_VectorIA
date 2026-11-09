from io import BytesIO
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib

from .schemas import (
    RespuestaCNN,
    SolicitudPronostico,
    RespuestaRNN,
    SolicitudANN,
    RespuestaANN,
)


class PredictCNN:
    """Inferencia oficial para clasificación de especies de mosquitos usando MobileNetV2."""
    def __init__(self, ruta_modelo="models/mobilenetv2_final.keras", tamano=224, es_mobilenet=True):
        self.ruta_modelo = ruta_modelo
        self.tamano = tamano
        self.es_mobilenet = es_mobilenet
        self.modelo = keras.models.load_model(ruta_modelo)
        # Nombres canónicos de las 3 clases (mismo orden que image_dataset_from_directory)
        self.clases = ["aedes aegypti", "aedes albopictus", "culex"]

    def preparar_imagen(self, imagen_bytes: bytes):
        """Carga, convierte a RGB y escala al rango de MobileNetV2 [-1, 1]."""
        imagen = Image.open(BytesIO(imagen_bytes))
        imagen = imagen.convert("RGB")
        imagen = imagen.resize((self.tamano, self.tamano))
        arreglo = np.array(imagen, dtype=np.float32)

        # Preprocesamiento nativo de MobileNetV2 (0-255 -> -1 a 1)
        if self.es_mobilenet:
            arreglo = preprocess_input(arreglo)

        return np.expand_dims(arreglo, axis=0)

    def predecir(self, imagen_bytes: bytes) -> RespuestaCNN:
        entrada = self.preparar_imagen(imagen_bytes)
        probabilidades = self.modelo.predict(entrada, verbose=0)[0]
        posicion = int(np.argmax(probabilidades))
        especie = self.clases[posicion]

        # Coincidencia robusta para detectar si es Aedes aegypti
        es_aegypti = int("aegypti" in especie.lower())

        return RespuestaCNN(
            especie=especie,
            probabilidad=float(probabilidades[posicion]),
            aegypti_detectado=es_aegypti,
        )


# Columnas de entrada del scaler_lstm.joblib (24 features con rezagos y componentes circulares)
# Deben coincidir EXACTAMENTE con self.feature_cols del LstmModel entrenado
_FEATURES_BASE_LSTM = ["casos_dengue", "lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]
_FEATURES_TEMPORAL_LSTM = ["semana_sin", "semana_cos", "semana_epi"]
_FEATURES_LAG_LSTM = [
    f"{col}_lag{lag}"
    for col in ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]
    for lag in [1, 2, 3, 4]
]
COLUMNAS_LSTM = _FEATURES_BASE_LSTM + _FEATURES_TEMPORAL_LSTM + _FEATURES_LAG_LSTM  # 24 columnas


def _preparar_ventana_lstm(df: pd.DataFrame, indice_corte: int, n_pasos: int) -> pd.DataFrame:
    """Reconstruye las 24 features (con rezagos y componentes circulares) para los últimos n_pasos."""
    # Ampliamos la ventana para poder calcular rezagos máximos (lag4)
    inicio = max(0, indice_corte - n_pasos - 4)
    ventana_ext = df.iloc[inicio:indice_corte].copy()
    ventana_ext["fecha_inicio_semana"] = pd.to_datetime(ventana_ext["fecha_inicio_semana"])

    # Componentes circulares de estacionalidad
    ventana_ext["semana_sin"] = np.sin(2 * np.pi * ventana_ext["semana_epi"] / 52.0)
    ventana_ext["semana_cos"] = np.cos(2 * np.pi * ventana_ext["semana_epi"] / 52.0)

    # Rezagos de las 4 variables climáticas
    for col in ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]:
        for lag in [1, 2, 3, 4]:
            ventana_ext[f"{col}_lag{lag}"] = ventana_ext[col].shift(lag)

    # Eliminamos las filas con NaN por los rezagos y tomamos las últimas n_pasos filas
    ventana_ext = ventana_ext.dropna(subset=COLUMNAS_LSTM).reset_index(drop=True)
    return ventana_ext.tail(n_pasos)


class PredictLSTM:
    """Inferencia para pronóstico semanal de casos de dengue (Módulo 2).
    
    El scaler_lstm.joblib fue entrenado con 24 features (5 base + 3 temporales + 16 rezagos).
    Este predictor reconstruye exactamente esas features antes de escalar.
    """
    def __init__(
        self,
        ruta_modelo="models/lstm_casos.keras",
        ruta_scaler="models/scaler_lstm.joblib",
        ruta_datos_nacional="data/raw/vectoria_dengue_nacional_real.csv",
        n_pasos=26,
    ):
        self.modelo = keras.models.load_model(ruta_modelo)
        self.scaler = joblib.load(ruta_scaler)
        self.n_pasos = n_pasos
        self.n_features = len(COLUMNAS_LSTM)  # 24
        self.df_nacional = self._cargar_datos(ruta_datos_nacional)

    def _cargar_datos(self, ruta_csv):
        ruta = Path(ruta_csv)
        if not ruta.exists():
            ruta = Path("../data/raw/vectoria_dengue_nacional_real.csv")
        df = pd.read_csv(ruta)
        df["fecha_inicio_semana"] = pd.to_datetime(df["fecha_inicio_semana"])
        return df.sort_values("fecha_inicio_semana").reset_index(drop=True)

    def predecir(self, solicitud: SolicitudPronostico) -> RespuestaRNN:
        df = self.df_nacional

        # Ubicar la fila de referencia por fecha o por año y semana epidemiológica
        if solicitud.fecha_inicio:
            fecha_dt = pd.to_datetime(solicitud.fecha_inicio)
            idx = df[df["fecha_inicio_semana"] == fecha_dt].index
        elif solicitud.anio and solicitud.semana_epi:
            idx = df[
                (df["anio"] == solicitud.anio) & (df["semana_epi"] == solicitud.semana_epi)
            ].index
        else:
            idx = [len(df) - 1]

        if len(idx) == 0:
            raise ValueError("No se encontraron registros históricos para la fecha o semana indicada.")

        indice_corte = idx[0]
        if indice_corte < self.n_pasos + 4:
            raise ValueError(
                f"Se requieren al menos {self.n_pasos + 4} semanas previas para calcular rezagos y proyectar."
            )

        # Reconstruir las 24 columnas con rezagos y componentes circulares
        ventana = _preparar_ventana_lstm(df, indice_corte, self.n_pasos)

        if len(ventana) < self.n_pasos:
            raise ValueError("No hay suficientes filas válidas tras calcular los rezagos.")

        datos_escalados = self.scaler.transform(ventana[COLUMNAS_LSTM].values)
        entrada_lstm = np.expand_dims(datos_escalados, axis=0)  # (1, 26, 24)

        pred_escalada = self.modelo.predict(entrada_lstm, verbose=0)[0]  # (4,)
        dummy = np.zeros((4, self.n_features))
        dummy[:, 0] = pred_escalada
        casos_pred = np.maximum(0, self.scaler.inverse_transform(dummy)[:, 0]).round(1).tolist()

        fecha_base = df.iloc[indice_corte]["fecha_inicio_semana"]
        fechas_futuras = [fecha_base.date() + timedelta(weeks=i + 1) for i in range(4)]

        sem_actual = int(df.iloc[indice_corte]["semana_epi"])
        casos_misma_semana = df[df["semana_epi"] == sem_actual]["casos_dengue"]
        umbral_75 = float(np.percentile(casos_misma_semana, 75))

        return RespuestaRNN(
            casos_pronosticados=casos_pred,
            casos_pred_4sem=float(sum(casos_pred)),
            fechas=fechas_futuras,
            casos_umbral_endemico=umbral_75,
        )


# Las 7 columnas exactas con las que se entrenó la ANN (notebook 04 y scaler_ann.pkl)
COLUMNAS_ANN = [
    "casos_pred_4sem",
    "casos_umbral_endemico",
    "lluvia_14d_mm",
    "temp_max",
    "temp_min",
    "humedad",
    "aegypti_detectado",
]

CLASES_ANN = [
    "alerta_sanitaria",
    "eliminacion_criaderos",
    "fumigacion_focalizada",
    "monitoreo_normal",
]

NOMBRES_LEGIBLES = {
    "alerta_sanitaria": "Alerta Sanitaria",
    "eliminacion_criaderos": "Eliminación de Criaderos",
    "fumigacion_focalizada": "Fumigación Focalizada",
    "monitoreo_normal": "Monitoreo Normal",
}

MOTIVOS = {
    "alerta_sanitaria": "Los casos pronosticados ({casos:.0f}) superan el doble del umbral endémico ({umbral:.0f}).",
    "fumigacion_focalizada": "Los casos pronosticados ({casos:.0f}) superan el umbral endémico ({umbral:.0f}) y se detectó Aedes aegypti.",
    "eliminacion_criaderos": "La lluvia acumulada ({lluvia:.1f} mm) y temperatura ({temp:.1f} °C) favorecen el desarrollo de criaderos.",
    "monitoreo_normal": "Los indicadores epidemiológicos y climáticos se mantienen dentro de los umbrales seguros.",
}


class PredictANN:
    """Inferencia para recomendación de acciones preventivas (Módulo 3)."""
    def __init__(self, ruta_modelo="models/ann.h5", ruta_scaler="models/scaler_ann.pkl", ruta_label_encoder=None):
        self.modelo = keras.models.load_model(ruta_modelo)
        self.scaler = joblib.load(ruta_scaler) if ruta_scaler and Path(ruta_scaler).exists() else None
        if ruta_label_encoder and Path(ruta_label_encoder).exists():
            le = joblib.load(ruta_label_encoder)
            self.clases = list(le.classes_)
        else:
            self.clases = CLASES_ANN

    def predecir(self, solicitud: SolicitudANN) -> RespuestaANN:
        valores = [getattr(solicitud, col) for col in COLUMNAS_ANN]
        entrada = np.array([valores], dtype=float)

        if self.scaler is not None:
            entrada = self.scaler.transform(entrada)

        probabilidades = self.modelo.predict(entrada, verbose=0)[0]
        recomendacion = self.clases[int(np.argmax(probabilidades))]

        motivo = MOTIVOS[recomendacion].format(
            casos=solicitud.casos_pred_4sem,
            umbral=solicitud.casos_umbral_endemico,
            lluvia=solicitud.lluvia_14d_mm,
            temp=solicitud.temp_max,
        )

        return RespuestaANN(
            recomendacion=recomendacion,
            recomendacion_legible=NOMBRES_LEGIBLES.get(recomendacion, recomendacion),
            motivo=motivo,
        )
