from io import BytesIO
from pathlib import Path
from datetime import timedelta
import json
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
import joblib

from api.schemas import (
    RespuestaCNN,
    SolicitudPronostico,
    RespuestaRNN,
    SolicitudANN,
    RespuestaANN,
)


class PredictCNN:
    """Inferencia para clasificación de especies de mosquitos.
    Tanto la CNN propia (Rescaling 1/255) como MobileNetV2 (Rescaling 1/127.5,
    offset=-1) incluyen su propio reescalado como parte del modelo guardado,
    así que aquí SIEMPRE se envía la imagen cruda en rango 0-255."""

    def __init__(self, ruta_modelo="models/cnn_propia.keras", tamano=224,
                 ruta_clases="models/clases_cnn.json"):
        self.ruta_modelo = ruta_modelo
        self.tamano = tamano
        self.modelo = keras.models.load_model(ruta_modelo)

        ruta_clases = Path(ruta_clases)
        if ruta_clases.exists():
            with open(ruta_clases, "r", encoding="utf-8") as f:
                self.clases = json.load(f)
        else:
            self.clases = ["aedes aegypti", "aedes albopictus", "culex"]
            print("⚠ No se encontró models/clases_cnn.json, usando orden por defecto sin verificar.")

    def preparar_imagen(self, imagen_bytes: bytes):
        """Carga, convierte a RGB y redimensiona. Sin reescalado externo:
        los dos modelos lo hacen internamente."""
        imagen = Image.open(BytesIO(imagen_bytes))
        imagen = imagen.convert("RGB")
        imagen = imagen.resize((self.tamano, self.tamano))
        arreglo = np.array(imagen, dtype=np.float32)
        return np.expand_dims(arreglo, axis=0)

    def predecir(self, imagen_bytes: bytes) -> RespuestaCNN:
        entrada = self.preparar_imagen(imagen_bytes)
        probabilidades = self.modelo.predict(entrada, verbose=0)[0]
        posicion = int(np.argmax(probabilidades))
        especie = self.clases[posicion]
        es_aegypti = int("aegypti" in especie.lower())

        return RespuestaCNN(
            especie=especie,
            probabilidad=float(probabilidades[posicion]),
            aegypti_detectado=es_aegypti,
        )


# Columnas de entrada del scaler_lstm.joblib (24 features con rezagos y componentes circulares)
_FEATURES_BASE_LSTM = ["casos_dengue", "lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]
_FEATURES_TEMPORAL_LSTM = ["semana_sin", "semana_cos", "semana_epi"]
_FEATURES_LAG_LSTM = [
    f"{col}_lag{lag}"
    for col in ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]
    for lag in [1, 2, 3, 4]
]
COLUMNAS_LSTM = _FEATURES_BASE_LSTM + _FEATURES_TEMPORAL_LSTM + _FEATURES_LAG_LSTM  # 24 columnas


def _preparar_ventana_lstm(df: pd.DataFrame, indice_corte: int, n_pasos: int) -> pd.DataFrame:
    fin = indice_corte + 1                  # +1 para INCLUIR la semana elegida
    inicio = max(0, fin - n_pasos - 4)      # 4 filas extra para calcular los rezagos
    ventana_ext = df.iloc[inicio:fin].copy()
    ventana_ext["fecha_inicio_semana"] = pd.to_datetime(ventana_ext["fecha_inicio_semana"])

    ventana_ext["semana_sin"] = np.sin(2 * np.pi * ventana_ext["semana_epi"] / 52.0)
    ventana_ext["semana_cos"] = np.cos(2 * np.pi * ventana_ext["semana_epi"] / 52.0)

    for col in ["lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]:
        for lag in [1, 2, 3, 4]:
            ventana_ext[f"{col}_lag{lag}"] = ventana_ext[col].shift(lag)

    ventana_ext = ventana_ext.dropna(subset=COLUMNAS_LSTM).reset_index(drop=True)
    return ventana_ext.tail(n_pasos)


class PredictLSTM:
    """Inferencia para pronóstico semanal de casos de dengue (Módulo 2)."""
    def __init__(
        self,
        ruta_modelo="models/lstm_casos.keras",
        ruta_scaler="models/scaler_lstm.joblib",
        ruta_datos_nacional="data/raw/vectoria_dengue_nacional_real.csv",
        n_pasos=26,
        ruta_scaler_y="models/scaler_y_lstm.joblib"
    ):
        self.modelo = keras.models.load_model(ruta_modelo)
        self.scaler = joblib.load(ruta_scaler)
        self.n_pasos = n_pasos
        self.n_features = len(COLUMNAS_LSTM)  # 24
        self.df_nacional = self._cargar_datos(ruta_datos_nacional)
        self.scaler_y = joblib.load(ruta_scaler_y)

    def _cargar_datos(self, ruta_csv):
        ruta = Path(ruta_csv)
        if not ruta.exists():
            ruta = Path("../data/raw/vectoria_dengue_nacional_real.csv")
        df = pd.read_csv(ruta)
        df["fecha_inicio_semana"] = pd.to_datetime(df["fecha_inicio_semana"])
        return df.sort_values("fecha_inicio_semana").reset_index(drop=True)

    def predecir(self, solicitud: SolicitudPronostico) -> RespuestaRNN:
        df = self.df_nacional

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

        ventana = _preparar_ventana_lstm(df, indice_corte, self.n_pasos)

        if len(ventana) < self.n_pasos:
            raise ValueError("No hay suficientes filas válidas tras calcular los rezagos.")

        datos_escalados = self.scaler.transform(ventana[COLUMNAS_LSTM].values)
        entrada_lstm = np.expand_dims(datos_escalados, axis=0)  # (1, 26, 24)

        pred_escalada = self.modelo.predict(entrada_lstm, verbose=0)[0]  # (4,)
        casos_pred = np.maximum(
            0, self.scaler_y.inverse_transform(pred_escalada.reshape(-1, 1)).flatten()
        ).round(1).tolist()

        fecha_base = df.iloc[indice_corte]["fecha_inicio_semana"]
        fechas_futuras = [fecha_base.date() + timedelta(weeks=i + 1) for i in range(4)]

        anio_actual = int(df.iloc[indice_corte]["anio"])
        sem_actual = int(df.iloc[indice_corte]["semana_epi"])
        suma_4sem_hist = df["casos_dengue"].rolling(4).sum().shift(-4)

        # canal endémico: misma semana epi, solo los 5 años anteriores
        historico = suma_4sem_hist[
            (df["semana_epi"] == sem_actual)
            & (df["anio"] >= anio_actual - 5)
            & (df["anio"] < anio_actual)
            ].dropna()

        if len(historico) == 0:  # años iniciales (2014) sin años previos: usa todo el historial
            historico = suma_4sem_hist[df["semana_epi"] == sem_actual].dropna()

        umbral_75 = float(np.percentile(historico, 75))

        return RespuestaRNN(
            casos_pronosticados=casos_pred,
            casos_pred_4sem=float(sum(casos_pred)),
            fechas=fechas_futuras,
            casos_umbral_endemico=umbral_75,
        )


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
    def __init__(self, ruta_modelo="models/ann.h5", ruta_scaler="models/scaler_ann.pkl", ruta_label_encoder=None,):
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