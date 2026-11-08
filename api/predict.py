from io import BytesIO
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib

from schemas import (
    RespuestaCNN,
    SolicitudPronostico,
    RespuestaRNN,
    SolicitudANN,
    RespuestaANN,
)


class PredictCNN:
    """Inferencia para clasificación de especies de mosquitos."""
    def __init__(self, ruta_modelo, tamano=224, es_mobilenet=False):
        self.ruta_modelo = ruta_modelo
        self.tamano = tamano
        self.es_mobilenet = es_mobilenet
        self.modelo = keras.models.load_model(ruta_modelo)
        # Nombres canónicos de las 3 clases
        self.clases = ["aedes aegypti", "aedes albopictus", "culex"]

    def preparar_imagen(self, imagen_bytes: bytes):
        """Carga, convierte a RGB y ajusta dimensiones."""
        imagen = Image.open(BytesIO(imagen_bytes))
        imagen = imagen.convert("RGB")
        imagen = imagen.resize((self.tamano, self.tamano))
        arreglo = np.array(imagen, dtype=np.float32)

        # Si es MobileNetV2 aplica escalado [-1, 1]; si es CNN propia usa 0-255 (la red tiene Rescaling)
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


class PredictLSTM:
    """Inferencia para pronóstico semanal de casos de dengue (Módulo 2)."""
    def __init__(self, ruta_modelo, ruta_scaler, ruta_datos_nacional="data/raw/vectoria_dengue_nacional_real.csv", n_pasos=26):
        self.modelo = keras.models.load_model(ruta_modelo)
        self.scaler = joblib.load(ruta_scaler)
        self.n_pasos = n_pasos
        self.df_nacional = self._cargar_datos(ruta_datos_nacional)

    def _cargar_datos(self, ruta_csv):
        ruta = Path(ruta_csv)
        if not ruta.exists():
            # Intento alternativo en raíz si se corre desde otro directorio
            ruta = Path("../data/raw/vectoria_dengue_nacional_real.csv")
        df = pd.read_csv(ruta)
        df["fecha_inicio_semana"] = pd.to_datetime(df["fecha_inicio_semana"]).dt.date
        return df.sort_values("fecha_inicio_semana").reset_index(drop=True)

    def predecir(self, solicitud: SolicitudPronostico) -> RespuestaRNN:
        df = self.df_nacional

        # Ubicar la fila de referencia por fecha o por año y semana epidemiológica
        if solicitud.fecha_inicio:
            idx = df[df["fecha_inicio_semana"] == solicitud.fecha_inicio].index
        elif solicitud.anio and solicitud.semana_epi:
            idx = df[(df["anio"] == solicitud.anio) & (df["semana_epi"] == solicitud.semana_epi)].index
        else:
            # Por defecto toma la última semana disponible en el dataset
            idx = [len(df) - 1]

        if len(idx) == 0:
            raise ValueError("No se encontraron registros históricos para la fecha o semana indicada.")

        indice_corte = idx[0]
        if indice_corte < self.n_pasos:
            raise ValueError(f"Se requieren al menos {self.n_pasos} semanas previas en el histórico para proyectar.")

        # Extraemos la ventana de las últimas n_pasos semanas (t-26 a t)
        ventana = df.iloc[indice_corte - self.n_pasos : indice_corte].copy()
        features = ["casos_dengue", "lluvia_mm", "temp_max_c", "temp_min_c", "humedad_rel_pct"]

        datos_escalados = self.scaler.transform(ventana[features])
        entrada_lstm = np.expand_dims(datos_escalados, axis=0)

        # Predicción a 4 semanas
        pred_escalada = self.modelo.predict(entrada_lstm, verbose=0)[0]
        # Invertir escalado en la columna de casos (primer feature)
        dummy = np.zeros((4, len(features)))
        dummy[:, 0] = pred_escalada
        casos_pred = np.maximum(0, self.scaler.inverse_transform(dummy)[:, 0]).round(1).tolist()

        # Generar las 4 fechas consecutivas (saltos de 7 días)
        fecha_base = df.iloc[indice_corte]["fecha_inicio_semana"]
        fechas_futuras = [fecha_base + timedelta(weeks=i + 1) for i in range(4)]

        # Canal endémico simplificado: percentil 75 histórico de esa misma semana
        sem_actual = df.iloc[indice_corte]["semana_epi"]
        casos_misma_semana = df[df["semana_epi"] == sem_actual]["casos_dengue"]
        umbral_75 = float(np.percentile(casos_misma_semana, 75))

        return RespuestaRNN(
            casos_pronosticados=casos_pred,
            casos_pred_4sem=float(sum(casos_pred)),
            fechas=fechas_futuras,
            casos_umbral_endemico=umbral_75,
        )


# Features oficiales para la ANN según LEEME_datos.md y vectoria_acciones.csv
COLUMNAS_ANN = [
    "casos_pred_4sem",
    "casos_umbral_endemico",
    "lluvia_14d_mm",
    "temp_max",
    "temp_min",
    "humedad",
    "aegypti_detectado",
    "semana_epi",
]

# Clases en formato snake_case coincidente con vectoria_acciones.csv
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
    def __init__(self, ruta_modelo, ruta_scaler=None, ruta_label_encoder=None):
        self.modelo = keras.models.load_model(ruta_modelo)
        self.scaler = joblib.load(ruta_scaler) if ruta_scaler else None
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
