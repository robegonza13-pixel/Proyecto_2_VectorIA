from datetime import date
from typing import Literal
from pydantic import BaseModel, Field


class RespuestaCNN(BaseModel):
    """Respuesta de la clasificación de especies de mosquitos."""
    especie: str
    probabilidad: float = Field(ge=0, le=1)
    aegypti_detectado: int = Field(ge=0, le=1, description="1 si es Aedes aegypti, 0 en otro caso")


class SolicitudPronostico(BaseModel):
    """Solicitud para pronosticar casos de dengue."""
    fecha_inicio: date | None = None
    anio: int | None = Field(default=None, ge=2014, le=2030)
    semana_epi: int | None = Field(default=None, ge=1, le=53)


class RespuestaRNN(BaseModel):
    """Respuesta con la proyección a 4 semanas y comparación contra canal endémico."""
    casos_pronosticados: list[float] = Field(min_length=4, max_length=4)
    casos_pred_4sem: float = Field(ge=0, description="Suma total de casos estimados para las 4 semanas")
    fechas: list[date] = Field(min_length=4, max_length=4)
    casos_umbral_endemico: float = Field(ge=0, description="Umbral endémico de referencia (P75)")


class SolicitudANN(BaseModel):
    """Variables de entrada para recomendar la acción preventiva (7 features del scaler_ann.pkl)."""
    casos_pred_4sem: float = Field(ge=0)
    casos_umbral_endemico: float = Field(ge=0)
    lluvia_14d_mm: float = Field(ge=0)
    temp_max: float
    temp_min: float
    humedad: float = Field(ge=0, le=100)
    aegypti_detectado: int = Field(ge=0, le=1)


class RespuestaANN(BaseModel):
    """Acción recomendada por la ANN y su justificación epidemiológica."""
    recomendacion: Literal[
        "alerta_sanitaria",
        "fumigacion_focalizada",
        "eliminacion_criaderos",
        "monitoreo_normal",
    ]
    recomendacion_legible: str
    motivo: str
