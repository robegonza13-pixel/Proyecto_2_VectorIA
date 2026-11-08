"""Módulo de entrenamiento de modelos para VectorIA."""
from .cnn import EntrenarCNN, DiagnosticoEvaluacion
from .cnn_mobilenetv2 import EntrenarMobileNetV2

__all__ = ["EntrenarCNN", "DiagnosticoEvaluacion", "EntrenarMobileNetV2"]
