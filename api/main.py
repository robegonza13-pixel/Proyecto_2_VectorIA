from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import (
    RespuestaCNN,
    SolicitudPronostico,
    RespuestaRNN,
    SolicitudANN,
    RespuestaANN,
)
from .predict import PredictCNN, PredictLSTM, PredictANN

app = FastAPI(
    title="VectorIA API",
    description="API de soporte para vigilancia epidemiológica del Dengue en Costa Rica (CNN, LSTM, ANN)",
    version="1.0.0",
)

# Permitir conexiones desde Streamlit o navegadores
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización diferida de predictores
predictor_cnn: PredictCNN | None = None
predictor_lstm: PredictLSTM | None = None
predictor_ann: PredictANN | None = None


@app.on_event("startup")
def cargar_modelos():
    """Carga los modelos entrenados si están disponibles en disco, priorizando MobileNetV2."""
    global predictor_cnn, predictor_lstm, predictor_ann
    try:
        # Prioridad absoluta al modelo de mejor rendimiento: MobileNetV2
        if Path("models/mobilenetv2_final.keras").exists():
            predictor_cnn = PredictCNN("models/mobilenetv2_final.keras", tamano=224, es_mobilenet=True)
            print("✓ Modelo oficial cargado en API: MobileNetV2 Final")
        elif Path("models/mobilenetv2_fase1.keras").exists():
            predictor_cnn = PredictCNN("models/mobilenetv2_fase1.keras", tamano=224, es_mobilenet=True)
            print("✓ Modelo cargado en API: MobileNetV2 Fase 1")
        elif Path("models/cnn_propia.keras").exists():
            predictor_cnn = PredictCNN("models/cnn_propia.keras", tamano=224, es_mobilenet=False)
            print("✓ Modelo cargado en API: CNN Propia")

        if Path("models/lstm_casos.keras").exists() and Path("models/scaler_lstm.joblib").exists():
            predictor_lstm = PredictLSTM("models/lstm_casos.keras", "models/scaler_lstm.joblib")

        if Path("models/ann.h5").exists() and Path("models/scaler_ann.pkl").exists():
            predictor_ann = PredictANN("models/ann.h5", ruta_scaler="models/scaler_ann.pkl")
        elif Path("models/ann_acciones.keras").exists():
            scaler_ann = "models/scaler_ann.joblib" if Path("models/scaler_ann.joblib").exists() else None
            le_ann = "models/label_encoder_ann.joblib" if Path("models/label_encoder_ann.joblib").exists() else None
            predictor_ann = PredictANN("models/ann_acciones.keras", ruta_scaler=scaler_ann, ruta_label_encoder=le_ann)
    except Exception as e:
        print(f"Advertencia al precargar modelos: {e}")


@app.get("/")
def estado():
    return {
        "sistema": "VectorIA API",
        "estado": "operativo",
        "modelo_cnn": "MobileNetV2 Transfer Learning",
        "aviso_etico": "Herramienta de apoyo a la vigilancia epidemiológica; no diagnostica dengue en personas.",
    }


@app.post("/predict/mosquito", response_model=RespuestaCNN)
async def predecir_mosquito(imagen: UploadFile = File(...)):
    """Clasifica la especie de mosquito en la fotografía utilizando MobileNetV2."""
    if predictor_cnn is None:
        raise HTTPException(status_code=503, detail="Modelo MobileNetV2 no disponible o no cargado en models/.")
    contenido = await imagen.read()
    return predictor_cnn.predecir(contenido)


@app.post("/predict/casos", response_model=RespuestaRNN)
def predecir_casos(solicitud: SolicitudPronostico):
    """Pronostica los casos para las próximas 4 semanas (Módulo 2 - LSTM)."""
    if predictor_lstm is None:
        raise HTTPException(status_code=503, detail="Modelo LSTM no disponible o no cargado en models/.")
    try:
        return predictor_lstm.predecir(solicitud)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/accion", response_model=RespuestaANN)
def predecir_accion(solicitud: SolicitudANN):
    """Recomienda la acción preventiva a ejecutar (Módulo 3 - ANN)."""
    if predictor_ann is None:
        raise HTTPException(status_code=503, detail="Modelo ANN no disponible o no cargado en models/.")
    return predictor_ann.predecir(solicitud)
