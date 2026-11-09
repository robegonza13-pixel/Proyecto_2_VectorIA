from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    RespuestaCNN,
    SolicitudPronostico,
    RespuestaRNN,
    SolicitudANN,
    RespuestaANN,
)
from api.predict import PredictCNN, PredictLSTM, PredictANN
from pathlib import Path

app = FastAPI(
    title="VectorIA API",
    description="API de soporte para vigilancia epidemiológica del Dengue en Costa Rica (CNN, LSTM, ANN)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_cnn: PredictCNN | None = None
predictor_lstm: PredictLSTM | None = None
predictor_ann: PredictANN | None = None

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # VectorIA/ (un nivel arriba de api/)
MODELS_DIR = BASE_DIR / "models"

@app.on_event("startup")
def cargar_modelos():
    """Carga los modelos entrenados si están disponibles en disco, priorizando el de mejor F1 (MobileNetV2)."""
    global predictor_cnn, predictor_lstm, predictor_ann
    try:
        ruta_mobilenet = MODELS_DIR / "mobilenetv2_final.keras"
        ruta_cnn_propia = MODELS_DIR / "cnn_propia.keras"
        ruta_mobilenet_f1 = MODELS_DIR / "mobilenetv2_fase1.keras"
        ruta_clases = str(MODELS_DIR / "clases_cnn.json")

        if ruta_mobilenet.exists():
            predictor_cnn = PredictCNN(str(ruta_mobilenet), tamano=224, ruta_clases=ruta_clases)
            print(f"✓ Modelo oficial cargado en API: MobileNetV2 (mejor F1) — {ruta_mobilenet}")
        elif ruta_cnn_propia.exists():
            predictor_cnn = PredictCNN(str(ruta_cnn_propia), tamano=224, ruta_clases=ruta_clases)
            print(f"✓ Modelo cargado en API: CNN Propia (fallback) — {ruta_cnn_propia}")
        elif ruta_mobilenet_f1.exists():
            predictor_cnn = PredictCNN(str(ruta_mobilenet_f1), tamano=224, ruta_clases=ruta_clases)
            print(f"✓ Modelo cargado en API: MobileNetV2 Fase 1 — {ruta_mobilenet_f1}")

        if (MODELS_DIR / "lstm_casos.keras").exists() \
                and (MODELS_DIR / "scaler_lstm.joblib").exists() \
                and (MODELS_DIR / "scaler_y_lstm.joblib").exists():
            predictor_lstm = PredictLSTM(
                str(MODELS_DIR / "lstm_casos.keras"),
                str(MODELS_DIR / "scaler_lstm.joblib"),
                ruta_scaler_y=str(MODELS_DIR / "scaler_y_lstm.joblib"),
            )

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
        "modelo_cnn": Path(predictor_cnn.ruta_modelo).name if predictor_cnn else "no cargado",
        "aviso_etico": "Herramienta de apoyo a la vigilancia epidemiológica; no diagnostica dengue en personas.",
    }


@app.post("/predict/mosquito", response_model=RespuestaCNN)
async def predecir_mosquito(imagen: UploadFile = File(...)):
    """Clasifica la especie de mosquito en la fotografía (CNN propia, con fallback a MobileNetV2)."""
    if predictor_cnn is None:
        raise HTTPException(status_code=503, detail="Modelo CNN no disponible o no cargado en models/.")
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