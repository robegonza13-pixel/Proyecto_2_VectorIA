import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib


class EntrenarANN:
    """Clase modular en POO para entrenar y evaluar el clasificador de acciones preventivas (Módulo 3).
    
    Atributos:
        filepath (Path): Ruta al archivo vectoria_acciones.csv
        columnas_x (list): Lista de 7 features predictoras oficiales
        scaler (MinMaxScaler): Escalador numérico entrenado sólo con train
        model (Sequential): Red neuronal densa
        clases (list): Nombres de las 4 clases de recomendación
        history: Registro histórico del entrenamiento
    """

    def __init__(self, filepath='data/raw/vectoria_acciones.csv'):
        self.filepath = Path(filepath)
        self.columnas_x = [
            'casos_pred_4sem',
            'casos_umbral_endemico',
            'lluvia_14d_mm',
            'temp_max',
            'temp_min',
            'humedad',
            'aegypti_detectado'
        ]
        self.scaler = MinMaxScaler()
        self.model = None
        self.clases = None
        self.history = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def cargar_y_preparar_datos(self, test_size=0.25, random_state=101):
        """Carga el dataset, realiza One-Hot Encoding y aplica GroupShuffleSplit por semana epidemiológica."""
        df = pd.read_csv(self.filepath, encoding="ISO-8859-1")

        X = df[self.columnas_x]
        y = df['recomendacion']

        # One-Hot Encoding conservando el orden canónico de clases
        y_dummies = pd.get_dummies(y)
        self.clases = list(y_dummies.columns)
        y_values = y_dummies.values

        # Partición agrupada por semana para evitar data leakage entre escenarios de la misma semana
        grupos = df['fecha_inicio_semana']
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y_values, groups=grupos))

        X_train_raw = X.iloc[train_idx].values
        X_test_raw = X.iloc[test_idx].values
        self.y_train = y_values[train_idx]
        self.y_test = y_values[test_idx]

        # Escalado con MinMaxScaler ajustado únicamente sobre entrenamiento
        self.X_train = self.scaler.fit_transform(X_train_raw)
        self.X_test = self.scaler.transform(X_test_raw)

        print(f"✓ Datos de ANN preparados con éxito:")
        print(f"  - X_train: {self.X_train.shape} | X_test: {self.X_test.shape}")
        print(f"  - Clases: {self.clases}")

    def calcular_pesos_clases(self):
        """Calcula pesos de balanceo suavizados para no ignorar clases minoritarias ni mayoritarias."""
        y_train_labels = np.argmax(self.y_train, axis=1)
        pesos_originales = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_labels),
            y=y_train_labels
        )
        pesos_suavizados = {i: float(max(1.0, peso)) for i, peso in enumerate(pesos_originales)}
        print(f"✓ Pesos calculados para balancear entrenamiento: {pesos_suavizados}")
        return pesos_suavizados

    def construir_modelo(self):
        """Construye la arquitectura ANN: Dense(16) -> Dropout(0.2) -> Dense(8) -> Softmax(4)."""
        num_neuronas_entrada = self.X_train.shape[1]
        num_neuronas_salida = self.y_train.shape[1]

        self.model = Sequential([
            Dense(units=16, activation='relu', input_shape=(num_neuronas_entrada,)),
            Dropout(0.2),
            Dense(units=8, activation='relu'),
            Dense(units=num_neuronas_salida, activation='softmax')
        ], name="ANN_Recomendacion_Acciones")

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=[
                'categorical_accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        self.model.summary()
        return self.model

    def entrenar(self, epochs=150, batch_size=32, class_weights=None):
        """Entrena el modelo con EarlyStopping sobre val_loss."""
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            class_weight=class_weights,
            verbose=1
        )
        print("✓ Entrenamiento de ANN completado exitosamente.")
        return self.history

    def evaluar(self):
        """Genera el reporte de clasificación (Precision, Recall, F1) y la matriz de confusión."""
        pred_prob = self.model.predict(self.X_test)
        y_pred = np.argmax(pred_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)

        print("\n" + "=" * 60)
        print("REPORTE DE CLASIFICACIÓN - ACCIONES PREVENTIVAS")
        print("=" * 60)
        reporte = classification_report(y_true, y_pred, target_names=self.clases, digits=4)
        print(reporte)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión - ANN Acciones")
        plt.colorbar()
        tick_marks = np.arange(len(self.clases))
        plt.xticks(tick_marks, self.clases, rotation=45, ha='right')
        plt.yticks(tick_marks, self.clases)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.show()

        return cm

    def guardar(self, ruta_modelo='models/ann.h5', ruta_scaler='models/scaler_ann.pkl'):
        """Guarda los artefactos entrenados para consumo en la API y Streamlit."""
        Path(ruta_modelo).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(ruta_modelo)
        joblib.dump(self.scaler, ruta_scaler)
        print(f"✓ Modelo guardado en {ruta_modelo}")
        print(f"✓ Escalador guardado en {ruta_scaler}")
