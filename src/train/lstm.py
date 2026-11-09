from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class LstmModel:
    def __init__(self, length=26, horizon=4):
        self.length = length    # ventana de 26 semanas
        self.horizon = horizon  # predicción de las próximas 4 semanas
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.df = None
        self.feature_cols = None
        self.test_size = None

        self.model_rnn = None
        self.model_lstm = None

    # ------------------------------------------------------------------
    # Preparación de datos
    # ------------------------------------------------------------------
    def preparar_datos(self, df, test_size=104, usar_lags=True):
        data = df.sort_values('fecha_inicio_semana').reset_index(drop=True).copy()
        data['fecha_inicio_semana'] = pd.to_datetime(data['fecha_inicio_semana'])
        self.test_size = test_size

        # componentes temporales circulares (estacionalidad)
        if 'semana_epi' in data.columns:
            data['semana_sin'] = np.sin(2 * np.pi * data['semana_epi'] / 52.0)
            data['semana_cos'] = np.cos(2 * np.pi * data['semana_epi'] / 52.0)

        # rezagos del clima: el clima actúa semanas después
        if usar_lags:
            for col in ['lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']:
                if col in data.columns:
                    for lag in [1, 2, 3, 4]:
                        data[f'{col}_lag{lag}'] = data[col].shift(lag)

        # los primeros registros quedan con nulos por los rezagos
        self.df = data.dropna().reset_index(drop=True)

        base_cols = ['casos_dengue', 'lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']
        base_cols = [c for c in base_cols if c in self.df.columns]
        temporal = [c for c in ['semana_sin', 'semana_cos', 'semana_epi'] if c in self.df.columns]
        lag_cols = [c for c in self.df.columns if 'lag' in c]
        self.feature_cols = base_cols + temporal + lag_cols

        X_raw = self.df[self.feature_cols].values
        Y_raw = self.df[['casos_dengue']].values

        # split temporal: el test arrastra las 'length' filas previas como contexto
        train_size = len(self.df) - test_size
        X_train_raw = X_raw[:train_size]
        X_test_raw = X_raw[train_size - self.length:]
        Y_train_raw = Y_raw[:train_size]
        Y_test_raw = Y_raw[train_size - self.length:]

        # los scalers se ajustan SOLO con train (evita fuga de datos)
        scaled_x_train = self.scaler_x.fit_transform(X_train_raw)
        scaled_x_test = self.scaler_x.transform(X_test_raw)
        scaled_y_train = self.scaler_y.fit_transform(Y_train_raw)
        scaled_y_test = self.scaler_y.transform(Y_test_raw)

        return scaled_x_train, scaled_y_train, scaled_x_test, scaled_y_test

    def crear_secuencias(self, data_x, data_y):
        x, y = [], []
        for i in range(len(data_x) - self.length - self.horizon + 1):
            x.append(data_x[i: i + self.length])
            y.append(data_y[i + self.length: i + self.length + self.horizon, 0])
        return np.array(x), np.array(y)

    def _desescalar_y(self, arr):
        """Vuelve de [0, 1] a casos reales, sin importar la forma del arreglo."""
        arr = np.asarray(arr)
        return self.scaler_y.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)

    @staticmethod
    def _callbacks():
        # callbacks nuevos para cada modelo, así ninguno hereda estado del otro
        return [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
        ]

    # ------------------------------------------------------------------
    # Entrenamiento (el test NO participa aquí)
    # ------------------------------------------------------------------
    def entrenar(self, scaled_x_train, scaled_y_train, epochs=150, batch_size=16, n_val=52):
        x_all, y_all = self.crear_secuencias(scaled_x_train, scaled_y_train)

        # validación = las últimas secuencias del train, en orden temporal.
        # 'hueco' evita que una ventana de train comparta semanas con una de validación.
        hueco = self.length + self.horizon
        x_train, y_train = x_all[:-(n_val + hueco)], y_all[:-(n_val + hueco)]
        x_val, y_val = x_all[-n_val:], y_all[-n_val:]
        print(f"Secuencias -> train: {len(x_train)} | validación: {len(x_val)}")

        n_features = scaled_x_train.shape[1]

        # modelo base: SimpleRNN
        print("Entrenando Modelo Base (SimpleRNN)...")
        self.model_rnn = Sequential([
            tf.keras.Input(shape=(self.length, n_features)),
            SimpleRNN(32, activation='tanh'),
            Dense(self.horizon),
        ])
        self.model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model_rnn.fit(
            x_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=self._callbacks(), verbose=0,
        )

        # modelo principal: LSTM 64 -> LSTM 32 -> Dropout 0.2 -> Dense 4 (como pide el PDF)
        print("Entrenando Modelo Principal (LSTM)...")
        self.model_lstm = Sequential([
            tf.keras.Input(shape=(self.length, n_features)),
            LSTM(64, activation='tanh', return_sequences=True),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(self.horizon),
        ])
        self.model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        history_lstm = self.model_lstm.fit(
            x_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=self._callbacks(), verbose=0,
        )

        print("Entrenamiento finalizado con éxito.")
        return history_lstm

    # ------------------------------------------------------------------
    # Evaluación sobre el test (primera vez que el modelo "ve" estos datos)
    # ------------------------------------------------------------------
    def evaluar_metricas(self, scaled_x_test, scaled_y_test, test_size=None):
        if test_size is None:
            test_size = self.test_size

        x_test, y_test_scaled = self.crear_secuencias(scaled_x_test, scaled_y_test)

        y_pred_rnn_scaled = self.model_rnn.predict(x_test, verbose=0)
        y_pred_lstm_scaled = self.model_lstm.predict(x_test, verbose=0)

        y_true = self._desescalar_y(y_test_scaled)
        y_pred_rnn = self._desescalar_y(y_pred_rnn_scaled)
        y_pred_lstm = self._desescalar_y(y_pred_lstm_scaled)

        # baseline estacional: las mismas 4 semanas del año anterior.
        # Empieza en NaN: si una fila no tiene año previo, se descarta y no cuenta como 0.
        y_baseline = np.full_like(y_true, np.nan, dtype=float)
        test_start_idx = len(self.df) - test_size

        for i in range(len(y_true)):
            ini = test_start_idx + i - 52
            if ini >= 0 and (ini + self.horizon) <= len(self.df):
                y_baseline[i] = self.df['casos_dengue'].iloc[ini: ini + self.horizon].values

        # los conteos de casos no pueden ser negativos
        y_true_f = y_true.flatten()
        y_rnn_f = np.maximum(0, y_pred_rnn.flatten())
        y_lstm_f = np.maximum(0, y_pred_lstm.flatten())
        y_base_f = y_baseline.flatten()

        mascara = (np.isfinite(y_true_f) & np.isfinite(y_rnn_f)
                   & np.isfinite(y_lstm_f) & np.isfinite(y_base_f))
        y_true_c, y_rnn_c = y_true_f[mascara], y_rnn_f[mascara]
        y_lstm_c, y_base_c = y_lstm_f[mascara], y_base_f[mascara]

        rmse_base = np.sqrt(mean_squared_error(y_true_c, y_base_c))
        mae_base = mean_absolute_error(y_true_c, y_base_c)
        rmse_rnn = np.sqrt(mean_squared_error(y_true_c, y_rnn_c))
        mae_rnn = mean_absolute_error(y_true_c, y_rnn_c)
        rmse_lstm = np.sqrt(mean_squared_error(y_true_c, y_lstm_c))
        mae_lstm = mean_absolute_error(y_true_c, y_lstm_c)

        print("\nComparativa de Métricas (test):")
        print(f"1. Baseline Estacional (Año Anterior) -> RMSE: {rmse_base:.2f} | MAE: {mae_base:.2f}")
        print(f"2. Modelo Base (SimpleRNN)            -> RMSE: {rmse_rnn:.2f} | MAE: {mae_rnn:.2f}")
        print(f"3. Modelo Principal (LSTM)            -> RMSE: {rmse_lstm:.2f} | MAE: {mae_lstm:.2f}")

        return {
            'metrics': {
                'baseline': {'rmse': rmse_base, 'mae': mae_base},
                'rnn': {'rmse': rmse_rnn, 'mae': mae_rnn},
                'lstm': {'rmse': rmse_lstm, 'mae': mae_lstm},
            },
            'y_true': y_true,
            'y_pred_rnn': y_pred_rnn,
            'y_pred_lstm': y_pred_lstm,
        }

    # ------------------------------------------------------------------
    # Uso posterior
    # ------------------------------------------------------------------
    def predecir_semanas(self, ultimos_datos_scaled_x):
        batch = ultimos_datos_scaled_x[-self.length:].reshape((1, self.length, len(self.feature_cols)))
        pred_scaled = self.model_lstm.predict(batch, verbose=0)
        return np.maximum(0, self._desescalar_y(pred_scaled)[0])

    def guardar(self, carpeta="../models"):
        """Guarda el LSTM y los DOS scalers que necesita la API."""
        carpeta = Path(carpeta)
        carpeta.mkdir(parents=True, exist_ok=True)
        self.model_lstm.save(str(carpeta / "lstm_casos.keras"))
        joblib.dump(self.scaler_x, carpeta / "scaler_lstm.joblib")
        joblib.dump(self.scaler_y, carpeta / "scaler_y_lstm.joblib")
        print(f"✓ Guardados en {carpeta.resolve()}")