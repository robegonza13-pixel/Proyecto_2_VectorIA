
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


class LstmModel:
    def __init__(self, length=26, horizon=4):
        self.length = length  # ventana de 26 semanas
        self.horizon = horizon  # predicción de las próximas 4 semanas
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.df = None
        self.feature_cols = None

        self.model_rnn = None
        self.model_lstm = None

    def preparar_datos(self, df, test_size=104, usar_lags=True):
        # se trabaja sobre una copia limpia del DataFrame
        data = df.sort_values('fecha_inicio_semana').reset_index(drop=True).copy()
        data['fecha_inicio_semana'] = pd.to_datetime(data['fecha_inicio_semana'])

        # contiene los componentes temporales a traves de transacciones trigometricas
        if 'semana_epi' in data.columns:
            data['semana_sin'] = np.sin(2 * np.pi * data['semana_epi'] / 52.0)
            data['semana_cos'] = np.cos(2 * np.pi * data['semana_epi'] / 52.0)

        # creación de rezagos climaticos, para capturar los efectos que se retardaron
        if usar_lags:
            features_to_lag = ['lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']
            for col in features_to_lag:
                if col in data.columns:
                    for lag in [1, 2, 3, 4]:
                        data[f'{col}_lag{lag}'] = data[col].shift(lag)

        # eliminación de valores nulos genrados por la desfase temporal
        self.df = data.dropna().reset_index(drop=True)

        base_cols = ['casos_dengue', 'lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']
        base_cols = [c for c in base_cols if c in self.df.columns]
        temporal = [c for c in ['semana_sin', 'semana_cos', 'semana_epi'] if c in self.df.columns]
        lag_cols = [c for c in self.df.columns if 'lag' in c]

        self.feature_cols = base_cols + temporal + lag_cols

        X_raw = self.df[self.feature_cols].values
        Y_raw = self.df[['casos_dengue']].values

        # contiene la división (split) secuencial incluyendo la ventana previa en el test
        train_size = len(self.df) - test_size

        X_train_raw = X_raw[:train_size]
        X_test_raw = X_raw[train_size - self.length:]

        Y_train_raw = Y_raw[:train_size]
        Y_test_raw = Y_raw[train_size - self.length:]

        # escalado independiente para evitar que ocurran fujas de datos
        # (se ajusta solamente con los datos de entrenamiento)
        scaled_x_train = self.scaler_x.fit_transform(X_train_raw)
        scaled_x_test = self.scaler_x.transform(X_test_raw)

        scaled_y_train = self.scaler_y.fit_transform(Y_train_raw)
        scaled_y_test = self.scaler_y.transform(Y_test_raw)

        return scaled_x_train, scaled_y_train, scaled_x_test, scaled_y_test

    def crear_secuencias(self, data_x, data_y):
        # genera secuencias temporales en el modelo
        x, y = [], []
        for i in range(len(data_x) - self.length - self.horizon + 1):
            x.append(data_x[i: i + self.length])
            y.append(data_y[i + self.length: i + self.length + self.horizon, 0])
        return np.array(x), np.array(y)

    def entrenar(self, scaled_x_train, scaled_y_train, scaled_x_test, scaled_y_test, epochs=150, batch_size=16):
        # se encarga de construir, copilar y entrenar el modelo de forma independiente
        x_train, y_train = self.crear_secuencias(scaled_x_train, scaled_y_train)
        x_test, y_test = self.crear_secuencias(scaled_x_test, scaled_y_test)

        n_features = scaled_x_train.shape[1]

        # se añaden callbacks para evitar que ocurran paradas prematuras y ajustar la tasa de aprendizaje
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]

        # modelo base: contiene SimpleRNN
        print("Entrenando Modelo Base (SimpleRNN)...")
        self.model_rnn = Sequential([
            tf.keras.Input(shape=(self.length, n_features)),
            SimpleRNN(32, activation='tanh'),
            Dense(self.horizon)
        ])
        self.model_rnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model_rnn.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=0
        )

        # modelo principal: LSTM
        print("Entrenando Modelo Principal (LSTM)...")
        self.model_lstm = Sequential([
            tf.keras.Input(shape=(self.length, n_features)),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.1),
            LSTM(32, activation='tanh'),
            Dense(self.horizon)
        ])
        self.model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        history_lstm = self.model_lstm.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=callbacks,
            verbose=0
        )

        print("Entrenamiento finalizado con éxito.")
        return history_lstm

    def evaluar_metricas(self, scaled_x_test, scaled_y_test, test_size=104):
        # evalúa el desempeño de los modelos entrenados frente el baseline estacional
        # calcula las métricas de error rmse y mae
        x_test, y_test_scaled = self.crear_secuencias(scaled_x_test, scaled_y_test)

        # realiza predicciones en escala [0, 1]
        y_pred_rnn_scaled = self.model_rnn.predict(x_test)
        y_pred_lstm_scaled = self.model_lstm.predict(x_test)

        # desescalado directo sobre Y
        y_true = self.scaler_y.inverse_transform(y_test_scaled)
        y_pred_rnn = self.scaler_y.inverse_transform(y_pred_rnn_scaled)
        y_pred_lstm = self.scaler_y.inverse_transform(y_pred_lstm_scaled)

        # se construye el de baseline estacional (semanas del año anterior)
        y_baseline_estacional = np.zeros_like(y_true)
        test_start_idx = len(self.df) - test_size

        for i in range(len(y_true)):
            abs_idx = test_start_idx + i
            target_historical_idx = abs_idx - 52
            if target_historical_idx >= 0 and (target_historical_idx + self.horizon) <= len(self.df):
                y_baseline_estacional[i] = self.df['casos_dengue'].iloc[
                    target_historical_idx: target_historical_idx + self.horizon
                ].values

        # aplanación de arreglos para evitar casos de dengues negativos
        y_true_flat = y_true.flatten()
        y_pred_rnn_flat = np.maximum(0, y_pred_rnn.flatten())  # Evitar valores negativos en conteo
        y_pred_lstm_flat = np.maximum(0, y_pred_lstm.flatten())
        y_baseline_flat = y_baseline_estacional.flatten()

        # se incluye un filtro para evitar posibles inconsistencias
        mascara = np.isfinite(y_true_flat) & np.isfinite(y_pred_lstm_flat) & np.isfinite(y_pred_rnn_flat) & np.isfinite(
            y_baseline_flat)

        y_true_clean = y_true_flat[mascara]
        y_pred_rnn_clean = y_pred_rnn_flat[mascara]
        y_pred_lstm_clean = y_pred_lstm_flat[mascara]
        y_baseline_clean = y_baseline_flat[mascara]

        # calculo de métricas finales
        rmse_base = np.sqrt(mean_squared_error(y_true_clean, y_baseline_clean))
        mae_base = mean_absolute_error(y_true_clean, y_baseline_clean)

        rmse_rnn = np.sqrt(mean_squared_error(y_true_clean, y_pred_rnn_clean))
        mae_rnn = mean_absolute_error(y_true_clean, y_pred_rnn_clean)

        rmse_lstm = np.sqrt(mean_squared_error(y_true_clean, y_pred_lstm_clean))
        mae_lstm = mean_absolute_error(y_true_clean, y_pred_lstm_clean)

        print("\nComparativa de Métricas Reales:")
        print(f"1. Baseline Estacional (Año Anterior) -> RMSE: {rmse_base:.2f} | MAE: {mae_base:.2f}")
        print(f"2. Modelo Base (SimpleRNN)            -> RMSE: {rmse_rnn:.2f} | MAE: {mae_rnn:.2f}")
        print(f"3. Modelo Principal (LSTM)            -> RMSE: {rmse_lstm:.2f} | MAE: {mae_lstm:.2f}")

        return {
            'metrics': {
                'baseline': {'rmse': rmse_base, 'mae': mae_base},
                'rnn': {'rmse': rmse_rnn, 'mae': mae_rnn},
                'lstm': {'rmse': rmse_lstm, 'mae': mae_lstm}
            },
            'y_true': y_true,
            'y_pred_rnn': y_pred_rnn,
            'y_pred_lstm': y_pred_lstm
        }

    def predecir_semanas(self, ultimos_datos_scaled_x):
        # predice las próximas 4 semanas utilizando la red LSTM
        batch_actual = ultimos_datos_scaled_x[-self.length:].reshape((1, self.length, len(self.feature_cols)))
        pred_scaled = self.model_lstm.predict(batch_actual)
        pred_inversos = self.scaler_y.inverse_transform(pred_scaled)[0]
        return np.maximum(0, pred_inversos)