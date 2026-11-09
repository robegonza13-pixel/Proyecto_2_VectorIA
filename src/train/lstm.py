
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator


class LstmModel:
    def __init__(self, filepath, length=26, horizon=4):
        self.filepath = filepath
        self.length = length  # ventana de 26 semanas
        self.horizon = horizon  # predicción de las próximas 4 semanas
        self.scaler = MinMaxScaler()
        self.df = None
        self.scaled_train = None
        self.scaled_test = None
        self.feature_cols = None

        # modelos
        self.model_rnn = None
        self.model_lstm = None

    def preparar_datos(self, df, test_size=104, usar_lags = True):
        # se prepara un dataframe: creando rezagos del clima y variables temporales
        self.df = df.sort_values('fecha_inicio_semana').reset_index(drop=True).copy()

        # asegura el formato de la fecha sea date para extraer los componentes
        self.df['fecha_inicio_semana'] = pd.to_datetime(df['fecha_inicio_semana'])

        # se crea un componente temporal donde se verifica si cumple la instruccion de usar la fecha
        if 'semana_epi' in df.columns:
            df['semana_sin'] = np.sin(2 * np.pi * df['semana_epi'] / 52.0)
            df['semana_cos'] = np.cos(2 * np.pi * df['semana_epi'] / 52.0)

        # se crean los rezagos del clima y selecciona las columnas del csv que tienen relación
        if usar_lags:
            features_to_lag = ['lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']
            for col in features_to_lag:
                for lag in [1, 2, 3, 4]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)

        self.df = df.dropna().reset_index(drop=True)

        # se definen las columnas de entrada
        base_cols = ['casos_dengue', 'lluvia_mm', 'temp_max_c', 'temp_min_c', 'humedad_rel_pct']
        temporal = [c for c in ['semana_sin', 'semana_cos', 'semana_epi'] if c in df.columns]
        lag_cols = [col for col in df.columns if 'lag' in col]

        self.feature_cols = base_cols + temporal + lag_cols

        data_values = df[self.feature_cols].values

        # Split train / test
        train_data = data_values[:-test_size]
        test_data = data_values[-test_size:]

        # escalado con train ( esto evita que ocurra una fuga de datos)
        self.scaler.fit(train_data)
        scaled_train = self.scaler.transform(train_data)
        scaled_test = self.scaler.transform(test_data)

        return scaled_train, scaled_test

    def crear_secuencias(self, data):
        #genera secuencias temporales para el modelo
        x, y = [], []
        target_idx = 0  # 'casos_dengue' está en la columna 0

        for i in range(len(data) - self.length - self.horizon + 1):
            x.append(data[i: i + self.length])
            y.append(data[i + self.length: i + self.length + self.horizon, target_idx])

        return np.array(x), np.array(y)

    def entrenar(self, scaled_train, scaled_test, epochs=100, batch_size=16):
        # Se encarga de contruir y entrenar el modelo
        x_train, y_train = self.crear_secuencias(scaled_train)
        x_test, y_test = self.crear_secuencias(scaled_test)

        n_features = scaled_train.shape[1]
        early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

        # modelo base: contiene SimpleRNN
        print("Entrenamiento de baseline: SimpleRNN")
        self.model_rnn = Sequential(
            [SimpleRNN(32, activation = 'relu', input_shape = (self.length, n_features)),
             Dense(self.horizon)])

        self.model_rnn.compile(optimizer='adam', loss='mse')
        self.model_rnn.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)

        # modelo principal: LSTM
        print("Entrenamiento del modelo LSTM")
        self.model_lstm = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=(self.length, n_features)),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(self.horizon)
        ])

        self.model_lstm.compile(optimizer='adam', loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

        history_lstm = self.model_lstm.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(x_test, y_test), callbacks=[early_stop], verbose=0)

        print("Entrenamiento finalizado con éxito")

        return history_lstm

    def evaluar_metricas(self, scaled_test, test_size=104):
        # calcula las métricas rmse y mae
        x_test, y_test_scaled = self.crear_secuencias(scaled_test)

        # predicción normalizada [0, 1]
        y_pred_rnn_scaled = self.model_rnn.predict(x_test)
        y_pred_lstm_scaled = self.model_lstm.predict(x_test)

        # función para desescalar y obtener los casos reales de dengue
        def desescalar(datos_scaled):
            n_samples = datos_scaled.shape[0]
            n_features = len(self.feature_cols)
            dummy = np.zeros((n_samples * self.horizon, n_features))
            dummy[:, 0] = datos_scaled.flatten()
            inv = self.scaler.inverse_transform(dummy)
            return inv[:, 0].reshape(n_samples, self.horizon)

        y_true = desescalar(y_test_scaled)
        y_pred_rnn = desescalar(y_pred_rnn_scaled)
        y_pred_lstm = desescalar(y_pred_lstm_scaled)

        # calculo de baseline estacional
        # se realiza construyendo un array de baseline estacional para el test set
        y_baseline_estacional = np.zeros_like(y_true)

        total_len = len(self.df)
        test_start_idx = total_len - test_size

        for i in range(len(y_true)):
            # indice de la semana actual dentro del dataframe
            # al crear secuencias, el primer target empieza en test_start_idx
            abs_idx = test_start_idx + self.length + i

            #buscamos los casos de 52 semanas (1 año) antes
            target_historical_idx = abs_idx - 52

            if  i >=0:
                # tomamos un bloque de 4 semanas de un añp atras
                y_baseline_estacional[i] =  self.df['casos_dengue'].iloc[target_historical_idx : target_historical_idx + self.horizon].values
            else:
                y_baseline_estacional[i] = y_true[0]

        # calculo de las métricas de SimpleRNN
        rmse_rnn = np.sqrt(mean_squared_error(y_true, y_pred_rnn))
        mae_rnn = mean_absolute_error(y_true, y_pred_rnn)

        # calculo de las métricas de LSTM
        rmse_lstm = np.sqrt(mean_squared_error(y_true, y_pred_lstm))
        mae_lstm = mean_absolute_error(y_true, y_pred_lstm)

        # calculo de métricas baseline
        rmse_base = np.sqrt(mean_squared_error(y_true, y_baseline_estacional))
        mae_base = mean_absolute_error(y_true, y_baseline_estacional)

        print(" Se muestra una comparativa de las métricas")
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

    def predecir_semanas(self, ultimos_datos_scaled):
        """Predice las próximas 4 semanas utilizando el modelo LSTM principal."""
        batch_actual = ultimos_datos_scaled[-self.length:].reshape((1, self.length, len(self.feature_cols)))
<<<<<<< HEAD
        pred_scaled = self.model_lstm.predict(batch_actual)[0]  # CORREGIDO: era self.model (inexistente)
=======
        pred_scaled = self.model_lstm.predict(batch_actual)[0]
>>>>>>> ba4631f009f736393f092c8b3f0995524c9396bf

        # Desescalar la salida de las 4 semanas de casos de dengue
        dummy = np.zeros((self.horizon, len(self.feature_cols)))
        dummy[:, 0] = pred_scaled
        pred_inversos = self.scaler.inverse_transform(dummy)[:, 0]

        return pred_inversos
