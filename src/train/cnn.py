from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class EntrenarCNN:
    """Carga los datos, construye el modelo propio, calcula pesos por clase y entrena la red.
    Atributos:
        data_dir (Path): Ruta donde se almacenan las imagenes procesadas (data/processed)
        img_size (tuple): Dimension en la que el modelo procesara las imagenes (alto x ancho)
        batch_size (int): Cantidad de imagenes a procesar por lote
        train_ds (tf.data.Dataset): Lote de imagenes de entrenamiento
        val_ds (tf.data.Dataset): Lote de imagenes de validacion
        test_ds (tf.data.Dataset): Lote de imagenes de test
        class_names (list): Lista de clases de mosquitos detectadas en las carpetas
        modelo (keras.Model): Modelo secuencial de red neuronal convolucional
        historia (keras.callbacks.History): Registro de entrenamiento con metricas por epoca
    """
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_names = []
        self.modelo = None
        self.historia = None

    def cargar_datos(self):
        """Carga datos de entrenamiento, validacion y test desde las carpetas procesadas"""
        self.train_ds = keras.utils.image_dataset_from_directory(
            self.data_dir / "train",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",  # Etiquetas en formato one-hot para clasificacion multiclase
            shuffle=True  # Se barajan los datos de entrenamiento para evitar sesgos por orden
        )

        self.val_ds = keras.utils.image_dataset_from_directory(
            self.data_dir / "val",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False  # No se baraja para mantener correspondencia con las metricas
        )

        self.test_ds = keras.utils.image_dataset_from_directory(
            self.data_dir / "test",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False  # No se baraja para que coincida con las etiquetas reales en la evaluacion
        )

        self.class_names = self.train_ds.class_names
        print(f"✓ Datos cargados exitosamente. Clases: {self.class_names}")

        AUTOTUNE = tf.data.AUTOTUNE  # Le dice a tensorflow que gestione recursos segun la CPU/GPU disponible
        # prefetch precarga lotes
        self.train_ds = self.train_ds.prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(buffer_size=AUTOTUNE)

    def construir_modelo(self):
        """Construye las capas del modelo CNN devolviendo el modelo listo para compilar"""
        self.modelo = keras.Sequential([
            layers.Input(shape=(self.img_size[0], self.img_size[1], 3)),

            # Data Augmentation integrado en el modelo para evitar memorizacion y sobreajuste
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.5),
            layers.RandomZoom(height_factor=(-0.1, 0.1)),
            layers.RandomTranslation(height_factor=0.05, width_factor=0.05),


            # Normalizacion de pixeles (0 a 255 -> 0.0 a 1.0) dentro de la red para que sea portable
            layers.Rescaling(1. / 255),

            # Bloque Convolucional 1: Extraccion de bordes y caracteristicas iniciales
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Bloque Convolucional 2: Detecta formas intermedias y texturas
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Bloque Convolucional 3: Captura partes del cuerpo y patrones de escamas
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Promedia espacialmente cada mapa de caracteristicas reduciendo drásticamente parametros frente a Flatten
            layers.GlobalAveragePooling2D(),

            # Capa densa de interpretacion
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),  # Apaga neuronas al azar para forzar representaciones redundantes

            # Capa de salida: 3 neuronas con softmax que entregan probabilidades para cada especie
            layers.Dense(3, activation='softmax')
        ], name="CNN_Mosquitos_Propia")

        self.modelo.summary()
        return self.modelo

    def calcular_pesos_clases(self):
        """Calcula pesos inversamente proporcionales a la frecuencia de imagenes para balancear el aprendizaje"""
        train_path = self.data_dir / "train"
        conteos = [len(list((train_path / clase).glob("*.*"))) for clase in self.class_names]
        total = sum(conteos)
        n_clases = len(self.class_names)

        # Formula: total / (n_clases * cantidad_en_clase)
        pesos = {i: total / (n_clases * count) for i, count in enumerate(conteos)}
        print(f"✓ Pesos calculados para balancear el entrenamiento:\n{pesos}")
        return pesos

    def compilar_y_entrenar(self, epochs=40, class_weights=None):
        """Compila el modelo con optimizador, perdida y callbacks para luego entrenarlo"""
        Path('models').mkdir(parents=True, exist_ok=True)
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            # Label smoothing ayuda cuando hay clases muy similares (evita confianza extrema)
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.F1Score(average="macro", name="f1_macro")
            ]
        )

        callbacks = [
            # Detiene el entrenamiento si el loss de validacion no mejora tras 5 epocas y restaura el mejor
            keras.callbacks.EarlyStopping(
                monitor='val_f1_macro',
                mode='max',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce la tasa de aprendizaje a la mitad si se estanca 3 epocas para afinar convergencia
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_macro',
                mode='max',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # Guarda automaticamente el archivo del mejor modelo encontrado en disco
            keras.callbacks.ModelCheckpoint(
                filepath='models/cnn_propia_best.keras',
                monitor='val_f1_macro',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]

        print(f"Iniciando entrenamiento por máximo {epochs} épocas:")
        self.historia = self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights  # Aplica ponderacion calculada para evitar sesgo hacia especies mayoritarias
        )

        print("✓ Entrenamiento completado exitosamente.")
        return self.historia

    def guardar_modelo(self):
        """Guarda los pesos y arquitectura final en la carpeta models"""
        Path('models').mkdir(parents=True, exist_ok=True)
        self.modelo.save('models/cnn_propia.keras')
        print("✓ Modelo guardado en models/cnn_propia.keras")


class DiagnosticoEvaluacion:
    """Evalua el rendimiento del modelo, genera curvas de entrenamiento y reportes con matriz de confusion
    Atributos:
        modelo (keras.Model): Modelo ya entrenado a evaluar
        historia (keras.callbacks.History): Registro de metricas durante el entrenamiento
        test_ds (tf.data.Dataset): Conjunto de prueba nunca antes visto
        class_names (list): Nombres de las clases en orden
    """
    def __init__(self, modelo, historia, test_ds, class_names):
        self.modelo = modelo
        self.historia = historia
        self.test_ds = test_ds
        self.class_names = class_names

        self.y_true = None
        self.y_pred = None

    def graficar_metricas(self):
        """Grafica curvas de loss y accuracy comparando train vs validacion para detectar overfitting"""
        hist = self.historia.history
        epocas = range(1, len(hist['loss']) + 1)

        plt.figure(figsize=(14, 5))

        # 1. Grafico de Perdida (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(epocas, hist['loss'], 'b-', label='Pérdida Entrenamiento')
        plt.plot(epocas, hist['val_loss'], 'r--', label='Pérdida Validación')
        plt.title('Evolución de la Pérdida (Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Grafico de Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epocas, hist['accuracy'], 'b-', label='Precisión Entrenamiento')
        plt.plot(epocas, hist['val_accuracy'], 'r--', label='Precisión Validación')
        plt.title('Evolución de la Precisión (Accuracy)')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def evaluar_test(self):
        """Calcula predicciones en el conjunto test y muestra el reporte formal con F1-Score y matriz"""
        print("Evaluando el modelo en el conjunto de prueba (Test)...")

        # Extraemos etiquetas reales de cada lote convirtiendo one-hot a indice de clase
        y_true = []
        for _, etiquetas in self.test_ds:
            y_true.extend(np.argmax(etiquetas.numpy(), axis=1))
        self.y_true = np.array(y_true)

        # Obtenemos probabilidades predichas y seleccionamos la clase con mayor valor
        predicciones_prob = self.modelo.predict(self.test_ds)
        self.y_pred = np.argmax(predicciones_prob, axis=1)

        print("\n" + "=" * 60)
        print("REPORTE DE CLASIFICACIÓN (MÉTRICAS OFICIALES)")
        print("=" * 60)
        # classification_report nos da Precision, Recall y F1-score por clase y promedio macro
        reporte = classification_report(y_true, self.y_pred, target_names=self.class_names, digits=4)
        print(reporte)

        # Matriz de confusion para ver exactamente que especie se confunde con cual
        cm = confusion_matrix(y_true, self.y_pred)
        plt.figure(figsize=(7, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión - Test")
        plt.colorbar()
        tick_marks = np.arange(len(self.class_names))
        plt.xticks(tick_marks, self.class_names, rotation=45)
        plt.yticks(tick_marks, self.class_names)

        # Anotamos los numeros dentro de cada celda de la matriz
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.ylabel('Etiqueta Real')
        plt.xlabel('Predicción del Modelo')
        plt.tight_layout()
        plt.show()
