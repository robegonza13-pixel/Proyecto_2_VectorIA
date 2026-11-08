from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

class EntrenarMobileNetV2:
    """Implementa Transfer Learning y Fine-Tuning usando MobileNetV2 pre-entrenado en ImageNet.
    Atributos:
        data_dir (Path): Ruta donde se encuentran las carpetas train/val/test
        img_size (tuple): Dimension de entrada requerida por MobileNetV2 (por defecto 224x224)
        batch_size (int): Tamano del lote de imagenes
        train_ds (tf.data.Dataset): Dataset de entrenamiento optimizado
        val_ds (tf.data.Dataset): Dataset de validacion optimizado
        test_ds (tf.data.Dataset): Dataset de prueba optimizado
        class_names (list): Nombres de las clases detectadas
        base_model (keras.Model): Red MobileNetV2 base sin su clasificador original
        modelo (keras.Model): Modelo completo con cabeza clasificadora para mosquitos
        historia_fase1 (keras.callbacks.History): Historial de entrenamiento de la Fase 1
        historia_fase2 (keras.callbacks.History): Historial de entrenamiento de la Fase 2 (Fine-tuning)
    """
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_names = []
        self.base_model = None
        self.modelo = None
        self.historia_fase1 = None
        self.historia_fase2 = None

    def cargar_datos(self):

        AUTOTUNE = tf.data.AUTOTUNE

        # 1. Cargamos Train y extraemos los nombres de clases
        train_ds_raw = keras.utils.image_dataset_from_directory(
            self.data_dir / "train",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=True
        )
        self.class_names = train_ds_raw.class_names

        # 2. Cargamos Val sin barajar
        val_ds_raw = keras.utils.image_dataset_from_directory(
            self.data_dir / "val",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )

        # 3. Cargamos Test sin barajar para mantener orden de evaluacion
        test_ds_raw = keras.utils.image_dataset_from_directory(
            self.data_dir / "test",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )

        # Aplicamos la normalizacion [-1, 1], prefetch para maxima velocidad de entrenamiento
        self.train_ds = train_ds_raw.prefetch(AUTOTUNE)
        self.val_ds = val_ds_raw.prefetch(AUTOTUNE)
        self.test_ds = test_ds_raw.prefetch(AUTOTUNE)

        print(f"✓ Datos cargados exitosamente. Clases: {self.class_names}")

    def construir_modelo(self):
        """Descarga MobileNetV2 con pesos de ImageNet y acopla un clasificador denso personalizado"""
        # 1. Cargar el cuerpo de MobileNetV2 sin la capa densa final (include_top=False)
        self.base_model = MobileNetV2(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights='imagenet'
        )

        # Congelamos todos los pesos para la Fase 1 (usamos el modelo solo como extractor de caracteristicas)
        self.base_model.trainable = False

        # 2. Construccion con la API funcional de Keras
        entradas = layers.Input(shape=(self.img_size[0], self.img_size[1], 3))

        # Data augmentation para mejorar generalizacion en orientaciones arbitrarias
        x = layers.RandomFlip("horizontal_and_vertical")(entradas)
        x = layers.RandomRotation(0.25)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomContrast(0.25)(x)
        x = layers.RandomBrightness(0.2)(x)

        # Conversión de 0-255 a [-1, 1] (lo mismo que hace preprocess_input de MobileNetV2)
        x = layers.Rescaling(1. / 127.5, offset=-1)(x)

        # Base pre-entrenada (training=False asegura que BatchNorm mantenga las estadisticas de ImageNet)
        x = self.base_model(x, training=False)

        # Cabeza de clasificacion adaptada a nuestras 3 especies de mosquitos
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        salidas = layers.Dense(3, activation='softmax')(x)

        self.modelo = keras.Model(inputs=entradas, outputs=salidas, name="MobileNetV2_Mosquitos")
        self.modelo.summary()
        return self.modelo

    def calcular_pesos_clases(self):
        """Calcula ponderaciones de clase para compensar el menor volumen de fotos de Culex"""
        train_path = self.data_dir / "train"
        conteos = [len(list((train_path / clase).glob("*.*"))) for clase in self.class_names]
        total = sum(conteos)
        n_clases = len(self.class_names)

        pesos = {i: total / (n_clases * count) for i, count in enumerate(conteos)}
        print(f"✓ Pesos calculados para balancear el entrenamiento:\n{pesos}")
        return pesos

    def entrenar_fase1(self, epochs=15, class_weights=None):
        """Fase 1: Entrena unicamente las capas densas nuevas manteniendo el cuerpo congelado"""
        Path('models').mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print("FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS (Base Congelada)")
        print("=" * 60)

        # Learning rate estandar (1e-3) porque la cabeza parte de pesos aleatorios
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.F1Score(average="macro", name="f1_macro")
            ]
        )

        callbacks = [
            # Si val_loss no mejora en 5 epocas, detiene para no sobreajustar la cabeza
            keras.callbacks.EarlyStopping(
                monitor='val_f1_macro',
                mode='max',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce la velocidad de aprendizaje si nos acercamos a una meseta
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_macro',
                mode='max',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            # Guarda el mejor checkpoint de la Fase 1
            keras.callbacks.ModelCheckpoint(
                filepath='models/mobilenetv2_fase1.keras',
                mode='max',
                monitor='val_f1_macro',
                save_best_only=True,
                verbose=1

            )
        ]

        self.historia_fase1 = self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )

        print("✓ Fase 1 completada exitosamente.")
        return self.historia_fase1

    def entrenar_fine_tuning(self, epochs_adicionales=20, capas_a_descongelar=40, class_weights=None):
        """Fase 2: Descongela las capas superiores de MobileNetV2 protegiendo el Batch Normalization"""
        Path('models').mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 60)
        print(f"FASE 2: FINE-TUNING (Descongelando últimas {capas_a_descongelar} capas)")
        print("=" * 60)

        # 1. Habilitamos entrenamiento en base_model y volvemos a congelar las capas más tempranas
        self.base_model.trainable = True
        for capa in self.base_model.layers[:-capas_a_descongelar]:
            capa.trainable = False

        # REGLA DE ORO: Nunca descongelar BatchNormalization en Fine-Tuning para no destruir estadísticas previas
        for capa in self.base_model.layers:
            if isinstance(capa, keras.layers.BatchNormalization):
                capa.trainable = False

        entrenables = sum(1 for c in self.base_model.layers if c.trainable)
        print(f"Capas entrenables en base_model: {entrenables} (Excluyendo BatchNormalization)")

        # 2. Recompilamos con Learning Rate del 10% respecto a la Fase 1 (1e-4)
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.F1Score(average="macro", name="f1_macro")
            ]
        )

        callbacks = [
            # Paciencia extendida a 8 para permitir que el ajuste fino micro-converja sin cortarse
            keras.callbacks.EarlyStopping(
                monitor='val_f1_macro',
                mode='max',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            # Reductor de tasa de aprendizaje más tolerante
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_f1_macro',
                mode='max',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            # Guarda el modelo definitivo listo para producción
            keras.callbacks.ModelCheckpoint(
                filepath='models/mobilenetv2_final.keras',
                monitor='val_f1_macro',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]

        # Continuamos el contador de épocas desde el final de la Fase 1
        epoca_inicial = len(self.historia_fase1.history['loss']) if self.historia_fase1 else 0
        total_epocas = epoca_inicial + epochs_adicionales

        self.historia_fase2 = self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=total_epocas,
            initial_epoch=epoca_inicial,
            callbacks=callbacks,
            class_weight=class_weights
        )

        print("✓ Fase 2 (Fine-Tuning) completada exitosamente.")
        return self.historia_fase2

    def guardar_modelo(self, ruta='models/mobilenetv2_final.keras'):
        """Guarda los pesos y arquitectura final en la carpeta models"""
        Path('models').mkdir(parents=True, exist_ok=True)
        self.modelo.save(ruta)
        print(f"✓ Modelo guardado exitosamente en {ruta}")

