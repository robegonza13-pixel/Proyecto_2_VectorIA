from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class EntrenarMobileNetV2:
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
        def aplicar_preprocesamiento(imagen, etiqueta):
            return preprocess_input(imagen), etiqueta

        AUTOTUNE = tf.data.AUTOTUNE

        # 1. Cargamos Train y extraemos los nombres de clases sin releer el disco
        train_ds_raw = keras.utils.image_dataset_from_directory(
            self.data_dir / "train",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=True
        )
        self.class_names = train_ds_raw.class_names

        # 2. Cargamos Val
        val_ds_raw = keras.utils.image_dataset_from_directory(
            self.data_dir / "val",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )

        # 3. Cargamos Test
        test_ds_raw = keras.utils.image_dataset_from_directory(
            self.data_dir / "test",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )

        # Aplicamos preprocesamiento (-1 a 1) y optimización en memoria
        self.train_ds = train_ds_raw.map(aplicar_preprocesamiento).prefetch(AUTOTUNE)
        self.val_ds = val_ds_raw.map(aplicar_preprocesamiento).prefetch(AUTOTUNE)
        self.test_ds = test_ds_raw.map(aplicar_preprocesamiento).prefetch(AUTOTUNE)

        print(f"✓ Datos cargados exitosamente. Clases: {self.class_names}")

    def construir_modelo(self):
        # 1. Cargar el cuerpo de MobileNetV2 pre-entrenado en ImageNet
        self.base_model = MobileNetV2(
            input_shape=(self.img_size[0], self.img_size[1], 3),
            include_top=False,
            weights='imagenet'
        )

        # Congelamos los pesos para la Fase 1 (no se tocan)
        self.base_model.trainable = False

        # 2. Construcción con la API funcional de Keras
        entradas = layers.Input(shape=(self.img_size[0], self.img_size[1], 3))

        # Data augmentation
        x = layers.RandomFlip("horizontal")(entradas)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomTranslation(height_factor=0.1, width_factor=0.1)(x)
        x = layers.RandomContrast(0.1)(x)
        x = layers.RandomBrightness(0.1)(x)

        # Base pre-entrenada (training=False asegura que BatchNorm conserve las estadísticas de ImageNet)
        x = self.base_model(x, training=False)

        # Cabeza de clasificación personalizada
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        salidas = layers.Dense(3, activation='softmax')(x)

        self.modelo = keras.Model(inputs=entradas, outputs=salidas, name="MobileNetV2_Mosquitos")
        self.modelo.summary()
        return self.modelo


    def calcular_pesos_clases(self):
        train_path = self.data_dir / "train"
        conteos = [len(list((train_path / clase).glob("*.*"))) for clase in self.class_names]
        total = sum(conteos)
        n_clases = len(self.class_names)

        pesos = {i: total / (n_clases * count) for i, count in enumerate(conteos)}
        print(f"✓ Pesos calculados para balancear el entrenamiento:\n{pesos}")
        return pesos

    def entrenar_fase1(self, epochs=15, class_weights=None):
        print("\n" + "=" * 60)
        print("FASE 1: EXTRACCIÓN DE CARACTERÍSTICAS (Base Congelada)")
        print("=" * 60)

        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall")
            ]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/mobilenetv2_fase1.keras',
                monitor='val_loss',
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

    def entrenar_fine_tuning(self, epochs_adicionales=15, capas_a_descongelar=10, class_weights=None):
        print("\n" + "=" * 60)
        print(f"FASE 2: FINE-TUNING (Descongelando últimas {capas_a_descongelar} capas)")
        print("=" * 60)

        # 1. Descongelar el modelo base y volver a congelar las primeras
        self.base_model.trainable = True
        for capa in self.base_model.layers[:-capas_a_descongelar]:
            capa.trainable = False

        entrenables = sum(1 for c in self.base_model.layers if c.trainable)
        print(f"Capas entrenables en base_model: {entrenables} de {len(self.base_model.layers)}")

        # 2. Recompilar con Learning Rate diminuto (1e-5)
        self.modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall")
            ]
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/mobilenetv2_final.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Continuamos desde la época en la que quedó la Fase 1
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
