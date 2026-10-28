from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class EntrenarCNN:
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
        self.train_ds = keras.utils.image_dataset_from_directory(
            self.data_dir / "train",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=True
        )

        self.val_ds = keras.utils.image_dataset_from_directory(
            self.data_dir / "val",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )

        self.test_ds = keras.utils.image_dataset_from_directory(
            self.data_dir / "test",
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode="categorical",
            shuffle=False
        )

        self.class_names = self.train_ds.class_names
        print(f"✓ Datos cargados exitosamente. Clases: {self.class_names}")


        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)


    def construir_modelo(self):
        self.modelo = keras.Sequential([

            layers.Input(shape=(self.img_size[0], self.img_size[1], 3)),

            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),

            layers.Rescaling(1. / 255),

            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.GlobalAveragePooling2D(),

            layers.Dense(128, activation='relu'),

            layers.Dropout(0.5),

            layers.Dense(3, activation='softmax')
        ], name="CNN_Mosquitos_Propia")

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

    def compilar_y_entrenar(self, epochs=30, class_weights=None):
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
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath='models/cnn_propia.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        print(f"Iniciando entrenamiento por máximo {epochs} épocas:")
        self.historia= self.modelo.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights
        )

        print("✓ Entrenamiento completado exitosamente.")
        return self.historia