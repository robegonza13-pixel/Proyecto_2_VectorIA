import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split
import hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import traceback  # Herramienta para detectar la línea exacta del error

class AuditorDataset:
    def __init__(self, ruta_dataset):
        self.ruta = Path(ruta_dataset)
        self.duplicadas = []
        self.corruptas = []
        self.pequenas = []
        self.borrosas = []
        self.mal_iluminadas = []
        self.df_metadatos = None

    def auditor_imagenes(self, min_res=100, umbral_borrosidad=30.0, umbral_oscuro=15.0, umbral_claro=240.0):
        """Audita el dataset filtrando imágenes corruptas, duplicadas, pequeñas o con mala iluminación."""
        registros = []
        hashes_vistos = set()

        for clase in self.ruta.iterdir():
            if clase.is_dir():
                for imagen in clase.iterdir():
                    if imagen.is_file():
                        try:
                            # 1. Integridad estructural y liberación de RAM
                            with Image.open(imagen) as img_temp:
                                img_temp.verify()

                            # 2. Duplicados exactos
                            hash_archivo = hashlib.md5(imagen.read_bytes()).hexdigest()
                            if hash_archivo in hashes_vistos:
                                self.duplicadas.append(imagen)
                                continue

                            # 3. Extracción de dimensiones y resolución mínima
                            with Image.open(imagen) as img_abierta:
                                img_pil = img_abierta.convert('RGB')
                                ancho, alto = img_pil.size

                            if ancho < min_res or alto < min_res:
                                self.pequenas.append(imagen)
                                continue

                            # Convertir a OpenCV solo para evaluar calidad visual (borrosidad/luz)
                            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                            gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

                            # 4. Filtro de Borrosidad
                            borrosidad = cv2.Laplacian(gris, cv2.CV_64F).var()
                            if borrosidad < umbral_borrosidad:
                                self.borrosas.append(imagen)
                                continue

                            # 5. Filtro de Iluminación extrema
                            iluminacion = np.mean(gris)
                            if iluminacion < umbral_oscuro or iluminacion > umbral_claro:
                                self.mal_iluminadas.append(imagen)
                                continue

                            # Guardamos los metadatos de la imagen original válida
                            registros.append({
                                "ruta": str(imagen),
                                "especie": clase.name,
                                "ancho_original": ancho,
                                "alto_original": alto,
                                "ancho_recorte": ancho,  # Mantenemos la llave para no romper la clase EdaImagenes
                                "alto_recorte": alto,    # Mantenemos la llave para no romper la clase EdaImagenes
                                "borrosidad": round(borrosidad, 2),
                                "iluminacion": round(iluminacion, 2)
                            })
                            hashes_vistos.add(hash_archivo)


                        except Exception as e:

                            print(f"\n❌ Error DETECTADO en la imagen {imagen.name}: {e}")
                            traceback.print_exc()
                            self.corruptas.append(imagen)
                            continue  # sigue con la siguiente imagen de la misma clase

        self.df_metadatos = pd.DataFrame(registros)

        print(f"✓ Auditoría finalizada (Sin recortes artificiales):")
        print(f"  - Imágenes válidas para entrenar: {len(self.df_metadatos)}")
        print(f"  - Descartadas (Corruptas: {len(self.corruptas)} | Duplicadas: {len(self.duplicadas)} | Pequeñas: {len(self.pequenas)} | Borrosas: {len(self.borrosas)} | Luz extrema: {len(self.mal_iluminadas)})")

        return self.df_metadatos

class EdaImagenes:
    """Realiza un analisis estadistico y visual con los datos de las imagenes validas
    Atributos:
        df_metadatos (DataFrame): Copia del inventario de imagenes limpias
    """

    def __init__(self, df_metadatos):
        self.df_metadatos = df_metadatos.copy()
        # Calculamos la relacion de aspecto (ancho/alto) para evaluar deformaciones al reescalar
        self.df_metadatos['aspect_ratio'] = self.df_metadatos['ancho_recorte'] / self.df_metadatos['alto_recorte']

    def reporte_estadistico_general(self):
        """Se muestran estadisticos basicos del dataset recibido"""
        print(f"El dataset cuenta con una cantidad de imagenes de:\n {self.df_metadatos['especie'].value_counts()}\n"
              f"Con un total de {len(self.df_metadatos)} imagenes \n")
        print(
            f"Porcentualmente corresponde a:\n {(self.df_metadatos['especie'].value_counts(normalize=True) * 100).round(2)}\n")
        print(
            f"El promedio de las dimensiones por clase es de:\n {(self.df_metadatos.groupby('especie')[['alto_recorte', 'ancho_recorte']].mean()).astype(int)}\n")
        print(
            f"El promedio de las dimensiones global es de: \n{(self.df_metadatos[['alto_recorte', 'ancho_recorte']].mean()).astype(int)}\n")

    def graficar_distribucion_clases(self):
        """Grafico de barras para visualizar el balance de clases de mosquitos"""
        plt.figure(figsize=(8, 5))

        conteos = self.df_metadatos['especie'].value_counts()
        conteos.plot(kind='bar', color=['#2b5c8f', '#4682b4', '#e74c3c'])

        plt.title("Distribución de Imágenes por Especie de Mosquito")
        plt.xlabel("Especie")
        plt.ylabel("Cantidad de Imágenes")
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def graficar_dimensiones(self):
        """Diagrama de dispersion para ver que tan dispersas estan las resoluciones"""
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.df_metadatos['ancho_recorte'],
            self.df_metadatos['alto_recorte'],
            alpha=0.4,
            color='#2b5c8f'
        )
        plt.title("Distribución de Dimensiones de las Imágenes (Ancho vs Alto)")
        plt.xlabel("Ancho (px)")
        plt.ylabel("Alto (px)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def graficar_relacion_aspecto(self):
        """Histograma para comprobar si la mayoria son cuadradas (1.0) o panoramicas"""
        plt.figure(figsize=(8, 5))
        plt.hist(
            self.df_metadatos['aspect_ratio'],
            bins=30,
            color='#2b5c8f',
            edgecolor='black',
            alpha=0.7
        )
        plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Aspecto Cuadrado (1:1)')
        plt.title("Distribución de la Relación de Aspecto (Ancho / Alto)")
        plt.xlabel("Relación de Aspecto")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def mostrar_galeria(self, n_por_clase=3):
        """Muestra una muestra aleatoria de imagenes por clase para inspeccion visual"""
        especies = self.df_metadatos['especie'].unique()
        filas = len(especies)
        columnas = n_por_clase
        fig, axes = plt.subplots(nrows=filas, ncols=columnas, squeeze=False, figsize=(columnas * 4, filas * 3.5))
        for i, especie in enumerate(especies):
            fotos_especie = self.df_metadatos[self.df_metadatos['especie'] == especie]
            muestras = fotos_especie.sample(n=n_por_clase)
            for j, (_, fila) in enumerate(muestras.iterrows()):
                ax = axes[i, j]
                img = Image.open(fila['ruta'])
                ax.imshow(img)
                ax.set_title(f"{especie}\n{fila['ancho_recorte']}x{fila['alto_recorte']} px", fontsize=10)
                ax.axis('off')
        plt.tight_layout()
        plt.show()


class DivisorDatos:
    """Divide datos en train, val y test de forma estratificada y copia a disco"""

    def __init__(self, df_metadatos, ruta_salida, seed=42):
        self.df_metadatos = df_metadatos
        self.ruta_salida = Path(ruta_salida)
        self.seed = seed

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def dividir_datos(self):
        """Se dividen los datos en entrenamiento (80%), validacion (10%) y test (10%) manteniendo proporciones"""
        self.train_df, temp_df = train_test_split(
            self.df_metadatos,
            test_size=0.2,
            stratify=self.df_metadatos['especie'],
            random_state=self.seed
        )

        self.val_df, self.test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df['especie'],
            random_state=self.seed
        )

        print(f"✓ División estratificada completada:")
        print(f"  - Train: {len(self.train_df)} imágenes ({len(self.train_df) / len(self.df_metadatos) * 100:.1f}%)")
        print(f"  - Val:   {len(self.val_df)} imágenes ({len(self.val_df) / len(self.df_metadatos) * 100:.1f}%)")
        print(f"  - Test:  {len(self.test_df)} imágenes ({len(self.test_df) / len(self.df_metadatos) * 100:.1f}%)")

    def copiar_a_disco(self):
        """Copia fisicamente las imagenes organizadas en carpetas train/val/test y por especie"""
        conjuntos = [
            ('train', self.train_df),
            ('val', self.val_df),
            ('test', self.test_df)
        ]
        for nombre_split, df_split in conjuntos:
            for _, fila in df_split.iterrows():
                carpeta_destino = self.ruta_salida / nombre_split / fila['especie']
                carpeta_destino.mkdir(parents=True, exist_ok=True)
                nombre_archivo = Path(fila['ruta']).name
                shutil.copy2(fila['ruta'], carpeta_destino / nombre_archivo)
        print("✓ Todas las imágenes fueron organizadas exitosamente en disco.")
