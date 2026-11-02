from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import shutil
from sklearn.model_selection import train_test_split


class ExploracionLimpiezaImagenes:
    """Revisa la cantidad de imagenes, verificar duplicadas y corruptas.
    Atributos:
    ruta(Path): Ruta de la carpeta donde se almacenan las imagenes sin procesar
    df_metadatos(Dataframe): Inventario de las imagenes existentes con sus respectivas rutas
    corrutas(Lista): Guarda ruta de imagen corrupta para excluirlas y registrarlas
    duplicadas(Lista): Guarda ruta de imagen duplicada para excluirlas y registrarlas
    """
    def __init__(self, ruta):
        """Inicializa la clase con la ruta de las imagenes crudas"""
        self.ruta = Path(ruta)
        self.df_metadatos = None
        self.corruptas = []
        self.duplicadas = []

    def auditor_imagenes(self):
        """Recorre cada carpeta de especie, verifica que no esté corrupta, calcula su hash para ver que no este duplicada
            y registra sus metadatos en un dataframe"""
        registros = []
        hashes_vistos = set()
        for clase in self.ruta.iterdir():
            if clase.is_dir():
                for imagen in clase.iterdir():
                    if imagen.is_file():
                        try:
                            Image.open(imagen).verify() # Verificamos la integridad de la imagen
                            hash_archivo = hashlib.md5(imagen.read_bytes()).hexdigest() # Se usa para detectar duplicado
                            if hash_archivo in hashes_vistos:
                                self.duplicadas.append(imagen)
                            else:
                                img = Image.open(imagen) #Otro open porque el verify deja el registro inutilizado
                                ancho , alto = img.size
                                modo = img.mode
                                formato = img.format
                                registros.append({
                                    "ruta": str(imagen),
                                    "especie": clase.name,
                                    "ancho": ancho,
                                    "alto": alto,
                                    "modo": modo,
                                    "formato": formato
                                })
                                hashes_vistos.add(hash_archivo)
                        except Exception: # Cualquier fallo se toma como corrupta
                            self.corruptas.append(imagen)

        self.df_metadatos = pd.DataFrame(registros)
        print(f"✓ Auditoría finalizada:")
        print(f"  - Imágenes válidas: {len(self.df_metadatos)}")
        print(f"  - Duplicadas detectadas: {len(self.duplicadas)}")
        print(f"  - Corruptas detectadas: {len(self.corruptas)}")

        return self.df_metadatos


class EdaImagenes:
    """Realiza un analisis estadistico de con datos de las imagenes válidas
        Atributos:
        metadatos(Dataframe): Copia de dataset inventario de las imagenes existentes recibido"""
    def __init__(self, df_metadatos):
        self.df_metadatos = df_metadatos.copy()
        self.df_metadatos['aspect_ratio'] = self.df_metadatos['ancho'] / self.df_metadatos['alto']

    def reporte_estadistico_general(self):
        """Se muestran estadisticos basicos del dataset recibido"""
        print(f"El dataset de cuenta con una cantidad de imagenes de:\n {self.df_metadatos['especie'].value_counts()}\n"
              f"Con un total de {len(self.df_metadatos)} imagenes \n")
        print(f"Porcentualmente corresponde a:\n {(self.df_metadatos['especie'].value_counts(normalize=True) * 100).round(2)}\n")
        print(f"El conteo de canales es de {self.df_metadatos['modo'].value_counts()}\n")
        print(f"El promedio de las dimensiones por clase es de: {(self.df_metadatos.groupby('especie')[['alto', 'ancho']].mean()).astype(int)}\n")
        print(f"El promedio de las dimensiones global es de: \n{(self.df_metadatos[['alto', 'ancho']].mean()).astype(int)}\n")

    def graficar_distribucion_clases(self):

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
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.df_metadatos['ancho'],
            self.df_metadatos['alto'],
            alpha=0.4,  # alpha le da transparencia a los puntos para ver dónde se amontonan
            color='#2b5c8f'
        )
        plt.title("Distribución de Dimensiones de las Imágenes (Ancho vs Alto)")
        plt.xlabel("Ancho (px)")
        plt.ylabel("Alto (px)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def graficar_relacion_aspecto(self):
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
        plt.legend()  # Para que se vea qué significa la línea roja
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    def mostrar_galeria(self, n_por_clase=3):
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
                ax.set_title(f"{especie}\n{fila['ancho']}x{fila['alto']} px", fontsize=10)
                ax.axis('off')
        plt.tight_layout()
        plt.show()

class DivisorDatos:
    """Divide datos procesados en train, test y val y guarda en el disco los datos divididos
        Atributos:
            df_metadatos(Dataframe): Inventario de imagenes recibido
            ruta_salida(Path): Ruta de la carpeta donde se almacenan las imagenes procesadas
            seed(int): Semilla para recrear division y dar trazabilidad
            train_df(Dataframe): Inventario de imagenes de entrenamiento
            val_df(Dataframe): Inventario de imagenes de validacion
            test_df(Dataframe): Inventario de imagenes de test"""

    def __init__(self, df_metadatos, ruta_salida, seed=42):
        self.df_metadatos = df_metadatos
        self.ruta_salida = Path(ruta_salida)
        self.seed = seed

        self.train_df = None
        self.val_df = None
        self.test_df = None

    def dividir_datos(self):
        """Se dividen los datos en entrenamientos, test y validacion"""
        self.train_df, temp_df = train_test_split(
            self.df_metadatos,
            test_size=0.2, # Se dividen los datos en 80/20
            stratify=self.df_metadatos['especie'],
            random_state=self.seed
        )

        self.val_df, self.test_df = train_test_split(
            temp_df,
            test_size=0.50, # El 20% restante se dividen 50/50 para validacion y test
            stratify=temp_df['especie'],
            random_state=self.seed
        )

        print(f"✓ División estratificada completada:")
        print(f"  - Train: {len(self.train_df)} imágenes ({len(self.train_df) / len(self.df_metadatos) * 100:.1f}%)")
        print(f"  - Val:   {len(self.val_df)} imágenes ({len(self.val_df) / len(self.df_metadatos) * 100:.1f}%)")
        print(f"  - Test:  {len(self.test_df)} imágenes ({len(self.test_df) / len(self.df_metadatos) * 100:.1f}%)")

    def copiar_a_disco(self):
        """Guarda en disco los datos divididos en la ruta processed"""
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

