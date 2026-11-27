# Music Pattern Analysis

Análisis avanzado de patrones musicales usando el Spotify Tracks Dataset.

## Estructura del proyecto

- `data/raw`: datos originales sin modificar
- `data/processed`: datos limpios y transformados
- `notebooks`: análisis exploratorio y modelos
- `src`: código reutilizable
- `reports`: resultados y figuras

## Objetivos

- Análisis exploratorio (EDA) de las características de audio
- Limpieza y preparación de datos
- Clustering de canciones por patrones musicales
- Modelos predictivos de popularidad

## Dataset

Se usa el [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) de Kaggle (~114.000 canciones con características de audio). Los datos no se incluyen en el repositorio.

### Cómo descargarlo

1. Crear un token en Kaggle (Settings → API → Create New Token) y guardarlo en `~/.kaggle/`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Descargar el dataset:

```bash
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data/raw --unzip
```

## Limpieza de datos

Para generar los datasets procesados en `data/processed/`:

```bash
python src/limpieza.py
```

Se crean dos archivos (no se suben al repo):

- `tracks_genres.csv`: una fila por canción y género (113.386 filas). Para análisis por género.
- `tracks_unique.csv`: una fila por `track_id` (89.578 filas), con todos sus géneros en `track_genres`. Para clustering y modelos.

Criterios aplicados: se eliminan duplicados exactos, nulos y filas con `tempo = 0` o `time_signature = 0` (fallos de análisis de audio). Se añade la columna `sin_popularidad` (`popularity == 0`) para poder filtrarlas en los modelos.

## Análisis por género

Scripts: `src/generos_perfil.py` (perfil medio y rankings) y `src/generos_heatmap.py` (heatmap agrupado y pares de géneros más parecidos).

Notas:
- El dataset está balanceado por diseño (862-1.000 canciones por género), por lo que no refleja la proporción real de cada género en Spotify.
- `songwriter` y `singer-songwriter` contienen exactamente las mismas 1.000 canciones; se excluye `songwriter` del análisis de similitud.
- Algunas etiquetas (`happy`, `chill`, `sad`, `sleep`) son más de ambiente que de género musical.

## Clustering de canciones

Scripts (ejecutar en este orden desde la raíz):

- `src/preparacion.py`: transformación `log1p` de variables sesgadas y `StandardScaler`
- `src/clustering_pca.py`: PCA para entender el peso de cada variable
- `src/clustering_k.py`: método del codo y silhouette
- `src/clustering_perfil.py`: K-Means con k=8 y perfil de cada cluster

Decisiones:

- `RobustScaler` se descartó: dejaba `instrumentalness` y `speechiness` dominando el PCA (70% de la varianza). Con `StandardScaler` ningún componente está dominado por una sola variable.
- k=8 por ser el máximo local de silhouette (0,164) entre los k útiles. Los silhouette son bajos (0,14-0,24): los estilos musicales forman un continuo y los clusters se solapan.

Clusters encontrados: acústico suave, latino bailable, rock/metal intenso, electrónica instrumental, comedia/habla, ambient y clásica, rítmico con voz hablada y directo/brasileño.

La popularidad apenas varía entre clusters (24-36): el audio por sí solo explica poco.
