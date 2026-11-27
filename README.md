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
