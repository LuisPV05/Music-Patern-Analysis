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
