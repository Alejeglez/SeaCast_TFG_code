# SeaCast_TFG_code

Repositorio con el código del TFG Predicción de Datos Oceanográficos con Redes Neuronales de Grafos y Aprendizaje por Conjuntos

# SeaCast

Este repositorio parte del proyecto original [SeaCast](https://github.com/deinal/seacast).  

## 🚀 Cómo usar
### 1. Clonar el repositorio:

```bash
git clone https://github.com/Alejeglez/SeaCast_TFG.git
```
### 2. Crear entorno:

```bash
conda create -n Seacast_env python=3.10.16
conda activate Seacast_env
```
### 3. Instalar dependencias:

```bash
pip install -r requirements.txt
```
---

## 🚀 Despliegue de SeaCast

### 1. Descarga de datos

Antes de descargar los datos, necesitas configurar tus credenciales de la API de CDS creando un archivo de configuración en $HOME/.cdsapirc.
Las instrucciones para crear este archivo se pueden encontrar en:
https://cds.climate.copernicus.eu/how-to-api

Descarga los datos necesarios para el período del 01-01-2003 al 31-12-2023:

```bash
user=[CMEMS-username]
password=[CMEMS-password]

python download_data.py --static -b data/atlantic/ -u $user -psw $password &&
python download_data.py -d reanalysis -s 2003-01-01 -e 2023-12-31 -u $user -psw $password &&
python download_data.py -d era5 -s 2003-01-01 -e 2023-12-31
```

### 2. Preparación de los datos

#### Divisiones del conjunto de datos
- **Conjunto de entrenamiento**: 2003-01-01 a 2019-12-31
- **Conjunto de validación**: 2020-01-01 a 2021-12-31
- **Conjunto de prueba**: 2022-01-01 a 2023-12-31

#### 3. Preparar Estados - Datos de Reanálisis (oceanográficos)

```bash
# Training set
python prepare_states.py -d data/atlantic/raw/reanalysis -o data/atlantic/samples/train -n 6 -p rea_data -s 2003-01-01 -e 2019-12-31

# Validation set
python prepare_states.py -d data/atlantic/raw/reanalysis -o data/atlantic/samples/val -n 6 -p rea_data -s 2020-01-01 -e 2021-12-31

# Test set
python prepare_states.py -d data/atlantic/raw/reanalysis -o data/atlantic/samples/test -n 17 -p rea_data -s 2022-01-01 -e 2023-12-31
```

#### 4. Preparar Estados - Datos ERA5 (forzantes atmosféricos)

```bash
# Training set
python prepare_states.py -d data/atlantic/raw/era5 -o data/atlantic/samples/train -n 6 -p forcing -s 2003-01-01 -e 2019-12-31

# Validation set
python prepare_states.py -d data/atlantic/raw/era5 -o data/atlantic/samples/val -n 6 -p forcing -s 2020-01-01 -e 2021-12-31

# Test set
python prepare_states.py -d data/atlantic/raw/era5 -o data/atlantic/samples/test -n 17 -p forcing -s 2022-01-01 -e 2023-12-31
```

### 5. Preparación de características y modelo

```bash
# Create grid features
python create_grid_features.py --dataset atlantic

# Create parameter weights
python create_parameter_weights.py --dataset atlantic --batch_size 4 --n_workers 4

# Create mesh
python create_mesh.py --dataset atlantic --graph hierarchical --levels 3 --hierarchical 1
```

### 6. Entrenamiento del modelo

Entrena el modelo Hi-LAM con la siguiente configuración:

```bash
python train_model.py --dataset atlantic \
                     --n_nodes 1 \
                     --n_workers 4 \
                     --epochs 150 \
                     --lr 0.001 \
                     --batch_size 1 \
                     --step_length 1 \
                     --ar_steps 1 \
                     --optimizer adamw \
                     --scheduler cosine \
                     --processor_layers 4 \
                     --hidden_dim 128 \
                     --model hi_lam \
                     --graph hierarchical \
                     --finetune_start 1 \
```

### Notas
- Asegúrate de que todas las dependencias estén instaladas antes de ejecutar los comandos.
- Ajusta el número de trabajadores (`--n_workers`) según las capacidades de tu sistema.
- El tamaño del lote (`--batch_size`) se puede modificar según la memoria disponible de la GPU.

### 7. Inferencia del modelo

Para evaluar un modelo Hi-LAM entrenado en el conjunto de prueba con ruido añadido a las condiciones iniciales:

```bash
python train_model.py --dataset atlantic \
                      --data_subset reanalysis \
                      --forcing_prefix forcing \
                      --n_workers 1 \
                      --batch_size 1 \
                      --step_length 1 \
                      --model hi_lam \
                      --graph hierarchical \
                      --processor_layers 4 \
                      --hidden_dim 128 \
                      --n_example_pred 1 \
                      --store_pred 1 \
                      --eval test \
                      --load saved_models/hi_lam-4x128-05_17_01-6834/last.ckpt \
                      --noise perlin_fractal
```

### Notes
- `--load` debe apuntar a la ruta del checkpoint del modelo guardado.
- `--store_pred 1` guardará las predicciones del modelo después de la inferencia.
- `--noise` controla el tipo de ruido añadido durante la inferencia. Las opciones válidas son:
    - gaussian
    - perlin
    - perlin_fractal
- Para generar un conjunto de predicciones (ensemble), se deben ejecutar múltiples veces el comando anterior usando la misma configuración. Cada ejecución producirá salidas diferentes debido al ruido añadido a las condiciones iniciales.
- Las salidas de cada ejecución se guardan en carpetas separadas (dentro de la carpeta `wandb`). Para construir el conjunto (ensemble), agrupa manualmente las carpetas de salida correspondientes a cada tipo de ruido en directorios separados para su análisis o agregación posterior.

### 8. Agregación del conjunto

Una vez que se hayan generado múltiples predicciones usando diferentes ejecuciones con la misma configuración y ajustes de ruido, puedes agregarlas en una única predicción de ensemble utilizando el siguiente comando:

```bash
python src/seacast_tools/ensemble_unifier.py "data/atlantic/predictions_mock/ensemble_5/gaussian_001" --method mean
```

### Notes

- El primer argumento posicional es la ruta al directorio que contiene múltiples ejecuciones de predicciones del mismo experimento (es decir, múltiples salidas generadas con el mismo tipo de ruido).

- `--method` especifica el método de agregación a utilizar. Actualmente soportado:

    - `mean:` calcula el promedio de todas las predicciones del ensemble para cada día.
    
- Las salidas se guardarán en una subcarpeta llamada `unified` dentro de la carpeta que contiene las ejecuciones individuales.

Para más información acerca de las métricas de evaluación, consulta la [documentación de métricas](docs/weatherbench_x_adapted.md).
