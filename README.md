# Analisis de precios de vino

Proyecto en Python vinculado al TFM sobre analisis de la oferta y comportamiento de precios del vino en los catalogos digitales de Walmart Costa Rica, Masxmenos Costa Rica y PriceSmart Costa Rica.

El TFM principal mantiene un enfoque descriptivo y de Business Intelligence, segun el alcance acordado con el tutor. Este trabajo de la Asignatura 7 desarrolla una extension exploratoria de Machine Learning sobre el mismo dataset, centrada en un modelo de clasificacion de segmentos de precio.

## Como ejecutar

Abre la carpeta del proyecto en VS Code, abre una terminal integrada en esa carpeta y ejecuta:

```powershell
python analisis_precios_vino.py
```

El CSV adjunto debe permanecer en la carpeta del proyecto. Para usar otro archivo o elegir otra carpeta de salida:

```powershell
python analisis_precios_vino.py "ruta/al/archivo.csv" --output-dir resultados
```

Si las dependencias no estan instaladas en el Python seleccionado en VS Code:

```powershell
python -m pip install -r requirements.txt
```

## Como ver los resultados

Despues de ejecutar el script, abre la carpeta `resultados` en el explorador de VS Code. Haz clic en un `.csv` para verlo como tabla, en `evolucion_precio_mediano.png` para abrir el grafico y en `informe_calidad.txt` para consultar el resumen de limpieza. Para regenerar todo, vuelve a ejecutar el comando anterior.

El informe tecnico con el alcance, metodo, resultados e interpretacion esta en [DOCUMENTACION_ANALISIS.md](DOCUMENTACION_ANALISIS.md).

## Vinculacion con el TFM

El TFM busca analizar la oferta, comparacion y evolucion de precios de vinos en supermercados costarricenses, usando datos obtenidos mediante scraping diario y normalizados al precio equivalente de una botella de 750 ml. Esta normalizacion permite comparar productos con distintas presentaciones y alimentar un tablero de Business Intelligence.

Aunque el TFM no depende de modelos predictivos como resultado principal, este proyecto cumple la consigna de la asignatura mediante:

- Un modelo de clasificacion para predecir `segmento_precio`.
- Una extension multiclase, porque el clasificador distingue varios segmentos comerciales.
- Un analisis no supervisado para segmentar productos.

Los modelos deben interpretarse como prueba de concepto y complemento metodologico, no como sustituto del alcance descriptivo aprobado para el TFM.

## Entregables de la asignatura

- [Notebook reproducible](TFM_vinos_modelado.ipynb): ejecutar sus celdas en orden. Incluye EDA, preprocesamiento, clasificacion multiclase con validacion temporal y clustering.
- [Memoria PDF](Memoria_Proyecto_Modelado_Predictivo_actualizada.pdf): memoria generada desde los resultados del notebook. Completar integrantes antes de entregar.
- [Generador de memoria](generar_memoria_pdf.py): vuelve a crear el PDF a partir de las salidas actuales.

Para regenerar la memoria despues de ejecutar todas las celdas del notebook:

```powershell
python generar_memoria_pdf.py
```

El PDF debe regenerarse despues de modificar los datos o modelos.

## Archivos generados

- `resultados/informe_calidad.txt`: cantidad de filas, periodo y filtro aplicado.
- `resultados/precios_sospechosos.csv`: registros con precio equivalente faltante o inferior a 500 CRC; se conservan separados, sin corregirlos automaticamente.
- `resultados/datos_limpios.csv`: observaciones usadas en los resumenes y el modelo.
- `resultados/resumen_por_categoria.csv` y `resultados/resumen_por_comercio.csv`: cantidad de observaciones y estadisticos del precio equivalente.
- `resultados/evolucion_precio_mediano.png`: mediana diaria del precio equivalente a 750 ml.
- `resultados/metricas_clasificacion_notebook.csv`: accuracy y F1 macro del clasificador y de la referencia de clase mayoritaria.
- `resultados/reporte_clasificacion_notebook.csv`: precision, recall y F1 por segmento de precio.
- `resultados/predicciones_clasificacion_notebook.csv`: segmento real, segmento predicho y acierto por fila de prueba.
- `resultados/silhouette_clusters_notebook.csv`, `perfil_clusters_notebook.csv` y `segmentacion_productos_notebook.csv`: resultados del analisis no supervisado.

La prueba de los modelos usa las fechas mas recientes. Los resultados son exploratorios: deben presentarse como una extension academica de Machine Learning sobre el dataset del TFM, manteniendo la cautela declarada en el primer avance sobre la corta profundidad historica disponible.
