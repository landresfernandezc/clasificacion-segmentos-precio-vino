# Analisis exploratorio y clasificacion de segmentos de precio de vino

## 1. Estado y alcance

Este documento describe la ejecucion actual del analisis del archivo `webscraping_precios_vino_clean (7).csv` y su relacion con el primer avance del TFM.

El TFM principal analiza la oferta y el comportamiento de precios del vino en los catalogos digitales de Walmart Costa Rica, Masxmenos Costa Rica y PriceSmart Costa Rica. Su enfoque aprobado es descriptivo y de Business Intelligence: comparacion de precios, evolucion de corto plazo, variabilidad, posicionamiento por comercio/categoria/segmento y presentacion de resultados mediante tablero.

Para mantener coherencia con ese enfoque, la opcion de modelo elegida para la Asignatura 7 es **un modelo de clasificacion**. El objetivo supervisado es predecir `segmento_precio`, una etiqueta categorica vinculada al posicionamiento comercial de los productos. No se presenta regresion de precio como modelo principal.

## 2. Evaluacion frente a la consigna de la Asignatura 7

**Conclusion: el proyecto cubre los componentes tecnicos y formatos principales solicitados por la asignatura, usando clasificacion como modelo supervisado principal.**

| Requisito | Estado actual | Evidencia y brecha |
| --- | --- | --- |
| Dataset relacionado con el TFM y justificacion de su eleccion | Implementado | El dataset es el mismo insumo del TFM: scraping diario de precios de vino en Walmart Costa Rica, Masxmenos Costa Rica y PriceSmart Costa Rica. |
| Analisis exploratorio de datos | Implementado | El notebook muestra estructura, faltantes, distribuciones, precio por segmento, estadisticos por categoria/comercio y evolucion diaria. |
| Preprocesamiento (20%) | Implementado con cautelas | Se eliminan duplicados exactos, se apartan valores sospechosos, se filtran filas con `segmento_precio` disponible y se aplican TF-IDF, one-hot encoding y StandardScaler. |
| Modelo supervisado principal (70%) | Implementado | Se entrena una clasificador logistico multiclase para predecir `segmento_precio`, con particion temporal, ajuste de `C` y evaluacion con accuracy, F1 macro y reporte por clase. |
| Modelo avanzado o extension (5%) | Implementado | La extension avanzada elegida es el enfoque multiclase: el clasificador distingue varios segmentos comerciales, no una salida binaria. |
| Tecnica no supervisada (5%) | Implementado, exploratorio | KMeans agrupa productos; se compara silhouette para k de 2 a 6 y se exportan perfiles. |
| Notebook Jupyter reproducible (`.ipynb`) | Implementado | `TFM_vinos_modelado.ipynb` incluye codigo, explicaciones y resultados guardados. |
| Memoria final en PDF | Implementada con pendiente menor | `Memoria_Proyecto_Modelado_Predictivo_actualizada.pdf` se genera desde las salidas del notebook. Debe completarse con los nombres de integrantes. |
| Apartados de la memoria | Implementado | La memoria cubre contexto, datos, metodologia, EDA, modelo de clasificacion, resultados, interpretacion, conclusiones, limitaciones y trabajo futuro. |
| Nombres de integrantes | Pendiente | No se han proporcionado los nombres del grupo. |

## 3. Datos

El CSV contiene 4.170 filas, 13 columnas y observaciones fechadas entre el 10 y el 29 de septiembre de 2026. Registra precios de tres comercios: Masxmenos Costa Rica, PriceSmart Costa Rica y Walmart Costa Rica. Las columnas incluyen fecha, categoria, precio de lista/oferta, descuento, volumen, precio final, producto, comercio, precio equivalente por 750 ml, segmento de precio y tipos de cambio.

La variable objetivo del modelo supervisado es `segmento_precio`. El precio equivalente por 750 ml se mantiene para analisis descriptivo, revision de coherencia de segmentos y clustering, pero no se usa como predictor del clasificador para evitar fuga de informacion.

## 4. Preparacion y control de calidad

El notebook convierte fechas y variables numericas, elimina duplicados exactos, separa observaciones con precio equivalente faltante o menor a 500 CRC y conserva para clasificacion las filas validas con `segmento_precio` informado.

Las variables predictoras del clasificador son:

- Texto de `producto`, representado mediante TF-IDF.
- Variables categoricas `categoria` y `retailer`, codificadas con OneHotEncoder.
- Volumen expresado en unidades de 750 ml, dias desde el inicio y `descuento_pct`, escalados con StandardScaler.

El preprocesamiento se ejecuta dentro del pipeline para evitar ajustar transformadores con datos de prueba.

## 5. Metodologia del modelo

Se construyo un pipeline de clasificador logistico multiclase. La particion es temporal: se entrena con fechas anteriores al bloque final y se evalua con las fechas mas recientes. La regularizacion `C` se selecciona mediante validacion cronologica por bloques dentro del conjunto de entrenamiento.

Las metricas principales son:

- Accuracy.
- F1 macro, para ponderar de forma equilibrada los segmentos aunque haya desbalance.
- Precision, recall y F1 por clase.

La referencia de comparacion predice siempre la clase mayoritaria del entrenamiento.

## 6. Analisis no supervisado

Ademas del clasificador, el notebook mantiene un analisis no supervisado con KMeans. Se agregan productos por descripcion, se combinan variables de texto reducidas con TruncatedSVD y variables numericas estandarizadas, y se evalua silhouette para distintos valores de k.

Este apartado cumple la parte de segmentacion no supervisada de la rubrica y complementa el analisis de posicionamiento del TFM.

## 7. Como reproducir y consultar

Abrir `TFM_vinos_modelado.ipynb` en VS Code y ejecutar todas las celdas en orden. El notebook guarda metricas, predicciones y perfiles en `resultados/`. Para regenerar la memoria PDF:

```powershell
python generar_memoria_pdf.py
```

Archivos clave:

- `resultados/metricas_clasificacion_notebook.csv`
- `resultados/reporte_clasificacion_notebook.csv`
- `resultados/predicciones_clasificacion_notebook.csv`
- `resultados/silhouette_clusters_notebook.csv`
- `resultados/perfil_clusters_notebook.csv`
- `resultados/segmentacion_productos_notebook.csv`

## 8. Limitaciones y trabajo pendiente

- Completar los nombres de integrantes en notebook y PDF.
- Revisar los umbrales o reglas usadas para crear `segmento_precio`.
- Validar manualmente los 46 precios sospechosos y equivalencias de volumen/paquetes.
- Normalizar nombres de producto y categorias para reducir variantes de scraping.
- Ampliar el periodo historico antes de presentar el clasificador como componente predictivo estable.
- Evaluar rendimiento sobre productos no vistos si el TFM lo requiere en etapas futuras.
