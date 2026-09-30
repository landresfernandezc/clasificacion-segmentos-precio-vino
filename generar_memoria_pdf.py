"""Genera la memoria PDF a partir del CSV y las salidas del notebook."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "resultados"
CSV_NAME = "webscraping_precios_vino_clean (7).csv"
PRICE_TARGET = "precio_equivalente_750ml_crc"
CLASS_TARGET = "segmento_precio"


def add_text_page(pdf: PdfPages, page_number: int, title: str, sections: list[tuple[str, str]]) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    axis = figure.add_axes([0.09, 0.07, 0.82, 0.86])
    axis.set_axis_off()
    axis.text(0, 1, title, fontsize=20, weight="bold", color="#155c52", va="top")
    cursor = 0.93
    for heading, paragraph in sections:
        axis.text(0, cursor, heading, fontsize=12, weight="bold", color="#263238", va="top")
        cursor -= 0.035
        wrapped = textwrap.wrap(paragraph, width=102, break_long_words=False)
        axis.text(0, cursor, "\n".join(wrapped), fontsize=9.5, color="#263238", va="top", linespacing=1.35)
        cursor -= max(0.045, len(wrapped) * 0.019 + 0.025)
    axis.text(0, -0.025, f"Asignatura 7 | Memoria de trabajo | {page_number}", fontsize=8, color="#607d8b")
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_table_page(pdf: PdfPages, page_number: int, title: str, dataframe: pd.DataFrame, subtitle: str) -> None:
    figure, axis = plt.subplots(figsize=(8.27, 11.69))
    axis.set_axis_off()
    axis.text(0, 0.96, title, transform=axis.transAxes, fontsize=20, weight="bold", color="#155c52", va="top")
    axis.text(0, 0.90, subtitle, transform=axis.transAxes, fontsize=10, color="#263238", va="top", wrap=True)
    table = axis.table(
        cellText=dataframe.values,
        colLabels=dataframe.columns,
        cellLoc="center",
        colLoc="center",
        loc="upper center",
        bbox=[0, 0.48, 1, min(0.34, 0.06 + 0.06 * len(dataframe))],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#155c52")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#edf3f1")
    axis.text(
        0,
        0.40,
        "Las metricas corresponden a la particion cronologica reservada y a los datos disponibles.",
        transform=axis.transAxes,
        fontsize=9,
        color="#455a64",
        va="top",
        wrap=True,
    )
    axis.text(
        0,
        0.03,
        f"Asignatura 7 | Memoria de trabajo | {page_number}",
        transform=axis.transAxes,
        fontsize=8,
        color="#607d8b",
    )
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_eda_chart(pdf: PdfPages, data: pd.DataFrame) -> None:
    clean = data.loc[data[PRICE_TARGET].notna() & (data[PRICE_TARGET] >= 500)].copy()
    figure, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    figure.suptitle("Analisis exploratorio", fontsize=20, weight="bold", color="#155c52", y=0.96)
    clean[PRICE_TARGET].plot(kind="hist", bins=35, ax=axes[0], color="#167d6a", edgecolor="white")
    axes[0].set_title("Distribucion del precio equivalente")
    axes[0].set_xlabel("CRC por 750 ml")
    clean.boxplot(column=PRICE_TARGET, by=CLASS_TARGET, ax=axes[1], rot=25, grid=False)
    axes[1].set_title("Precio por segmento")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("CRC por 750 ml")
    figure.suptitle("Analisis exploratorio", fontsize=20, weight="bold", color="#155c52", y=0.97)
    figure.text(0.08, 0.02, "Se excluyen del grafico registros con precio equivalente inferior a 500 CRC.", fontsize=9)
    figure.tight_layout(rect=[0.04, 0.05, 0.96, 0.93])
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)

    daily = clean.groupby(clean["fecha_extraccion"].dt.date)[PRICE_TARGET].median()
    figure, axis = plt.subplots(figsize=(8.27, 5.2))
    daily.plot(ax=axis, marker="o", color="#167d6a")
    axis.set_title("Mediana diaria del precio equivalente")
    axis.set_xlabel("Fecha")
    axis.set_ylabel("CRC por 750 ml")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def add_model_charts(pdf: PdfPages, predictions: pd.DataFrame, products: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    comparison = pd.crosstab(predictions[CLASS_TARGET], predictions["prediccion_segmento"])
    image = axes[0].imshow(comparison.values, cmap="Greens")
    axes[0].set_title("Segmento real y predicho")
    axes[0].set_xticks(range(len(comparison.columns)), labels=comparison.columns, rotation=35, ha="right")
    axes[0].set_yticks(range(len(comparison.index)), labels=comparison.index)
    axes[0].set_xlabel("Prediccion")
    axes[0].set_ylabel("Real")
    for row in range(comparison.shape[0]):
        for column in range(comparison.shape[1]):
            axes[0].text(column, row, int(comparison.iloc[row, column]), ha="center", va="center", color="#263238")
    figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].scatter(
        products["mediana_presentacion"],
        products["mediana_precio"],
        c=products["cluster"],
        cmap="viridis",
        alpha=0.75,
    )
    axes[1].set(title="Segmentos de productos", xlabel="Volumen mediano (ml)", ylabel="Precio mediano (CRC)")
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def generate(output_path: Path) -> None:
    csv_path = ROOT / CSV_NAME
    required = [
        RESULTS / "metricas_clasificacion_notebook.csv",
        RESULTS / "reporte_clasificacion_notebook.csv",
        RESULTS / "silhouette_clusters_notebook.csv",
        RESULTS / "perfil_clusters_notebook.csv",
        RESULTS / "predicciones_clasificacion_notebook.csv",
        RESULTS / "segmentacion_productos_notebook.csv",
    ]
    missing = [path.name for path in [csv_path, *required] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan el CSV o salidas del notebook. Ejecuta todas sus celdas primero: " + ", ".join(missing)
        )

    data = pd.read_csv(csv_path, encoding="utf-8-sig", parse_dates=["fecha_extraccion"])
    classification = pd.read_csv(RESULTS / "metricas_clasificacion_notebook.csv")
    report = pd.read_csv(RESULTS / "reporte_clasificacion_notebook.csv", index_col=0)
    silhouette = pd.read_csv(RESULTS / "silhouette_clusters_notebook.csv")
    profile = pd.read_csv(RESULTS / "perfil_clusters_notebook.csv")
    predictions = pd.read_csv(RESULTS / "predicciones_clasificacion_notebook.csv")
    products = pd.read_csv(RESULTS / "segmentacion_productos_notebook.csv")

    clean_count = int(((data[PRICE_TARGET] >= 500) & data[PRICE_TARGET].notna()).sum())
    suspicious_count = len(data) - clean_count
    best_cluster = silhouette.sort_values("silhouette", ascending=False).iloc[0]
    best_classification = classification.loc[classification["modelo"].str.startswith("Clasificador")].iloc[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        add_text_page(pdf, 1, "Clasificacion de segmentos de precio de vino", [
            ("Master y asignatura", "Master en Big Data & Business Intelligence 2025/2026 | Asignatura 7: Modelado Predictivo con Machine Learning."),
            ("Integrantes", "[Completar con los nombres de todas las personas del grupo.]"),
            ("Vinculacion con el TFM", "El TFM analiza la oferta y el comportamiento de precios del vino en los catalogos digitales de Walmart Costa Rica, Masxmenos Costa Rica y PriceSmart Costa Rica. Este trabajo usa el mismo dataset y se centra solo en clasificacion, alineada con el analisis de segmentos del TFM."),
            ("Opcion de modelo elegida", "Se eligio un modelo de clasificacion supervisada. No se presenta regresion como modelo principal."),
            ("Objetivo de este trabajo", "Predecir el segmento de precio del producto a partir de atributos del catalogo y complementar el analisis con una segmentacion no supervisada de productos."),
        ])
        add_text_page(pdf, 2, "1. Introduccion y dataset", [
            ("Contexto", "Costa Rica depende principalmente de vinos importados, por lo que el precio final en supermercados refleja formato, cadena comercial, posicionamiento y dinamicas de catalogo. El TFM estudia este mercado minorista con datos publicados en linea."),
            ("Fuente y cobertura", f"El archivo contiene {len(data):,} observaciones, 13 variables, tres comercios y fechas entre {data.fecha_extraccion.min().date()} y {data.fecha_extraccion.max().date()}. La unidad observacional es un producto registrado en una fecha y comercio."),
            ("Variable objetivo", "La variable objetivo es segmento_precio. El precio equivalente a 750 ml se mantiene para analisis exploratorio y control de coherencia, pero no se usa como predictor del clasificador para evitar fuga de informacion."),
            ("Justificacion", "Clasificar segmentos aporta valor al TFM porque ayuda a analizar el posicionamiento comercial de la oferta por supermercado, categoria y periodo observado."),
        ])
        add_text_page(pdf, 3, "2. Preparacion y analisis exploratorio", [
            ("Calidad", f"No se detectaron duplicados exactos. Se apartaron {suspicious_count} filas con precio equivalente faltante o inferior a 500 CRC y se conservaron {clean_count} para analisis descriptivo. Para clasificacion se usan las filas validas con segmento informado."),
            ("Transformaciones", "Se convierten fechas y columnas numericas a tipos adecuados. Las descripciones se representan con TF-IDF, categoria/comercio con OneHotEncoder, y volumen, descuento y fecha con StandardScaler. El preprocesamiento se ajusta dentro del pipeline."),
            ("Datos incompletos", "La columna segmento_precio tiene faltantes; estos registros no se usan para entrenar/evaluar el clasificador. Los tipos de cambio incompletos no se incorporan al modelo."),
            ("Lectura exploratoria", "Las distribuciones por segmento permiten revisar si las etiquetas reflejan niveles diferenciados de precio. Las diferencias observadas son descriptivas y dependen de la composicion del catalogo."),
        ])
        add_eda_chart(pdf, data)

        class_table = classification[["modelo", "accuracy", "F1_macro", "filas_prueba"]].copy()
        class_table["accuracy"] = class_table["accuracy"].map(lambda value: f"{value:.3f}")
        class_table["F1_macro"] = class_table["F1_macro"].map(lambda value: f"{value:.3f}")
        class_table["filas_prueba"] = class_table["filas_prueba"].astype(int).astype(str)
        add_table_page(
            pdf,
            5,
            "3. Modelo supervisado principal",
            class_table,
            "Clasificador logistico multiclase para predecir segmento_precio. Se usa particion temporal y validacion cronologica por bloques para seleccionar C.",
        )

        report_table = report.loc[[idx for idx in report.index if idx not in {"accuracy", "macro avg", "weighted avg"}]].reset_index()
        report_table = report_table.rename(columns={"index": "segmento"})
        for column in ["precision", "recall", "f1-score"]:
            report_table[column] = report_table[column].map(lambda value: f"{value:.3f}")
        report_table["support"] = report_table["support"].astype(int).astype(str)
        add_table_page(
            pdf,
            6,
            "4. Resultados por segmento",
            report_table,
            f"El clasificador elegido obtiene accuracy de {best_classification['accuracy']:.3f} y F1 macro de {best_classification['F1_macro']:.3f}.",
        )

        add_text_page(pdf, 7, "5. Extension y no supervisado", [
            ("Extension multiclase", f"La extension avanzada elegida es un modelo multiclase: se predicen varios segmentos comerciales, no una decision binaria. F1 macro en prueba: {best_classification['F1_macro']:.3f} sobre {int(best_classification['filas_prueba'])} observaciones."),
            ("Clustering", f"Se agregaron observaciones por descripcion de producto, se combinaron TF-IDF reducido mediante TruncatedSVD con precio, volumen y frecuencia estandarizados, y se evaluo KMeans para k de 2 a 6. El mayor silhouette fue {best_cluster['silhouette']:.3f} para k={int(best_cluster['k'])}."),
            ("Interpretacion de perfiles", "Los grupos separan patrones de producto asociados a volumen, precio mediano y frecuencia de observacion. Es una segmentacion exploratoria que requiere validacion de negocio antes de asignarle significado comercial definitivo."),
        ])

        cluster_table = profile.copy()
        cluster_table["cluster"] = cluster_table["cluster"].astype(int).astype(str)
        add_table_page(pdf, 8, "6. Perfiles de producto", cluster_table,
                       f"Perfiles de los {len(products):,} nombres de producto agrupados por KMeans.")
        add_model_charts(pdf, predictions, products)
        add_text_page(pdf, 10, "7. Conclusiones y limitaciones", [
            ("Conclusiones", "Se eligio solo clasificacion. El modelo predice segmento_precio y se alinea con el TFM porque apoya el analisis de posicionamiento comercial de la oferta de vinos."),
            ("Limitaciones", "El periodo observado es corto; los nombres de producto y categorias tienen variantes; los segmentos tienen faltantes y sus umbrales deben documentarse; el desempeno sobre productos nuevos no se ha medido."),
            ("Trabajo futuro", "Completar nombres del grupo, revisar umbrales de segmento, normalizar catalogo, ampliar el periodo, comparar clasificadores y validar la utilidad de las predicciones dentro del tablero BI."),
        ])

    print(f"Memoria PDF generada: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Memoria_Proyecto_Modelado_Predictivo_actualizada.pdf",
        help="Ruta de salida del PDF",
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
