"""Analisis exploratorio y clasificacion de segmentos de precio de vino."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRICE_TARGET = "precio_equivalente_750ml_crc"
CLASS_TARGET = "segmento_precio"
MIN_VALID_PRICE_CRC = 500
RANDOM_STATE = 42


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and parse fields used by the analysis."""
    data = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_columns = {
        "fecha_extraccion",
        "categoria",
        "presentacion_ml",
        "descuento_pct",
        "producto",
        "retailer",
        PRICE_TARGET,
        CLASS_TARGET,
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(sorted(missing_columns))}")

    data["fecha_extraccion"] = pd.to_datetime(data["fecha_extraccion"], errors="coerce")
    for column in ("presentacion_ml", "descuento_pct", PRICE_TARGET):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def create_outputs(data: pd.DataFrame, output_dir: Path) -> None:
    """Write quality summaries, cleaned data, classification results, and charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    original_count = len(data)
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates().copy()

    invalid_price = data[PRICE_TARGET].isna() | (data[PRICE_TARGET] < MIN_VALID_PRICE_CRC)
    anomalies = data.loc[invalid_price].copy()
    clean = data.loc[~invalid_price].copy()
    model_data = clean.dropna(subset=["fecha_extraccion", "presentacion_ml", "producto", "retailer", CLASS_TARGET]).copy()
    model_data[CLASS_TARGET] = model_data[CLASS_TARGET].astype(str)

    anomalies.to_csv(output_dir / "precios_sospechosos.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(output_dir / "datos_limpios.csv", index=False, encoding="utf-8-sig")

    by_category = clean.groupby("categoria", dropna=False)[PRICE_TARGET].agg(
        registros="count", mediana="median", promedio="mean", minimo="min", maximo="max"
    ).reset_index()
    by_category.to_csv(output_dir / "resumen_por_categoria.csv", index=False, encoding="utf-8-sig")

    by_retailer = clean.groupby("retailer", dropna=False)[PRICE_TARGET].agg(
        registros="count", mediana="median", promedio="mean", minimo="min", maximo="max"
    ).reset_index()
    by_retailer.to_csv(output_dir / "resumen_por_comercio.csv", index=False, encoding="utf-8-sig")

    daily = clean.groupby(clean["fecha_extraccion"].dt.date)[PRICE_TARGET].median().dropna()
    if not daily.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.plot(pd.to_datetime(daily.index), daily.values, marker="o", color="#147d6f")
        axis.set_title("Mediana diaria del precio equivalente a 750 ml")
        axis.set_xlabel("Fecha de extraccion")
        axis.set_ylabel("Colones costarricenses (CRC)")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "evolucion_precio_mediano.png", dpi=160)
        plt.close(figure)

    train, test = temporal_split(model_data)
    metrics, report, predictions = fit_classification_model(train, test)
    metrics.to_csv(output_dir / "metricas_modelo.csv", index=False, encoding="utf-8-sig")
    report.to_csv(output_dir / "reporte_clasificacion.csv", encoding="utf-8-sig")
    predictions.to_csv(output_dir / "predicciones_prueba.csv", index=False, encoding="utf-8-sig")

    report_lines = [
        "ANALISIS DE PRECIOS DE VINO",
        "",
        f"Filas originales: {original_count}",
        f"Duplicados exactos eliminados: {duplicate_count}",
        f"Filas sin precio valido o con precio equivalente menor a {MIN_VALID_PRICE_CRC} CRC: {len(anomalies)}",
        f"Filas disponibles para analisis descriptivo: {len(clean)}",
        f"Filas etiquetadas para clasificacion: {len(model_data)}",
        f"Fechas: {clean['fecha_extraccion'].min().date()} a {clean['fecha_extraccion'].max().date()}",
        f"Precio equivalente mediano (CRC por 750 ml): {clean[PRICE_TARGET].median():.2f}",
        "",
        "Modelo elegido: clasificacion supervisada de segmento_precio.",
        "Los precios sospechosos se excluyen del modelo, no se corrigen automaticamente.",
        "La prueba usa las fechas mas recientes; el resultado es exploratorio por la corta profundidad historica.",
    ]
    (output_dir / "informe_calidad.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print("Analisis completado.")
    print(f"Filas originales: {original_count}; filas para clasificacion: {len(model_data)}")
    print(f"Precios sospechosos aislados: {len(anomalies)}")
    print(f"Resultados: {output_dir.resolve()}")


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so the test set contains only later observations."""
    dates = np.sort(data["fecha_extraccion"].dropna().unique())
    if len(dates) < 5:
        raise ValueError("Se necesitan al menos cinco fechas para la particion temporal.")
    test_start = dates[max(1, int(len(dates) * 0.8))]
    train = data.loc[data["fecha_extraccion"] < test_start].copy()
    test = data.loc[data["fecha_extraccion"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("La particion temporal produjo un conjunto vacio.")
    return add_model_features(train, data["fecha_extraccion"].min()), add_model_features(test, data["fecha_extraccion"].min())


def add_model_features(frame: pd.DataFrame, first_date: pd.Timestamp) -> pd.DataFrame:
    result = frame.copy()
    result["dias_desde_inicio"] = (result["fecha_extraccion"] - first_date).dt.days
    result["unidades_750ml"] = result["presentacion_ml"] / 750
    result["descuento_pct"] = result["descuento_pct"].fillna(0)
    result["producto"] = result["producto"].fillna("").astype(str)
    for column in ("categoria", "retailer"):
        result[column] = result[column].fillna("desconocido").astype(str)
    return result


def fit_classification_model(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a multiclass logistic regression and compare it with the majority class."""
    features = ColumnTransformer(
        transformers=[
            ("producto", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2), "producto"),
            ("categoricas", OneHotEncoder(handle_unknown="ignore"), ["categoria", "retailer"]),
            ("numericas", StandardScaler(), ["unidades_750ml", "dias_desde_inicio", "descuento_pct"]),
        ]
    )
    model = Pipeline(
        [
            ("variables", features),
            ("clasificacion", LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    model.fit(train, train[CLASS_TARGET])
    model_predictions = model.predict(test)
    majority_class = train[CLASS_TARGET].mode().iloc[0]
    baseline_predictions = np.full(len(test), majority_class)

    metrics = pd.DataFrame(
        [
            _classification_metrics(f"Base: clase mayoritaria ({majority_class})", test[CLASS_TARGET], baseline_predictions),
            _classification_metrics("Clasificador logistico multiclase", test[CLASS_TARGET], model_predictions),
        ]
    )
    report = pd.DataFrame(classification_report(test[CLASS_TARGET], model_predictions, zero_division=0, output_dict=True)).T
    predictions = test[["fecha_extraccion", "categoria", "producto", "retailer", PRICE_TARGET, CLASS_TARGET]].copy()
    predictions["prediccion_segmento"] = model_predictions
    predictions["acierto"] = predictions[CLASS_TARGET] == predictions["prediccion_segmento"]
    return metrics, report, predictions


def _classification_metrics(name: str, actual: pd.Series, predicted: np.ndarray) -> dict[str, float | str]:
    return {
        "modelo": name,
        "accuracy": accuracy_score(actual, predicted),
        "F1_macro": f1_score(actual, predicted, average="macro", zero_division=0),
        "filas_prueba": len(actual),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_csv = Path(__file__).with_name("webscraping_precios_vino_clean (7).csv")
    parser.add_argument("csv", nargs="?", type=Path, default=default_csv, help="Ruta al CSV de precios")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("resultados"),
        help="Carpeta donde se guardaran los resultados",
    )
    args = parser.parse_args()
    if not args.csv.exists():
        parser.error(f"No se encontro el archivo CSV: {args.csv}")
    create_outputs(load_data(args.csv), args.output_dir)


if __name__ == "__main__":
    main()
