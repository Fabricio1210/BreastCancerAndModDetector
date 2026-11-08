from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class CancerDetectionError(ValueError):
    """Error legible al validar datos o cargar artefactos de CancerDetection."""


@dataclass(frozen=True)
class CancerArtifacts:
    model: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame
    report: dict[str, Any]
    comparison: pd.DataFrame
    feature_names: tuple[str, ...]
    train_min: pd.Series
    train_max: pd.Series


@dataclass(frozen=True)
class CSVValidation:
    data: pd.DataFrame
    features: pd.DataFrame
    valid_mask: pd.Series
    errors: list[dict[str, Any]]


def _read_dataset(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise CancerDetectionError(f"No se pudo leer el archivo {path.name}: {error}") from error
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise CancerDetectionError(
            f"{path.name} contiene columnas duplicadas: {duplicates}."
        )
    return frame


def _validate_model(model: Any, feature_names: tuple[str, ...]) -> None:
    if not isinstance(model, Pipeline):
        raise CancerDetectionError(
            "best_model.joblib no contiene el pipeline completo esperado."
        )
    if "scale" not in model.named_steps or not isinstance(
        model.named_steps["scale"], StandardScaler
    ):
        raise CancerDetectionError(
            "El pipeline no contiene el StandardScaler esperado; no se ajustará "
            "ningún preprocesador durante la inferencia."
        )
    if "model" not in model.named_steps or not isinstance(model.named_steps["model"], SVC):
        raise CancerDetectionError("El clasificador del pipeline no es el SVM esperado.")
    if not hasattr(model, "predict_proba"):
        raise CancerDetectionError("El pipeline no expone predict_proba.")
    classes = list(model.classes_)
    if set(classes) != {0, 1} or len(classes) != 2:
        raise CancerDetectionError(
            f"Clases inesperadas {classes}; se requiere benigno=0 y maligno=1."
        )
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and tuple(model_features) != feature_names:
        raise CancerDetectionError(
            "El orden o esquema de feature_names_in_ no coincide con train.csv."
        )


def load_artifacts(root: Path) -> CancerArtifacts:
    root = Path(root)
    required = {
        "modelo": root / "models" / "best_model.joblib",
        "entrenamiento": root / "data" / "processed" / "train.csv",
        "prueba": root / "data" / "processed" / "test.csv",
        "métricas": root / "reports" / "metrics.json",
        "comparación": root / "reports" / "model_comparison.csv",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise CancerDetectionError(
            "Faltan artefactos locales requeridos: " + "; ".join(missing)
        )
    train = _read_dataset(required["entrenamiento"])
    test = _read_dataset(required["prueba"])
    metadata = {"source_index", "malignant"}
    if not metadata.issubset(train.columns) or not metadata.issubset(test.columns):
        raise CancerDetectionError(
            "train.csv y test.csv deben incluir source_index y malignant."
        )
    feature_names = tuple(column for column in train.columns if column not in metadata)
    if len(feature_names) != 30:
        raise CancerDetectionError(
            f"Se esperaban 30 variables predictoras; train.csv contiene {len(feature_names)}."
        )
    if tuple(column for column in test.columns if column not in metadata) != feature_names:
        raise CancerDetectionError(
            "El esquema de variables de test.csv no coincide con train.csv."
        )
    for name, frame in (("train.csv", train), ("test.csv", test)):
        if frame[list(feature_names)].isna().any().any():
            raise CancerDetectionError(f"{name} contiene valores faltantes.")
        try:
            values = frame[list(feature_names)].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise CancerDetectionError(f"{name} tiene variables no numéricas.") from error
        if not np.isfinite(values).all():
            raise CancerDetectionError(f"{name} contiene valores no finitos.")
        numeric_labels = pd.to_numeric(frame["malignant"], errors="coerce")
        labels = set(numeric_labels.dropna().unique())
        if (
            not labels.issubset({0, 1})
            or numeric_labels.isna().any()
            or not np.isfinite(numeric_labels.to_numpy(dtype=float)).all()
        ):
            raise CancerDetectionError(
                f"{name}: malignant debe contener solo 0=benigno y 1=maligno."
            )

    try:
        with required["métricas"].open(encoding="utf-8") as file:
            report = json.load(file)
        comparison = pd.read_csv(required["comparación"])
    except (OSError, json.JSONDecodeError, pd.errors.ParserError) as error:
        raise CancerDetectionError(f"No se pudieron leer los reportes locales: {error}") from error
    if report.get("target_mapping") != {"benign": 0, "malignant": 1}:
        raise CancerDetectionError(
            "El reporte no confirma la codificación esperada benigno=0, maligno=1."
        )
    try:
        model = joblib.load(required["modelo"])
    except Exception as error:
        raise CancerDetectionError(
            "No fue posible abrir models/best_model.joblib. Puede haberse creado "
            "con otra versión de scikit-learn; instala las versiones indicadas "
            f"en requirements.txt. Detalle: {error}"
        ) from error
    _validate_model(model, feature_names)
    return CancerArtifacts(
        model=model,
        train=train,
        test=test,
        report=report,
        comparison=comparison,
        feature_names=feature_names,
        train_min=train[list(feature_names)].min(),
        train_max=train[list(feature_names)].max(),
    )


def validate_measurements(
    measurements: dict[str, Any],
    feature_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    names = feature_names
    if names is None:
        raise CancerDetectionError("No se configuró el esquema de las variables.")
    missing = [name for name in names if name not in measurements or measurements[name] is None]
    extra = [name for name in measurements if name not in names]
    if missing:
        raise CancerDetectionError(
            "Faltan mediciones obligatorias: " + ", ".join(missing) + "."
        )
    if extra:
        raise CancerDetectionError("Columnas no esperadas: " + ", ".join(extra) + ".")
    values: list[float] = []
    for name in names:
        value = measurements[name]
        if isinstance(value, (bool, np.bool_)):
            raise CancerDetectionError(f"{name}: se requiere un valor numérico.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise CancerDetectionError(f"{name}: se requiere un valor numérico.") from error
        if not math.isfinite(numeric):
            raise CancerDetectionError(f"{name}: el valor debe ser finito.")
        values.append(numeric)
    return pd.DataFrame([values], columns=names)


def predict_frame(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    if tuple(features.columns) != tuple(getattr(model, "feature_names_in_", features.columns)):
        raise CancerDetectionError(
            "Las columnas u orden de entrada no coinciden con feature_names_in_ del modelo."
        )
    values = features.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CancerDetectionError("Las entradas deben ser numéricas y finitas.")
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise CancerDetectionError("El modelo no tiene ambas clases 0 y 1.")
    probability_matrix = model.predict_proba(features)
    benign_index = classes.index(0)
    malignant_index = classes.index(1)
    probabilities_benign = probability_matrix[:, benign_index]
    probabilities_malignant = probability_matrix[:, malignant_index]
    predicted = model.predict(features)
    result = pd.DataFrame(index=features.index)
    result["clase_predicha"] = [
        "maligno" if int(label) == 1 else "benigno" for label in predicted
    ]
    result["probabilidad_benigno"] = probabilities_benign
    result["probabilidad_maligno"] = probabilities_malignant
    return result


def evaluate_threshold(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if isinstance(threshold, bool) or not math.isfinite(float(threshold)):
        raise CancerDetectionError("El umbral debe ser un número finito entre 0 y 1.")
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise CancerDetectionError("El umbral debe estar entre 0 y 1.")
    classes = list(model.classes_)
    if 0 not in classes or 1 not in classes:
        raise CancerDetectionError("El modelo no tiene ambas clases 0 y 1.")
    probabilities = model.predict_proba(features)[:, classes.index(1)]
    predictions = (probabilities >= threshold).astype(int)
    observed = np.asarray(labels, dtype=int)
    if len(observed) != len(predictions) or not set(np.unique(observed)).issubset({0, 1}):
        raise CancerDetectionError("Las etiquetas observadas deben ser 0 o 1 y alinearse con las filas.")
    matrix = confusion_matrix(observed, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = float(recall_score(observed, predictions, pos_label=1, zero_division=0))
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics = {
        "accuracy": float(accuracy_score(observed, predictions)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(precision_score(observed, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(observed, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    return metrics, predictions, probabilities


def validate_csv_upload(payload: bytes, feature_names: tuple[str, ...]) -> CSVValidation:
    if not payload:
        raise CancerDetectionError("El archivo CSV está vacío.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise CancerDetectionError("El CSV debe estar codificado en UTF-8.") from error
    try:
        header = next(csv.reader(StringIO(text)))
    except StopIteration as error:
        raise CancerDetectionError("El archivo CSV no contiene encabezados.") from error
    duplicates = sorted({name for name in header if header.count(name) > 1})
    try:
        data = pd.read_csv(StringIO(text), dtype=object, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise CancerDetectionError(f"No se pudo interpretar el CSV: {error}") from error
    if len(header) != len(data.columns):
        raise CancerDetectionError("El encabezado CSV no coincide con el contenido.")

    allowed = set(feature_names) | {"source_index", "malignant"}
    missing = [name for name in feature_names if name not in header]
    additional = [name for name in header if name not in allowed]
    errors: list[dict[str, Any]] = []
    if duplicates:
        errors.append({
            "fila": "Todas",
            "columna": ", ".join(duplicates),
            "error": "Columna duplicada en el encabezado.",
        })
    if missing:
        errors.append({
            "fila": "Todas",
            "columna": ", ".join(missing),
            "error": "Faltan columnas predictoras obligatorias.",
        })
    if additional:
        errors.append({
            "fila": "Todas",
            "columna": ", ".join(additional),
            "error": "Columna adicional no permitida.",
        })
    schema_valid = not (duplicates or missing or additional)
    valid_values: list[list[float] | None] = []
    valid_flags: list[bool] = []
    for index, (_, row) in enumerate(data.iterrows()):
        row_valid = schema_valid
        numeric_values: list[float] = []
        if schema_valid:
            for feature in feature_names:
                raw = row[feature]
                if pd.isna(raw) or str(raw).strip() == "":
                    errors.append({
                        "fila": index + 2,
                        "columna": feature,
                        "error": "Valor faltante.",
                    })
                    row_valid = False
                    continue
                try:
                    number = float(raw)
                except (TypeError, ValueError, OverflowError):
                    errors.append({
                        "fila": index + 2,
                        "columna": feature,
                        "error": "El valor no es numérico.",
                    })
                    row_valid = False
                    continue
                if not math.isfinite(number):
                    errors.append({
                        "fila": index + 2,
                        "columna": feature,
                        "error": "El valor no es finito.",
                    })
                    row_valid = False
                    continue
                numeric_values.append(number)
            if "malignant" in data.columns:
                raw_label = row["malignant"]
                try:
                    label = float(raw_label)
                except (TypeError, ValueError, OverflowError):
                    label = math.nan
                if not math.isfinite(label) or label not in (0, 1):
                    errors.append({
                        "fila": index + 2,
                        "columna": "malignant",
                        "error": "La etiqueta observada opcional debe ser 0 o 1.",
                    })
        valid_flags.append(row_valid)
        valid_values.append(numeric_values if row_valid else None)

    feature_rows = [
        values if values is not None else [np.nan] * len(feature_names)
        for values in valid_values
    ]
    features = pd.DataFrame(feature_rows, columns=feature_names, index=data.index)
    valid_mask = pd.Series(valid_flags, index=data.index, dtype=bool)
    return CSVValidation(data=data, features=features, valid_mask=valid_mask, errors=errors)
