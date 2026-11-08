from dataclasses import dataclass
import math
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC


@dataclass(frozen=True)
class CancerResult:
    label: str
    benign_probability: float
    malignant_probability: float


@dataclass(frozen=True)
class EmotionResult:
    label: str


def load_cancer_model(model_path: Path):
    model = joblib.load(model_path)
    if not hasattr(model, "feature_names_in_") or not hasattr(model, "predict_proba"):
        raise ValueError("El archivo no contiene un modelo SVM compatible.")
    if list(model.classes_) != [0, 1]:
        raise ValueError("El modelo de cáncer debe usar 0 = benigno y 1 = maligno.")
    return model


def predict_cancer(model, measurements: dict[str, float]) -> CancerResult:
    feature_names = list(model.feature_names_in_)
    missing = [name for name in feature_names if name not in measurements]
    extra = [name for name in measurements if name not in feature_names]
    if missing or extra:
        raise ValueError("Se requiere exactamente el conjunto de 30 variables del modelo.")

    values = [measurements[name] for name in feature_names]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Las mediciones deben ser números finitos mayores o iguales a cero.")

    row = pd.DataFrame([values], columns=feature_names)
    probabilities = model.predict_proba(row)[0]
    predicted_class = int(model.predict(row)[0])
    return CancerResult(
        label="Maligno" if predicted_class == 1 else "Benigno",
        benign_probability=float(probabilities[0]),
        malignant_probability=float(probabilities[1]),
    )


def train_emotion_model(training_data_path: Path):
    data = pd.read_csv(training_data_path)
    if not {"text", "emotion"}.issubset(data.columns):
        raise ValueError("El conjunto de emociones debe incluir las columnas 'text' y 'emotion'.")
    if data.empty or data["text"].isna().any() or data["emotion"].isna().any():
        raise ValueError("El conjunto de emociones está vacío o contiene valores faltantes.")
    if data["emotion"].nunique() < 2:
        raise ValueError("Se necesitan al menos dos emociones para entrenar el clasificador.")

    model = Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(
                ngram_range=(1, 2),
                strip_accents="unicode",
                sublinear_tf=True,
            )),
            ("characters", TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                strip_accents="unicode",
                sublinear_tf=True,
            )),
        ])),
        ("classifier", SVC(
            C=2.0,
            class_weight="balanced",
            kernel="linear",
            random_state=42,
        )),
    ])
    model.fit(data["text"], data["emotion"])
    return model


def predict_emotion(model, text: str) -> EmotionResult:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Escribe una oración para analizar.")

    prediction = model.predict([cleaned_text])[0]
    return EmotionResult(
        label=str(prediction).capitalize(),
    )
