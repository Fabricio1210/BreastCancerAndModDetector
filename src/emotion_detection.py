from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import SVC


class EmotionDetectionError(ValueError):
    """Error loading the local training data or classifying text."""


@dataclass(frozen=True)
class EmotionArtifacts:
    model: Pipeline
    sample_count: int
    emotion_counts: tuple[tuple[str, int], ...]


def load_emotion_artifacts(training_data_path: Path) -> EmotionArtifacts:
    try:
        data = pd.read_csv(training_data_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise EmotionDetectionError(
            f"No se pudo leer el conjunto local de frases: {error}"
        ) from error

    required_columns = {"text", "emotion"}
    if not required_columns.issubset(data.columns):
        raise EmotionDetectionError(
            "El CSV de emociones debe incluir las columnas 'text' y 'emotion'."
        )
    if data.empty:
        raise EmotionDetectionError("El conjunto local de frases está vacío.")
    if data["text"].isna().any() or data["emotion"].isna().any():
        raise EmotionDetectionError(
            "El conjunto local contiene frases o etiquetas faltantes."
        )

    data["text"] = data["text"].astype(str).str.strip()
    data["emotion"] = data["emotion"].astype(str).str.strip()
    if data["text"].eq("").any() or data["emotion"].eq("").any():
        raise EmotionDetectionError(
            "El conjunto local contiene frases o etiquetas vacías."
        )
    if data["emotion"].nunique() < 2:
        raise EmotionDetectionError(
            "Se necesitan al menos dos categorías para entrenar el clasificador."
        )

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
    counts = data["emotion"].value_counts().sort_index()
    return EmotionArtifacts(
        model=model,
        sample_count=len(data),
        emotion_counts=tuple((str(label), int(count)) for label, count in counts.items()),
    )


def predict_emotion(model: Pipeline, text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise EmotionDetectionError("Escribe una oración para analizar.")
    return str(model.predict([text.strip()])[0])
