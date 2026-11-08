from pathlib import Path

import pandas as pd
import pytest

from src.emotion_detection import (
    EmotionDetectionError,
    load_emotion_artifacts,
    predict_emotion,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "emotion_samples.csv"


@pytest.fixture(scope="module")
def emotion_artifacts():
    return load_emotion_artifacts(SAMPLES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Estoy muy feliz y agradecida por este día tan bonito.", "alegría"),
        ("Me siento triste desde que mi mejor amigo se fue.", "tristeza"),
        ("Estoy muy enojado porque no respetaron el acuerdo.", "enojo"),
        ("Me da miedo caminar sola por esta calle oscura.", "miedo"),
        ("¡No esperaba verte aquí; qué sorpresa!", "sorpresa"),
        ("La reunión comienza a las nueve de la mañana.", "neutral"),
    ],
)
def test_svm_classifies_examples_from_each_emotion_category(
    emotion_artifacts, text, expected
):
    assert predict_emotion(emotion_artifacts.model, text) == expected


def test_emotion_sample_set_has_six_categories_and_balanced_examples(
    emotion_artifacts,
):
    assert emotion_artifacts.sample_count == 72
    assert emotion_artifacts.emotion_counts == (
        ("alegría", 12),
        ("enojo", 12),
        ("miedo", 12),
        ("neutral", 12),
        ("sorpresa", 12),
        ("tristeza", 12),
    )


@pytest.mark.parametrize("text", ["", "   ", None])
def test_emotion_prediction_rejects_empty_or_non_text_input(
    emotion_artifacts, text
):
    with pytest.raises(EmotionDetectionError, match="Escribe una oración"):
        predict_emotion(emotion_artifacts.model, text)


def test_emotion_loader_reports_invalid_training_data(tmp_path: Path):
    invalid_csv = tmp_path / "invalid.csv"
    pd.DataFrame({"sentence": ["Texto"], "label": ["alegría"]}).to_csv(
        invalid_csv, index=False
    )
    with pytest.raises(EmotionDetectionError, match="columnas"):
        load_emotion_artifacts(invalid_csv)
