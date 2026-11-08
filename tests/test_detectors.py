from pathlib import Path

import pytest
from sklearn.datasets import load_breast_cancer

from detectors import (
    load_cancer_model,
    predict_cancer,
    predict_emotion,
    train_emotion_model,
)


APP_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cancer_model():
    return load_cancer_model(APP_DIR / "models" / "best_model_variation.joblib")


@pytest.fixture(scope="module")
def emotion_model():
    return train_emotion_model(APP_DIR / "data" / "emotion_samples.csv")


def test_cancer_model_predicts_both_dataset_classes(cancer_model):
    dataset = load_breast_cancer()
    for dataset_target, expected_label in ((1, "Benigno"), (0, "Maligno")):
        index = next(i for i, target in enumerate(dataset.target) if target == dataset_target)
        values = dict(zip(dataset.feature_names, dataset.data[index]))
        result = predict_cancer(cancer_model, values)

        assert result.label == expected_label
        assert result.benign_probability + result.malignant_probability == pytest.approx(1.0)


def test_cancer_prediction_rejects_missing_features(cancer_model):
    with pytest.raises(ValueError, match="exactamente"):
        predict_cancer(cancer_model, {})


def test_cancer_prediction_rejects_non_finite_measurements(cancer_model):
    dataset = load_breast_cancer()
    values = dict(zip(dataset.feature_names, dataset.data[0]))
    values["mean radius"] = float("nan")

    with pytest.raises(ValueError, match="números finitos"):
        predict_cancer(cancer_model, values)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Estoy muy feliz y agradecida por este día tan bonito.", "Alegría"),
        ("Me siento triste desde que mi mejor amigo se fue.", "Tristeza"),
        ("Estoy muy enojado porque no respetaron el acuerdo.", "Enojo"),
        ("Me da miedo caminar sola por esta calle oscura.", "Miedo"),
        ("¡No esperaba verte aquí; qué sorpresa!", "Sorpresa"),
        ("La reunión comienza a las nueve de la mañana.", "Neutral"),
    ],
)
def test_emotion_model_predicts_sample_emotions(emotion_model, text, expected):
    result = predict_emotion(emotion_model, text)

    assert result.label == expected


def test_emotion_prediction_rejects_empty_text(emotion_model):
    with pytest.raises(ValueError, match="Escribe una oración"):
        predict_emotion(emotion_model, "   ")
