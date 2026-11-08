from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.cancer_detection import (
    CancerDetectionError,
    evaluate_threshold,
    load_artifacts,
    predict_frame,
    validate_csv_upload,
    validate_measurements,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts(ROOT)


def test_schema_has_30_features_in_model_order_and_excludes_metadata(artifacts):
    assert len(artifacts.feature_names) == 30
    assert tuple(artifacts.model.feature_names_in_) == artifacts.feature_names
    assert "source_index" not in artifacts.feature_names
    assert "malignant" not in artifacts.feature_names
    assert tuple(artifacts.test.loc[:, artifacts.feature_names].columns) == artifacts.feature_names


def test_class_mapping_is_benign_zero_and_malignant_one(artifacts):
    assert list(artifacts.model.classes_) == [0, 1]
    assert artifacts.report["target_mapping"] == {"benign": 0, "malignant": 1}
    benign = artifacts.test.loc[artifacts.test["malignant"] == 0].iloc[[0]]
    malignant = artifacts.test.loc[artifacts.test["malignant"] == 1].iloc[[0]]
    assert predict_frame(artifacts.model, benign[list(artifacts.feature_names)]).iloc[0][
        "clase_predicha"
    ] == "benigno"
    assert predict_frame(artifacts.model, malignant[list(artifacts.feature_names)]).iloc[0][
        "clase_predicha"
    ] == "maligno"


def test_manual_inference_matches_complete_saved_pipeline(artifacts):
    row = artifacts.test.loc[artifacts.test["malignant"] == 1].iloc[[0]]
    actual = predict_frame(artifacts.model, row[list(artifacts.feature_names)]).iloc[0]
    probabilities = artifacts.model.predict_proba(row[list(artifacts.feature_names)])[0]
    class_index = list(artifacts.model.classes_).index(1)
    expected_prediction = int(artifacts.model.predict(row[list(artifacts.feature_names)])[0])
    assert actual["clase_predicha"] == ("maligno" if expected_prediction == 1 else "benigno")
    assert actual["probabilidad_maligno"] == pytest.approx(probabilities[class_index])
    assert actual["probabilidad_benigno"] == pytest.approx(
        probabilities[list(artifacts.model.classes_).index(0)]
    )


def test_measurement_validation_rejects_missing_nonnumeric_nan_and_infinity(artifacts):
    valid = {name: 1.0 for name in artifacts.feature_names}
    with pytest.raises(CancerDetectionError, match="Faltan"):
        validate_measurements({}, artifacts.feature_names)
    invalid_cases = [
        ("not-a-number", "numérico"),
        (float("nan"), "finito"),
        (float("inf"), "finito"),
        (None, "Faltan"),
    ]
    for bad_value, message in invalid_cases:
        row = dict(valid)
        row[artifacts.feature_names[0]] = bad_value
        with pytest.raises(CancerDetectionError, match=message):
            validate_measurements(row, artifacts.feature_names)


def test_measurement_validation_preserves_column_order_and_excludes_metadata(artifacts):
    values = {name: float(i) for i, name in enumerate(reversed(artifacts.feature_names))}
    ordered = validate_measurements(values, artifacts.feature_names)
    assert tuple(ordered.columns) == artifacts.feature_names
    assert ordered.iloc[0].tolist() == [float(i) for i in reversed(range(30))]
    assert "source_index" not in ordered and "malignant" not in ordered


def test_csv_validation_reports_invalid_rows_and_ignores_metadata_as_features(artifacts):
    row = artifacts.test.iloc[0]
    csv = pd.DataFrame([row.to_dict()]).to_csv(index=False).encode("utf-8")
    result = validate_csv_upload(csv, artifacts.feature_names)
    assert result.valid_mask.tolist() == [True]
    assert tuple(result.features.columns) == artifacts.feature_names
    assert "source_index" not in result.features
    assert "malignant" not in result.features

    invalid = row[list(artifacts.feature_names)].to_dict()
    invalid[artifacts.feature_names[0]] = "no-numérico"
    csv = pd.DataFrame([invalid]).to_csv(index=False).encode("utf-8")
    result = validate_csv_upload(csv, artifacts.feature_names)
    assert result.valid_mask.tolist() == [False]
    assert any(error["columna"] == artifacts.feature_names[0] for error in result.errors)


def test_invalid_optional_observed_label_does_not_block_valid_input(artifacts):
    features = list(artifacts.feature_names)
    values = ["1"] * len(features)
    content = ",".join(features + ["malignant"]) + "\n"
    content += ",".join(values + ["etiqueta inválida"]) + "\n"
    result = validate_csv_upload(content.encode("utf-8"), artifacts.feature_names)
    assert result.valid_mask.tolist() == [True]
    assert any(error["columna"] == "malignant" for error in result.errors)


def test_csv_validation_rejects_duplicate_and_additional_columns(artifacts):
    names = list(artifacts.feature_names)
    row = ",".join(["1"] * len(names))
    duplicate_csv = (",".join(names + [names[0]]) + "\n" + row + ",1\n").encode()
    duplicate = validate_csv_upload(duplicate_csv, artifacts.feature_names)
    assert not duplicate.valid_mask.any()
    assert any("duplicada" in error["error"] for error in duplicate.errors)

    extra_csv = (
        ",".join(names + ["identificador"])
        + "\n"
        + row
        + ",abc\n"
    ).encode()
    extra = validate_csv_upload(extra_csv, artifacts.feature_names)
    assert not extra.valid_mask.any()
    assert any("adicional" in error["error"] for error in extra.errors)


@pytest.mark.parametrize(
    ("bad_value", "expected_error"),
    [
        ("", "faltante"),
        ("NaN", "no es finito"),
        ("inf", "no es finito"),
        ("sin-dato", "no es numérico"),
    ],
)
def test_csv_validation_reports_missing_nan_infinite_and_nonnumeric(
    artifacts, bad_value, expected_error
):
    features = list(artifacts.feature_names)
    values = ["1"] * len(features)
    values[0] = bad_value
    content = ",".join(features) + "\n" + ",".join(values) + "\n"
    result = validate_csv_upload(content.encode("utf-8"), artifacts.feature_names)
    assert result.valid_mask.tolist() == [False]
    assert any(
        error["columna"] == features[0] and expected_error in error["error"].lower()
        for error in result.errors
    )


def test_saved_report_metrics_match_recalculated_test_metrics(artifacts):
    report = artifacts.report["test_metrics"]
    calculated, predictions, probabilities = evaluate_threshold(
        artifacts.model,
        artifacts.test[list(artifacts.feature_names)],
        artifacts.test["malignant"],
        0.50,
    )
    for report_name, calculated_name in (
        ("accuracy", "accuracy"),
        ("sensitivity_recall", "sensitivity"),
        ("specificity", "specificity"),
        ("precision", "precision"),
        ("f1", "f1"),
    ):
        assert calculated[calculated_name] == pytest.approx(report[report_name], abs=1e-8)
    assert calculated["confusion_matrix"] == report["confusion_matrix"]
    assert roc_auc_score(artifacts.test["malignant"], probabilities) == pytest.approx(
        report["roc_auc"], abs=1e-8
    )
    assert np.array_equal(
        predictions, (probabilities >= 0.50).astype(int)
    )


def test_threshold_rule_is_probability_one_greater_or_equal(artifacts):
    metrics, predictions, probabilities = evaluate_threshold(
        artifacts.model,
        artifacts.test[list(artifacts.feature_names)],
        artifacts.test["malignant"],
        0.75,
    )
    assert np.array_equal(predictions, (probabilities >= 0.75).astype(int))
    assert sum(metrics["confusion_matrix"][0]) + sum(metrics["confusion_matrix"][1]) == len(
        artifacts.test
    )
    with pytest.raises(CancerDetectionError, match="entre 0 y 1"):
        evaluate_threshold(
            artifacts.model,
            artifacts.test[list(artifacts.feature_names)],
            artifacts.test["malignant"],
            1.1,
        )
