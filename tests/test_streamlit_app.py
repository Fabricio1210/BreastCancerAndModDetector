from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def test_all_navigation_pages_render_without_app_exceptions():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception

    for section in (
        "Clasificación individual",
        "Carga CSV",
        "Laboratorio",
        "Evaluación",
        "Detector de emociones",
        "Inicio",
    ):
        app.radio[0].set_value(section).run()
        assert not app.exception


def test_individual_example_can_be_classified_and_lab_threshold_changes():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value("Clasificación individual").run()
    assert len(app.number_input) == 30

    app.button[0].click().run()
    app.button[-1].click().run()
    assert "individual_result" in app.session_state
    assert not app.exception

    app.radio[0].set_value("Laboratorio").run()
    assert len(app.get("plotly_chart")) == 2
    app.slider[0].set_value(0.75).run()
    assert app.slider[0].value == 0.75
    assert not app.exception


def test_emotion_detector_classifies_text_and_rejects_empty_input():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.radio[0].set_value("Detector de emociones").run()
    assert not app.exception

    app.text_area[0].set_value("Estoy muy feliz y agradecida por este día.")
    app.button[0].click().run()
    assert any(
        "Categoría textual predicha: Alegría" in header.value
        for header in app.subheader
    )
    assert not app.exception

    app.text_area[0].set_value("   ")
    app.button[0].click().run()
    assert any("Escribe una oración" in error.value for error in app.error)
    assert not app.exception
