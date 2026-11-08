from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.cancer_detection import (
    CancerArtifacts,
    CancerDetectionError,
    evaluate_threshold,
    load_artifacts,
    predict_frame,
    validate_csv_upload,
    validate_measurements,
)
from src.emotion_detection import (
    EmotionDetectionError,
    load_emotion_artifacts,
    predict_emotion,
)


ROOT = Path(__file__).resolve().parent
ACADEMIC_NOTICE = (
    "**Uso académico. Esta predicción no constituye un diagnóstico médico.**"
)
EMOTION_NOTICE = (
    "Este clasificador identifica patrones de texto en ejemplos de práctica; "
    "no determina el estado emocional real de una persona."
)

st.set_page_config(
    page_title="CancerDetection — Laboratorio interactivo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Cargando el modelo SVM local…")
def get_artifacts() -> CancerArtifacts:
    return load_artifacts(ROOT)


@st.cache_resource(show_spinner="Preparando el clasificador de texto SVM…")
def get_emotion_artifacts():
    return load_emotion_artifacts(ROOT / "data" / "emotion_samples.csv")


@st.cache_data(show_spinner=False)
def get_translations() -> dict[str, str]:
    return {
        "radius": "radio",
        "texture": "textura",
        "perimeter": "perímetro",
        "area": "área",
        "smoothness": "suavidad",
        "compactness": "compacidad",
        "concavity": "concavidad",
        "concave points": "puntos cóncavos",
        "symmetry": "simetría",
        "fractal dimension": "dimensión fractal",
    }


def spanish_feature_name(feature: str) -> str:
    translations = get_translations()
    if feature.startswith("mean "):
        return f"Media de {translations[feature[5:]]} ({feature})"
    if feature.startswith("worst "):
        return f"Peor valor de {translations[feature[6:]]} ({feature})"
    if feature.endswith(" error"):
        return f"Error de {translations[feature[:-6]]} ({feature})"
    return feature


def prediction_chart(result: pd.DataFrame, title: str = "Probabilidades estimadas") -> go.Figure:
    probabilities = result.iloc[0]
    chart_data = pd.DataFrame({
        "Clase": ["Benigno (0)", "Maligno (1)"],
        "Probabilidad estimada": [
            probabilities["probabilidad_benigno"],
            probabilities["probabilidad_maligno"],
        ],
    })
    figure = px.bar(
        chart_data,
        x="Clase",
        y="Probabilidad estimada",
        text=chart_data["Probabilidad estimada"].map(lambda value: f"{value:.1%}"),
        color="Clase",
        color_discrete_map={"Benigno (0)": "#38BDF8", "Maligno (1)": "#FB7185"},
        title=title,
    )
    figure.update_layout(
        yaxis={"range": [0, 1], "tickformat": ".0%"},
        showlegend=False,
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        height=330,
    )
    return figure


def show_prediction(result: pd.DataFrame, chart_key: str = "individual-prediction") -> None:
    prediction = result.iloc[0]
    label = prediction["clase_predicha"]
    icon = "🟠" if label == "maligno" else "🔵"
    st.subheader(f"{icon} Clase predicha: {label}")
    st.caption(
        "Probabilidades estimadas por el modelo; no equivalen a riesgo clínico validado."
    )
    first, second = st.columns(2)
    first.metric("Benigno (0)", f"{prediction['probabilidad_benigno']:.2%}")
    second.metric("Maligno (1)", f"{prediction['probabilidad_maligno']:.2%}")
    st.plotly_chart(
        prediction_chart(result),
        width="stretch",
        key=chart_key,
    )


def show_reference_range_warnings(
    row: pd.DataFrame, artifacts: CancerArtifacts
) -> None:
    warnings = []
    for feature in artifacts.feature_names:
        value = float(row.iloc[0][feature])
        low = artifacts.train_min[feature]
        high = artifacts.train_max[feature]
        if value < low or value > high:
            warnings.append(
                f"**{spanish_feature_name(feature)}** = `{value:.8g}` está fuera del "
                f"rango observado en entrenamiento (`{low:.8g}` a `{high:.8g}`)."
            )
    if warnings:
        st.warning(
            "Una o más entradas quedan fuera del rango observado en `train.csv`. "
            "Estos rangos son referencias estadísticas del conjunto de entrenamiento, "
            "no límites clínicos."
        )
        for warning in warnings:
            st.markdown(f"- {warning}")


def page_home(artifacts: CancerArtifacts) -> None:
    st.title("CancerDetection — Laboratorio interactivo de clasificación")
    st.caption(ACADEMIC_NOTICE)
    st.write(
        "Explora cómo un clasificador Support Vector Machine (SVM) procesa las "
        "30 mediciones numéricas del conjunto Wisconsin Diagnostic Breast Cancer. "
        "Esta aplicación es un recurso de aprendizaje sobre clasificación, no una "
        "herramienta para tomar decisiones médicas."
    )
    st.info(
        "**Codificación de este proyecto:** benigno = `0`; maligno = `1`. "
        "Esto se verifica con la columna `malignant` de los CSV y los metadatos "
        "del reporte. Scikit-learn entrega originalmente benigno = 1 y maligno = 0; "
        "el proyecto invirtió esa codificación."
    )

    metrics = artifacts.report["test_metrics"]
    cards = st.columns(4)
    cards[0].metric("Registros de entrenamiento", f"{len(artifacts.train):,}")
    cards[1].metric("Registros de prueba", f"{len(artifacts.test):,}")
    cards[2].metric("Variables predictoras", f"{len(artifacts.feature_names)}")
    cards[3].metric("Modelo seleccionado", "SVM (RBF)")

    st.subheader("Resultados del reporte original")
    st.caption(
        "Métricas leídas de `reports/metrics.json`; describen el conjunto de prueba "
        "del experimento, no garantizan el resultado de un registro individual."
    )
    metric_cards = st.columns(5)
    metric_cards[0].metric("Exactitud (accuracy)", f"{metrics['accuracy']:.1%}")
    metric_cards[1].metric("Precisión (PPV)", f"{metrics['precision']:.1%}")
    metric_cards[2].metric("Sensibilidad", f"{metrics['sensitivity_recall']:.1%}")
    metric_cards[3].metric("Especificidad", f"{metrics['specificity']:.1%}")
    metric_cards[4].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    st.caption(
        "**Exactitud** = proporción total de aciertos. "
        "**Precisión (valor predictivo positivo)** = proporción de predicciones "
        "malignas que fueron malignas en el conjunto evaluado."
    )

    st.subheader("¿Qué datos contiene el dataset?")
    st.write(
        "El conjunto Wisconsin Diagnostic Breast Cancer contiene 569 registros y "
        "30 variables numéricas calculadas a partir de imágenes digitalizadas de "
        "aspirados con aguja fina de masas mamarias. Las variables describen diez "
        "características celulares (radio, textura, perímetro, área, suavidad, "
        "compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal) "
        "como media, error y peor valor. El dataset no proporciona unidades para "
        "estas columnas en los archivos de este proyecto."
    )
    st.dataframe(
        pd.DataFrame({
            "Nombre de columna": artifacts.feature_names,
            "Etiqueta en español": [
                spanish_feature_name(name) for name in artifacts.feature_names
            ],
        }),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Flujo del experimento")
    st.markdown(
        """
        1. **Entrenamiento y selección:** comparación con validación cruzada estratificada
           sobre los datos de entrenamiento; el SVM fue seleccionado por ROC AUC.
        2. **Estandarización:** el `StandardScaler` ya forma parte del pipeline guardado.
           La app usa el pipeline completo y no vuelve a ajustarlo.
        3. **Predicción:** se ordenan las 30 variables como espera el artefacto y se
           consultan la clase y las probabilidades del modelo local.
        4. **Evaluación:** las métricas del reporte provienen del conjunto de prueba
           existente. Cambiar umbrales o variables sirve para explorar el modelo, no
           para demostrar causalidad ni validar su uso clínico.
        """
    )
    st.subheader("Actividad breve de clase")
    st.markdown(
        """
        1. En **Clasificación individual**, carga el ejemplo benigno o maligno del
           benchmark y observa la clase y las probabilidades.
        2. En **Laboratorio**, modifica una medición y compara el resultado antes
           y después.
        3. Ajusta el umbral y observa la matriz de confusión, sensibilidad,
           especificidad, precisión y F1 en `test.csv`.
        4. Discute por qué estos cambios no prueban causalidad ni validan un
           clasificador para uso médico.
        """
    )


def _set_individual_example(artifacts: CancerArtifacts, label: int | None) -> None:
    if label is None:
        for feature in artifacts.feature_names:
            st.session_state.pop(f"individual_{feature}", None)
        st.session_state.pop("individual_result", None)
        st.session_state["individual_example_note"] = ""
        return
    examples = artifacts.test.loc[artifacts.test["malignant"] == label]
    if examples.empty:
        st.session_state["individual_example_note"] = (
            "No existe un ejemplo de esta clase en el conjunto test.csv."
        )
        return
    sample = examples.iloc[0]
    for feature in artifacts.feature_names:
        st.session_state[f"individual_{feature}"] = float(sample[feature])
    st.session_state.pop("individual_result", None)
    class_label = "maligno" if label == 1 else "benigno"
    st.session_state["individual_example_note"] = (
        f"Ejemplo precargado: fila de prueba source_index={sample['source_index']} "
        f"(clase observada: {class_label}). Pertenece al benchmark existente; "
        "no es una validación independiente."
    )


def page_individual(artifacts: CancerArtifacts) -> None:
    st.title("Clasificación individual")
    st.caption(ACADEMIC_NOTICE)
    st.write(
        "Ingresa las 30 mediciones que requiere el modelo. No se solicitan datos "
        "identificables, edad, síntomas ni antecedentes. Ninguna variable se "
        "completará automáticamente."
    )

    buttons = st.columns(3)
    buttons[0].button(
        "Cargar ejemplo benigno",
        on_click=_set_individual_example,
        args=(artifacts, 0),
        width="stretch",
    )
    buttons[1].button(
        "Cargar ejemplo maligno",
        on_click=_set_individual_example,
        args=(artifacts, 1),
        width="stretch",
    )
    buttons[2].button(
        "Restablecer formulario",
        on_click=_set_individual_example,
        args=(artifacts, None),
        width="stretch",
    )
    note = st.session_state.get("individual_example_note", "")
    if note:
        st.info(note)

    group_names = {
        "mean": (
            "Valores medios",
            [name for name in artifacts.feature_names if name.startswith("mean ")],
        ),
        "error": (
            "Errores",
            [name for name in artifacts.feature_names if name.endswith(" error")],
        ),
        "worst": (
            "Valores worst (peor valor)",
            [name for name in artifacts.feature_names if name.startswith("worst ")],
        ),
    }
    entered: dict[str, float | None] = {}
    with st.form("individual_form"):
        st.caption(
            "Las etiquetas en español conservan entre paréntesis el nombre original "
            "de cada columna. Las variables no tienen unidades indicadas en los CSV."
        )
        for group_title, features in group_names.values():
            st.subheader(group_title)
            columns = st.columns(3)
            for index, feature in enumerate(features):
                with columns[index % 3]:
                    entered[feature] = st.number_input(
                        spanish_feature_name(feature),
                        value=None,
                        format="%.8g",
                        step=0.000001,
                        key=f"individual_{feature}",
                        help="Captura el valor numérico del dataset; no se infieren datos faltantes.",
                    )
        submitted = st.form_submit_button("Clasificar registro", type="primary")

    if submitted:
        try:
            row = validate_measurements(entered, artifacts.feature_names)
            result = predict_frame(artifacts.model, row)
            st.session_state["individual_result"] = result
            st.session_state["individual_input"] = row
        except CancerDetectionError as error:
            st.error(str(error))

    if "individual_result" in st.session_state:
        show_prediction(st.session_state["individual_result"])
        row = st.session_state["individual_input"]
        show_reference_range_warnings(row, artifacts)
        st.subheader("Entradas utilizadas")
        localized = row.T.reset_index()
        localized.columns = ["Columna original", "Valor"]
        localized.insert(1, "Variable en español", localized["Columna original"].map(spanish_feature_name))
        st.dataframe(localized, hide_index=True, width="stretch")


def page_csv(artifacts: CancerArtifacts) -> None:
    st.title("Clasificación por CSV")
    st.caption(ACADEMIC_NOTICE)
    st.write(
        "Los archivos se procesan solo en memoria y no se guardan por defecto. "
        "La columna `malignant`, si existe, se conserva como etiqueta observada "
        "opcional; nunca se usa como predictor."
    )
    template = pd.DataFrame(columns=artifacts.feature_names)
    st.download_button(
        "Descargar plantilla (30 columnas)",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="plantilla_cancer_detection.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Selecciona un archivo CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        validation = validate_csv_upload(uploaded.getvalue(), artifacts.feature_names)
    except CancerDetectionError as error:
        st.error(str(error))
        return

    if validation.errors:
        st.warning("Se encontraron problemas de estructura o de datos por fila/columna.")
        st.dataframe(pd.DataFrame(validation.errors), hide_index=True, width="stretch")
    if validation.data.empty:
        st.error("El CSV no contiene registros para procesar.")
        return

    valid_rows = validation.valid_mask
    outputs = pd.DataFrame(index=validation.data.index)
    outputs["fila_csv"] = np.arange(2, len(validation.data) + 2)
    if "source_index" in validation.data:
        outputs["source_index"] = validation.data["source_index"].astype(str).to_numpy()
    outputs["estado_validacion"] = np.where(valid_rows, "Válida", "Inválida")
    row_errors: dict[int, list[str]] = {}
    for error in validation.errors:
        row_number = error.get("fila")
        if isinstance(row_number, int):
            row_errors.setdefault(row_number - 2, []).append(
                f"{error.get('columna', 'archivo')}: {error.get('error', '')}"
            )
    outputs["errores_validacion"] = [
        " | ".join(row_errors.get(index, [])) for index in outputs.index
    ]
    global_errors = [
        f"{error['columna']}: {error['error']}"
        for error in validation.errors
        if error.get("fila") == "Todas"
    ]
    if global_errors:
        outputs["errores_validacion"] = outputs["errores_validacion"].where(
            outputs["errores_validacion"].ne(""),
            " | ".join(global_errors),
        )
    outputs["clase_predicha"] = ""
    outputs["probabilidad_benigno"] = np.nan
    outputs["probabilidad_maligno"] = np.nan
    outputs["alertas_rango_train"] = ""

    if valid_rows.any():
        valid_features = validation.features.loc[valid_rows]
        predictions = predict_frame(artifacts.model, valid_features)
        outputs.loc[valid_rows, "clase_predicha"] = predictions["clase_predicha"].to_numpy()
        outputs.loc[valid_rows, "probabilidad_benigno"] = predictions["probabilidad_benigno"].to_numpy()
        outputs.loc[valid_rows, "probabilidad_maligno"] = predictions["probabilidad_maligno"].to_numpy()
        range_alerts = []
        for _, row in valid_features.iterrows():
            outside = [
                feature
                for feature in artifacts.feature_names
                if row[feature] < artifacts.train_min[feature]
                or row[feature] > artifacts.train_max[feature]
            ]
            range_alerts.append(
                "Fuera del rango train (no clínico): " + ", ".join(outside)
                if outside
                else ""
            )
        outputs.loc[valid_rows, "alertas_rango_train"] = range_alerts
    observed = validation.data.get("malignant")
    if observed is not None:
        outputs["malignant_observado"] = observed.astype(str).to_numpy()

    st.write(
        f"Filas: **{len(outputs)}** · válidas: **{int(valid_rows.sum())}** · "
        f"inválidas: **{int((~valid_rows).sum())}**. "
        "Las filas inválidas no fueron clasificadas."
    )
    st.dataframe(outputs, hide_index=True, width="stretch")
    st.download_button(
        "Descargar resultados CSV",
        data=outputs.to_csv(index=False).encode("utf-8"),
        file_name="resultados_cancer_detection.csv",
        mime="text/csv",
    )
    if observed is not None and "malignant" in validation.data:
        numeric_observed = pd.to_numeric(observed, errors="coerce")
        usable = valid_rows & numeric_observed.isin([0, 1])
        if usable.any():
            st.subheader("Evaluación opcional con etiquetas observadas")
            st.caption(
                f"Se conservaron {int(usable.sum())} etiquetas `malignant` válidas "
                "como observaciones opcionales; no participaron en la predicción."
            )
            metrics, _, _ = evaluate_threshold(
                artifacts.model,
                validation.features.loc[usable],
                numeric_observed.loc[usable],
                0.50,
            )
            evaluation_columns = st.columns(5)
            evaluation_columns[0].metric("Exactitud", f"{metrics['accuracy']:.1%}")
            evaluation_columns[1].metric("Sensibilidad", f"{metrics['sensitivity']:.1%}")
            evaluation_columns[2].metric("Especificidad", f"{metrics['specificity']:.1%}")
            evaluation_columns[3].metric("Precisión (PPV)", f"{metrics['precision']:.1%}")
            evaluation_columns[4].metric("F1", f"{metrics['f1']:.1%}")


def _reset_lab_values() -> None:
    for key in list(st.session_state):
        if key.startswith("laboratory_value_"):
            del st.session_state[key]
    st.session_state.pop("laboratory_before", None)
    st.session_state.pop("laboratory_after", None)


def _select_lab_example(artifacts: CancerArtifacts) -> None:
    index = st.session_state.get("laboratory_example_index", 0)
    row = artifacts.test.iloc[index]
    _reset_lab_values()
    for feature in artifacts.feature_names:
        st.session_state[f"laboratory_value_{feature}"] = float(row[feature])


def page_laboratory(artifacts: CancerArtifacts) -> None:
    st.title("Laboratorio interactivo")
    st.caption(ACADEMIC_NOTICE)
    st.warning(
        "Modificar variables permite explorar el comportamiento del modelo; no "
        "demuestra relaciones causales y puede producir combinaciones poco realistas. "
        "Las métricas exploratorias usan `test.csv`; ajustar el umbral con estos datos "
        "no equivale a validarlo. Una selección formal requiere entrenamiento o validación."
    )
    test = artifacts.test
    labels = [
        f"source_index={row.source_index} · etiqueta observada: "
        f"{'maligno' if row.malignant == 1 else 'benigno'}"
        for row in test.itertuples(index=False)
    ]
    chosen = st.selectbox(
        "Selecciona un ejemplo del benchmark de prueba",
        range(len(test)),
        format_func=lambda index: labels[index],
        key="laboratory_example_index",
        on_change=_select_lab_example,
        args=(artifacts,),
    )
    if f"laboratory_value_{artifacts.feature_names[0]}" not in st.session_state:
        _select_lab_example(artifacts)
    example = test.iloc[chosen]
    base_row = pd.DataFrame(
        [[float(example[name]) for name in artifacts.feature_names]],
        columns=artifacts.feature_names,
    )
    original_prediction = predict_frame(artifacts.model, base_row)

    threshold = st.slider(
        "Umbral de clase maligna",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01,
        help="Se clasifica como maligno si la probabilidad de clase 1 es mayor o igual al umbral.",
    )
    metrics, _, _ = evaluate_threshold(
        artifacts.model,
        artifacts.test[list(artifacts.feature_names)],
        artifacts.test["malignant"],
        threshold,
    )
    st.caption(
        f"Regla activa: clase maligna si P(clase 1) ≥ {threshold:.2f}; "
        "en otro caso, clase benigna."
    )
    st.subheader("Métricas exploratorias sobre test.csv")
    st.caption(
        "Estas métricas usan el umbral elegido y el conjunto de prueba del benchmark; "
        "son exploratorias y no deben confundirse con las métricas del reporte original."
    )
    cols = st.columns(5)
    cols[0].metric("Sensibilidad", f"{metrics['sensitivity']:.1%}")
    cols[1].metric("Especificidad", f"{metrics['specificity']:.1%}")
    cols[2].metric("Precisión (PPV)", f"{metrics['precision']:.1%}")
    cols[3].metric("F1", f"{metrics['f1']:.1%}")
    cols[4].metric("Exactitud", f"{metrics['accuracy']:.1%}")
    matrix = metrics["confusion_matrix"]
    matrix_frame = pd.DataFrame(
        matrix,
        index=["Observado: benigno (0)", "Observado: maligno (1)"],
        columns=["Predicho: benigno (0)", "Predicho: maligno (1)"],
    )
    st.subheader("Matriz de confusión exploratoria")
    st.dataframe(matrix_frame, width="stretch")
    st.caption(
        f"Verdaderos negativos: {matrix[0][0]} · Falsos positivos: {matrix[0][1]} · "
        f"Falsos negativos: {matrix[1][0]} · Verdaderos positivos: {matrix[1][1]}"
    )

    st.subheader("Modifica una o varias variables")
    st.info(
        f"Ejemplo original del benchmark: source_index={example['source_index']}; "
        f"etiqueta observada: {'maligno' if example['malignant'] == 1 else 'benigno'}. "
        "Usa «Restablecer ejemplo» para recuperar sus valores."
    )
    if st.button("Restablecer ejemplo original"):
        _select_lab_example(artifacts)
        st.rerun()

    selected_features = st.multiselect(
        "Variables que deseas modificar",
        artifacts.feature_names,
        format_func=spanish_feature_name,
    )
    edited = {name: float(example[name]) for name in artifacts.feature_names}
    if selected_features:
        columns = st.columns(3)
        for index, feature in enumerate(selected_features):
            with columns[index % 3]:
                edited[feature] = st.number_input(
                    spanish_feature_name(feature),
                    step=max(abs(float(example[feature])) * 0.05, 0.000001),
                    format="%.8g",
                    key=f"laboratory_value_{feature}",
                )
    changed_row = validate_measurements(edited, artifacts.feature_names)
    changed_prediction = predict_frame(artifacts.model, changed_row)
    before, after = st.columns(2)
    with before:
        st.markdown("**Antes — ejemplo original**")
        show_prediction(original_prediction, chart_key="laboratory-before")
    with after:
        st.markdown("**Después — valores modificados**")
        show_prediction(changed_prediction, chart_key="laboratory-after")

    comparison = pd.DataFrame({
        "Variable": artifacts.feature_names,
        "Etiqueta en español": [
            spanish_feature_name(name) for name in artifacts.feature_names
        ],
        "Valor original": [
            float(base_row.iloc[0][name]) for name in artifacts.feature_names
        ],
        "Valor modificado": [
            float(changed_row.iloc[0][name]) for name in artifacts.feature_names
        ],
    })
    comparison["Cambio"] = comparison["Valor modificado"] - comparison["Valor original"]
    st.subheader("Comparación de entradas")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(
        "Las predicciones del ejemplo usan `model.predict`, como el clasificador "
        "guardado. El umbral ajustable se aplica explícitamente a las métricas del "
        "conjunto de prueba."
    )


def page_evaluation(artifacts: CancerArtifacts) -> None:
    st.title("Evaluación y explicación")
    st.caption(ACADEMIC_NOTICE)
    st.subheader("Comparación de modelos")
    comparison = artifacts.comparison.copy()
    if "mean_roc_auc" in comparison.columns:
        st.plotly_chart(
            px.bar(
                comparison.sort_values("mean_roc_auc"),
                x="mean_roc_auc",
                y="model",
                orientation="h",
                labels={"mean_roc_auc": "ROC AUC medio (validación cruzada)", "model": "Modelo"},
                title="Comparación en entrenamiento",
            ),
            width="stretch",
        )
    st.dataframe(comparison, hide_index=True, width="stretch")

    y_true = artifacts.test["malignant"].astype(int).to_numpy()
    predictions = predict_frame(
        artifacts.model, artifacts.test[list(artifacts.feature_names)]
    )
    y_pred = (predictions["clase_predicha"] == "maligno").astype(int).to_numpy()
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicho benigno (0)", "Predicho maligno (1)"],
            y=["Observado benigno (0)", "Observado maligno (1)"],
            text=matrix,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    figure.update_layout(title="Matriz de confusión del modelo guardado", height=380)
    st.plotly_chart(figure, width="stretch", key="evaluation-confusion-matrix")
    st.markdown(
        "- **Falso positivo:** un registro benigno del conjunto se clasifica como maligno.\n"
        "- **Falso negativo:** un registro maligno del conjunto se clasifica como benigno."
    )

    class_one_index = list(artifacts.model.classes_).index(1)
    malignant_probability = artifacts.model.predict_proba(
        artifacts.test[list(artifacts.feature_names)]
    )[:, class_one_index]
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, malignant_probability)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    roc_figure = go.Figure()
    roc_figure.add_trace(
        go.Scatter(
            x=false_positive_rate,
            y=true_positive_rate,
            mode="lines",
            name=f"Modelo SVM (AUC = {roc_auc:.3f})",
        )
    )
    roc_figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Azar",
            line={"dash": "dash", "color": "#94A3B8"},
        )
    )
    roc_figure.update_layout(
        title="Curva ROC sobre test.csv",
        xaxis_title="Tasa de falsos positivos",
        yaxis_title="Sensibilidad (tasa de verdaderos positivos)",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        height=420,
    )
    st.plotly_chart(roc_figure, width="stretch", key="evaluation-roc-curve")
    st.metric("ROC AUC recalculado", f"{roc_auc:.4f}")
    st.caption(
        "La curva ROC y el AUC se recalculan con las probabilidades del pipeline "
        "guardado y las etiquetas observadas de test.csv."
    )

    st.subheader("Métricas originales y comprobación")
    original = artifacts.report["test_metrics"]
    recalculated, _, _ = evaluate_threshold(
        artifacts.model,
        artifacts.test[list(artifacts.feature_names)],
        artifacts.test["malignant"],
        0.50,
    )
    st.dataframe(
        pd.DataFrame([
            {"Métrica": "Exactitud", "Reporte original": original["accuracy"], "Recalculada (umbral 0.50)": recalculated["accuracy"]},
            {"Métrica": "Sensibilidad", "Reporte original": original["sensitivity_recall"], "Recalculada (umbral 0.50)": recalculated["sensitivity"]},
            {"Métrica": "Especificidad", "Reporte original": original["specificity"], "Recalculada (umbral 0.50)": recalculated["specificity"]},
            {"Métrica": "Precisión (PPV)", "Reporte original": original["precision"], "Recalculada (umbral 0.50)": recalculated["precision"]},
            {"Métrica": "F1", "Reporte original": original["f1"], "Recalculada (umbral 0.50)": recalculated["f1"]},
            {"Métrica": "ROC AUC", "Reporte original": original["roc_auc"], "Recalculada (umbral 0.50)": roc_auc},
        ]),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Limitaciones")
    st.markdown(
        """
        - El dataset es pequeño y proviene de una fuente y población determinadas; no
          representa necesariamente otros centros, equipos o poblaciones.
        - El conjunto de prueba contiene 114 registros. Sus métricas pueden variar y
          no garantizan la predicción de un caso individual.
        - El umbral 0.50 es una regla de demostración, no una decisión clínica.
        - Las características están correlacionadas; las probabilidades no son riesgo
          clínico calibrado y el modelo no establece causalidad.
        - La aplicación requiere exactamente las 30 mediciones del dataset y no
          sustituye evaluación profesional.
        """
    )


def page_emotions() -> None:
    st.title("Detector de emociones en texto")
    st.caption(EMOTION_NOTICE)
    st.write(
        "Escribe una oración y el modelo SVM la asignará a la categoría textual "
        "más parecida entre alegría, tristeza, enojo, miedo, sorpresa y neutral."
    )
    try:
        emotion_artifacts = get_emotion_artifacts()
    except EmotionDetectionError as error:
        st.error(f"No se pudo preparar el detector local de emociones: {error}")
        return

    st.info(
        f"Modelo local entrenado al iniciar con {emotion_artifacts.sample_count} "
        f"frases etiquetadas de `data/emotion_samples.csv` "
        f"({len(emotion_artifacts.emotion_counts)} categorías). "
        "No se envía texto a servicios externos."
    )
    with st.form("emotion_form"):
        text = st.text_area(
            "Oración para analizar",
            placeholder="Ejemplo: Me siento muy feliz por compartir este día.",
            max_chars=2000,
        )
        submitted = st.form_submit_button("Analizar texto", type="primary")
    if submitted:
        try:
            label = predict_emotion(emotion_artifacts.model, text)
            st.subheader(f"Categoría textual predicha: {label.capitalize()}")
            st.caption(
                "La etiqueta describe la categoría que el modelo reconoce en el texto; "
                "no es una medición de la emoción real de quien lo escribió."
            )
        except EmotionDetectionError as error:
            st.error(str(error))

    with st.expander("Categorías y tamaño del conjunto de práctica"):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Categoría": label.capitalize(), "Frases de entrenamiento": count}
                    for label, count in emotion_artifacts.emotion_counts
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.warning(
            "El conjunto es pequeño y está formado por frases de ejemplo; el modelo "
            "puede fallar ante ironía, negación, contexto, diferencias culturales o "
            "textos distintos a los ejemplos. No es una herramienta psicológica ni clínica."
        )


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(160deg, #0b1120 0%, #101a2d 52%, #0b1120 100%); }
        .block-container { max-width: 1440px; padding-top: 1.7rem; }
        [data-testid="stMetric"] {
            background: rgba(23, 36, 58, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            padding: 1rem 1.15rem;
            border-radius: 14px;
        }
        div[data-testid="stAlert"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    try:
        artifacts = get_artifacts()
    except (CancerDetectionError, OSError, ValueError, ImportError) as error:
        st.error(
            "No se pudo cargar el modelo o los artefactos locales. Comprueba que el "
            "entorno use las versiones de requirements.txt y que existan "
            f"`models/best_model.joblib`, `data/processed/train.csv`, "
            f"`data/processed/test.csv` y los reportes. Detalle: {error}"
        )
        st.stop()

    sections = [
        "Inicio",
        "Clasificación individual",
        "Carga CSV",
        "Laboratorio",
        "Evaluación",
        "Detector de emociones",
    ]
    selected = st.radio(
        "Navegación",
        sections,
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected == "Inicio":
        page_home(artifacts)
    elif selected == "Clasificación individual":
        page_individual(artifacts)
    elif selected == "Carga CSV":
        page_csv(artifacts)
    elif selected == "Laboratorio":
        page_laboratory(artifacts)
    elif selected == "Evaluación":
        page_evaluation(artifacts)
    else:
        page_emotions()


if __name__ == "__main__":
    main()
