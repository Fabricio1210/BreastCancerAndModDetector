# Breast Cancer & Emotion Detector

Aplicación de escritorio en español con dos clasificadores Support Vector Machine (SVM):

- **Cáncer de mama:** usa el modelo `Support vector machine` entrenado en la variante `variation` de `Challenge1_Cancer_Detection`. Recibe las 30 mediciones del conjunto Breast Cancer Wisconsin Diagnostic y muestra la clase y las probabilidades estimadas.
- **Emoción en texto:** entrena un SVM de texto separado con TF-IDF de palabras y caracteres. Clasifica una oración en alegría, tristeza, enojo, miedo, sorpresa o neutral.

Los dos clasificadores usan SVM, pero no pueden compartir el mismo modelo entrenado: el primero se entrenó con 30 variables numéricas y el segundo necesita características de texto y etiquetas de emociones. El modelo de cáncer se distribuye en `models/best_model_variation.joblib`; los ejemplos de entrenamiento del clasificador de emociones están en `data/emotion_samples.csv`.

## Requisitos

- Python 3.10 o posterior
- Tkinter (incluido en la mayoría de las instalaciones de Python para Windows)

Instala las dependencias desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
```

Inicia la interfaz:

```powershell
python app.py
```

En la pestaña de cáncer, captura las 30 mediciones del estudio o carga un ejemplo del conjunto didáctico. En la pestaña de emociones, escribe una oración en español y selecciona **Detectar emoción**.

## Resultados y limitaciones

El artefacto de cáncer se evaluó en un conjunto de prueba de 114 filas: ROC AUC 0.997, exactitud 0.982, sensibilidad 0.952 y especificidad 1.000. Las métricas se reportan como contexto experimental; no garantizan resultados equivalentes para otras muestras.

El clasificador emocional es demostrativo: usa un corpus pequeño escrito para este proyecto, por lo que su cobertura y generalización son limitadas. El modelo no interpreta ironía, contexto ni estados clínicos; su resultado no es una probabilidad clínica.