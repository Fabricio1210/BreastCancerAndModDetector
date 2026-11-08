# CancerDetection — Laboratorio interactivo de clasificación

Aplicación educativa en español para explorar un clasificador Support Vector Machine
(SVM) sobre Breast Cancer Wisconsin Diagnostic.

> **Uso académico. Esta predicción no constituye un diagnóstico médico.**

La aplicación presenta seis secciones: **Inicio**, **Clasificación individual**,
**Carga CSV**, **Laboratorio**, **Evaluación** y **Detector de emociones**. El aviso
de uso académico se muestra discretamente en las secciones de cáncer y no constituye
un diagnóstico médico.

## Datos y artefactos

Este proyecto usa 30 mediciones numéricas del dataset Wisconsin Diagnostic Breast
Cancer, con 455 filas en entrenamiento y 114 en prueba. La variable `source_index`
es identificadora y `malignant` es la etiqueta observada opcional: ambas se excluyen
de los predictores.

La codificación de este proyecto es **benigno = 0** y **maligno = 1**. La carga
comprueba el esquema del CSV, el reporte y `classes_` del modelo. Es distinta de
la codificación del dataset original de scikit-learn, que es benigno = 1 y
maligno = 0.

El artefacto `models/best_model.joblib` es el SVM de la variante `variation` con
kernel RBF, `C=10` y `gamma=0.01`. El entrenamiento original usa estandarización
en un pipeline y selección por validación cruzada estratificada de diez pliegues.
No se inventan unidades para las columnas: los archivos del proyecto no las
especifican.

El reporte original indica ROC AUC 0.997, exactitud (accuracy) 0.982, precisión
(valor predictivo positivo) 1.000, sensibilidad 0.952 y especificidad 1.000 en
el conjunto de prueba. Exactitud es la fracción total correcta; precisión es la
fracción de predicciones positivas que fueron positivas. Son resultados de un
benchmark de 114 filas, no una garantía para un registro individual ni una
validación clínica. La app muestra también la comprobación de esas métricas.

Los datos originales están en `data/processed/train.csv` y `test.csv`; los
reportes, en `reports/metrics.json` y `reports/model_comparison.csv`. No se
modifican al explorar la interfaz. La carpeta de este proyecto no contenía los
archivos solicitados `notebooks/01_cancer_detection_challenge.ipynb`,
`src/cancer_detection.py` ni `reports/model_card.md`; se creó el módulo
reutilizable `src/cancer_detection.py`. No se inventó ni alteró un notebook
original.

## Windows 11, Anaconda y VS Code

1. Abre Anaconda Prompt o una terminal de VS Code y cambia a la carpeta del proyecto:

   ```powershell
   cd "C:\ruta\BreastCancerAndModDetector"
   ```

2. Crea el entorno con el nombre conservado `CancerDetection`:

   ```powershell
   conda env create -f environment.yml
   conda activate CancerDetection
   ```

   Como alternativa, crea un entorno limpio de Python 3.14 y usa las versiones
   verificadas de `requirements.txt`:

   ```powershell
   conda create -n CancerDetection python=3.14 pip -y
   conda activate CancerDetection
   python -m pip install -r requirements.txt
   ```

3. En VS Code selecciona el intérprete del entorno `CancerDetection`
   (**Python: Select Interpreter**) y abre una terminal nueva. Confirma el
   intérprete con `python --version`.

4. Inicia la aplicación desde la raíz del proyecto:

   ```powershell
   python -m streamlit run app.py
   ```

Streamlit abre la interfaz en el navegador local. No requiere claves API ni
servicios externos.

## Uso de la aplicación

- **Inicio:** muestra objetivo, esquema de 30 variables, tamaños de partición,
  modelo, métricas y flujo de entrenamiento/evaluación.
- **Clasificación individual:** solicita las 30 entradas, permite precargar
  registros benignos o malignos del conjunto de prueba y restablecer el formulario.
  Los ejemplos son del benchmark existente y no constituyen una validación nueva.
  El formulario no rellena silenciosamente variables faltantes.
- **Carga CSV:** ofrece una plantilla con las 30 columnas. Puede incluir de forma
  opcional `source_index` y `malignant`; las columnas adicionales, faltantes o
  duplicadas y los valores inválidos se reportan. Las filas inválidas no se clasifican.
  La carga y descarga se procesan en memoria.
- **Laboratorio:** permite variar una o más variables de un ejemplo, comparar
  probabilidades y cambiar el umbral. La regla exploratoria es
  `P(clase 1) >= umbral`. Las métricas se recalculan con `test.csv`; explorar sus
  umbrales no equivale a validarlos.
- **Evaluación:** compara modelos desde el reporte, calcula matriz de confusión y
  curva ROC a partir del modelo local y explica falsos positivos/falsos negativos.
- **Detector de emociones:** analiza una oración con un SVM lineal de texto y
  TF-IDF de palabras y caracteres. Se entrena localmente al iniciar con las 72
  frases de `data/emotion_samples.csv` (12 por categoría: alegría, tristeza, enojo,
  miedo, sorpresa y neutral). Esta muestra pequeña es solo didáctica; la categoría
  inferida no determina el estado emocional de una persona ni sustituye una evaluación
  psicológica. El texto se procesa localmente y no se guarda por defecto.

Las referencias de rango se calculan solo con `train.csv`. Estar fuera del rango
observado genera una advertencia, pero esos rangos no son límites clínicos.
Las probabilidades son estimaciones del modelo y no un riesgo clínico validado.

## Actividad de clase (10–15 minutos)

1. Carga un ejemplo benigno o maligno en **Clasificación individual** y registra
   su clase y probabilidades.
2. Elige ese registro en **Laboratorio** y cambia una medición; compara las
   entradas y probabilidades antes y después.
3. Ajusta el umbral y observa la matriz de confusión y las métricas.
4. Explica por qué la variación no prueba causalidad y por qué las métricas de un
   conjunto de prueba pequeño no garantizan predicciones individuales.

## Pruebas

Con el entorno activo, ejecuta:

```powershell
python -m pytest -q
```

Las pruebas validan el esquema y orden de las 30 columnas, el mapeo de clases,
la exclusión de metadatos, entradas inválidas, equivalencia con el pipeline,
clasificación CSV, regla de umbral, consistencia con los reportes y las seis
categorías del detector de emociones.
