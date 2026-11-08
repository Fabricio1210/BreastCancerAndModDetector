# Ficha del modelo — CancerDetection

## Identificación y propósito

- Modelo: pipeline Support Vector Machine con kernel RBF.
- Artefacto local: `models/best_model.joblib`.
- Uso previsto: demostración educativa de clasificación binaria en una clase
  universitaria.
- Uso no previsto: diagnóstico, triaje, recomendación clínica o decisión sobre
  una persona.

## Datos y codificación

- Dataset: Breast Cancer Wisconsin Diagnostic, 569 filas y 30 mediciones.
- Particiones locales: 455 filas de entrenamiento y 114 de prueba.
- Predictoras: las 30 columnas numéricas del dataset; `source_index` se excluye
  por ser identificador y `malignant` se usa solo como etiqueta observada.
- Etiquetas de este proyecto: `0 = benigno`, `1 = maligno`. El target original
  de scikit-learn se invirtió para producir esta codificación.
- No se especifican unidades de las mediciones en los CSV del proyecto.

## Entrenamiento y evaluación reportada

El pipeline guarda `StandardScaler` seguido de SVC, con `C=10`,
`gamma=0.01`, kernel RBF y probabilidades habilitadas. La selección se hizo por
ROC AUC promedio en validación cruzada estratificada de diez pliegues sobre
entrenamiento. El conjunto de prueba se reservó para la evaluación reportada.

| Métrica reportada en prueba | Valor |
|---|---:|
| ROC AUC | 0.9970 |
| Exactitud | 0.9825 |
| Precisión (PPV) | 1.0000 |
| Sensibilidad | 0.9524 |
| Especificidad | 1.0000 |
| F1 | 0.9756 |
| Verdaderos negativos | 72 |
| Falsos positivos | 0 |
| Falsos negativos | 2 |
| Verdaderos positivos | 40 |

Estos son los valores del reporte local `reports/metrics.json`; la aplicación
recalcula métricas en test.csv y presenta las dos versiones para comprobar su
consistencia. Una divergencia debe investigarse, no ocultarse modificando el
reporte.

## Limitaciones y riesgos

- Solo se dispone de un dataset relativamente pequeño; no se demuestra
  generalización a otras poblaciones, instituciones o procesos de medición.
- Las métricas de test no son una garantía para un registro individual.
- Las probabilidades no deben interpretarse como riesgo clínico calibrado.
- Los errores de clasificación tienen consecuencias distintas en contextos
  clínicos; el umbral educativo de 0.50 no es una recomendación clínica.
- Explorar cambios de una o más columnas no prueba causalidad. Las variables
  pueden estar correlacionadas y las combinaciones modificadas pueden ser
  poco realistas.
- No se cuenta con validación externa ni con unidades documentadas para las
  mediciones en los archivos locales.

**Uso académico. Esta predicción no constituye un diagnóstico médico.**
