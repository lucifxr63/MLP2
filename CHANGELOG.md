# Changelog

Todas las modificaciones relevantes del proyecto se registran en este documento.

## [1.0.0] - 2025-06-22
### Añadido
- Implementación completa del modelo CNN baseline para clasificación de imágenes médicas (BreastMNIST)
- Implementación del modelo Mixture of Experts (MoE) con arquitectura modular
- Script de comparación de modelos (`compare_models.py`) que genera reportes y gráficos comparativos
- Sistema de guardado automático de resultados en carpetas con timestamp
- Visualizaciones de métricas, curvas de aprendizaje, matrices de confusión y curvas ROC
- Manejo de desbalance de clases mediante pesos de clase
- Aumento de datos para mejorar la generalización

### Mejorado
- Refactorización del código para mejor legibilidad y mantenibilidad
- Documentación detallada de funciones y parámetros
- Manejo de errores y validación de entradas
- Optimización del preprocesamiento de datos

## [0.1.0] - 2025-06-22
- Se agregó `models/cnn_baseline.py` con una CNN de referencia.
- Se añadió el notebook `notebooks/breast_mnist_moe.ipynb` con pruebas de MoE.
- Se actualizó el `README.md` para mencionar el script.

## [0.0.1] - Inicio
- Creación del repositorio y del README inicial.

