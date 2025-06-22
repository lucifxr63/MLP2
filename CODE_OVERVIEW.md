# Explicación del Código

Este repositorio contiene un ejemplo de detección de cáncer de mama usando el dataset **BreastMNIST**.
Se incluye un script de referencia con una CNN simple y un notebook con una arquitectura Mixture of Experts (MoE).

## cnn_baseline.py

- Carga los datos de BreastMNIST mediante `medmnist`.
- Preprocesa las imágenes para que tengan forma `(64, 64, 1)` y valores en `[0,1]`.
- Aplica *data augmentation* (flip, rotación, zoom) durante el entrenamiento.
- Calcula pesos de clase para mitigar el desbalance.
- Define una CNN pequeña con dos capas convolucionales y `Dropout`.
- Entrena el modelo y evalúa en el conjunto de prueba mostrando F1 y AUC.

## breast_mnist_moe.ipynb

El notebook repite la exploración de datos, la CNN base y además implementa una arquitectura **Mixture of Experts** con un *gating network* que combina varios expertos convolucionales. Se prueban configuraciones con 2, 4 y 8 expertos.

Para cada experimento se grafican las curvas de entrenamiento y se reportan F1-score, AUC y la matriz de confusión.

