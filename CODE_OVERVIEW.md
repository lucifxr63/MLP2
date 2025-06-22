# Visión General del Código

Este documento proporciona una descripción detallada de la implementación del sistema de clasificación de imágenes médicas para la detección temprana de cáncer de mama.

## 🏗️ Arquitectura General

El proyecto sigue una arquitectura modular con los siguientes componentes principales:

1. **Módulo de Datos**
   - Carga y preprocesamiento de imágenes
   - Aumento de datos
   - Generación de lotes (batching)

2. **Modelos**
   - CNN estándar
   - Mixture of Experts (MoE)

3. **Evaluación**
   - Métricas de rendimiento
   - Visualizaciones
   - Generación de reportes

## 📂 Estructura de Archivos

```
models/
├── cnn_baseline.py     # Implementación de la CNN de referencia
├── moe_model.py        # Implementación del modelo MoE
compare_models.py       # Script para comparar modelos
```

## 🧩 Componentes Clave

### 1. cnn_baseline.py

#### Funcionalidades
- Carga del conjunto de datos BreastMNIST
- Preprocesamiento de imágenes (reescalado, normalización)
- Definición de la arquitectura CNN
- Entrenamiento con early stopping
- Evaluación y generación de métricas
- Guardado de resultados

#### Arquitectura CNN
```
Input(28, 28, 1)
├─ Conv2D(32, 3x3, relu)
├─ MaxPooling2D
├─ Conv2D(64, 3x3, relu)
├─ MaxPooling2D
├─ Flatten
├─ Dense(64, relu)
└─ Dense(1, sigmoid)
```

### 2. moe_model.py

#### Características
- Implementación de la arquitectura Mixture of Experts
- Múltiples expertos CNN trabajando en paralelo
- Capa de gating para combinar las salidas
- Manejo de desbalance de clases

#### Estructura MoE
```
Input
├─ Múltiples Expertos (CNNs)
├─ Capa de Gating
└─ Combinación Ponderada
```

### 3. compare_models.py

#### Funcionalidades
- Carga de resultados de múltiples ejecuciones
- Comparación de métricas de rendimiento
- Generación de gráficos comparativos
- Creación de reportes detallados

## 🔄 Flujo de Datos

1. **Carga de Datos**
   - Se cargan las imágenes y etiquetas
   - Se aplica preprocesamiento básico

2. **Aumento de Datos** (solo entrenamiento)
   - Rotaciones aleatorias
   - Volteos horizontales/verticales
   - Zoom aleatorio

3. **Entrenamiento**
   - Se entrena el modelo por épocas
   - Se monitorea el rendimiento en validación
   - Se aplica early stopping

4. **Evaluación**
   - Cálculo de métricas en el conjunto de prueba
   - Generación de visualizaciones
   - Guardado de resultados

## ⚙️ Configuración

### Hiperparámetros

| Parámetro         | Valor por Defecto | Descripción                     |
|-------------------|-------------------|---------------------------------|
| batch_size        | 32                | Tamaño del lote                |
| epochs           | 50                | Número máximo de épocas        |
| learning_rate    | 1e-3              | Tasa de aprendizaje inicial    |
| patience         | 10                | Paciencia para early stopping   |
| n_experts        | 4                 | Número de expertos (solo MoE)  |

## 🛠️ Uso Avanzado

### Entrenamiento Personalizado
```python
from models.cnn_baseline import train_model

history = train_model(
    train_ds,
    val_ds,
    learning_rate=0.001,
    batch_size=32,
    epochs=100
)
```

### Evaluación de Modelos
```python
from compare_models import compare_models

results = compare_models(
    model_paths=['path/to/model1', 'path/to/model2'],
    test_ds=test_dataset
)
```

## 📊 Métricas Implementadas

- Precisión
- Recall
- F1-Score
- AUC-ROC
- Matriz de confusión
- Curvas de aprendizaje (pérdida y precisión)

## 📝 Notas de Implementación

- Se utiliza `TensorFlow` como backend principal
- Los modelos se guardan en formato Keras (`.h5`)
- Las visualizaciones se generan con `matplotlib` y `seaborn`
- El código sigue las mejores prácticas de PEP 8

## 🚀 Mejoras Futuras

- Implementar búsqueda de hiperparámetros
- Añadir soporte para otros conjuntos de datos médicos
- Implementar técnicas avanzadas de regularización
- Añadir soporte para entrenamiento distribuido
- Calcula pesos de clase para mitigar el desbalance.
- Define una CNN pequeña con dos capas convolucionales y `Dropout`.
- Entrena el modelo y evalúa en el conjunto de prueba mostrando F1 y AUC.

## breast_mnist_moe.ipynb

El notebook repite la exploración de datos, la CNN base y además implementa una arquitectura **Mixture of Experts** con un *gating network* que combina varios expertos convolucionales. Se prueban configuraciones con 2, 4 y 8 expertos.

Para cada experimento se grafican las curvas de entrenamiento y se reportan F1-score, AUC y la matriz de confusión.

