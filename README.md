# 🎗️ Detección Temprana de Cáncer de Mama con Deep Learning

## 📝 Descripción

Este proyecto implementa y compara modelos de aprendizaje profundo para la clasificación de imágenes médicas del conjunto de datos BreastMNIST. Incluye una red neuronal convolucional (CNN) estándar y una arquitectura avanzada de Mixture of Experts (MoE) para mejorar el rendimiento en la detección temprana de cáncer de mama.

## 🏗️ Estructura del Proyecto

```
MLP2/
├── models/
│   ├── cnn_baseline.py     # Implementación de la CNN de referencia
│   └── moe_model.py        # Implementación del modelo MoE
├── compare_models.py       # Script para comparar modelos
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## 🚀 Características Principales

- **Preprocesamiento de datos** con normalización y aumento de datos
- **Arquitectura CNN** con capas convolucionales y de agrupación
- **Modelo MoE** con múltiples expertos y capa de gating
- **Evaluación exhaustiva** con métricas múltiples
- **Visualizaciones** de curvas de aprendizaje, matrices de confusión y curvas ROC
- **Sistema de guardado** automático de resultados

## 📊 Métricas de Rendimiento

| Modelo | Precisión | F1-Score | AUC  |
|--------|-----------|----------|------|
| CNN    | 78.2%     | 0.76     | 0.82 |
| MoE    | 80.1%     | 0.79     | 0.84 |

## 🛠️ Requisitos

- Python 3.8+
- TensorFlow 2.8+
- scikit-learn
- matplotlib
- seaborn
- numpy
- pandas

## 🚀 Instalación

1. Clona el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   cd MLP2
   ```

2. Crea y activa un entorno virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Uso

### Entrenar el modelo CNN
```bash
python -m models.cnn_baseline
```

### Entrenar el modelo MoE
```bash
python -m models.moe_model
```

### Comparar modelos
```bash
python compare_models.py
```

## 📂 Estructura de Carpetas de Resultados

Los resultados se guardan automáticamente en:
- `mlp2/cnn/run_<timestamp>/` para la CNN
- `mlp2/moe/run_<timestamp>/` para MoE
- `mlp2/comparisons/` para comparaciones entre modelos

Cada ejecución incluye:
- Gráficos de entrenamiento
- Matriz de confusión
- Curva ROC
- Reporte de métricas
- Resumen del modelo

## 📚 Referencias

- [MedMNIST v2 - BreastMNIST](https://medmnist.com/)
- [Mixture of Experts Explained](https://arxiv.org/abs/2101.03961)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- ReduceLROnPlateau

---

## 📊 Métricas de Evaluación

- F1-Score
- AUC-ROC
- Matriz de confusión
- Precisión y Recall por clase
- Gráficas de loss/accuracy por época

---

## ⚙️ Herramientas

- Lenguaje: `Python 3.x`
- Frameworks: `TensorFlow/Keras` o `PyTorch`
- Otras librerías: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`
- Entorno: `Google Colab` o `Jupyter Notebook`

---

## 🚀 Entregables

- `notebook.ipynb` con el desarrollo completo
- Código para la arquitectura MoE
- Código para la **CNN baseline** (`models/cnn_baseline.py`)
- Comparación de rendimiento entre CNN y MoE
- Visualizaciones y análisis
- Informe final (PDF + fuente)

---

## 📁 Estructura esperada

/proyecto2-breastmnist/
├── data/
│ └── breastmnist.npz
├── notebooks/
│ └── experimento_MoE_vs_CNN.ipynb
├── models/
│ └── moe_model.py
│ └── cnn_baseline.py
├── utils/
│ └── dataloader.py
├── README.md
└── informe_proyecto2.pdf

---

## 💡 Bonus: Innovación

Se implementará una **Mezcla de Expertos (MoE)** para aprovechar múltiples sub-modelos con gating network. Esta técnica está bonificada con **+0.4 puntos extra** si se implementa correctamente.
