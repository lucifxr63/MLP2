# Proyecto 2 - Detección Temprana de Cáncer de Mama con Deep Learning y Mezcla de Expertos (MoE)

## 🎯 Objetivo del Proyecto

Desarrollar un sistema de diagnóstico asistido mediante inteligencia artificial que permita **clasificar imágenes de ecografía mamaria** (BreastMNIST) en benignas o malignas. Como innovación, se utilizará una arquitectura **Mixture of Experts (MoE)** para mejorar la precisión del modelo y su capacidad de generalización.

---

## 🧠 Dataset

**Fuente:** [MedMNIST v2 - BreastMNIST](https://medmnist.com/)

- Formato: Imágenes en escala de grises 64x64
- Etiquetas:
  - `0`: Lesión benigna
  - `1`: Lesión maligna
- Divisiones:
  - Entrenamiento: 546 imágenes
  - Validación: 78 imágenes
  - Prueba: 156 imágenes
- Distribución:
  - Benigno: 147
  - Maligno: 399 (⚠️ Dataset desbalanceado)

---

## 🔍 Análisis Exploratorio

- Visualización de imágenes por clase
- Imágenes promedio por clase
- Detección de desbalance de clases

---

## 🧪 Diseño Experimental

### Modelos a Comparar

1. **CNN estándar**
2. **MoE (Mixture of Experts)** con:
   - 2 expertos
   - 4 expertos
   - 8 expertos

### Métodos complementarios

- Data Augmentation (rotación, zoom, flips)
- Class Weights o sobremuestreo para manejar desbalance
- Regularización: Dropout, BatchNormalization
- Early Stopping
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
