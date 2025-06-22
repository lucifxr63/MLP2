import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from medmnist import BreastMNIST, INFO
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
from datetime import datetime


def load_dataset(batch_size=32):
    data_flag = 'breastmnist'
    info = INFO[data_flag]
    DataClass = BreastMNIST
    train = DataClass(split='train', download=True)
    val = DataClass(split='val', download=True)
    test = DataClass(split='test', download=True)

    def preprocess(img, label):
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.expand_dims(img, -1)
        label = tf.cast(label, tf.int32)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train.imgs, train.labels))
    train_ds = train_ds.map(preprocess).shuffle(1000)
    val_ds = tf.data.Dataset.from_tensor_slices((val.imgs, val.labels)).map(preprocess)
    test_ds = tf.data.Dataset.from_tensor_slices((test.imgs, test.labels)).map(preprocess)

    return train_ds, val_ds, test_ds


def build_expert(expert_id):
    """Construye un modelo experto individual"""
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', name=f'expert_{expert_id}_conv1')(inputs)
    x = layers.MaxPooling2D(name=f'expert_{expert_id}_pool1')(x)
    x = layers.Conv2D(64, 3, activation='relu', name=f'expert_{expert_id}_conv2')(x)
    x = layers.MaxPooling2D(name=f'expert_{expert_id}_pool2')(x)
    x = layers.Flatten(name=f'expert_{expert_id}_flatten')(x)
    x = layers.Dense(64, activation='relu', name=f'expert_{expert_id}_dense')(x)
    return keras.Model(inputs, x, name=f'expert_{expert_id}')


def build_moe(n_experts=4):
    """Construye el modelo Mixture of Experts"""
    inputs = keras.Input(shape=(28, 28, 1), name='input')
    
    # Capa de expertos
    experts = [build_expert(i)(inputs) for i in range(n_experts)]
    expert_outputs = [layers.Dense(1, name=f'expert_{i}_output')(e) for i, e in enumerate(experts)]
    experts_concat = layers.Concatenate(axis=1, name='experts_concat')(expert_outputs)

    # Capa de compuerta (gating network)
    gate = layers.Flatten(name='gate_flatten')(inputs)
    gate = layers.Dense(32, activation='relu', name='gate_dense1')(gate)
    gate = layers.Dense(n_experts, activation='softmax', name='gate_output')(gate)

    # Combinar salidas ponderadas
    weighted = layers.Multiply(name='weighted_experts')([experts_concat, gate])
    out = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1, keepdims=True), name='sum_weighted')(weighted)
    outputs = layers.Activation('sigmoid', name='output')(out)
    
    return keras.Model(inputs, outputs, name=f'moe_{n_experts}_experts')


def plot_training_history(history, save_dir):
    """Plot training & validation metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, save_dir):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()


def save_metrics_report(y_true, y_pred, y_pred_proba, save_dir):
    """Save classification report and metrics to a text file"""
    report = classification_report(y_true, y_pred, target_names=['Benign', 'Malignant'])
    conf_matrix = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    
    with open(os.path.join(save_dir, 'metrics_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(conf_matrix))
        f.write(f"\n\nROC AUC Score: {roc_auc:.4f}")
        f.write(f"\nF1 Score: {f1:.4f}")


def main(n_experts=4, epochs=20, batch_size=32):
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create mlp2 directory in the project root if it doesn't exist
    mlp2_dir = os.path.abspath(os.path.join('..', 'mlp2', 'moe'))
    os.makedirs(mlp2_dir, exist_ok=True)
    # Create timestamped results directory inside mlp2/moe
    results_dir = os.path.join(mlp2_dir, f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and prepare dataset
    train_ds, val_ds, test_ds = load_dataset(batch_size)

    # Data augmentation
    data_aug = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    def augment(img, label):
        return data_aug(img), label

    # Prepare datasets
    train_ds = train_ds.map(augment).batch(batch_size).prefetch(2)
    val_ds = val_ds.batch(batch_size).prefetch(2)
    test_ds = test_ds.batch(batch_size)

    # Calculate class weights
    labels = [int(l) for _, l in train_ds.unbatch()]
    neg, pos = np.bincount(labels)
    total = neg + pos
    class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}

    # Build and compile model
    model = build_moe(n_experts)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights
    )
    
    # Evaluate model
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")
    
    # Make predictions
    y_true = np.concatenate([y.numpy() for x, y in test_ds.unbatch().batch(len(test_ds))], axis=0)
    y_pred_proba = model.predict(test_ds)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate and save plots and metrics
    plot_training_history(history, results_dir)
    plot_confusion_matrix(y_true, y_pred, results_dir)
    plot_roc_curve(y_true, y_pred_proba, results_dir)
    save_metrics_report(y_true, y_pred, y_pred_proba, results_dir)
    
    # Save model summary with proper encoding
    summary_path = os.path.join(results_dir, 'model_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Get model summary as string
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Alternative method if the above still has issues
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))
            f.write('\n'.join(string_list))
    except Exception as e:
        print(f"Warning: Could not save model summary with special characters: {e}")
        # Save a simplified summary instead
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model.name}")
            f.write(f"\nNumber of layers: {len(model.layers)}")
            f.write(f"\nTotal params: {model.count_params()}")
            f.write("\n\nNote: Full model summary could not be saved due to encoding issues.")
    
    print(f"\nAll metrics and plots have been saved to: {os.path.abspath(results_dir)}")


if __name__ == '__main__':
    main(n_experts=4, epochs=20)
