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


def build_cnn():
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)


def plot_training_history(history, save_dir):
    """Plot training & validation metrics"""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
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

def main(epochs=20, batch_size=32):
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create mlp2 directory in the project root if it doesn't exist
    mlp2_dir = os.path.abspath(os.path.join('..', 'mlp2'))
    os.makedirs(mlp2_dir, exist_ok=True)
    # Create timestamped results directory inside mlp2
    results_dir = os.path.join(mlp2_dir, f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    train_ds, val_ds, test_ds = load_dataset(batch_size)

    data_aug = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    def augment(img, label):
        return data_aug(img), label

    train_ds = train_ds.map(augment).batch(batch_size).prefetch(2)
    val_ds = val_ds.batch(batch_size).prefetch(2)
    test_ds = test_ds.batch(batch_size)

    labels = [int(l) for _, l in train_ds.unbatch()]
    neg, pos = np.bincount(labels)
    total = neg + pos
    class_weights = {0: (1 / neg) * (total / 2.0), 1: (1 / pos) * (total / 2.0)}

    model = build_cnn()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        class_weight=class_weights)
    
    # Evaluate the model
    test_metrics = model.evaluate(test_ds, return_dict=True)
    test_loss = test_metrics['loss']
    test_acc = test_metrics['accuracy']
    print(f'Test accuracy: {test_acc:.4f}, Test AUC: {test_metrics["auc"]:.4f}')
    
    # Make predictions
    y_true = np.concatenate([y.numpy() for _, y in test_ds.unbatch().batch(len(test_ds))], axis=0)
    y_pred_proba = model.predict(test_ds)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate and save plots and metrics
    plot_training_history(history, results_dir)
    plot_confusion_matrix(y_true, y_pred, results_dir)
    plot_roc_curve(y_true, y_pred_proba, results_dir)
    save_metrics_report(y_true, y_pred, y_pred_proba, results_dir)
    
    print(f"\nAll metrics and plots have been saved to: {os.path.abspath(results_dir)}")


if __name__ == '__main__':
    main()
