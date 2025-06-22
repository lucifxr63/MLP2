import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from medmnist import BreastMNIST, INFO
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score


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
    inputs = keras.Input(shape=(64, 64, 1))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)


def main(epochs=20, batch_size=32):
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

    probs = model.predict(test_ds)
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    y_pred = (probs.ravel() > 0.5).astype(int)

    print(classification_report(y_true, y_pred))
    print('F1:', f1_score(y_true, y_pred))
    print('AUC:', roc_auc_score(y_true, probs))
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:\n', cm)


if __name__ == '__main__':
    main()
