import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(dataset_dir):
    data_splits = {}
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        X, y = [], []
        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.endswith(".npy"):
                    path = os.path.join(root, file)
                    data = np.load(path)
                    if data.ndim == 2:
                        data = np.stack([data]*3, axis=-1)
                    label = 1 if "positive" in root.lower() else 0
                    X.append(data)
                    y.append(label)
        X = np.array(X).astype('float32') / 255.0
        y = np.array(y)
        data_splits[split] = (X, y)
    return data_splits

def build_model(input_shape):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    for layer in base_model.layers[-20:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0001,
        decay_steps=5000,
        decay_rate=0.96,
        staircase=True
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, x_train, y_train, x_val, y_val, save_path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        x_train, y_train,
        epochs=25,
        batch_size=32,
        validation_data=(x_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    model.save(save_path)
    return history

def evaluate_model(model, X_test, y_test):
    print("\n[EVALUATION] Résultats sur le test set:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Prédictions
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype("int32")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes, digits=4))

    # Matrice de confusion
    cm = confusion_matrix(y_test, predicted_classes)
    print("\nMatrice de Confusion:")
    print(cm)

    # Affichage
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Classe 0", "Classe 1"],
                yticklabels=["Classe 0", "Classe 1"])
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vrais Labels')
    plt.show()

    return loss, accuracy

def perform_inference(model, image_path):
    # Charger et prétraiter l'image
    image = np.load(image_path)  # Charger l'image (ici, on suppose que l'image est au format .npy)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)  # Convertir en image RGB
    image = image.astype('float32') / 255.0  # Normaliser les pixels

    # Ajouter une dimension pour simuler un batch
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    prediction = model.predict(image)
    predicted_class = (prediction > 0.5).astype("int32")

    # Afficher la prédiction
    print(f"Prédiction: {'Positive' if predicted_class == 1 else 'Negative'}")

    return predicted_class
