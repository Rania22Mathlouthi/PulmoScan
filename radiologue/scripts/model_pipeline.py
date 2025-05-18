import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os


def prepare_data(dataset_path):
    """
    Charge les données image, cliniques et labels depuis le chemin donné.
    """
    X_train = np.load(os.path.join(dataset_path, "X_train.npy"))
    X_val = np.load(os.path.join(dataset_path, "X_val.npy"))
    X_test = np.load(os.path.join(dataset_path, "X_test.npy"))

    y_train = np.load(os.path.join(dataset_path, "y_train.npy"))
    y_val = np.load(os.path.join(dataset_path, "y_val.npy"))
    y_test = np.load(os.path.join(dataset_path, "y_test.npy"))

    clinical_train = np.load(os.path.join(dataset_path, "clinical_train.npy"))
    clinical_val = np.load(os.path.join(dataset_path, "clinical_val.npy"))
    clinical_test = np.load(os.path.join(dataset_path, "clinical_test.npy"))

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        clinical_train,
        clinical_val,
        clinical_test,
    )


def build_vgg_model(img_shape=(128, 128, 2), clinical_shape=(6,), num_classes=4):
    """
    Construit un modèle VGG16 avec deux entrées : image et données cliniques.
    """
    img_input = tf.keras.Input(shape=img_shape)
    img = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(img_input)
    img = tf.keras.layers.Resizing(224, 224)(img)

    base_model = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model(img)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    clin_input = tf.keras.Input(shape=clinical_shape)
    y = tf.keras.layers.Dense(32, activation="relu")(clin_input)
    y = tf.keras.layers.Dropout(0.4)(y)

    combined = tf.keras.layers.concatenate([x, y])
    z = tf.keras.layers.Dense(128, activation="relu")(combined)
    z = tf.keras.layers.Dropout(0.5)(z)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(z)

    model = tf.keras.Model(inputs=[img_input, clin_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_model(model, X_train, y_train, clinical_train, X_val, y_val, clinical_val):
    """
    Entraîne le modèle avec early stopping.
    """
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    history = model.fit(
        [X_train, clinical_train],
        y_train,
        validation_data=([X_val, clinical_val], y_val),
        epochs=40,
        batch_size=16,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            )
        ],
    )
    return history
def perform_vgg_inference(image_path, clinical_array, vgg_model_path):
    print("🧠 Inférence VGG + Données cliniques...")
    print(f"📁 Chargement de l'image depuis : {image_path}")

    model = tf.keras.models.load_model(vgg_model_path)

    # L’image est déjà prétraitée (format .npy, normalisée, reshape, etc.)
    img_array = np.load(image_path)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter batch dim

    clinical_array = np.array(clinical_array).reshape(1, -1)

    prediction = model.predict([img_array, clinical_array])
    predicted_label = np.argmax(prediction, axis=1)[0]

    print(f"🧬 Classe finale prédite (VGG + clinique) : {predicted_label}")
    return predicted_label