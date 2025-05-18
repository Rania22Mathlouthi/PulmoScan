# model_pipeline.py

import os
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import zipfile
import requests
import os

# 📦 1. Préparation des données (extraction + chargement)
def prepare_single_image(image_path, image_size=(224, 224)):
    """
    Charge une seule image et la prépare pour l'inférence
    """
    print(f"📁 Chargement de l'image depuis : {image_path}")
    
    # Chargement et prétraitement de l'image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)  # Ajouter une dimension batch

    # Normalisation si nécessaire
    img_array = img_array / 255.0

    print("✅ Image préparée pour l'inférence.")
    return img_array


# 🧠 2. Entraînement du modèle CNN simple
from ultralytics import YOLO

def train_model_yolov8(data_yaml_path, model_name='yolov8n.pt', epochs=20):
    """
    Entraîne un modèle YOLOv8 à partir d’un fichier data.yaml
    """
    print("🚀 Entraînement YOLOv8...")

    model = YOLO(model_name)  # ex: 'yolov8n.pt', 'yolov8s.pt'
    results = model.train(data=data_yaml_path, epochs=epochs)

    print("✅ Entraînement terminé.")
    return model, results


def detect_and_send(image_path, clinical_vector, api_url):
    print(f"🔍 YOLOv8 detection on: {image_path}")
    model = YOLO("model_lung_yolo.pt")
    results = model.predict(source=image_path, conf=0.25, save=True, save_txt=True, name='predict2')

    # Locate the predicted image (saved by YOLO)
    predicted_dir = 'runs/detect/predict2'
    files = os.listdir(predicted_dir)
    image_file = [f for f in files if f.lower().endswith(('.jpg', '.png'))]
    if not image_file:
        print("❌ No image found after YOLO prediction.")
        return

    image_path = os.path.join(predicted_dir, image_file[0])
    print("📤 Sending image and clinical data to PC2...")

    with open(image_path, 'rb') as f:
        response = requests.post(api_url, files={"image": f}, json={"clinical": clinical_vector})

    if response.status_code == 200:
        res = response.json()
        print("✅ Prediction received from PC2:")
        print("Predicted class index:", res["class_index"])
        print("Confidence:", res["confidence"])
    else:
        print(f"❌ Failed to get response: {response.status_code} - {response.text}")


# 📊 3. Évaluation
def evaluate_model(model, val_ds):
    """
    Évalue les performances du modèle sur le jeu de test
    """
    print("📊 Évaluation du modèle...")
    loss, accuracy = model.evaluate(val_ds)
    print(f"✅ Accuracy: {accuracy:.2f}")
    return accuracy

# 💾 4. Sauvegarde du modèle
def save_model(model, path='lung_model.h5'):
    model.save(path)
    print(f"📁 Modèle sauvegardé sous : {path}")

# 📂 5. Chargement du modèle
def load_model(path='lung_model.h5'):
    print(f"📂 Chargement du modèle depuis : {path}")
    return keras_load_model(path)

def perform_yolo_inference(image_path, model):
    class_names = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
    print(f"🔍 YOLOv8 - Inférence sur : {image_path}")
    results = model.predict(image_path)

    if results and len(results[0].boxes.cls) > 0:
        for result in results[0].boxes:
            class_index = int(result.cls[0])
            confidence = result.conf[0]
            predicted_class = class_names[class_index]
            print(f"✅ Classe prédite : {predicted_class} avec une confiance de {confidence:.2f}")
            return predicted_class, confidence
    print("❌ Aucune détection effectuée.")
    return None, None