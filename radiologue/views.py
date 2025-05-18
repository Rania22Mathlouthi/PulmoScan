from django.shortcuts import render, get_object_or_404, redirect
from .models import Radiologue
from django.contrib.auth.decorators import login_required
import joblib
import os
from django.conf import settings
import random
import numpy as np
import tensorflow as tf
from django.http import HttpResponse
from .forms import UploadImageForm
from ultralytics import YOLO
from .scripts.luna_pipeline import perform_inference
from .scripts.model_classification_malignant import perform_yolo_inference
from .scripts.model_pipeline import perform_vgg_inference
from datetime import datetime
from django.template.loader import get_template
from xhtml2pdf import pisa
import io
from account.models import CustomUser
def radiologue_upload(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            for file in request.FILES.getlist('image'):
                save_path = os.path.join(settings.MEDIA_ROOT, 'luna', file.name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    for chunk in file.chunks():
                        f.write(chunk)
            return HttpResponse("Files uploaded successfully!")
    else:
        form = UploadImageForm()
    return render(request, 'radiologue.html', {'form': form})

def Radiologue_list(request):
    Radiologues = Radiologue.objects.all()
    return render(request, 'radiologue.html', {'Radiologues': Radiologues})

def download_report_pdf(request):
    report = request.session.get('report')
    if not report:
        return HttpResponse("No report found.")
    template = get_template('medical_report.html')
    html = template.render(report)
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="medical_report.pdf"'
    pisa_status = pisa.CreatePDF(io.BytesIO(html.encode('utf-8')), dest=response)
    return response

def Predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        image_name = uploaded_file.name
        # === Identifier le patient à partir du nom de l'image (ex: PT-3.jpg)

        patient_id_str = os.path.splitext(image_name)[0].replace("PT-", "")
        try:
            user = CustomUser.objects.get(id=int(patient_id_str))
            patient_email = user.email
        except CustomUser.DoesNotExist:
            return HttpResponse("❌ Patient introuvable pour cette image.")

        luna_dir = os.path.join(settings.MEDIA_ROOT, 'luna')
        os.makedirs(luna_dir, exist_ok=True)
        image_path = os.path.join(luna_dir, image_name)

        # Save the uploaded image
        with open(image_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        print(f"🖼️ Image uploaded and saved to: {image_path}")

        # === LUNA16 Inference ===
        luna_model_path = "../efficient_model.keras"
        if not os.path.exists(luna_model_path):
            return HttpResponse("❌ LUNA16 model not found.")
        model = tf.keras.models.load_model(luna_model_path)
        prediction_class = perform_inference(model, image_path)
        luna_result = "Positive" if prediction_class == 1 else "Negative"

        yolo_result = "N/A"
        yolo_confidence = "N/A"
        vgg_result = "N/A"
        malignant_type = "N/A"

        if luna_result == "Positive":
            # === YOLOv8 Inference ===
            print("🔍 YOLOv8 - starting inference")
            yolo_model_path = "../model_lung_yolo.pt"
            yolo_model = YOLO(yolo_model_path)
            malignant_dir = settings.MALIGNANT_IMAGES_DIR

            malignant_images = [f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not malignant_images:
                return HttpResponse("❌ No valid image found in 'malignant' directory for YOLOv8.")

            selected_image = random.choice(malignant_images)
            malignant_image_path = os.path.join(malignant_dir, selected_image)
            predicted_class, confidence = perform_yolo_inference(malignant_image_path, yolo_model)

            # Ensure compatibility
            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            yolo_confidence = round(float(confidence) * 100, 2)

            if predicted_class == "normal":
                yolo_result = "Nodule bénin"
                vgg_result = "N/A"  # No VGG needed
            else:
                yolo_result = "Malignant (type: " + predicted_class + ")"
                malignant_type = predicted_class

                # === VGG + Clinical Inference ===
                print("🧠 VGG model - starting stage prediction")
                stage_vgg_dir = settings.STAGE_IMAGES_DIR
                vgg_images = [f for f in os.listdir(stage_vgg_dir) if f.endswith('.npy')]
                if not vgg_images:
                    return HttpResponse("❌ No .npy files found in 'stage' directory.")

                selected_vgg = os.path.join(stage_vgg_dir, random.choice(vgg_images))
                
                clinical_input = [65, 1, 3.5, 120, 85, 0.8]
                vgg_model_path = "../trained_vgg_model.h5"
                vgg_result = perform_vgg_inference(selected_vgg, clinical_input, vgg_model_path)
                # 🧠 Identifier le patient via le nom du fichier npy
                

                if hasattr(vgg_result, 'item'):
                    vgg_result = int(vgg_result.item())
                else:
                    vgg_result = int(vgg_result)

        request.session['report'] = {
            'image_name': image_name,
            'luna_result': luna_result,
            'yolo_result': yolo_result,
            'yolo_confidence': yolo_confidence,
            'vgg_result': vgg_result,
            'malignant_type': malignant_type,
            'patient_id': f"PT-{random.randint(1000,9999)}",
            'date': datetime.now().strftime('%Y-%m-%d'),
        }

        # === Immediately Generate & Return PDF ===
        template = get_template('medical_report.html')
        html = template.render(request.session['report'])

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="medical_report.pdf"'
        pisa.CreatePDF(io.BytesIO(html.encode('utf-8')), dest=response)
        # ✉️ Générer le PDF
        pdf_buffer = io.BytesIO()
        pisa.CreatePDF(io.BytesIO(html.encode('utf-8')), dest=pdf_buffer)
        pdf_buffer.seek(0)

        # ✉️ Envoyer le rapport par mail
        from django.core.mail import EmailMessage

        email = EmailMessage(
            subject="🩺 Rapport médical Mediplus",
            body=f"Bonjour,\n\nVeuillez trouver ci-joint votre rapport médical.",
            from_email="noreply@mediplus.tn",
            to=[patient_email]
        )
        email.attach('rapport_medical.pdf', pdf_buffer.read(), 'application/pdf')
        email.send()

        return response

    form = UploadImageForm()
    return render(request, 'radiologue.html', {'form': form})
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        image_name = uploaded_file.name
        luna_dir = os.path.join(settings.MEDIA_ROOT, 'luna')
        os.makedirs(luna_dir, exist_ok=True)
        image_path = os.path.join(luna_dir, image_name)

        # Save the uploaded image
        with open(image_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        print(f"🖼️ Image uploaded and saved to: {image_path}")

        # === LUNA16 Inference ===
        luna_model_path = "../efficient_model.keras"
        if not os.path.exists(luna_model_path):
            return HttpResponse("❌ LUNA16 model not found.")
        model = tf.keras.models.load_model(luna_model_path)
        prediction_class = perform_inference(model, image_path)
        luna_result = "Positive" if prediction_class == 1 else "Negative"

        yolo_result = "N/A"
        yolo_confidence = "N/A"
        vgg_result = "N/A"

        if luna_result == "Positive":
            print("🔍 YOLOv8 - starting inference")
            yolo_model_path = "../model_lung_yolo.pt"
            yolo_model = YOLO(yolo_model_path)
            malignant_dir = settings.MALIGNANT_IMAGES_DIR

            malignant_images = [f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not malignant_images:
                return HttpResponse("❌ No valid image found in 'malignant' directory for YOLOv8.")

            selected_image = random.choice(malignant_images)
            malignant_image_path = os.path.join(malignant_dir, selected_image)
            predicted_class, confidence = perform_yolo_inference(malignant_image_path, yolo_model)

            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            yolo_result = "Nodule bénin" if predicted_class == "normal" else "Malignant"
            yolo_confidence = round(float(confidence) * 100, 2)

            print("🧠 VGG model - starting stage prediction")
            stage_vgg_dir = settings.STAGE_IMAGES_DIR
            vgg_images = [f for f in os.listdir(stage_vgg_dir) if f.endswith('.npy')]
            if not vgg_images:
                return HttpResponse("❌ No .npy files found in 'stage' directory.")

            selected_vgg = os.path.join(stage_vgg_dir, random.choice(vgg_images))
            clinical_input = [65, 1, 3.5, 120, 85, 0.8]  # Example clinical data
            vgg_model_path = "../trained_vgg_model.h5"
            vgg_result = perform_vgg_inference(selected_vgg, clinical_input, vgg_model_path)
            if hasattr(vgg_result, 'item'):
                vgg_result = int(vgg_result.item())
            else:
                vgg_result = int(vgg_result)

        request.session['report'] = {
            'image_name': image_name,
            'luna_result': luna_result,
            'yolo_result': yolo_result,
            'yolo_confidence': yolo_confidence,
            'vgg_result': vgg_result,
            'patient_id': f"PT-{random.randint(1000,9999)}",
            'date': datetime.now().strftime('%Y-%m-%d'),
        }
        request.session['report_ready'] = True

        form = UploadImageForm()
            # Generate PDF and return as download
        template = get_template('medical_report.html')
        html = template.render(request.session['report'])

        response = HttpResponse(content_type='application/pdf')
        response['Content-Disposition'] = 'attachment; filename="medical_report.pdf"'
        pisa.CreatePDF(io.BytesIO(html.encode('utf-8')), dest=response)

        return response

    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        image_name = uploaded_file.name
        luna_dir = os.path.join(settings.MEDIA_ROOT, 'luna')
        os.makedirs(luna_dir, exist_ok=True)
        image_path = os.path.join(luna_dir, image_name)

        # Save the uploaded image
        with open(image_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        print(f"🖼️ Image uploaded and saved to: {image_path}")

        # === LUNA16 Inference ===
        luna_model_path = "../efficient_model.keras"
        if not os.path.exists(luna_model_path):
            return HttpResponse("❌ LUNA16 model not found.")
        model = tf.keras.models.load_model(luna_model_path)
        prediction_class = perform_inference(model, image_path)
        luna_result = "Positive" if prediction_class == 1 else "Negative"

        # Default values in case LUNA is negative
        yolo_result = "N/A"
        yolo_confidence = "N/A"
        vgg_result = "N/A"

        if luna_result == "Positive":
            # === YOLOv8 Inference ===
            print("🔍 YOLOv8 - starting inference")
            yolo_model_path = "../model_lung_yolo.pt"
            yolo_model = YOLO(yolo_model_path)
            malignant_dir = settings.MALIGNANT_IMAGES_DIR
            malignant_images = [f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not malignant_images:
                return HttpResponse("❌ No valid image found in 'malignant' for YOLOv8.")

            selected_image = random.choice(malignant_images)
            malignant_image_path = os.path.join(malignant_dir, selected_image)
            predicted_class, confidence = perform_yolo_inference(malignant_image_path, yolo_model)
            yolo_result = "Nodule bénin" if predicted_class == "normal" else "Malignant"
            yolo_confidence = round(float(confidence * 100), 2)  # ✅ cast to float

            # === VGG + Clinical Inference ===
            print("🧠 VGG model - starting stage prediction")
            stage_vgg_dir = settings.STAGE_IMAGES_DIR
            vgg_images = [f for f in os.listdir(stage_vgg_dir) if f.endswith('.npy')]
            if not vgg_images:
                return HttpResponse("❌ No .npy files found in 'stage' directory.")
            selected_vgg = os.path.join(stage_vgg_dir, random.choice(vgg_images))
            clinical_input = [65, 1, 3.5, 120, 85, 0.8]  # Example
            vgg_model_path = "../trained_vgg_model.h5"
            vgg_result = int(perform_vgg_inference(selected_vgg, clinical_input, vgg_model_path))  # ✅ cast to int

        # === Store in session (use only JSON-serializable types) ===
        request.session['report'] = {
            'image_name': image_name,
            'luna_result': luna_result,
            'yolo_result': yolo_result,
            'yolo_confidence': yolo_confidence,
            'vgg_result': vgg_result,
            'patient_id': f"PT-{random.randint(1000,9999)}",
            'date': datetime.now().strftime('%Y-%m-%d'),
        }
        request.session['report_ready'] = True

    # Always render form, show download if ready
    form = UploadImageForm()
    return render(request, 'radiologue.html', {
        'form': form,
        'report_ready': request.session.get('report_ready', False)
    })
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        image_name = uploaded_file.name
        luna_dir = os.path.join(settings.MEDIA_ROOT, 'luna')
        os.makedirs(luna_dir, exist_ok=True)
        image_path = os.path.join(luna_dir, image_name)

        with open(image_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        print(f"🖼️ Image uploaded and saved to: {image_path}")

        # === LUNA16 Inference ===
        luna_model_path = "../efficient_model.keras"
        if not os.path.exists(luna_model_path):
            return HttpResponse("❌ LUNA16 model not found.")

        model = tf.keras.models.load_model(luna_model_path)
        prediction_class = perform_inference(model, image_path)
        luna_result = "Positive" if prediction_class == 1 else "Negative"

        yolo_result = "N/A"
        yolo_confidence = "N/A"
        vgg_result = "N/A"

        if luna_result == "Positive":
            print("🔍 YOLOv8 - starting inference")
            yolo_model_path = "../model_lung_yolo.pt"
            yolo_model = YOLO(yolo_model_path)
            malignant_dir = settings.MALIGNANT_IMAGES_DIR

            malignant_images = [f for f in os.listdir(malignant_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not malignant_images:
                return HttpResponse("❌ No valid image found in 'malignant' directory for YOLOv8.")

            selected_image = random.choice(malignant_images)
            malignant_image_path = os.path.join(malignant_dir, selected_image)
            predicted_class, confidence = perform_yolo_inference(malignant_image_path, yolo_model)
            if hasattr(confidence, 'item'):
                confidence = confidence.item()
            yolo_result = "Nodule bénin" if predicted_class == "normal" else "Malignant"
            yolo_confidence = round(confidence * 100, 2)

            print("🧠 VGG model - starting stage prediction")
            stage_vgg_dir = settings.STAGE_IMAGES_DIR
            vgg_images = [f for f in os.listdir(stage_vgg_dir) if f.endswith('.npy')]
            if not vgg_images:
                return HttpResponse("❌ No .npy files found in 'stage' directory.")

            selected_vgg = os.path.join(stage_vgg_dir, random.choice(vgg_images))
            clinical_input = [65, 1, 3.5, 120, 85, 0.8]  # Example clinical data
            vgg_model_path = "../trained_vgg_model.h5"
            vgg_result = perform_vgg_inference(selected_vgg, clinical_input, vgg_model_path)

        request.session['report'] = {
            'image_name': image_name,
            'luna_result': luna_result,
            'yolo_result': yolo_result,
            'yolo_confidence': yolo_confidence,
            'vgg_result': vgg_result,
            'patient_id': f"PT-{random.randint(1000,9999)}",
            'date': datetime.now().strftime('%Y-%m-%d'),
        }
        request.session['report_ready'] = True

    form = UploadImageForm()
    return render(request, 'radiologue.html', {
        'form': form,
        'report_ready': request.session.get('report_ready', False)
    })

def show_report(request):
    return render(request, 'show_report_button.html')
def dashboard_view(request):
    return render(request, 'dashboard.html')