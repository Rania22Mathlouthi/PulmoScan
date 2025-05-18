from django.shortcuts import render, get_object_or_404, redirect
from .models import Patient
from django.contrib.auth.decorators import login_required
import joblib
import os
from django.conf import settings
import numpy as np

def Patient_list(request):
    Patients = Patient.objects.all()
    return render(request, 'index.html', {'Patients': Patients})
# Charger une seule fois
model = joblib.load('../early_model.pkl')
if isinstance(model, dict):
    model = model.get('model')  # S'assure que c'est le modèle scikit-learn

def precoce_view(request):
    result = None
    if request.method == 'POST':
        features = [
            int(request.POST.get('GENDER')),
            int(request.POST.get('AGE')),
            int(request.POST.get('SMOKING')),
            int(request.POST.get('YELLOW_FINGERS')),
            int(request.POST.get('ANXIETY')),
            int(request.POST.get('PEER_PRESSURE')),
            int(request.POST.get('CHRONIC_DISEASE')),
            int(request.POST.get('FATIGUE')),
            int(request.POST.get('ALLERGY')),
            int(request.POST.get('WHEEZING')),
            int(request.POST.get('ALCOHOL_CONSUMING')),
            int(request.POST.get('COUGHING')),
            int(request.POST.get('SHORTNESS_OF_BREATH')),
            int(request.POST.get('SWALLOWING_DIFFICULTY')),
            int(request.POST.get('CHEST_PAIN')),
        ]

        input_data = np.array([features])  # (1, 15)
        prediction = model.predict(input_data)[0]
        result = "🧠 High risk of lung cancer" if prediction == 1 else "✅ Low risk"

    return render(request, 'Precoce.html', {'result': result})
