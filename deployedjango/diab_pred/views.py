import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import numpy as np

# Chemin absolu au fichier model.pkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'model.pkl')

print(f"Chemin du modèle : {model_path}")

# Charger le modèle
try:
    model = joblib.load(model_path)
    print("Modèle chargé avec succès:", model)
except Exception as e:
    print("Erreur lors du chargement du modèle:", e)
    model = None

def form_view(request):
    return render(request, 'form.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        if model is None:
            return JsonResponse({'error': 'Le modèle n\'a pas pu être chargé.'})
        try:
            # Récupérer les données du formulaire
            pregnancies = int(request.POST.get('pregnancies'))
            glucose = int(request.POST.get('glucose'))
            bloodpressure = int(request.POST.get('bloodpressure'))
            skinthickness = int(request.POST.get('skinthickness'))
            insulin = int(request.POST.get('insulin'))
            bmi = float(request.POST.get('bmi'))
            dpf = float(request.POST.get('dpf'))
            age = int(request.POST.get('age'))

            features = np.array([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]).reshape(1, -1)
            prediction = model.predict(features)
            probabilities = model.predict_proba(features)
            confidence = probabilities[0][prediction[0]] * 100
            has_diabetes = bool(prediction[0] == 1)

            return JsonResponse({
                'prediction': int(prediction[0]),
                'confidence': confidence,
                'has_diabetes': has_diabetes
            })
        except Exception as e:
            return JsonResponse({'error': str(e)})
    return JsonResponse({'error': 'Invalid request method'})
