from flask import Flask, redirect, render_template, request
from markupsafe import Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.predict import predict_crop
from utils.fertilizer import recommend_fertilizer
import requests
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)

disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

crop_recommendation_model_path = 'models/trained_model.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction


@app.route('/')
def home():
    title = 'Home page'
    return render_template('index.html', title=title)


@app.route('/crop-recommend')
def crop_recommend():
    title = 'Crop recommendation page'
    return render_template('crop.html', title=title)


@app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Fertilizer Suggestion Page'
    return render_template('fertilizer.html', title=title)


@app.route('/disease.html')
def disease():
    return render_template('disease.html')


@app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Crop Recommending'
    try:
        if request.method == 'POST':
            user_inputs = {
                'crop_temp': float(request.form['water_temp']),
                'crop_humidity': float(request.form['humidity']),
                'crop_ph': float(request.form['water_ph']),
                'crop_ec': float(request.form['water_ec']),
                'crop_nitrogen': float(request.form['water_nitrogen']),
                'crop_phosphorus': float(request.form['water_phosphorus']),
                'crop_potassium': float(request.form['water_potassium']),
                'crop_calcium': float(request.form['water_calcium']),
                'crop_magnesium': float(request.form['water_magnesium']),
                'crop_zinc': float(request.form['water_zinc']),
                'crop_sodium': float(request.form['water_sodium']),
                'crop_light': float(request.form['light'])
            }

            predicted_crop = predict_crop(user_inputs)

            return render_template('crop-result.html', predicted_crop=predicted_crop, title=title)

    except ValueError as e:
        return str(e)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Fertilizer Suggesting'
    try:
        crop_name = request.form.get('crop_name')
        crop_growth_stage = request.form.get('growth_stage')
        water_content = {
            'ph': float(request.form.get('water_ph')),
            'ec': float(request.form.get('water_ec')),
            'nitrogen': float(request.form.get('water_nitrogen')),
            'phosphorus': float(request.form.get('water_phosphorus')),
            'potassium': float(request.form.get('water_potassium')),
            'calcium': float(request.form.get('water_calcium')),
            'magnesium': float(request.form.get('water_magnesium')),
            'sulfur': float(request.form.get('water_sulfur')),
            'copper': float(request.form.get('water_copper')),
            'chlorine': float(request.form.get('water_chlorine')),
            'boron': float(request.form.get('water_boron')),
            'iron': float(request.form.get('water_iron')),
            'zinc': float(request.form.get('water_zinc')),
            'manganese': float(request.form.get('water_manganese')),
            'molybdenum': float(request.form.get('water_molybdenum')),
            'nickel': float(request.form.get('water_nickel')),
            'cobalt': float(request.form.get('water_cobalt')),
            'sodium': float(request.form.get('water_sodium'))
        }

        response = recommend_fertilizer(crop_name, crop_growth_stage, water_content)

        return render_template('fertilizer-result.html', recommendation=response, title=title)

    except ValueError as e:
        return str(e)



@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except Exception as e:
            return str(e)
    return render_template('disease.html', title=title)


if __name__ == '__main__':
    app.run(debug=False)
