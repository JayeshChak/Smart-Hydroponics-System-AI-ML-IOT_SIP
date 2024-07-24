import pandas as pd
import pickle
import numpy as np

# Load your crop data (replace with your actual data loading code)
crop_data = pd.read_csv('Data/crop_data.csv')

# Load your trained model (replace with your actual model loading code)
with open('models/trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature columns (adjust according to your data)
feature_cols = ['crop_temp', 'crop_humidity', 'crop_ph', 'crop_nitrogen', 'crop_phosphorus',
                'crop_potassium', 'crop_calcium', 'crop_magnesium', 'crop_zinc', 'crop_sodium', 'crop_ec', 'crop_light']

# Function to preprocess input data and predict top crops
def predict_crop(user_inputs):
    # Convert user_inputs to DataFrame
    input_data = pd.DataFrame([user_inputs])

    # Use the loaded model to predict
    probas = model.predict_proba(input_data[feature_cols])
    top_indices = np.argsort(probas[0])[-1:][::-1]
    top_crop = model.classes_[top_indices][0]  # Get the top predicted crop

    # Fetch details of the predicted crop from crop_data
    crop_details = crop_data[crop_data['crop_name'] == top_crop].iloc[0]

    # Return relevant details as a dictionary
    return {
        'crop_name': crop_details['crop_name'],
        'description': crop_details['crop_type'],
        #'image_url': crop_details['image_url']
    }
  