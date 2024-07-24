# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('./pro_crop_data_final.csv')

# Define feature columns and target column
feature_cols = ['crop_temp', 'crop_humidity', 'crop_ph', 'crop_nitrogen', 'crop_phosphorus',
                'crop_potassium', 'crop_calcium', 'crop_magnesium', 'crop_zinc', 'crop_sodium',
                'crop_ec', 'crop_light']
target_col = 'crop_name'

# Check for missing columns
missing_cols = set(feature_cols + [target_col]) - set(data.columns)
if missing_cols:
    raise ValueError(f"Missing columns in the dataset: {missing_cols}")

# Convert categorical features to numerical values using one-hot encoding
data = pd.get_dummies(data, columns=['crop_growth_stage', 'crop_type'])

# Encode target variable
label_encoder = LabelEncoder()
data[target_col] = label_encoder.fit_transform(data[target_col])

# Save the label encoder
joblib.dump(label_encoder, './label_encoder.pkl')

# Define feature matrix and target vector
X = data[feature_cols]
y = data[target_col]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, './scaler.pkl')

# Initialize lists to store model names and accuracies
acc = []
modelname = []

# Function to train and evaluate models


def train_and_evaluate(model, model_name):
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    acc.append(accuracy)
    modelname.append(model_name)
    print(f'{model_name} Training Accuracy: {accuracy*100:.2f}%')
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))
    cv_scores = cross_val_score(model, X_scaled, y, cv=4)
    print(f'{model_name} Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}%')
    # Save the trained model
    with open(f'./{model_name}.pkl', 'wb') as file:
        joblib.dump(model, file)
    print(f'Model saved as ./{model_name}.pkl')


# Train and evaluate Decision Tree
dt_model = DecisionTreeClassifier(
    criterion="entropy", random_state=42, max_depth=8)
train_and_evaluate(dt_model, 'DecisionTreeClassifier')

# Train and evaluate Gaussian Naive Bayes
gnb_model = GaussianNB()
train_and_evaluate(gnb_model, 'GaussianNaiveBayes')

# Train and evaluate Support Vector Machine (SVM)
svm_model = SVC(kernel='linear', probability=True, random_state=2)
train_and_evaluate(svm_model, 'SVMModel')

# Train and evaluate Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=10)
train_and_evaluate(lr_model, 'LogisticRegression')

# Train and evaluate Random Forest
rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
train_and_evaluate(rf_model, 'RandomForest')

# Train and evaluate XGBoost
xgb_model = XGBClassifier(use_label_encoder=False,
                          eval_metric='mlogloss', random_state=2)
train_and_evaluate(xgb_model, 'XGBoost')

# Plot accuracy comparison
plt.figure(figsize=(10, 5))
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x=acc, y=modelname, palette='dark')
plt.show()

# Print accuracy comparison
accuracy_models = dict(zip(modelname, acc))
for k, v in accuracy_models.items():
    print(f'{k} --> {v}')


# Sample input data
input_data = {
    'crop_temp': 55.0,
    'crop_humidity': 240.0,
    'crop_ph': 6,
    'crop_nitrogen': 50.0,
    'crop_phosphorus': 23.0,
    'crop_potassium': 13.5,
    'crop_calcium': 1.5,
    'crop_magnesium': 0.25,
    'crop_zinc': 0.2,
    'crop_sodium': 0.4,
    'crop_ec': 1.3,
    'crop_light': 400.0
}

# Define model paths and their weights
model_paths = {
    'DecisionTreeClassifier': ('./DecisionTreeClassifier.pkl', 1.0),
    'GaussianNaiveBayes': ('./GaussianNaiveBayes.pkl', 1.0),
    'SVMModel': ('./SVMModel.pkl', 0.8795811518324608),
    'LogisticRegression': ('./LogisticRegression.pkl', 0.41492146596858637),
    'RandomForestClassifier': ('./RandomForest.pkl', 0.975130890052356),
    'XGBoost': ('./XGBoost.pkl', 0.943717277486911)
}

# Load models
models = {name: (joblib.load(path), weight)
          for name, (path, weight) in model_paths.items()}

# Function to predict the top N crops using weighted average of models


def predict_top_n_crops_weighted(models, input_data, label_encoder, scaler, top_n=5):
    input_df = pd.DataFrame([input_data])

    # Scaling input data
    input_df_scaled = scaler.transform(input_df)

    crop_probabilities_weighted = {}
    for model_name, (model, weight) in models.items():
        probas = model.predict_proba(input_df_scaled)
        top_n_indices = np.argsort(probas[0])[-top_n:][::-1]
        top_n_crops = label_encoder.inverse_transform(top_n_indices)
        for crop, prob in zip(top_n_crops, probas[0][top_n_indices]):
            if crop not in crop_probabilities_weighted:
                crop_probabilities_weighted[crop] = 0.0
            crop_probabilities_weighted[crop] += prob * weight

    # Select top 5 crops based on weighted average probabilities
    top_5_crops_weighted = sorted(
        crop_probabilities_weighted, key=crop_probabilities_weighted.get, reverse=True)[:5]
    return top_5_crops_weighted


# Load label encoder and scaler
label_encoder = joblib.load('./label_encoder.pkl')
scaler = joblib.load('./scaler.pkl')

# Predict the top 5 crops using weighted average of models
top_5_crops_weighted = predict_top_n_crops_weighted(
    models, input_data, label_encoder, scaler, top_n=5)

print('Top 5 crops that can be grown in hydroponics system (Multi-model weighted prediction):', top_5_crops_weighted)
