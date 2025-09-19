import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import io

# ==============================================================================
# --- NEW MODEL TRAINING SECTION ---
# We are now training a more advanced model on a new dataset.
# ==============================================================================

# 1. This new dataset includes a 'disease' column with specific condition names.
#    The data has been synthetically generated for this educational project.
csv_data = """age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,disease
52,1,0,125,212,0,1,168,0,1.0,2,2,3,Coronary Artery Disease
53,1,0,140,203,1,0,155,1,3.1,0,0,3,Coronary Artery Disease
70,1,0,145,174,0,1,125,1,2.6,0,0,3,Coronary Artery Disease
61,1,0,148,203,0,1,161,0,0.0,2,1,3,No Disease Detected
62,0,0,140,268,0,0,160,0,3.6,0,2,2,Arrhythmia
62,0,0,138,294,1,1,106,0,1.9,1,3,2,Cardiomyopathy
58,0,0,100,248,0,0,122,0,1.0,1,0,2,No Disease Detected
41,1,1,135,203,0,1,132,0,0.0,1,0,1,Tachycardia
60,1,0,117,230,1,1,160,1,1.4,2,2,3,Coronary Artery Disease
45,1,0,104,208,0,0,148,1,3.0,1,0,2,Coronary Artery Disease
63,0,0,108,269,0,1,169,1,1.8,1,2,2,Cardiomyopathy
59,1,0,164,176,1,0,90,0,1.0,1,2,1,Coronary Artery Disease
57,1,0,152,274,0,1,88,1,1.2,1,1,3,Coronary Artery Disease
64,1,3,110,211,0,0,144,1,1.8,1,0,2,No Disease Detected
42,1,0,140,226,0,1,178,0,0.0,2,0,2,Tachycardia
58,1,0,128,216,0,0,131,1,2.2,1,3,3,Cardiomyopathy
51,1,2,110,175,0,1,123,0,0.6,2,0,2,No Disease Detected
43,1,0,150,247,0,1,171,0,1.5,2,0,2,Tachycardia
66,0,3,150,226,0,1,114,0,2.6,0,0,2,Arrhythmia
58,1,2,120,340,0,1,172,0,0.0,2,0,2,Tachycardia
50,0,2,120,219,0,1,158,0,1.6,1,0,2,No Disease Detected
49,1,1,130,266,0,1,171,0,0.6,2,0,2,Tachycardia
54,1,0,140,239,0,1,160,0,1.2,2,0,2,No Disease Detected
48,0,2,130,275,0,1,139,0,0.2,2,0,2,No Disease Detected
57,1,2,150,168,0,1,174,0,1.6,2,0,2,Tachycardia
69,0,3,140,239,0,1,151,0,1.8,2,2,2,Arrhythmia
58,0,3,150,283,1,0,162,0,1.0,2,0,2,No Disease Detected
67,1,0,120,229,0,0,129,1,2.6,1,2,3,Coronary Artery Disease
43,1,0,132,247,1,0,143,1,0.1,1,4,3,Coronary Artery Disease
67,0,1,115,564,0,0,160,0,1.6,1,0,3,Arrhythmia
54,1,0,122,286,0,0,116,1,3.2,1,2,2,Cardiomyopathy
71,0,0,112,149,0,1,125,0,1.6,1,0,2,No Disease Detected
56,0,1,140,294,0,0,153,0,1.3,1,0,2,No Disease Detected
57,1,1,130,131,0,1,115,1,1.2,1,1,3,Coronary Artery Disease
46,1,0,140,311,0,1,120,1,1.8,1,2,3,Cardiomyopathy
"""
df = pd.read_csv(io.StringIO(csv_data))

# 2. Separate features (X) from the target (y, which is now the 'disease' column)
X = df.drop('disease', axis=1)
y = df['disease']

# 3. Use LabelEncoder to convert the disease names (strings) into numbers for the model.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Scale the features. This remains the same.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. UPGRADE: Train a RandomForestClassifier. This model is better for multi-class problems.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_encoded)

print("--- Specific Heart Disease Model has been trained successfully ---")

# ==============================================================================
# --- FLASK API SECTION ---
# This part remains mostly the same, but we'll decode the prediction.
# ==============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = [
            float(data['age']), float(data['sex']), float(data['cp']),
            float(data['trestbps']), float(data['chol']), float(data['fbs']),
            float(data['restecg']), float(data['thalach']), float(data['exang']),
            float(data['oldpeak']), float(data['slope']), float(data['ca']),
            float(data['thal'])
        ]
        
        final_features = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(final_features)
        
        # The model predicts a number (e.g., 0, 1, 2...)
        prediction_encoded = model.predict(scaled_features)
        
        # We use the LabelEncoder to turn the number back into the disease name string
        prediction_text = le.inverse_transform(prediction_encoded)[0]
        
        return jsonify({'prediction': prediction_text})

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'An error occurred while processing your request.'}), 400

if __name__ == "__main__":
    app.run(port=5000, debug=True)

