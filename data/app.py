from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- CONFIG ----------------
MODEL_PATH = "disease_predictor_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
SYMPTOM_MAP_PATH = "disease_symptom_map.pkl"
IMG_SIZE = (128, 128)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load environment variables from .env
load_dotenv()

# ---------------- OPENAI API KEY ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- LOAD MODEL & FILES ----------------
print("Loading model and encoders...")
model = load_model(MODEL_PATH)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(SYMPTOM_MAP_PATH, "rb") as f:
    disease_symptom_map = pickle.load(f)

print("✅ Model and GPT client loaded successfully!\n")

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------- GPT CHAT FUNCTION ----------------
def ask_gpt(prompt):
    """Get a response from GPT"""
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            print("Error: No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("Error: Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Preprocess image
            print("Preprocessing image...")
            X_img = preprocess_image(filepath)
            n_symptoms = scaler.mean_.shape[0]
            X_symptoms = np.zeros((1, n_symptoms))  # dummy symptom vector
            
            # Predict disease
            print("Predicting disease...")
            preds = model.predict([X_img, X_symptoms], verbose=0)
            pred_idx = np.argmax(preds)
            disease = label_encoder.inverse_transform([pred_idx])[0]
            conf = float(preds[0][pred_idx] * 100)  # Convert numpy float32 to Python float
            
            # Get related symptoms
            symptoms = disease_symptom_map.get(disease, [])
            print(f"Prediction: {disease} ({conf:.2f}%)")
            
            return jsonify({
                'disease': disease,
                'confidence': round(conf, 2),
                'symptoms': symptoms,
                'image_url': f'/uploads/{filename}'
            })
        
        print(f"Error: Invalid file type - {file.filename if file else 'None'}")
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Exception in predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/evaluate_symptoms', methods=['POST'])
def evaluate_symptoms():
    data = request.json
    disease = data.get('disease')
    symptom_responses = data.get('symptom_responses', {})  # {symptom: yes/no}
    
    symptoms = disease_symptom_map.get(disease, [])
    yes_count = sum(1 for s in symptoms if symptom_responses.get(s, '').lower() in ['yes', 'y'])
    ratio = yes_count / len(symptoms) if symptoms else 0
    
    advice = None
    advice_type = None
    
    if ratio >= 0.5:
        prompt = (
            f"Give short, safe, step-by-step first-hand aid suggestions for someone showing symptoms of {disease}. "
            f"Keep it under 5 bullet points and avoid medication names."
        )
        advice = ask_gpt(prompt)
        advice_type = "first_aid"
        hospital_alert = True  # Add flag for severe symptoms
    elif ratio > 0:
        prompt = f"Give simple monitoring or prevention tips for {disease}."
        advice = ask_gpt(prompt)
        advice_type = "preventive"
        hospital_alert = False
    else:
        advice = "No matching symptoms — the symptoms may be mild or early stage."
        advice_type = "mild"
        hospital_alert = False
    
    return jsonify({
        'ratio': round(ratio, 2),
        'yes_count': yes_count,
        'total_symptoms': len(symptoms),
        'advice': advice,
        'advice_type': advice_type,
        'hospital_alert': hospital_alert
    })

@app.route('/analyze_extra_symptoms', methods=['POST'])
def analyze_extra_symptoms():
    data = request.json
    disease = data.get('disease')
    user_symptoms = data.get('user_symptoms', [])  # list of strings
    
    if not user_symptoms:
        return jsonify({'error': 'No additional symptoms provided'}), 400
    
    symptoms = disease_symptom_map.get(disease, [])
    
    extra_prompt = (
        f"The predicted disease is {disease} with common symptoms: {', '.join(symptoms)}.\n"
        f"The user also reports these additional symptoms: {', '.join(user_symptoms)}.\n"
        f"Analyze whether these extra symptoms match the predicted disease, and suggest any extra precautions, "
        f"first-hand care, or advice."
    )
    extra_advice = ask_gpt(extra_prompt)
    
    return jsonify({
        'analysis': extra_advice
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)