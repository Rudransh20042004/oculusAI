from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
EYE_FOLDER = 'eyes'
os.makedirs(EYE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load your model and label encoder once
model = load_model("disease_predictor_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

SYMPTOM_MAP_PATH = "disease_symptom_map.pkl"
with open(SYMPTOM_MAP_PATH, "rb") as f:
    disease_symptom_map = pickle.load(f)

def predict_disease(img_path):
    # Open image with PIL and preprocess like test.py
    img = Image.open(img_path).resize((128,128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Placeholder symptom vector
    symptoms = np.zeros((1, 16))

    # Predict
    pred_probs = model.predict([img_array, symptoms])
    pred_class = np.argmax(pred_probs, axis=1)[0]
    pred_disease = le.inverse_transform([pred_class])[0]
    return pred_disease

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    # Save original uploaded file in uploads folder
    upload_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_path)

    # Also save a copy in eyes folder (if needed for later)
    eye_path = os.path.join(EYE_FOLDER, 'uploaded_eye.png')
    img_cv = cv2.imread(upload_path)
    cv2.imwrite(eye_path, img_cv)

    # Run prediction
    prediction = predict_disease(upload_path)

    return render_template(
        'index.html',
        img_file=f"{UPLOAD_FOLDER}/{file.filename}",
        result=prediction,
        disease_symptom_map=disease_symptom_map
    )

def ask_gpt(prompt):
    """
    Ask GPT for safe clinical advice.
    Adds a system message to guide the model.
    """
    
    response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful medical assistant. Provide safe general advice or first-aid guidance. "
"Avoid diagnosing, prescribing, or giving medications. Focus on home care and precautions."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=500
        )

        # Debug: log raw response to help troubleshoot empty outputs
    print("GPT raw response:", response)

    advice = response.choices[0].message.content.strip()
    if not advice:
        advice = "GPT did not return advice. Please consult a medical professional if needed."
    return advice

@app.route('/get_advice', methods=['POST'])
def get_advice():
    """
    Receives disease and symptoms, builds a safe prompt,
    and returns GPT-generated advice as JSON.
    """
    data = request.get_json()
    disease = data.get('disease')
    common_selected = data.get('common_selected', [])
    extra_symptoms = data.get('extra_symptoms', [])

    # Compute symptom ratio
    common_symptoms = disease_symptom_map.get(disease, [])
    ratio = len(common_selected) / len(common_symptoms) if common_symptoms else 0

    # Summarize symptoms
    summary_symptoms = ", ".join(common_symptoms[:3])
    if len(common_symptoms) > 3:
        summary_symptoms += ", etc."

    # Build user prompt safely
    if ratio >= 0.5:
        base_prompt = (
            f"The patient shows multiple symptoms of {disease}. "
            f"Provide safe first-aid or home care guidance, avoiding medications."
        )
    elif ratio > 0:
        base_prompt = (
            f"The patient shows some symptoms of {disease}. "
            f"Common symptoms include: {summary_symptoms}. "
            f"Provide safe monitoring tips or precautions."
        )
    else:
        base_prompt = (
            f"The patient shows no common symptoms of {disease}. "
            f"Common symptoms include: {summary_symptoms}. "
            f"Provide general advice or preventive guidance."
        )

    # Add extra symptoms if present
    if extra_symptoms:
        extra_prompt = (
            f"Additional reported symptoms: {', '.join(extra_symptoms)}. "
            f"Include any relevant precautions or home care advice."
        )
        full_prompt = f"{base_prompt}\n\n{extra_prompt}"
    else:
        full_prompt = base_prompt

    advice = ask_gpt(full_prompt)
    return jsonify({"advice": advice})



if __name__ == "__main__":
    app.run(debug=True)