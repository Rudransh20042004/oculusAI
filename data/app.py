from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
EYE_FOLDER = 'eyes'
os.makedirs(EYE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load your model and label encoder once
model = load_model("disease_predictor_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

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
        result=prediction
    )


if __name__ == "__main__":
    app.run(debug=True)