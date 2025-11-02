import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle

# Load model and label encoder
model = load_model("disease_predictor_model.h5")
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load and preprocess image
img_path = "vipul.jpg"
img = load_img(img_path, target_size=(128,128))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # batch dimension

# If you have symptom data, use it here
# For demo, use zeros
symptoms = np.zeros((1, 16))  

# Predict
pred_probs = model.predict([img_array, symptoms])
pred_class = np.argmax(pred_probs, axis=1)[0]
pred_disease = le.inverse_transform([pred_class])[0]

print("Predicted disease:", pred_disease)