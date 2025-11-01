import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = load_model("symptom_predictor_model.h5")
print("✅ Model loaded successfully!")

# Load your CSV
data = pd.read_csv("new_imgsym.csv")
image_dir = "renamed_images/"

# Prepare predictions
predictions = []

for idx, row in data.iterrows():
    img_path = image_dir + row['image']
    try:
        img = load_img(img_path, target_size=(128,128))
        x = img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)

        pred = model.predict(x)[0]
        pred_labels = (pred > 0.5).astype(int)
        predictions.append(pred_labels)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        predictions.append([np.nan]*(data.shape[1]-1))  # fill with NaNs if image fails

# Create a new DataFrame with predicted labels
pred_df = pd.DataFrame(predictions, columns=data.columns[1:])
pred_df.insert(0, "image", data['image'])

# Save to CSV
pred_df.to_csv("predicted_symptoms.csv", index=False)
print("✅ Predictions saved to predicted_symptoms.csv")