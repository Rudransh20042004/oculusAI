import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

# --------- PARAMETERS ---------
IMAGE_DIR = "renamed_images/"      # folder with images
CSV_FILE = "merged.csv"  # CSV with images & symptom columns
IMG_SIZE = (128, 128)              # resize images
BATCH_SIZE = 32
EPOCHS = 30

# --------- LOAD CSV ---------
df = pd.read_csv(CSV_FILE)

# Drop unwanted last column if exists
if "Unnamed: 17" in df.columns:
    df = df.iloc[:, :-1]

# Symptom features
symptom_cols = df.columns[1:-1]  # all except 'image' and 'disease_label'
X_symptoms = df[symptom_cols].values

# Standardize symptoms
scaler = StandardScaler()
X_symptoms = scaler.fit_transform(X_symptoms)

# Encode disease labels
le = LabelEncoder()
y = le.fit_transform(df['disease_label'])
num_classes = len(le.classes_)
y_categorical = to_categorical(y, num_classes=num_classes)

# --------- LOAD IMAGES ---------
def load_images(df, img_dir, img_size):
    images = []
    for img_name in df['image']:
        path = os.path.join(img_dir, img_name)
        img = load_img(path, target_size=img_size)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
    return np.array(images)

X_images = load_images(df, IMAGE_DIR, IMG_SIZE)

# --------- TRAIN TEST SPLIT ---------
X_img_train, X_img_val, X_sym_train, X_sym_val, y_train, y_val = train_test_split(
    X_images, X_symptoms, y_categorical, test_size=0.2, random_state=42, stratify=y
)

# --------- MODEL ---------
# Image branch
img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = Conv2D(32, (3,3), activation='relu')(img_input)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

# Symptom branch
sym_input = Input(shape=(X_symptoms.shape[1],))
y_branch = Dense(32, activation='relu')(sym_input)
y_branch = Dense(32, activation='relu')(y_branch)

# Merge branches
combined = Concatenate()([x, y_branch])
z = Dense(64, activation='relu')(combined)
output = Dense(num_classes, activation='softmax')(z)  # multi-class classification

model = Model(inputs=[img_input, sym_input], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --------- TRAIN ---------
model.fit(
    [X_img_train, X_sym_train], y_train,
    validation_data=([X_img_val, X_sym_val], y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# --------- SAVE MODEL ---------
model.save("disease_predictor_model.h5")
print("Model saved as disease_predictor_model.h5")

# --------- SAVE LABEL ENCODER ---------
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("Label encoder saved as label_encoder.pkl")


# --------- MAP DISEASE → RELEVANT SYMPTOMS ---------
disease_symptom_map = {}
for disease in df["disease_label"].unique():
    subset = df[df["disease_label"] == disease]
    # find symptoms that are most common (value >= 0.5 on average)
    avg_symptom_values = subset[symptom_cols].mean()
    relevant = avg_symptom_values[avg_symptom_values > 0.5].index.tolist()
    disease_symptom_map[disease] = relevant

with open("disease_symptom_map.pkl", "wb") as f:
    pickle.dump(disease_symptom_map, f)
print("✅ Disease-symptom map saved as disease_symptom_map.pkl")