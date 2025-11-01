'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

data = pd.read_csv("new_imgsym.csv")

# Separate X (filenames) and y (symptom labels)
image_paths = "renamed_images/" + data["image"]  # add folder prefix
y = data.drop(columns=["image"]) # all symptom columns as numpy array
num_labels = y.shape[1]

df_small = data.sample(frac=0.45, random_state=42)  # use 10% for quicker testing




datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=df_small,
    directory="renamed_images/",
    x_col="image",
    y_col=list(data.columns[1:]),  # all symptom columns
    target_size=(128, 128),
    batch_size=32,
    class_mode="raw",   # <-- important for multi-label (not categorical)
    subset="training"
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df_small,
    directory="renamed_images/",
    x_col="image",
    y_col=list(data.columns[1:]),
    target_size=(128, 128),
    batch_size=32,
    class_mode="raw",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_labels, activation='sigmoid')  # multi-label output
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15
)

new_img = tf.keras.utils.load_img("renamed_images/1.jpg", target_size=(128,128))
x = tf.keras.utils.img_to_array(new_img) / 255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)[0]
predicted_symptoms = (pred > 0.5).astype(int)

print(predicted_symptoms)
    
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, average_precision_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# --- Load CSV data ---
data = pd.read_csv("new_imgsym.csv")
image_dir = "renamed_images/"

# Split into train/test so that images are unique in each set
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

# Number of symptom labels
num_labels = data.shape[1] - 1

# --- Compute class weights for each symptom ---
# This helps with imbalance
class_weights = {}
for i, col in enumerate(data.columns[1:]):
    counts = data[col].value_counts()
    # weight = total_samples / (2 * number_of_samples_in_class)
    weight_for_0 = counts.get(0,0)
    weight_for_1 = counts.get(1,0)
    if weight_for_1 == 0:
        class_weights[i] = 1.0
    else:
        class_weights[i] = weight_for_0 / weight_for_1

print("Class weights:", class_weights)

# --- Image data generators ---
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=image_dir,
    x_col="image",
    y_col=list(data.columns[1:]),
    target_size=(128, 128),
    batch_size=32,
    class_mode='raw'
)

test_gen = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=image_dir,
    x_col="image",
    y_col=list(data.columns[1:]),
    target_size=(128, 128),
    batch_size=32,
    class_mode='raw',
    shuffle=False
)

# --- Build model ---
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128,128,3))
base_model.trainable = False  # freeze base initially

inputs = Input(shape=(128,128,3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(num_labels, activation='sigmoid')(x)
model = Model(inputs, outputs)

model.compile(optimizer=Adam(0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# --- Train model ---
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=15
)

# --- Evaluate model ---
y_true = test_df.iloc[:, 1:].values
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute multi-label metrics
f1 = f1_score(y_true, y_pred, average='macro')
avg_precision = average_precision_score(y_true, y_pred_prob, average='macro')

print(f"\n✅ Test Accuracy: {(y_pred==y_true).mean():.2%}")
print(f"🧮 Macro F1-score: {f1:.4f}")
print(f"📊 Average Precision: {avg_precision:.4f}")

# Save the trained model
model.save("symptom_predictor_model.h5")
print("✅ Model saved as symptom_predictor_model.h5")

# --- Predict a single image example ---
new_img_path = image_dir + "1.jpg"
new_img = tf.keras.utils.load_img(new_img_path, target_size=(128,128))
x = tf.keras.utils.img_to_array(new_img) / 255.0
x = np.expand_dims(x, axis=0)
pred = model.predict(x)[0]
predicted_symptoms = (pred > 0.5).astype(int)
print("Predicted symptoms (0=absent, 1=present):", predicted_symptoms)

