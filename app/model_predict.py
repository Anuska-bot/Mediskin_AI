import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_disease_model.keras")


model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "Atomic Dermatitis",
    "Basal Cell Carcinoma",
    "Eczema",
    "Melanocytic Nevi",
    "Melanoma",
    "Psoriasis",
    "Tinea Ringworm",
    "Warts Molluscum"
]


def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    confidence = round(float(predictions[0][index]) * 100, 2)


    if confidence < 50:
        return "Low confidence – please consult a dermatologist", confidence

    return classes[index], confidence

print(model.output_shape)
