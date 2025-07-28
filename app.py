import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image
import joblib
import os

# Constantes
MODEL_PATH = "models/vgg16_hypermodel.h5"
LABELS_PATH = "models/labels.pkl"
IMAGE_SIZE = (224, 224)

def load_model(model_path):
    return keras.models.load_model(model_path)

def load_labels(labels_path):
    return joblib.load(labels_path)

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image, model, encoder):
    processed_img = preprocess_image(image, IMAGE_SIZE)
    prediction = model.predict(processed_img)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

def extract_true_label(file_name):
    # Récupère le nom de la race avant le premier '_'
    base = os.path.basename(file_name)
    name_part = os.path.splitext(base)[0]
    return name_part.split('_')[0].lower()

# Interface
st.set_page_config(page_title="Prédicteur de Race de Chien", page_icon="🐶", layout="centered")
st.title("🐶 Prédicteur de race de chien")
st.markdown("""
Téléversez une ou plusieurs images de chiens.  
Le nom du fichier doit contenir la race réelle, par exemple : `beagle_01.jpg`.

L'application affichera la **race réelle extraite du nom de fichier** et la **race prédite** par le modèle.
""")

uploaded_files = st.file_uploader(
    label="⬇️ Déposez vos images ici et cliquez sur 'Browse files'",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    try:
        model = load_model(MODEL_PATH)
        encoder = load_labels(LABELS_PATH)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

    for idx, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(f"📷 Image {idx+1} : {uploaded_file.name}")

        img = Image.open(uploaded_file)
        st.image(img, caption="🖼️ Image chargée", use_column_width=True)

        real_race = extract_true_label(uploaded_file.name)

        with st.spinner("🔍 Prédiction en cours..."):
            predicted_race = predict(img, model, encoder)

        st.info(f"🎯 **Race réelle :** {real_race}")
        st.success(f"✅ **Race prédite :** {predicted_race}")

        if real_race.lower() == predicted_race.lower():
            st.success("✔️ La prédiction est correcte !")
        else:
            st.warning("❌ La prédiction est incorrecte.")
