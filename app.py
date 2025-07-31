import os
import re
import json
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

# ========================
# Chemins / constantes
# ========================
# Modifie MODEL_PATH si ton .h5 a un autre nom
MODEL_PATH = "models/vgg16_hypermodel.h5"
LABELS_PATH = "models/labels"                # <-- ton fichier "labels" (joblib list)
WNID_TO_NAME_PATH = "assets/wnid_to_name.json"  # optionnel (WNID -> nom lisible)

# Pillow ≥ 10 : ANTIALIAS n'existe plus
try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS

WNID_RE = re.compile(r"^n\d{8}$")

# ========================
# Helpers divers
# ========================
def canonicalize(label: str) -> str:
    s = label.strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s

def extract_true_label_display_and_canonical(file_name: str, wnid_to_name: dict[str, str]):
    """Retourne (affichage lisible, forme canonique) pour la 'race réelle' d'après le nom de fichier."""
    base = os.path.basename(file_name)
    token = os.path.splitext(base)[0].split("_")[0].strip()

    if WNID_RE.match(token) and wnid_to_name:
        display = wnid_to_name.get(token, token)
        return display, canonicalize(display)

    display = token.replace("-", " ").replace("_", "_")
    return display, canonicalize(display)

# ========================
# Chargements avec cache
# ========================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    return keras.models.load_model(path, compile=False)

@st.cache_data(show_spinner=False)
def load_labels(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de labels introuvable : {path}")
    labels = joblib.load(path)  # ton fichier /models/labels est un joblib list
    if not isinstance(labels, (list, tuple, np.ndarray)):
        raise ValueError("Le fichier de labels doit contenir une liste/tuple/ndarray.")
    return list(labels)

@st.cache_data(show_spinner=False)
def load_wnid_map(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# ========================
# Backbones supportés
# ========================
def get_preprocess_and_size(backbone: str):
    b = (backbone or "").lower()
    if b == "xception":
        from tensorflow.keras.applications.xception import preprocess_input
        return preprocess_input, (299, 299)
    # défaut: VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input
    return preprocess_input, (224, 224)

def infer_backbone_from_model(model) -> str:
    """Devine le backbone via la taille d'entrée du modèle."""
    h, w = model.input_shape[1:3]
    if (h, w) == (224, 224):
        return "VGG16"
    if (h, w) == (299, 299):
        return "Xception"
    # fallback sûr
    return "VGG16"

# ========================
# Prétraitement & prédiction
# ========================
def preprocess_image(image: Image.Image, size: tuple[int, int], preprocess_fn):
    img = image.convert("RGB").resize(size, RESAMPLE)
    x = keras.preprocessing.image.img_to_array(img)
    x = preprocess_fn(x)
    x = np.expand_dims(x, axis=0)
    return x

def predict_one(
    image: Image.Image,
    model,
    labels: list[str],
    preprocess_fn,
    size: tuple[int, int],
    topk: int = 5,
):
    """
    Retourne (pred_label, confidence, probs, topk_list)
    - probs: np.ndarray shape (1, n_classes)
    - topk_list: [(label, prob), ...] trié décroissant
    """
    x = preprocess_image(image, size, preprocess_fn)
    probs = model.predict(x)
    idx = int(np.argmax(probs, axis=1)[0])
    pred_label = labels[idx]
    confidence = float(np.max(probs, axis=1)[0])

    idxs = np.argsort(probs[0])[::-1][:topk]
    topk_list = [(labels[i], float(probs[0][i])) for i in idxs]
    return pred_label, confidence, probs, topk_list

# ========================
# UI Streamlit
# ========================
st.set_page_config(page_title="Prédicteur de race de chien à partir d'une image", page_icon="🐶", layout="centered")
st.title("🐶 Prédicteur de race de chien à partir d'une image")

uploaded = st.file_uploader("Veuillez charger une image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Affiche l'image
    img = Image.open(uploaded)
    st.image(img, caption="Image chargée", width=350)

    # Chargements
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        st.stop()

    try:
        labels = load_labels(LABELS_PATH)
    except Exception as e:
        st.error(f"Erreur chargement labels: {e}")
        st.stop()

    # Guardrail: nb classes cohérent
    num_out = model.output_shape[-1]
    if len(labels) != num_out:
        st.error(
            f"Incohérence labels vs modèle : labels={len(labels)} ≠ sorties_modèle={num_out}.\n"
            f"Le fichier '{LABELS_PATH}' doit contenir exactement {num_out} classes dans le même ordre que l'entraînement."
        )
        st.stop()

    # Choix auto du backbone (prétraitement + taille)
    forced_backbone = infer_backbone_from_model(model)
    preprocess_fn, size = get_preprocess_and_size(forced_backbone)
    st.caption(f"Backbone détecté automatiquement : **{forced_backbone}** (entrée {size[0]}×{size[1]}).")

    # Optionnel: afficher la "race réelle" (depuis nom de fichier)
    wnid_to_name = load_wnid_map(WNID_TO_NAME_PATH)
    real_display, real_canon = extract_true_label_display_and_canonical(uploaded.name, wnid_to_name)
    st.info(f"🎯 **Race réelle (d'après le nom du fichier)** : {real_display}")

    if st.button("Lancer la détection de race"):
        with st.spinner("Prédiction en cours..."):
            pred, conf, probs, topk_list = predict_one(img, model, labels, preprocess_fn, size)

        st.success(f"✅ **Race prédite :** {pred} _(confiance {conf:.2%})_")
        pred_canon = canonicalize(pred)
        if real_canon and (real_canon == pred_canon):
            st.success("✔️ La prédiction est **correcte**.")
        else:
            st.warning("❌ La prédiction est **incorrecte**.")

        with st.expander("Top‑5 des classes"):
            for i, (name, p) in enumerate(topk_list, start=1):
                st.write(f"{i}. {name} — {p:.2%}")