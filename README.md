# 🐶 Dog Breed Classifier

Ce projet Streamlit permet de prédire ou de détecter la race d'un chien à partir d'une image.
Il prend des images de chiens en entrée et retourne les races réelles et les races prédites des chiens.

## 🚀 Lancement

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

2. Placez votre modèle (`best_model.h5`) et l'encodeur (`label_encoder.pkl`) dans le dossier `models/`.

3. Lancez l'application :
```bash
streamlit run app.py
```

## 📁 Structure

- `app.py` : Interface Streamlit
- `models/` : Contient le modèle et le label encoder
- `requirements.txt` : Dépendances Python
