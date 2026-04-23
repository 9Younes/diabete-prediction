"""
🏥 APPLICATION WEB DE PRÉDICTION DU DIABÈTE
Interface Streamlit pour utiliser le modèle.
"""

import streamlit as st
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Diabète",
    page_icon="🏥",
    layout="centered"
)

# Charger le modèle et le scaler
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ═══════════════════════════════════════════════════════════
# EN-TÊTE
# ═══════════════════════════════════════════════════════════
st.title("🏥 Prédiction du Diabète")
st.markdown("### Outil d'aide au diagnostic basé sur l'Intelligence Artificielle")
st.markdown("---")

st.info("💡 Entrez les données médicales du patient pour obtenir une prédiction.")

# ═══════════════════════════════════════════════════════════
# FORMULAIRE DE SAISIE
# ═══════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Informations générales")
    grossesses = st.number_input("Nombre de grossesses", min_value=0, max_value=20, value=1)
    age = st.number_input("Âge (années)", min_value=1, max_value=120, value=30)
    bmi = st.number_input("BMI (Indice de masse corporelle)", min_value=10.0, max_value=70.0, value=25.0, step=0.1)
    heredite = st.number_input("Antécédents familiaux (DiabetesPedigreeFunction)", 
                                min_value=0.0, max_value=3.0, value=0.5, step=0.01)

with col2:
    st.subheader("🩺 Données médicales")
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
    tension = st.number_input("Tension artérielle (mm Hg)", min_value=0, max_value=200, value=70)
    peau = st.number_input("Épaisseur de la peau (mm)", min_value=0, max_value=100, value=20)
    insuline = st.number_input("Insuline (mu U/ml)", min_value=0, max_value=900, value=80)

st.markdown("---")

# ═══════════════════════════════════════════════════════════
# BOUTON DE PRÉDICTION
# ═══════════════════════════════════════════════════════════
if st.button("🔍 PRÉDIRE", use_container_width=True, type="primary"):
    
    # Préparer les données
    donnees = np.array([[grossesses, glucose, tension, peau, 
                         insuline, bmi, heredite, age]])
    donnees_scaled = scaler.transform(donnees)
    
    # Prédire
    prediction = model.predict(donnees_scaled)[0]
    probabilite = model.predict_proba(donnees_scaled)[0]
    
    st.markdown("---")
    st.subheader("🎯 Résultat de la prédiction")
    
    # Afficher le résultat
    col_a, col_b = st.columns(2)
    
    with col_a:
        if prediction == 1:
            st.error("### ⚠️ RISQUE DE DIABÈTE DÉTECTÉ")
            st.markdown("**Le modèle prédit que le patient est probablement diabétique.**")
        else:
            st.success("### ✅ PAS DE RISQUE DÉTECTÉ")
            st.markdown("**Le modèle prédit que le patient n'est probablement pas diabétique.**")
    
    with col_b:
        st.metric(
            label="Probabilité d'être diabétique",
            value=f"{probabilite[1]*100:.1f}%"
        )
        st.metric(
            label="Probabilité d'être non-diabétique",
            value=f"{probabilite[0]*100:.1f}%"
        )
    
    # Barre de progression
    st.progress(float(probabilite[1]))
    
    # Recommandations
    st.markdown("### 💡 Recommandations")
    if probabilite[1] > 0.7:
        st.warning("🚨 **Risque élevé** : Consultation médicale urgente recommandée. "
                   "Des examens complémentaires (HbA1c, glycémie à jeun) sont nécessaires.")
    elif probabilite[1] > 0.4:
        st.info("⚠️ **Risque modéré** : Surveillance médicale conseillée. "
                "Adopter un mode de vie sain (alimentation, activité physique).")
    else:
        st.success("✅ **Risque faible** : Continuer à maintenir un mode de vie sain. "
                   "Suivi médical de routine.")
    
    # Avertissement
    st.markdown("---")
    st.caption("⚠️ **Important** : Cet outil est une aide au diagnostic et ne remplace "
               "en aucun cas l'avis d'un professionnel de santé.")

# ═══════════════════════════════════════════════════════════
# SIDEBAR - INFORMATIONS
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.title("ℹ️ À propos")
    st.markdown("""
    ### 🎯 Objectif
    Cette application utilise un modèle de **Machine Learning** pour prédire 
    le risque de diabète à partir de données médicales.
    
    ### 📊 Modèle
    - Entraîné sur le **Pima Indians Diabetes Dataset**
    - 768 patients analysés
    - Algorithme optimisé avec GridSearch
    
    ### 📋 Variables utilisées
    1. Nombre de grossesses
    2. Taux de glucose
    3. Tension artérielle
    4. Épaisseur de la peau
    5. Taux d'insuline
    6. BMI
    7. Antécédents familiaux
    8. Âge
    
    ### ⚠️ Avertissement
    Cet outil est à but **éducatif** uniquement.
    """)