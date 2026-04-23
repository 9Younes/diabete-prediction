# 🏥 Diabetes Prediction - Machine Learning Web App

> Application web d'aide au diagnostic précoce du diabète basée sur l'Intelligence Artificielle.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=flat&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

## 📸 Démo

<!-- Ajoute ici ta capture d'écran ou GIF après -->
<!-- ![Demo](screenshots/demo.gif) -->

## 🎯 Objectif

Développer un modèle de Machine Learning capable de prédire le **risque de diabète** 
chez un patient à partir de **8 paramètres médicaux**, déployé via une interface web interactive.

Le diabète touche plus de **500 millions de personnes dans le monde**. 
Un dépistage précoce peut sauver des vies. 🩺

## ✨ Fonctionnalités

- 🔍 Prédiction en temps réel du risque de diabète
- 📊 Affichage des probabilités et recommandations
- 🎨 Interface web moderne et intuitive (Streamlit)
- 🤖 Comparaison automatique de 5 algorithmes ML
- ⚙️ Optimisation des hyperparamètres avec GridSearchCV

## 📊 Dataset

- **Source** : [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)
- **Taille** : 768 patients
- **Variables** : 8 features médicales + 1 variable cible

| Variable | Description |
|----------|-------------|
| Pregnancies | Nombre de grossesses |
| Glucose | Taux de glucose dans le sang |
| BloodPressure | Tension artérielle (mm Hg) |
| SkinThickness | Épaisseur du pli cutané (mm) |
| Insulin | Taux d'insuline (mu U/ml) |
| BMI | Indice de masse corporelle |
| DiabetesPedigreeFunction | Antécédents familiaux |
| Age | Âge (années) |
| **Outcome** | 0 = Non-diabétique, 1 = Diabétique |

## 🤖 Modèles testés

| Algorithme | Accuracy | F1-Score |
|------------|----------|----------|
| **Logistic Regression** 🏆 | **78%** | **66%** |
| Random Forest | 76% | 63% |
| Gradient Boosting | 77% | 65% |
| XGBoost | 75% | 63% |
| SVM | 75% | 63% |

Le meilleur modèle a été optimisé avec **GridSearchCV** (5-fold cross-validation).

## 🛠️ Technologies utilisées

- **Python 3.11**
- **Scikit-learn** - Algorithmes de Machine Learning
- **XGBoost** - Gradient Boosting
- **Streamlit** - Interface web
- **Pandas / NumPy** - Traitement des données
- **Matplotlib / Seaborn** - Visualisation
- **Joblib** - Sérialisation du modèle

## 🚀 Installation

### Prérequis
- Python 3.11+
- pip

### Étapes

1️⃣ **Cloner le repo**
\`\`\`bash
git clone https://github.com/9Younes/diabete-prediction.git
cd diabete-prediction
\`\`\`

2️⃣ **Créer un environnement virtuel**
\`\`\`bash
python -m venv venv
venv\\Scripts\\activate  # Windows
# source venv/bin/activate  # Mac/Linux
\`\`\`

3️⃣ **Installer les dépendances**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

4️⃣ **Entraîner le modèle**
\`\`\`bash
python 1_train_model.py
\`\`\`

5️⃣ **Lancer l'application**
\`\`\`bash
streamlit run 2_app.py
\`\`\`

L'application s'ouvre sur `http://localhost:8501` 🌐

## 📁 Structure du projet

\`\`\`
diabete-prediction/
│
├── 📄 1_train_model.py       # Script d'entraînement et comparaison des modèles
├── 📄 2_app.py                # Application web Streamlit
├── 📄 requirements.txt        # Dépendances Python
├── 📄 model.pkl               # Modèle ML entraîné
├── 📄 scaler.pkl              # StandardScaler entraîné
├── 📄 README.md               # Documentation
└── 📄 .gitignore              # Fichiers ignorés par Git
\`\`\`

## 💡 Méthodologie

1. **Exploration des données** (EDA) : distribution, corrélations, outliers
2. **Nettoyage** : traitement des valeurs manquantes (zéros aberrants)
3. **Préparation** : split train/test (80/20), normalisation (StandardScaler)
4. **Entraînement** : comparaison de 5 algorithmes de classification
5. **Optimisation** : GridSearchCV sur le meilleur modèle
6. **Évaluation** : Accuracy, Precision, Recall, F1-Score, Matrice de confusion
7. **Déploiement** : Interface Streamlit

## 📈 Résultats

- ✅ **78% de précision** sur les données de test
- ✅ Interface web utilisable et intuitive
- ✅ Temps de prédiction < 100 ms
- ✅ Modèle léger (< 5 MB)

## ⚠️ Avertissement

> Cet outil est à but **éducatif** et ne remplace **PAS** l'avis d'un professionnel de santé. 
> Il sert uniquement d'aide au diagnostic.

## 🔮 Améliorations futures

- [ ] Intégrer plus de données (dataset plus large)
- [ ] Tester le Deep Learning (réseaux de neurones)
- [ ] Déployer sur le cloud (Streamlit Cloud, Heroku)
- [ ] Ajouter un système de suivi des patients
- [ ] Supporter plusieurs langues

## 👤 Auteur

**Younes** 
- 🐙 GitHub : [@9Younes](https://github.com/9Younes)
- 💼 LinkedIn : https://www.linkedin.com/in/younes-bennini-103511256/

## 📜 Licence

Ce projet est sous licence **MIT** - voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

⭐ Si ce projet t'a plu, n'hésite pas à lui mettre une étoile sur GitHub !