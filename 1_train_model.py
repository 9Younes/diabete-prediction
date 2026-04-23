"""
🏥 ENTRAÎNEMENT DU MODÈLE DE PRÉDICTION DU DIABÈTE
Ce script teste plusieurs algorithmes et garde le meilleur.
"""

import pandas as pd
import numpy as np
import kagglehub
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

print("=" * 60)
print("🏥 PROJET DE PRÉDICTION DU DIABÈTE")
print("=" * 60)

# ═══════════════════════════════════════════════════════════
# 1️⃣ TÉLÉCHARGER LES DONNÉES
# ═══════════════════════════════════════════════════════════
print("\n📥 Étape 1 : Téléchargement des données...")
path = kagglehub.dataset_download("akshaydattatraykhare/diabetes-dataset")
df = pd.read_csv(os.path.join(path, "diabetes.csv"))
print(f"✅ Dataset téléchargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ═══════════════════════════════════════════════════════════
# 2️⃣ NETTOYER LES DONNÉES
# ═══════════════════════════════════════════════════════════
print("\n🧹 Étape 2 : Nettoyage des données...")
# Les zéros dans ces colonnes sont en fait des valeurs manquantes
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    df[col] = df[col].replace(0, df[col].median())
print("✅ Valeurs manquantes traitées")

# ═══════════════════════════════════════════════════════════
# 3️⃣ PRÉPARER LES DONNÉES
# ═══════════════════════════════════════════════════════════
print("\n🔧 Étape 3 : Préparation des données...")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"✅ Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")

# ═══════════════════════════════════════════════════════════
# 4️⃣ TESTER PLUSIEURS ALGORITHMES
# ═══════════════════════════════════════════════════════════
print("\n🤖 Étape 4 : Comparaison des algorithmes...")
print("-" * 60)

modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

resultats = {}
for nom, modele in modeles.items():
    modele.fit(X_train_scaled, y_train)
    y_pred = modele.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    resultats[nom] = {"accuracy": acc, "f1": f1, "modele": modele}
    print(f"📊 {nom:25s} → Accuracy: {acc:.2%} | F1: {f1:.2%}")

# ═══════════════════════════════════════════════════════════
# 5️⃣ CHOISIR LE MEILLEUR MODÈLE
# ═══════════════════════════════════════════════════════════
print("\n🏆 Étape 5 : Sélection du meilleur modèle...")
meilleur_nom = max(resultats, key=lambda x: resultats[x]["f1"])
meilleur_modele = resultats[meilleur_nom]["modele"]
print(f"✅ Meilleur modèle : {meilleur_nom}")
print(f"   Accuracy : {resultats[meilleur_nom]['accuracy']:.2%}")
print(f"   F1-Score : {resultats[meilleur_nom]['f1']:.2%}")

# ═══════════════════════════════════════════════════════════
# 6️⃣ OPTIMISER LE MEILLEUR MODÈLE (GridSearch)
# ═══════════════════════════════════════════════════════════
print("\n⚙️ Étape 6 : Optimisation avec GridSearch...")

if meilleur_nom == "Random Forest":
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    base_model = RandomForestClassifier(random_state=42)
elif meilleur_nom == "XGBoost":
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    base_model = XGBClassifier(eval_metric='logloss', random_state=42)
else:
    param_grid = {}
    base_model = meilleur_modele

if param_grid:
    grid = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    modele_final = grid.best_estimator_
    print(f"✅ Meilleurs paramètres : {grid.best_params_}")
else:
    modele_final = meilleur_modele

# Évaluation finale
y_pred_final = modele_final.predict(X_test_scaled)
acc_final = accuracy_score(y_test, y_pred_final)
print(f"\n🎯 ACCURACY FINALE : {acc_final:.2%}")
print("\n📋 Rapport détaillé :")
print(classification_report(y_test, y_pred_final))

# ═══════════════════════════════════════════════════════════
# 7️⃣ SAUVEGARDER LE MODÈLE
# ═══════════════════════════════════════════════════════════
print("\n💾 Étape 7 : Sauvegarde du modèle...")
joblib.dump(modele_final, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Modèle sauvegardé : model.pkl")
print("✅ Scaler sauvegardé : scaler.pkl")

print("\n" + "=" * 60)
print("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
print("=" * 60)
print("\n👉 Lance maintenant l'application avec : streamlit run 2_app.py")