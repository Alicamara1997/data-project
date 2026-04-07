# 🏠 ImmoPred AI — Projet Data Science Immobilier

> **Prédiction de prix immobilier par Machine Learning**  
> Dataset California Housing · 5 modèles ML · Dashboard Streamlit interactif

---

## 🗂️ Structure du projet

```
data_project/
├── data/
│   ├── raw/                  ← Données brutes (auto-générées)
│   └── processed/            ← Données nettoyées et enrichies
├── models/                   ← Modèles entraînés (.pkl) + métriques
├── src/
│   ├── data_loader.py        ← Chargement + features synthétiques
│   ├── preprocessing.py      ← Nettoyage, outliers, valeurs manquantes
│   ├── feature_engineering.py← Nouvelles variables (prix/m², distances…)
│   ├── train_models.py       ← Entraînement 5 modèles ML
│   └── evaluate.py           ← Métriques & visualisations
├── app/
│   └── streamlit_app.py      ← Dashboard interactif
├── main.py                   ← Script pipeline complet
├── requirements.txt
└── README.md
```

---

## ⚡ Démarrage rapide

### 1. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. Lancer le pipeline ML
```bash
python main.py
```

### 3. Démarrer l'application
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Dataset

**California Housing** (scikit-learn) enrichi avec des features synthétiques :

| Feature originale | Feature créée |
|---|---|
| MedInc (revenu médian) | price_per_sqm |
| HouseAge | age_category, income_category |
| AveRooms / AveBedrms | rooms_per_person |
| Latitude / Longitude | dist_to_coast, region |
| — | surface_sqm, has_garage, has_pool, school_score |

---

## 🤖 Modèles entraînés

| Modèle | Type | Avantages |
|---|---|---|
| Linear Regression | Baseline | Interprétable |
| Ridge Regression | Régularisé | Robuste aux colinéarités |
| Random Forest | Ensemble | Robuste, peu de tuning |
| Gradient Boosting | Boosting | Précis |
| **XGBoost** | **Boosting** | **Meilleur R²** |

---

## 📈 Métriques d'évaluation

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Square Error
- **R²** — Coefficient de détermination
- **MAPE** — Mean Absolute Percentage Error

---

## 🖥️ App Streamlit

| Onglet | Contenu |
|---|---|
| 🏠 Accueil | KPIs, distribution des prix, carte géo |
| 📊 Exploration | Corrélations, boxplots, scatter plots |
| 🤖 Modèles | Comparaison R²/MAE, radar, résidus, feature importance |
| 🔮 Prédiction | Formulaire interactif + jauge de prix |
