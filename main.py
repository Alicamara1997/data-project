"""
Pipeline principal — ImmoPred AI France
=======================================
Exécute les étapes dans l'ordre :
  1. Téléchargement des données DVF réelles (data.gouv.fr)
  2. Nettoyage
  3. Ingénierie des features
  4. Entraînement de 5 modèles ML
  5. Évaluation & rapport
  6. Sauvegarde des modèles

Usage :
    python main.py

Ensuite, lancez l'application Streamlit :
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import joblib
import pandas as pd

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_raw_data, save_data
from src.preprocessing import clean_data
from src.feature_engineering import feature_engineering_pipeline, get_model_features
from src.train_models import train_all_models
from src.evaluate import print_evaluation_report, get_feature_importance


def main():
    print("=" * 60)
    print("🏠  IMMOPRED AI — DONNÉES RÉELLES DVF")
    print("    Source : Demandes de Valeurs Foncières 2023")
    print("    Licence : data.gouv.fr — Données officielles")
    print("=" * 60)

    # ─── 1. Chargement ────────────────────────────────────────────
    print("\n📥 Étape 1 — Chargement des données...")
    df_raw = load_raw_data()
    save_data(df_raw, 'data/raw/housing_data.csv')
    print(f"   → {len(df_raw):,} propriétés chargées  |  {df_raw.shape[1]} colonnes")

    # ─── 2. Nettoyage ─────────────────────────────────────────────
    print("\n🧹 Étape 2 — Nettoyage des données...")
    df_clean = clean_data(df_raw)
    os.makedirs('data/processed', exist_ok=True)
    df_clean.to_csv('data/processed/housing_clean.csv', index=False)
    print(f"   → {len(df_clean):,} lignes après nettoyage")

    # ─── 3. Feature Engineering ───────────────────────────────────
    print("\n⚙️  Étape 3 — Ingénierie des features...")
    df_eng = feature_engineering_pipeline(df_clean)
    df_eng.to_csv('data/processed/housing_engineered.csv', index=False)

    # ─── 4. Entraînement ──────────────────────────────────────────
    print("\n🤖 Étape 4 — Entraînement des modèles...")
    feature_cols = get_model_features()

    results, X_train, X_test, y_train, y_test, used_features = train_all_models(
        df_eng,
        feature_cols,
        target_col='price',
        model_dir='models'
    )

    # ─── 5. Évaluation ────────────────────────────────────────────
    print("\n📊 Étape 5 — Rapport d'évaluation...")
    print_evaluation_report(results)

    # Feature importance du meilleur modèle
    with open('models/results.json', 'r') as f:
        meta = json.load(f)
    best_name = meta['best_model']
    best_file = best_name.lower().replace(' ', '_') + '.pkl'
    best_model = joblib.load(os.path.join('models', best_file))
    fi = get_feature_importance(best_model, used_features)
    if fi is not None:
        fi.to_csv('models/feature_importance.csv', index=False)
        print(f"\n🔑 Top 5 features importantes ({best_name}) :")
        print(fi.head(5).to_string(index=False))

    # ─── 6. Résumé final ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ Pipeline terminé avec succès !")
    print(f"   • Données     : data/processed/housing_engineered.csv")
    print(f"   • Modèles     : models/")
    print(f"   • Résultats   : models/results.json")
    print(f"   • Meilleur    : {best_name}")
    print("\n🚀 Lancer l'application :")
    print("   streamlit run app/streamlit_app.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
