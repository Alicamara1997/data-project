"""
Module d'entraînement des modèles de Machine Learning.
Modèles : Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost.
"""

import pandas as pd
import numpy as np
import time
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib


def create_models() -> dict:
    """Instancie tous les modèles à entraîner."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=10.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule les métriques d'évaluation."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    # MAPE en évitant la division par zéro
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


def train_all_models(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'price',
    model_dir: str = 'models',
    test_size: float = 0.2
) -> dict:
    """
    Entraîne tous les modèles, évalue et sauvegarde les résultats.

    Returns:
        results (dict), X_train, X_test, y_train, y_test
    """
    os.makedirs(model_dir, exist_ok=True)

    # Filtrer les features disponibles
    available = [f for f in feature_cols if f in df.columns]
    missing   = [f for f in feature_cols if f not in df.columns]
    if missing:
        print(f"  ⚠ Features absentes ignorées : {missing}")

    X = df[available].copy().fillna(df[available].median(numeric_only=True))
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Scaler (utile pour Linear / Ridge)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    print(f"\n🏋️  Entraînement — Train : {len(X_train):,}  |  Test : {len(X_test):,}  |  Features : {len(available)}")

    models  = create_models()
    results = {}
    linear_models = {'Linear Regression', 'Ridge Regression'}

    for name, model in models.items():
        print(f"\n  ▶ {name} ...")
        t0 = time.time()

        if name in linear_models:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        elapsed = time.time() - t0
        metrics = compute_metrics(y_test.values, y_pred)
        metrics['training_time'] = round(elapsed, 2)
        metrics['predictions']   = y_pred.tolist()
        results[name] = metrics

        print(f"    MAE  : ${metrics['MAE']:>10,.0f}")
        print(f"    RMSE : ${metrics['RMSE']:>10,.0f}")
        print(f"    R²   :  {metrics['R2']:>10.4f}")
        print(f"    MAPE :  {metrics['MAPE']:>9.1f}%")
        print(f"    Temps:  {elapsed:.1f}s")

        # Sauvegarde du modèle
        fname = name.lower().replace(' ', '_') + '.pkl'
        joblib.dump(model, os.path.join(model_dir, fname))

    # Meilleur modèle (selon R²)
    best_name = max(results, key=lambda k: results[k]['R2'])
    print(f"\n🏆 Meilleur modèle : {best_name}  (R²={results[best_name]['R2']:.4f})")

    # Copier le meilleur modèle
    best_src  = best_name.lower().replace(' ', '_') + '.pkl'
    best_mdl  = joblib.load(os.path.join(model_dir, best_src))
    joblib.dump(best_mdl, os.path.join(model_dir, 'best_model.pkl'))

    # Sauvegarder les métriques en JSON
    to_save = {
        k: {mk: mv for mk, mv in v.items() if mk != 'predictions'}
        for k, v in results.items()
    }
    with open(os.path.join(model_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump({'results': to_save, 'best_model': best_name, 'features': available}, f, indent=2)

    # Sauvegarder les prédictions pour la visualisation
    preds_df = X_test.copy()
    preds_df['y_true'] = y_test.values
    for name_m, res in results.items():
        preds_df[f'y_pred_{name_m}'] = res['predictions']
    preds_df.to_csv(os.path.join(model_dir, 'test_predictions.csv'), index=False)

    # Sauvegarder liste features
    with open(os.path.join(model_dir, 'features.json'), 'w') as f:
        json.dump(available, f)

    return results, X_train, X_test, y_train, y_test, available
