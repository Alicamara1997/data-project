"""
Module d'évaluation et de rapport des performances des modèles.
"""
from __future__ import annotations
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def print_evaluation_report(results: dict) -> None:
    """Affiche un rapport d'évaluation structuré dans la console."""
    print("\n" + "=" * 65)
    print("📊  RAPPORT D'ÉVALUATION DES MODÈLES")
    print("=" * 65)

    rows = {}
    for name, m in results.items():
        rows[name] = {
            'MAE ($)':    f"{m['MAE']:>12,.0f}",
            'RMSE ($)':   f"{m['RMSE']:>12,.0f}",
            'R²':         f"{m['R2']:>12.4f}",
            'MAPE (%)':   f"{m['MAPE']:>11.1f}",
            'Temps (s)':  f"{m['training_time']:>10.1f}"
        }

    report_df = pd.DataFrame(rows).T
    print("\n" + report_df.to_string())

    best = max(results, key=lambda k: results[k]['R2'])
    print(f"\n🏆 Meilleur modèle : {best}")
    print(f"   R²   = {results[best]['R2']:.4f}")
    print(f"   MAE  = ${results[best]['MAE']:,.0f}")
    print(f"   RMSE = ${results[best]['RMSE']:,.0f}")
    print("=" * 65)


def get_feature_importance(model, feature_names: list) -> Optional[pd.DataFrame]:
    """
    Extrait l'importance des features d'un modèle.
    Fonctionne avec RandomForest, GradientBoosting, XGBoost, et modèles linéaires.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None

    fi_df = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    fi_df['importance_pct'] = (fi_df['importance'] / fi_df['importance'].sum() * 100).round(2)
    return fi_df


def plot_predictions_vs_actual(y_true, y_pred, model_name: str, save_path: str = None):
    """Graphique Prédit vs Réel."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Évaluation — {model_name}', fontsize=14, fontweight='bold')

    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.3, color='steelblue', s=10)
    mn = min(y_true.min(), min(y_pred))
    mx = max(y_true.max(), max(y_pred))
    axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    axes[0].set_xlabel('Prix réel ($)')
    axes[0].set_ylabel('Prix prédit ($)')
    axes[0].set_title('Prédit vs Réel')

    # Résidus
    residuals = np.array(y_pred) - np.array(y_true)
    axes[1].hist(residuals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Résidu ($)')
    axes[1].set_ylabel('Fréquence')
    axes[1].set_title('Distribution des résidus')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
