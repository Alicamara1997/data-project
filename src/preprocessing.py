"""
Nettoyage des données DVF réelles.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧹 Nettoyage des données DVF...")
    n0 = len(df)

    # ── 1. Garder uniquement les vraies ventes résidentielles ─────────────────
    if 'nature_mutation' in df.columns:
        df = df[df['nature_mutation'] == 'Vente']
    if 'property_type' in df.columns:
        df = df[df['property_type'].isin(['Appartement', 'Maison'])]

    # ── 2. Colonnes obligatoires non-nulles ───────────────────────────────────
    required = ['price', 'surface_sqm', 'rooms', 'latitude', 'longitude']
    df = df.dropna(subset=[c for c in required if c in df.columns])

    # ── 3. Conversions numériques ─────────────────────────────────────────────
    for col in ['price', 'surface_sqm', 'rooms']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[c for c in ['price', 'surface_sqm', 'rooms'] if c in df.columns])

    # ── 4. Filtres de cohérence ────────────────────────────────────────────────
    # Prix : entre 10 000 € et 10 000 000 €
    df = df[(df['price'] >= 10_000) & (df['price'] <= 10_000_000)]
    # Surface : entre 6 m² et 700 m²
    if 'surface_sqm' in df.columns:
        df = df[(df['surface_sqm'] >= 6) & (df['surface_sqm'] <= 700)]
    # Chambres : 1 à 15
    if 'rooms' in df.columns:
        df = df[(df['rooms'] >= 1) & (df['rooms'] <= 15)]
    # Coordonnées en France métropolitaine
    if 'latitude' in df.columns:
        df = df[(df['latitude'] >= 41.0) & (df['latitude'] <= 51.5)]
    if 'longitude' in df.columns:
        df = df[(df['longitude'] >= -5.5) & (df['longitude'] <= 10.0)]

    # ── 5. Prix au m² cohérent ────────────────────────────────────────────────
    price_per_sqm = df['price'] / df['surface_sqm']
    df = df[(price_per_sqm >= 100) & (price_per_sqm <= 30_000)]

    # ── 6. Supprimer les doublons ─────────────────────────────────────────────
    df = df.drop_duplicates()

    # ── 7. Reset index ────────────────────────────────────────────────────────
    df = df.reset_index(drop=True)
    df['rooms'] = df['rooms'].astype(int)

    pct = (1 - len(df) / n0) * 100
    print(f"   → {n0:,} → {len(df):,} lignes  ({pct:.1f}% filtrés)")
    print(f"   → Prix médian : {df['price'].median():,.0f} €")
    print(f"   → Surface médiane : {df['surface_sqm'].median():.0f} m²")
    return df
