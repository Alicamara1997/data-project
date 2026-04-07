"""
Feature Engineering — Données DVF réelles françaises.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Paris centre (référence)
_PARIS = (48.8566, 2.3522)

# Points côtiers français (approx.)
_COAST_FR = np.array([
    (51.0, 2.5), (50.7, 1.6), (49.5, -1.5), (48.7, -2.0),
    (47.5, -2.5), (47.1, -2.2), (46.6, -1.4), (46.2, -1.1),
    (45.7, -1.1), (44.7, -1.2), (43.7, -1.5), (43.5, -1.8),
    (43.3, -1.5), (43.3, 3.5), (43.1, 5.9), (43.4, 6.8),
    (43.8, 7.4), (41.4, 9.2),   # Corse
])


def _haversine_km(lat1: float, lon1: float,
                  lats2: np.ndarray, lons2: np.ndarray) -> np.ndarray:
    """Distance Haversine approx. en km (vectorisée)."""
    R = 6371.0
    dlat = np.radians(lats2 - lat1)
    dlon = np.radians(lons2 - lon1)
    a = (np.sin(dlat / 2)**2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lats2)) * np.sin(dlon / 2)**2)
    return R * 2 * np.arcsin(np.sqrt(a))


def add_price_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    df['price_per_sqm'] = (df['price'] / df['surface_sqm']).round(2)
    return df


def add_distance_to_paris(df: pd.DataFrame) -> pd.DataFrame:
    lats = df['latitude'].values
    lons = df['longitude'].values
    df['dist_paris_km'] = _haversine_km(
        _PARIS[0], _PARIS[1],
        lats, lons
    ).round(1)
    return df


def add_distance_to_coast(df: pd.DataFrame) -> pd.DataFrame:
    lats = df['latitude'].values
    lons = df['longitude'].values
    dists = np.full(len(df), np.inf)
    for clat, clon in _COAST_FR:
        d = _haversine_km(clat, clon, lats, lons)
        dists = np.minimum(dists, d)
    df['dist_coast_km'] = dists.round(1)
    return df


def add_idf_flag(df: pd.DataFrame) -> pd.DataFrame:
    IDF = {'75', '77', '78', '91', '92', '93', '94', '95'}
    df['is_idf'] = df['department'].isin(IDF).astype(int)
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'month' in df.columns:
        m = df['month'].values
        df['month_sin'] = np.sin(2 * np.pi * m / 12).round(4)
        df['month_cos'] = np.cos(2 * np.pi * m / 12).round(4)
    return df


def add_log_surface(df: pd.DataFrame) -> pd.DataFrame:
    df['log_surface'] = np.log1p(df['surface_sqm']).round(4)
    return df


def encode_property_type(df: pd.DataFrame) -> pd.DataFrame:
    df['is_apartment'] = (df['property_type'] == 'Appartement').astype(int)
    df['is_house']     = (df['property_type'] == 'Maison').astype(int)
    return df


def encode_region(df: pd.DataFrame) -> pd.DataFrame:
    region_median = df.groupby('region')['price'].median()
    region_rank   = region_median.rank(method='dense').astype(int).to_dict()
    df['region_enc'] = df['region'].map(region_rank).fillna(0).astype(int)
    return df


def encode_department(df: pd.DataFrame) -> pd.DataFrame:
    dept_median = df.groupby('department')['price'].median()
    dept_rank   = dept_median.rank(method='dense').astype(int).to_dict()
    df['dept_enc'] = df['department'].map(dept_rank).fillna(0).astype(int)
    return df


def feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("\n⚙️  Feature Engineering (DVF réel)...")
    df = add_price_per_sqm(df)
    df = add_distance_to_paris(df)
    df = add_distance_to_coast(df)
    df = add_idf_flag(df)
    df = add_seasonal_features(df)
    df = add_log_surface(df)
    df = encode_property_type(df)
    df = encode_region(df)
    df = encode_department(df)
    print(f"  ✅ {len(df.columns)} colonnes | {len(df):,} transactions")
    return df


def get_model_features() -> list:
    """Features utilisées pour l'entraînement des modèles ML."""
    return [
        'surface_sqm',      # Surface habitable (m²)
        'rooms',            # Nombre de pièces
        'latitude',         # Latitude GPS
        'longitude',        # Longitude GPS
        'is_apartment',     # Appartement = 1
        'is_house',         # Maison = 1
        'dept_enc',         # Département (encodé par prix médian)
        'region_enc',       # Région (encodée par prix médian)
        'dist_paris_km',    # Distance à Paris (km)
        'dist_coast_km',    # Distance à la côte (km)
        'is_idf',           # Île-de-France = 1
        'log_surface',      # log(surface)
        'month_sin',        # Saisonnalité (sin)
        'month_cos',        # Saisonnalité (cos)
        'year',             # Année de vente
    ]
