"""
ImmoPred AI — Données DVF Réelles
===================================
Source officielle : data.gouv.fr — Demandes de Valeurs Foncières
100% données réelles de transactions immobilières en France (2023)

Téléchargement automatique par département via l'API publique.
"""
from __future__ import annotations
import os
import requests
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─── Départements clés (couverture nationale équilibrée) ──────────────────────
KEY_DEPARTMENTS = [
    # Île-de-France
    '75', '77', '78', '91', '92', '93', '94', '95',
    # PACA
    '06', '13', '83', '84',
    # Occitanie
    '31', '34', '66',
    # Nouvelle-Aquitaine
    '33', '64', '87',
    # Auvergne-Rhône-Alpes
    '38', '69', '74',
    # Pays de la Loire
    '44', '85',
    # Bretagne
    '29', '35',
    # Hauts-de-France
    '59', '62',
    # Grand Est
    '57', '67', '68',
    # Normandie
    '14', '76',
    # Centre-Val de Loire
    '37', '45',
]

BASE_URL = (
    "https://files.data.gouv.fr/geo-dvf/latest/csv/2023/"
    "departements/{dep}.csv.gz"
)

# ─── Mapping département → région ─────────────────────────────────────────────
DEPT_REGION = {
    '75': 'Île-de-France', '77': 'Île-de-France', '78': 'Île-de-France',
    '91': 'Île-de-France', '92': 'Île-de-France', '93': 'Île-de-France',
    '94': 'Île-de-France', '95': 'Île-de-France',
    '06': "PACA", '13': "PACA", '83': "PACA", '84': "PACA",
    '09': 'Occitanie', '11': 'Occitanie', '12': 'Occitanie',
    '30': 'Occitanie', '31': 'Occitanie', '32': 'Occitanie',
    '34': 'Occitanie', '46': 'Occitanie', '48': 'Occitanie',
    '65': 'Occitanie', '66': 'Occitanie', '81': 'Occitanie', '82': 'Occitanie',
    '16': 'Nouvelle-Aquitaine', '17': 'Nouvelle-Aquitaine',
    '19': 'Nouvelle-Aquitaine', '23': 'Nouvelle-Aquitaine',
    '24': 'Nouvelle-Aquitaine', '33': 'Nouvelle-Aquitaine',
    '40': 'Nouvelle-Aquitaine', '47': 'Nouvelle-Aquitaine',
    '64': 'Nouvelle-Aquitaine', '79': 'Nouvelle-Aquitaine',
    '86': 'Nouvelle-Aquitaine', '87': 'Nouvelle-Aquitaine',
    '01': 'Auvergne-Rhône-Alpes', '03': 'Auvergne-Rhône-Alpes',
    '07': 'Auvergne-Rhône-Alpes', '15': 'Auvergne-Rhône-Alpes',
    '26': 'Auvergne-Rhône-Alpes', '38': 'Auvergne-Rhône-Alpes',
    '42': 'Auvergne-Rhône-Alpes', '43': 'Auvergne-Rhône-Alpes',
    '63': 'Auvergne-Rhône-Alpes', '69': 'Auvergne-Rhône-Alpes',
    '73': 'Auvergne-Rhône-Alpes', '74': 'Auvergne-Rhône-Alpes',
    '44': 'Pays de la Loire', '49': 'Pays de la Loire',
    '53': 'Pays de la Loire', '72': 'Pays de la Loire', '85': 'Pays de la Loire',
    '22': 'Bretagne', '29': 'Bretagne', '35': 'Bretagne', '56': 'Bretagne',
    '02': 'Hauts-de-France', '59': 'Hauts-de-France', '60': 'Hauts-de-France',
    '62': 'Hauts-de-France', '80': 'Hauts-de-France',
    '08': 'Grand Est', '10': 'Grand Est', '51': 'Grand Est',
    '52': 'Grand Est', '54': 'Grand Est', '55': 'Grand Est',
    '57': 'Grand Est', '67': 'Grand Est', '68': 'Grand Est', '88': 'Grand Est',
    '14': 'Normandie', '27': 'Normandie', '50': 'Normandie',
    '61': 'Normandie', '76': 'Normandie',
    '18': 'Centre-Val de Loire', '28': 'Centre-Val de Loire',
    '36': 'Centre-Val de Loire', '37': 'Centre-Val de Loire',
    '41': 'Centre-Val de Loire', '45': 'Centre-Val de Loire',
    '21': 'Bourgogne-Franche-Comté', '25': 'Bourgogne-Franche-Comté',
    '39': 'Bourgogne-Franche-Comté', '58': 'Bourgogne-Franche-Comté',
    '70': 'Bourgogne-Franche-Comté', '71': 'Bourgogne-Franche-Comté',
    '89': 'Bourgogne-Franche-Comté', '90': 'Bourgogne-Franche-Comté',
    '2A': 'Corse', '2B': 'Corse',
}

# Colonnes utiles dans le fichier DVF géolocalisé
DVF_COLS = [
    'nature_mutation', 'valeur_fonciere', 'code_departement', 'nom_commune',
    'type_local', 'surface_reelle_bati', 'nombre_pieces_principales',
    'longitude', 'latitude', 'date_mutation',
]


def _download_one(dep: str, save_dir: str) -> pd.DataFrame | None:
    """Télécharge et décompresse le CSV.GZ DVF d'un département."""
    path = os.path.join(save_dir, f"dvf_{dep}.csv.gz")

    if not os.path.exists(path):
        url = BASE_URL.format(dep=dep)
        try:
            r = requests.get(url, timeout=120, stream=True)
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            print(f"  ✅ Dépt {dep} téléchargé")
        except Exception as e:
            print(f"  ⚠ Dépt {dep} — erreur : {e}")
            return None

    try:
        df = pd.read_csv(
            path,
            compression='gzip',
            usecols=lambda c: c in DVF_COLS,
            dtype={'code_departement': str, 'valeur_fonciere': float,
                   'surface_reelle_bati': float, 'nombre_pieces_principales': float,
                   'longitude': float, 'latitude': float},
            low_memory=False
        )
        df['code_departement'] = dep
        return df
    except Exception as e:
        print(f"  ⚠ Lecture dépt {dep} — erreur : {e}")
        return None


def load_raw_data(
    raw_dir: str = 'data/raw/dvf',
    departments: list | None = None,
    max_workers: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Télécharge et charge les données DVF réelles depuis data.gouv.fr.
    Retourne un DataFrame consolidé de transactions immobilières réelles.
    """
    os.makedirs(raw_dir, exist_ok=True)
    depts = departments or KEY_DEPARTMENTS

    print(f"\n📥 Téléchargement DVF ({len(depts)} départements) depuis data.gouv.fr...")
    print("   Source : https://files.data.gouv.fr/geo-dvf/latest/csv/2023/")

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, dep, raw_dir): dep for dep in depts}
        for fut in as_completed(futures):
            df_dep = fut.result()
            if df_dep is not None and len(df_dep) > 0:
                frames.append(df_dep)

    if not frames:
        raise RuntimeError("Aucun fichier DVF téléchargé. Vérifiez votre connexion.")

    df = pd.concat(frames, ignore_index=True)

    # ── Nommage standardisé ──────────────────────────────────────────────────
    df = df.rename(columns={
        'valeur_fonciere':        'price',
        'surface_reelle_bati':    'surface_sqm',
        'nombre_pieces_principales': 'rooms',
        'nom_commune':            'city',
        'code_departement':       'department',
        'type_local':             'property_type',
    })

    # ── Ajout région ────────────────────────────────────────────────────────
    df['region'] = df['department'].map(DEPT_REGION).fillna('Autre')

    # ── Date ────────────────────────────────────────────────────────────────
    if 'date_mutation' in df.columns:
        df['date_mutation'] = pd.to_datetime(df['date_mutation'], errors='coerce')
        df['year']  = df['date_mutation'].dt.year
        df['month'] = df['date_mutation'].dt.month

    print(f"\n✅ DVF chargé : {len(df):,} lignes brutes | "
          f"{df['department'].nunique()} depts | "
          f"{df['city'].nunique() if 'city' in df.columns else '?'} communes")
    return df


def save_data(df: pd.DataFrame, path: str = 'data/raw/housing_data.csv') -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"   → Sauvegardé : {path}")


def load_data_from_csv(path: str = 'data/raw/housing_data.csv') -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Introuvable : {path} — lancez : python main.py")
    return pd.read_csv(path)
