"""
Fonctions de nettoyage et de transformation des données Airbnb Paris.
"""
import ast

import numpy as np
import pandas as pd


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Charge le fichier listings.csv brut."""
    return pd.read_csv(filepath, low_memory=False)


def parse_price(price_series: pd.Series) -> pd.Series:
    """Convertit la colonne price de string '$1,234.00' en float."""
    return (
        price_series
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )


def parse_amenities(amenities_series: pd.Series) -> pd.DataFrame:
    """Parse la colonne amenities et retourne un DataFrame de booléens."""
    parsed = amenities_series.apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else []
    )
    all_amenities = set()
    for amenity_list in parsed:
        all_amenities.update(amenity_list)
    return parsed, all_amenities


def create_amenity_features(parsed_amenities: pd.Series, selected: list[str]) -> pd.DataFrame:
    """Crée des colonnes booléennes pour les amenities sélectionnées."""
    df = pd.DataFrame()
    for amenity in selected:
        col_name = "has_" + amenity.lower().replace(" ", "_").replace("/", "_")
        df[col_name] = parsed_amenities.apply(lambda x: amenity in x).astype(int)
    return df
