# Airbnb Paris — Prédiction du prix par nuitée

## Contexte

Modèle de Machine Learning pour prédire le prix optimal d'une nuitée Airbnb à Paris, à partir des caractéristiques du logement (quartier, type, capacité, équipements, avis, etc.).

**Dataset** : [Inside Airbnb — Paris](https://insideairbnb.com/get-the-data/) (~70 000 logements, ~75 colonnes)

## Structure du projet

```
airbnb-paris-price-predictor/
├── data/
│   ├── raw/                # Données brutes (listings.csv) — NON versionné
│   └── processed/          # Données nettoyées
├── notebooks/
│   └── 01_exploration_and_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py  # Fonctions de nettoyage et parsing
│   └── modeling.py         # Pipelines et entraînement
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/<votre-org>/airbnb-paris-price-predictor.git
cd airbnb-paris-price-predictor

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Données

1. Aller sur https://insideairbnb.com/get-the-data/
2. Section **Paris, France** → télécharger `listings.csv.gz` (version détaillée, ~75 colonnes)
3. Décompresser et placer le fichier dans `data/raw/listings.csv`

> ⚠️ Ne pas prendre la version résumée (~16 colonnes). Prendre la version **détaillée**.

## Approche technique

- **Cible** : `price` (parsé depuis string `"$120.00"` → float)
- **Features** : numériques + catégorielles + amenities (booléens)
- **Transformation** : `TransformedTargetRegressor` avec `log1p` / `expm1`
- **Pipeline** : `ColumnTransformer` (StandardScaler + OneHotEncoder)
- **Modèles comparés** : LinearRegression, Ridge, RandomForest, GradientBoosting
- **Évaluation** : Cross-validation 5-fold (R² et MAE, mean ± std)

## Équipe

- À compléter

## Licence

Projet académique — YNOV
