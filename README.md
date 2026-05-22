# Airbnb Paris — Prédiction du prix par nuitée

Modèle de régression pour estimer le prix d'une nuitée Airbnb à Paris à partir des caractéristiques du logement, du quartier et des équipements. Le travail complet (EDA, pipeline, comparaison de modèles, analyse des résidus) est dans `notebooks/airbnb.ipynb`.

## Résultats

| Indicateur | Valeur |
|------------|--------|
| Modèle retenu | `GradientBoosting` (100 arbres, `random_state=42`) |
| Échantillon modélisé | 52 532 annonces (après filtre 10–1000 €) |
| Features | 14 (4 numériques, 3 catégorielles, 7 booléens amenities) |
| R² test (holdout 20 %) | **0,4089** |
| MAE test | **77,41 €** |
| Erreur absolue médiane | **45,32 €** |
| P90 des erreurs absolues | **167,78 €** |

Comparaison CV 5-fold (`KFold`, `shuffle=True`, `random_state=42`) : les modèles linéaires échouent (R² très négatif) ; `GradientBoosting` bat `RandomForest` sur le MAE moyen (77,33 € vs 79,28 €). Détail dans `metrics.json` et `decisions.md`.

## Structure

```
airbnb-paris-price-predictor/
├── data/               # listings.csv (gitignored)
├── notebooks/
│   └── airbnb.ipynb    # Notebook principal (EDA → export)
├── model.joblib        # pipeline entraîné (généré par le notebook)
├── metrics.json        # métriques CV + holdout
├── README.md
├── decisions.md        # log des choix techniques (soutenance)
├── pyproject.toml
└── .gitignore
```

## Setup

```bash
git clone https://github.com/Ismaelito91/airbnb-paris-price-predictor.git
cd airbnb-paris-price-predictor
uv venv -p 3.11 .venv
uv sync
uv run jupyter notebook notebooks/airbnb.ipynb
```

Prérequis : Python ≥ 3.11, dépendances dans `pyproject.toml` (scikit-learn, pandas, numpy, matplotlib, seaborn, jupyter, joblib).

## Données

1. [Inside Airbnb](https://insideairbnb.com/get-the-data/) → section **Paris, France**
2. Télécharger `listings.csv.gz` (version **détaillée**, ~75 colonnes — pas le résumé à 16 colonnes)
3. Décompresser en `data/listings.csv`

Le notebook charge **84 055** annonces (79 colonnes), supprime les colonnes non informatives (~40 restantes), puis filtre les prix entre **10 € et 1000 €** → **52 532** lignes exploitables. Le symbole `$` dans le CSV correspond bien à des **euros** pour Paris (pas de conversion de devise).

## Notebook `airbnb.ipynb`

| Section | Contenu |
|---------|---------|
| 1 | Chargement, drop de colonnes (URLs, IDs, redondances, disponibilité), nettoyage `price` |
| 2 | EDA : distribution (skew 2,02), prix par quartier et `room_type` |
| 3 | Feature engineering : parsing `amenities` → 7 booléens `has_*` |
| 4 | `ColumnTransformer` + `Pipeline` + `TransformedTargetRegressor` (`log1p` / `expm1`) |
| 5 | Comparaison de 4 modèles en CV 5-fold |
| 6 | Holdout test, tableau d'erreurs, erreurs par quartier / `room_type` |
| 7 | Visualisations résidus vs prédictions, réel vs prédit |
| 8 | Export `../model.joblib` et `../metrics.json` |

Exécuter **toutes les cellules** depuis `notebooks/` (chemins relatifs vers `data/` et la racine du projet).

## Livrables générés

- **`model.joblib`** : `TransformedTargetRegressor` + preprocessing + `GradientBoostingRegressor` entraîné sur tout `X` / `y`
- **`metrics.json`** : meilleur modèle, scores CV par algorithme, métriques holdout (MAE, R², médiane, P90, taux de sous/sur-estimation)
