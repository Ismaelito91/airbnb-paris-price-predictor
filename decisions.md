# Decisions Log

Historique des choix techniques et justifications pour la soutenance. Aligné sur `notebooks/airbnb.ipynb` et `metrics.json`.

---

## Données & nettoyage

| Étape | Décision | Justification |
|-------|----------|---------------|
| Source | Inside Airbnb — `listings.csv` Paris | Dataset public, à jour, adapté au cas d'usage |
| Volume brut | 84 055 lignes × 79 colonnes | Snapshot complet avant filtrage |
| Drop colonnes | 79 → 40 colonnes | URLs, texte libre, IDs scraping, doublons (`neighbourhood` vs `neighbourhood_cleansed`), fenêtres de nuits redondantes, dates sans FE, colonnes de **disponibilité** (`availability_30/60/90`, `has_availability`) |
| Cible `price` | Regex `$` + virgules → float en **€** | Le `$` est un format Inside Airbnb ; valeurs parisiennes = euros |
| Filtre prix | Strictement ]10 € ; 1000 €] | Exclut saisies aberrantes et premium extrême ; **52 532** annonces retenues |
| Stats prix filtré | moyenne 213 €, médiane ~158 €, skew 2,02 | Queue à droite → transformation `log1p` sur `y` |

---

## Features retenues (14)

| Feature | Type | Justification |
|---------|------|---------------|
| `accommodates` | num | Capacité → impact direct sur le prix |
| `bedrooms` | num | Plus de chambres = logement plus grand / plus cher |
| `beds` | num | Proxy de la taille |
| `bathrooms` | num | Indicateur de standing |
| `neighbourhood_cleansed` | cat | Levier géographique #1 (EDA : Élysée 254 € vs Ménilmontant 107 €, ~2,4×) ; OneHot plus interprétable en soutenance que lat/long bruts |
| `room_type` | cat | Hotel room 338 € >> Shared room 52 € (médianes EDA) |
| `property_type` | cat | Loft, studio, rental unit → segments de prix différents |
| `has_wifi`, `has_tv`, `has_elevator`, `has_washer`, `has_dryer`, `has_air_conditioning`, `has_balcony` | bool | Parsées depuis `amenities` (JSON) ; confort perçu sans NLP |

Imputation et scaling : **dans le pipeline uniquement** (`SimpleImputer` median / constant `"Unknown"`, puis `StandardScaler` ou `OneHotEncoder`), pour éviter la fuite en CV.

---

## Features exclues (et pourquoi)

| Exclusion | Raison |
|-----------|--------|
| `name`, `description`, URLs, `host_about` | Texte libre → NLP hors scope |
| `host_response_rate`, `host_acceptance_rate` | Beaucoup de NaN ; lien indirect avec le prix |
| `latitude` / `longitude` | Gardées en EDA mais pas en modèle : `neighbourhood_cleansed` capture déjà l'effet quartier |
| `review_scores_*`, `reviews_per_month` | NaN fréquents sur annonces récentes ; signal avis non intégré |
| `estimated_revenue_l365d`, `estimated_occupancy_l365d` | Dérivées business, risque de fuite / hors périmètre prix « par nuit » |
| `availability_*`, `has_availability` | Disponibilité calendrier ≠ prix affiché |
| Colonnes hôte (`host_listings_count`, superhost, etc.) | Conservées après drop initial mais **non** dans `X` : priorité logement + lieu |

---

## Choix techniques

- **`log1p` sur la cible** via `TransformedTargetRegressor` : distribution asymétrique (skew 2,02) ; prédiction en euros via `expm1`
- **`ColumnTransformer`** : numériques + booléens → `SimpleImputer(median)` → `StandardScaler` ; catégorielles → `SimpleImputer(constant="Unknown")` → `OneHotEncoder(handle_unknown="ignore")`
- **`Pipeline`** : preprocessor + régresseur ; même chaîne pour tous les modèles
- **Validation** : `KFold(n_splits=5, shuffle=True, random_state=42)` puis holdout **20 %** (`train_test_split`, `random_state=42`)
- **Modèles testés** : `LinearRegression`, `Ridge(alpha=1)`, `RandomForestRegressor(n_estimators=100)`, `GradientBoostingRegressor(n_estimators=100)`
- **Sélection** : meilleur **R² moyen** en CV → `GradientBoosting` ; confirmation sur MAE holdout le plus bas parmi les modèles à arbres

Les modèles **linéaires** affichent un R² CV catastrophique (ordre −10⁴) : la matrice creuse issue du OneHot + échelles de prix rendent la régression linéaire inadaptée sans tuning ; les modèles à arbres gèrent mieux les interactions quartier × type de logement.

---

## Résultats

Valeurs issues du dernier run du notebook (`metrics.json`).

### Cross-validation 5-fold

| Modèle | R² (mean ± std) | MAE (mean ± std) |
|--------|------------------|------------------|
| LinearRegression | −69445,42 ± 138884,89 | 268,64 € ± 370,07 € |
| Ridge | −68806,89 ± 137607,89 | 267,75 € ± 368,35 € |
| RandomForest | 0,3939 ± 0,0047 | 79,28 € ± 0,92 € |
| **GradientBoosting** | **0,4057 ± 0,0056** | **77,33 € ± 1,18 €** |

### Modèle retenu

- **Final** : `GradientBoosting`
- **Critère** : plus haut R² moyen en CV ; meilleur MAE moyen parmi les modèles exploitables

### Holdout test (20 %)

| Métrique | Valeur |
|----------|--------|
| R² test | 0,4089 |
| MAE test | 77,41 € |
| Erreur absolue médiane | 45,32 € |
| P90 erreur absolue | 167,78 € |
| Sous-estimations (`réel > prédit`) | 45,9 % |
| Sur-estimations | 54,1 % |

### Interprétation

- Le modèle capture la **tendance générale** du marché parisien (MAE ~77 €, médiane d'erreur ~45 €).
- Les **plus grosses erreurs** concernent surtout des logements **premium ou atypiques** (prix réels proches de 1000 €, prédictions ~100–170 €) : `Entire home/apt` en quartiers comme Passy, Observatoire, Hôtel-de-Ville.
- **Quartiers les plus difficiles** (erreur moyenne élevée, `count ≥ 30`) : Louvre, Luxembourg, Élysée, Passy, Hôtel-de-Ville — forte variabilité intra-quartier.
- **Types de logement** : `Hotel room` MAE moyenne ~144 € (peu d'exemples, n=84) ; `Entire home/apt` ~78 € sur la majorité du test set.
- Écart **MAE (77 €) vs médiane (45 €)** : bonnes prédictions sur une large base, mais **queue d'erreurs** lourde sur l'extrême haut de gamme.
- Pistes d'amélioration (hors scope actuel) : NLP sur `description`, géo fine (lat/long + POI), scores d'avis imputés, hyperparamètres / `HistGradientBoosting`.

---

## Export

- `model.joblib` : pipeline complet réentraîné sur les 52 532 observations
- `metrics.json` : traçabilité CV + holdout pour la soutenance et la reproductibilité (`random_state=42`)
