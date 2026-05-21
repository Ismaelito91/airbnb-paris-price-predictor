# Decisions Log

Historique des choix techniques et justifications pour la soutenance.

---

## Features retenues

| Feature | Type | Justification |
|---------|------|---------------|
| `accommodates` | num | Capacité → impact direct sur le prix |
| `bedrooms` | num | Plus de chambres = plus cher |
| `beds` | num | Proxy de la taille du logement |
| `bathrooms` | num | Indicateur de standing |
| `neighbourhood_cleansed` | cat | Quartier parisien = facteur prix #1 (EDA : top quartiers +30–50 % vs médiane globale) |
| `room_type` | cat | Appart entier vs chambre privée = écart de prix majeur |
| `property_type` | cat | Loft vs studio → pricing différent |
| amenities (7 booléens) | bin | Wifi, TV, ascenseur, etc. → confort perçu |

## Features exclues (et pourquoi)

- `description`, `name` : texte libre → NLP hors scope pour 1.5j
- `host_response_rate` : trop de NaN, pas directement lié au prix
- `latitude` / `longitude` : utile mais nécessite du feature engineering géo plus fin
- `review_scores_*` : beaucoup de NaN pour les nouveaux logements

## Choix techniques

- `log1p` sur la cible : la distribution du prix est très asymétrique à droite
- `TransformedTargetRegressor` : applique `log1p` au fit puis `expm1` à la prédiction
- `Pipeline` + `ColumnTransformer` : même preprocessing pour tous les modèles
- `SimpleImputer` dans le pipeline : évite la fuite de données pendant la CV
- Filtrage des prix : on conserve les annonces entre `10€` et `1000€` pour limiter les erreurs et outliers extrêmes

## Résultats

Valeurs actuellement observées dans le notebook après rerun avec `KFold(shuffle=True, random_state=SEED)`.

| Modèle | R² (mean ± std) | MAE (mean ± std) |
|--------|------------------|------------------|
| LinearRegression | -69445.4245 ± 138884.8931 | 268.64€ ± 370.07€ |
| Ridge | -68806.8943 ± 137607.8948 | 267.75€ ± 368.35€ |
| RandomForest | 0.3940 ± 0.0046 | 79.28€ ± 0.93€ |
| GradientBoosting | 0.4057 ± 0.0056 | 77.33€ ± 1.18€ |

### Modèle retenu

- Modèle final retenu : `GradientBoosting`
- Pourquoi : meilleur compromis global, avec le plus haut `R²` moyen et le plus faible `MAE` moyen en cross-validation

### Évaluation finale sur test set

- `R² test` : `0.4089`
- `MAE test` : `77.41€`
- Erreur absolue médiane : `45.32€`
- 90e percentile des erreurs : `167.78€`

### Interprétation

- Le modèle est cohérent sur le marché parisien standard, mais il sous-estime fortement les annonces premium
- Les plus grosses erreurs concernent surtout des logements très chers ou atypiques
- Les quartiers centraux et haut de gamme comme `Louvre`, `Luxembourg`, `Élysée`, `Passy` ou `Hôtel-de-Ville` sont les plus difficiles
- La différence entre `MAE = 77.41€` et l'erreur médiane `45.32€` montre qu'une partie des annonces est bien prédite, mais qu'il existe une queue d'erreurs très importantes
