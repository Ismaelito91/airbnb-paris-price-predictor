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
| amenities (7 booléens) | bin | Wifi, TV, Ascenseur, etc. → confort perçu |

## Features exclues (et pourquoi)

- `description`, `name` : texte libre → NLP hors scope pour 1.5j
- `host_response_rate` : trop de NaN, pas directement lié au prix
- `latitude`/`longitude` : utile mais nécessite feature engineering geo (distance monuments), pas le temps
- `review_scores_*` : beaucoup de NaN pour les nouveaux logements (notre use case)

## Choix techniques

- **log1p sur la cible** : distribution skewed à droite → log normalise, le modèle apprend mieux les ratios
- **TransformedTargetRegressor** : applique log1p avant fit, expm1 après predict → transparent
- **Filtrage prix** : exclu price=0 (erreurs) et price>10000€ (villas de luxe non représentatives)

## Résultats

*(À remplir après exécution)*

| Modèle | R² (mean ± std) | MAE (mean ± std) |
|--------|-----------------|-------------------|
| LinearRegression | | |
| Ridge | | |
| RandomForest | | |
| GradientBoosting | | |
