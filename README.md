# Airbnb Paris — Prédiction du prix par nuitée

Modèle ML pour prédire le prix optimal d'une nuitée Airbnb à Paris.

## Structure

```
airbnb-paris-price-predictor/
├── data/               # listings.csv ici (gitignored)
├── notebooks/
│   ├── eda.ipynb       # Personne C — exploration
│   └── airbnb.ipynb    # Personne A+B — notebook principal à rendre
├── model.joblib        # généré à la fin (gitignored pendant le dev)
├── metrics.json        # métriques du meilleur modèle
├── README.md
├── decisions.md        # log des choix techniques
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone <url-du-repo>
cd airbnb-paris-price-predictor
uv venv
uv pip install -r requirements.txt
```

## Données

1. Aller sur https://insideairbnb.com/get-the-data/
2. Section **Paris, France** → télécharger `listings.csv.gz` (version détaillée, ~75 colonnes)
3. Décompresser et placer dans `data/listings.csv`

> ⚠️ Prendre la version **détaillée** (pas le résumé à 16 colonnes).

## Branches

```
main          → stable, ce qu'on rend
dev           → branche de travail commune
feat/pipeline → Personne A
feat/models   → Personne B
feat/eda      → Personne C
```

## Workflow

1. Chacun travaille sur sa branche `feat/*`
2. Merge dans `dev` au fur et à mesure
3. Merge `dev` → `main` avant le rendu

## Équipe

- Personne A : pipeline & preprocessing
- Personne B : modélisation & évaluation
- Personne C : EDA & visualisations
