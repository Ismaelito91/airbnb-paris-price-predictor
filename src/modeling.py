"""
Pipelines de modélisation et évaluation.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construit le ColumnTransformer pour features numériques et catégorielles."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough",
    )


def get_models() -> dict:
    """Retourne un dictionnaire des modèles à comparer."""
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }


def build_pipeline(preprocessor: ColumnTransformer, model) -> TransformedTargetRegressor:
    """Construit un pipeline complet avec TransformedTargetRegressor (log1p/expm1)."""
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def evaluate_model(pipeline, X, y, cv=5) -> dict:
    """Évalue un modèle en cross-validation 5-fold (R² et MAE)."""
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    return {
        "r2_mean": results["test_r2"].mean(),
        "r2_std": results["test_r2"].std(),
        "mae_mean": -results["test_mae"].mean(),
        "mae_std": results["test_mae"].std(),
    }
