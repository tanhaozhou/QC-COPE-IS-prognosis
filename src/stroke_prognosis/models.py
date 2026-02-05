"""
Model zoo used in the manuscript comparisons.

Notes:
- Keep model hyperparameters here for transparency and reproducibility.
- For clinical deployment, consider calibration and external validation.
"""
from __future__ import annotations

from typing import Dict, Tuple

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier


def get_classifiers(seed: int = 42):
    """Return a list of (name, estimator) matching the notebook's intent."""
    return [
        ("LogisticRegression", LogisticRegression(max_iter=2000, random_state=seed)),
        ("DecisionTree", DecisionTreeClassifier(random_state=seed)),
        ("KNN", KNeighborsClassifier()),
        ("GaussianNB", GaussianNB()),
        ("MLP", MLPClassifier(max_iter=2000, random_state=seed)),
        ("GaussianProcess", GaussianProcessClassifier(random_state=seed)),
        ("QDA", QuadraticDiscriminantAnalysis()),
        ("AdaBoost", AdaBoostClassifier(random_state=seed)),
        ("Bagging", BaggingClassifier(random_state=seed)),
        ("GradientBoosting", GradientBoostingClassifier(random_state=seed)),
        ("ExtraTrees", ExtraTreesClassifier(random_state=seed)),
        ("XGBoost", xgb.XGBClassifier(
            learning_rate=0.01,
            max_delta_step=1,
            sampling_method="uniform",
            random_state=seed,
            eval_metric="logloss",
            use_label_encoder=False,
        )),
        ("CatBoost", cb.CatBoostClassifier(
            random_state=seed,
            verbose=False,
        )),
        ("LightGBM", lgb.LGBMClassifier(
            random_state=seed,
        )),
    ]
