import os
from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import load_train_features
from .modeling import best_f1_threshold
from .settings import (
    DEFAULT_LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    N_ESTIMATORS,
    N_SPLITS_TUNING,
    N_TRIALS,
    PARAMS_PATH,
    RANDOM_STATE,
)


def build_params(trial: optuna.Trial) -> dict[str, object]:
    return {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_estimators": trial.suggest_int("n_estimators", 900, 2200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
    }


def make_objective(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS_TUNING,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        params = build_params(trial)
        oof = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
            )
            oof[val_idx] = model.predict_proba(X_val)[:, 1]

        threshold, score = best_f1_threshold(y, oof)
        trial.set_user_attr("threshold", threshold)
        return score

    return objective


def save_params(params: dict[str, object], threshold: float, score: float) -> None:
    lines = [
        f"BEST_F1 = {score:.8f}",
        f"BEST_THRESHOLD = {threshold:.8f}",
        "LGBM_PARAMS = {",
    ]
    for key, value in params.items():
        lines.append(f"    {key!r}: {value!r},")
    lines.append("}")
    PARAMS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tuning() -> None:
    X, y_series, groups = load_train_features(use_cache=True)
    y = y_series.to_numpy(dtype=int)

    study = optuna.create_study(direction="maximize", study_name="cocrystalboost_lgbm_f1")
    study.enqueue_trial(
        {
            "n_estimators": DEFAULT_LGBM_PARAMS.get("n_estimators", N_ESTIMATORS),
            "learning_rate": DEFAULT_LGBM_PARAMS["learning_rate"],
            "num_leaves": DEFAULT_LGBM_PARAMS["num_leaves"],
            "max_depth": 8,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 1e-4,
            "reg_lambda": 1e-4,
            "min_split_gain": 0.0,
        }
    )
    study.optimize(make_objective(X, y, groups), n_trials=N_TRIALS, n_jobs=min(4, os.cpu_count() or 1))

    best_trial = study.best_trial
    best_params = build_params(best_trial)
    best_threshold = float(best_trial.user_attrs["threshold"])
    save_params(best_params, best_threshold, float(best_trial.value))

    print(f"Best F1: {best_trial.value:.5f}")
    print(f"Best threshold: {best_threshold:.4f}")
    print("Best params:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Saved to: {PARAMS_PATH}")
