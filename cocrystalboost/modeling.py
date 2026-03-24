import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .params import load_lgbm_params
from .settings import EARLY_STOPPING_ROUNDS, N_SPLITS_INNER, N_SPLITS_OUTER, RANDOM_STATE


def slice_rows(data: pd.DataFrame | np.ndarray, indices: np.ndarray) -> pd.DataFrame | np.ndarray:
    if isinstance(data, pd.DataFrame):
        return data.iloc[indices]
    return data[indices]


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y_true, y_score >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def fit_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**load_lgbm_params())
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def choose_threshold(
    X_train: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> float:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS_INNER,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    inner_oof = np.zeros(len(y_train), dtype=float)

    for train_idx, val_idx in splitter.split(X_train, y_train, groups=groups):
        X_tr = slice_rows(X_train, train_idx)
        X_val = slice_rows(X_train, val_idx)
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        model = fit_model(X_tr, y_tr, X_val, y_val)
        inner_oof[val_idx] = model.predict_proba(X_val)[:, 1]

    threshold, _ = best_f1_threshold(y_train, inner_oof)
    return threshold


def train_and_predict(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | None]], float]:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS_OUTER,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_proba = np.zeros(len(y), dtype=float)
    test_proba = np.zeros(len(X_test), dtype=float)
    thresholds: list[float] = []
    summaries: list[dict[str, float | int | None]] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_train = slice_rows(X, train_idx)
        X_val = slice_rows(X, val_idx)
        y_train = y[train_idx]
        y_val = y[val_idx]
        train_groups = groups[train_idx]

        threshold = choose_threshold(X_train, y_train, train_groups)
        model = fit_model(X_train, y_train, X_val, y_val)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_f1 = f1_score(y_val, val_proba >= threshold, zero_division=0)

        thresholds.append(threshold)
        oof_proba[val_idx] = val_proba
        test_proba += model.predict_proba(X_test)[:, 1] / N_SPLITS_OUTER
        summaries.append(
            {
                "fold": fold,
                "threshold": round(threshold, 4),
                "f1": round(float(val_f1), 5),
                "best_iteration": getattr(model, "best_iteration_", None),
                "train_size": len(train_idx),
                "val_size": len(val_idx),
            }
        )

    return oof_proba, test_proba, summaries, float(np.median(thresholds))
