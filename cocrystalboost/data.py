import numpy as np
import pandas as pd

from .features import make_groups, prepare_features
from .settings import FEATURE_CACHE_PATH, TRAIN_PATH


def load_train_features(use_cache: bool = False) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    train_df = pd.read_csv(TRAIN_PATH)
    if use_cache and FEATURE_CACHE_PATH.exists():
        X = pd.read_pickle(FEATURE_CACHE_PATH)
    else:
        X = prepare_features(train_df)
        if use_cache:
            X.to_pickle(FEATURE_CACHE_PATH)
    y = train_df["result"].astype(int)
    groups = make_groups(train_df)
    return X, y, groups
