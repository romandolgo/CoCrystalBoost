from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train_dataset" / "train_extended.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
OUTPUT_PATH = BASE_DIR / "submission.csv"
PARAMS_PATH = BASE_DIR / "lgbm_params_generated.py"
FEATURE_CACHE_PATH = BASE_DIR / "train_features_cache.pkl"

RANDOM_STATE: int = 42
N_SPLITS_OUTER: int = 5
N_SPLITS_INNER: int = 3
N_SPLITS_TUNING: int = 5
N_ESTIMATORS: int = 1500
EARLY_STOPPING_ROUNDS: int = 100
FP_SIZE: int = 2048
N_TRIALS: int = 40

DEFAULT_LGBM_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_estimators": N_ESTIMATORS,
    "verbosity": -1,
}
