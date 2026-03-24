import pandas as pd
from sklearn.metrics import f1_score

from .features import make_groups, prepare_features
from .modeling import train_and_predict
from .params import params_source_name
from .settings import OUTPUT_PATH, SAMPLE_SUBMISSION_PATH, TEST_PATH, TRAIN_PATH


def main() -> None:
    for path in (TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    if "result" not in train_df.columns:
        raise ValueError("Train data must contain 'result' column.")

    print("Generating features...")
    X = prepare_features(train_df)
    X_test = prepare_features(test_df)
    y = train_df["result"].to_numpy(dtype=int)
    groups = make_groups(train_df)

    print(f"Train shape: {X.shape}, test shape: {X_test.shape}")
    print(f"LGBM params source: {params_source_name()}")

    oof_proba, test_proba, summaries, threshold = train_and_predict(X, y, X_test, groups)

    print("\nFold summary:")
    for summary in summaries:
        print(summary)

    oof_f1 = f1_score(y, oof_proba >= threshold, zero_division=0)
    print(f"\nOOF F1: {oof_f1:.5f}")
    print(f"Threshold: {threshold:.4f}")

    sample_submission["result"] = (test_proba >= threshold).astype(int)
    sample_submission.to_csv(OUTPUT_PATH, index=False)
    print(f"Submission saved to: {OUTPUT_PATH}")
