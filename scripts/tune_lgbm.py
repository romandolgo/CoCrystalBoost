from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cocrystalboost.tuning import run_tuning


if __name__ == "__main__":
    run_tuning()
