import importlib.util
from types import ModuleType

from .settings import DEFAULT_LGBM_PARAMS, N_ESTIMATORS, PARAMS_PATH, RANDOM_STATE


def load_module() -> ModuleType | None:
    if not PARAMS_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("lgbm_params_generated", PARAMS_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_lgbm_params() -> dict[str, object]:
    module = load_module()
    params = getattr(module, "LGBM_PARAMS", None) if module is not None else None
    if not isinstance(params, dict):
        return DEFAULT_LGBM_PARAMS.copy()

    merged = DEFAULT_LGBM_PARAMS.copy()
    merged.update(params)
    merged["random_state"] = RANDOM_STATE
    merged["n_estimators"] = int(merged.get("n_estimators", N_ESTIMATORS))
    return merged


def params_source_name() -> str:
    return PARAMS_PATH.name if PARAMS_PATH.exists() else "built-in defaults"
