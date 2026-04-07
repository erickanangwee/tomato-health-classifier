"""
Singleton loader for the champion model and scaler.
Loaded once at FastAPI startup; reused for every request.
"""
import joblib
from pathlib import Path

_model  = None
_scaler = None


def get_model():
    global _model
    if _model is None:
        path = Path("models/champion/best_model.joblib")
        if not path.exists():
            raise FileNotFoundError(
                f"Champion model not found at {path}. "
                "Run `dvc repro` or set MODEL_DIR env variable."
            )
        _model = joblib.load(path)
        print(f"Champion model loaded from {path}")
    return _model


def get_scaler():
    global _scaler
    if _scaler is None:
        path = Path("models/champion/scaler.joblib")
        if path.exists():
            _scaler = joblib.load(path)
            if _scaler is not None:
                print(f"Scaler loaded from {path}")
    return _scaler