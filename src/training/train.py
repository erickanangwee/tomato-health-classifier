"""
Stage: train_models
────────────────────
Trains Logistic Regression, Random Forest, and XGBoost on the processed
features. Each model is tuned with Optuna (n_trials per params.yaml).
All trials and the best run per model are logged to MLflow.

Outputs
───────
models/
  logistic_regression/   best_model.joblib  +  best_params.json
  random_forest/         best_model.joblib  +  best_params.json
  xgboost/               best_model.joblib  +  best_params.json
  training_summary.json
"""
import json
import argparse
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import optuna
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Suppress Optuna's per-trial output — MLflow handles logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def eval_metrics(model, X, y) -> dict:
    preds = model.predict(X)
    return {
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y, preds, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
        "f1_binary":   float(f1_score(y, preds, average="binary", zero_division=0)),
    }


# ── Objective factories ─────────────────────────────────────────────────────

def make_lr_objective(X_train, y_train, p, cv):
    lp = p["logistic_regression"]

    def objective(trial: optuna.Trial) -> float:
        C        = trial.suggest_float("C", lp["C_low"], lp["C_high"], log=True)
        solver   = trial.suggest_categorical("solver", lp["solver"])
        model = LogisticRegression(
            C=C, solver=solver, max_iter=lp["max_iter"],
            class_weight="balanced", random_state=p["data"]["seed"]
        )
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=p["optuna"]["metric"])
        return float(scores.mean())

    return objective


def make_rf_objective(X_train, y_train, p, cv):
    rp = p["random_forest"]

    def objective(trial: optuna.Trial) -> float:
        model = RandomForestClassifier(
            n_estimators  = trial.suggest_int("n_estimators",
                                               rp["n_estimators_low"],
                                               rp["n_estimators_high"]),
            max_depth      = trial.suggest_int("max_depth",
                                               rp["max_depth_low"],
                                               rp["max_depth_high"]),
            min_samples_split = trial.suggest_int("min_samples_split",
                                               rp["min_samples_split_low"],
                                               rp["min_samples_split_high"]),
            class_weight="balanced",
            random_state=p["data"]["seed"],
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=p["optuna"]["metric"])
        return float(scores.mean())

    return objective


def make_xgb_objective(X_train, y_train, p, cv):
    xp = p["xgboost"]

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(
            n_estimators   = trial.suggest_int("n_estimators",
                                               xp["n_estimators_low"],
                                               xp["n_estimators_high"]),
            max_depth      = trial.suggest_int("max_depth",
                                               xp["max_depth_low"],
                                               xp["max_depth_high"]),
            learning_rate  = trial.suggest_float("learning_rate",
                                               xp["learning_rate_low"],
                                               xp["learning_rate_high"], log=True),
            subsample      = trial.suggest_float("subsample",
                                               xp["subsample_low"],
                                               xp["subsample_high"]),
            eval_metric    = "logloss",
            use_label_encoder=False,
            random_state   = p["data"]["seed"],
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=p["optuna"]["metric"])
        return float(scores.mean())

    return objective


# ── Main tuning + logging loop ──────────────────────────────────────────────

def tune_and_log(
    model_name: str,
    build_fn,          # callable: best_params → fitted sklearn model
    objective_fn,
    X_train, y_train,
    X_val,   y_val,
    p: dict,
    output_dir: Path,
) -> dict:
    """Run Optuna study, log best trial to MLflow, save model artifact."""
    op    = p["optuna"]
    seed  = p["data"]["seed"]

    print(f"\n{'─'*60}")
    print(f"Tuning {model_name}  ({op['n_trials']} Optuna trials)...")

    study = optuna.create_study(
        direction=op["direction"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective_fn, n_trials=op["n_trials"],
                   timeout=op["timeout"], show_progress_bar=True)

    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"  Best CV {op['metric']}: {best_cv_score:.4f}")
    print(f"  Best params: {best_params}")

    # ── Refit on full training set with best params ─────────────────────────
    model = build_fn(best_params)
    model.fit(X_train, y_train)

    train_metrics = eval_metrics(model, X_train, y_train)
    val_metrics   = eval_metrics(model, X_val,   y_val)

    # ── Log to MLflow ───────────────────────────────────────────────────────
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_f1_weighted",   best_cv_score)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        mlflow.log_metric("n_optuna_trials", op["n_trials"])
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            registered_model_name=f"{p['mlflow']['model_name']}-{model_name}",
        )

    # ── Save locally for DVC tracking ──────────────────────────────────────
    model_dir = output_dir / model_name.lower().replace(" ", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.joblib")
    with open(model_dir / "best_params.json", "w") as f:
        json.dump({"params": best_params, "cv_score": best_cv_score,
                   "val_metrics": val_metrics}, f, indent=2)

    print(f"  Val accuracy : {val_metrics['accuracy']:.4f}")
    print(f"  Val F1       : {val_metrics['f1_weighted']:.4f}")

    return {"model_name": model_name, "val_metrics": val_metrics,
            "best_params": best_params, "cv_score": best_cv_score}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir",    default="models")
    parser.add_argument("--params",        default="params.yaml")
    args = parser.parse_args()

    p   = load_params(args.params)
    op  = p["optuna"]
    out = Path(args.output_dir)
    prc = Path(args.processed_dir)

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    # ── Load processed splits ───────────────────────────────────────────────
    X_train = np.load(prc / "X_train.npy")
    y_train = np.load(prc / "y_train.npy")
    X_val   = np.load(prc / "X_val.npy")
    y_val   = np.load(prc / "y_val.npy")
    print(f"Loaded  Train:{X_train.shape}  Val:{X_val.shape}")

    cv = StratifiedKFold(n_splits=op["cv_folds"], shuffle=True,
                         random_state=p["data"]["seed"])
    seed = p["data"]["seed"]

    results = []

    # ── 1. Logistic Regression ──────────────────────────────────────────────
    def build_lr(bp):
        return LogisticRegression(
            C=bp["C"], solver=bp["solver"],
            max_iter=p["logistic_regression"]["max_iter"],
            class_weight="balanced", random_state=seed
        )
    results.append(tune_and_log(
        "LogisticRegression",
        build_lr,
        make_lr_objective(X_train, y_train, p, cv),
        X_train, y_train, X_val, y_val, p, out,
    ))

    # ── 2. Random Forest ────────────────────────────────────────────────────
    def build_rf(bp):
        return RandomForestClassifier(
            n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
            min_samples_split=bp["min_samples_split"],
            class_weight="balanced", random_state=seed, n_jobs=-1
        )
    results.append(tune_and_log(
        "RandomForest",
        build_rf,
        make_rf_objective(X_train, y_train, p, cv),
        X_train, y_train, X_val, y_val, p, out,
    ))

    # ── 3. XGBoost ──────────────────────────────────────────────────────────
    def build_xgb(bp):
        return XGBClassifier(
            n_estimators=bp["n_estimators"], max_depth=bp["max_depth"],
            learning_rate=bp["learning_rate"], subsample=bp["subsample"],
            eval_metric="logloss", use_label_encoder=False,
            random_state=seed, n_jobs=-1
        )
    results.append(tune_and_log(
        "XGBoost",
        build_xgb,
        make_xgb_objective(X_train, y_train, p, cv),
        X_train, y_train, X_val, y_val, p, out,
    ))

    # ── Save summary ─────────────────────────────────────────────────────────
    with open(out / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'Model':<25} {'Val F1':>10} {'Val Acc':>10}")
    print("─" * 50)
    for r in results:
        print(f"{r['model_name']:<25} "
              f"{r['val_metrics']['f1_weighted']:>10.4f} "
              f"{r['val_metrics']['accuracy']:>10.4f}")


if __name__ == "__main__":
    main()