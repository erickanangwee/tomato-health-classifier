"""
Stage: evaluate
────────────────
Evaluates all trained models on the held-out TEST set.
Selects the champion (best val F1) and copies it to models/champion/.
Also computes and saves the tomato centroid vector used by the API guard.

Outputs
───────
models/
  champion/
    best_model.joblib
    scaler.joblib         (copied from processed/)
    champion_info.json
    tomato_centroid.npy   (mean MobileNet embedding of ALL training images)
  evaluation/
    test_metrics.json
    classification_report.txt
"""
import json
import shutil
import argparse
import yaml
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
import mlflow


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def compute_tomato_centroid(raw_dir: Path, params: dict) -> np.ndarray:
    """
    Compute the mean MobileNetV2 embedding of all training images.
    This centroid is used at inference time to detect non-tomato images.
    """
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    img_size = params["data"]["image_size"]
    model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embeddings = []
    img_files = sorted((raw_dir / "images").glob("*.jpg"))
    for fp in tqdm(img_files, desc="Computing tomato centroid"):
        img = Image.open(fp).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = model(x).squeeze().numpy()
        embeddings.append(emb)

    centroid = np.mean(embeddings, axis=0)
    print(f"Centroid computed from {len(embeddings)} images, shape: {centroid.shape}")
    return centroid


def evaluate(
    models_dir: str,
    processed_dir: str,
    raw_dir: str,
    output_dir: str,
    params_path: str = "params.yaml",
):
    p   = load_params(params_path)
    out = Path(output_dir)
    prc = Path(processed_dir)
    mdir = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    champion_dir = mdir / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    X_test = np.load(prc / "X_test.npy")
    y_test = np.load(prc / "y_test.npy")
    print(f"Test set: {X_test.shape[0]} samples")

    # ── Evaluate all models on test set ─────────────────────────────────────
    model_dirs = {
        "LogisticRegression": mdir / "logisticregression" / "best_model.joblib",
        "RandomForest":       mdir / "randomforest"       / "best_model.joblib",
        "XGBoost":            mdir / "xgboost"            / "best_model.joblib",
    }

    test_results = {}
    for name, weight_path in model_dirs.items():
        if not weight_path.exists():
            print(f"  Skipping {name} — weights not found at {weight_path}")
            continue
        model = joblib.load(weight_path)
        preds = model.predict(X_test)
        f1    = f1_score(y_test, preds, average="weighted", zero_division=0)
        test_results[name] = {
            "f1_weighted": float(f1),
            "report": classification_report(y_test, preds,
                                            target_names=["HEALTHY", "UNHEALTHY"],
                                            output_dict=True, zero_division=0),
        }
        print(f"  {name:<25}  Test F1: {f1:.4f}")

    if not test_results:
        raise RuntimeError("No models found to evaluate.")

    # ── Select champion ──────────────────────────────────────────────────────
    champion_name = max(test_results, key=lambda n: test_results[n]["f1_weighted"])
    champion_metrics = test_results[champion_name]
    print(f"\nChampion model: {champion_name}  "
          f"(Test F1={champion_metrics['f1_weighted']:.4f})")

    # ── Copy champion weights + scaler ───────────────────────────────────────
    src_path = model_dirs[champion_name]
    shutil.copy(src_path, champion_dir / "best_model.joblib")
    shutil.copy(prc / "scaler.joblib", champion_dir / "scaler.joblib")

    champion_info = {
        "champion_model": champion_name,
        "test_f1_weighted": champion_metrics["f1_weighted"],
        "all_test_results": {k: v["f1_weighted"] for k, v in test_results.items()},
    }
    with open(champion_dir / "champion_info.json", "w") as f:
        json.dump(champion_info, f, indent=2)

    # ── Log champion to MLflow ────────────────────────────────────────────────
    champ_model = joblib.load(champion_dir / "best_model.joblib")
    with mlflow.start_run(run_name=f"Champion-{champion_name}"):
        mlflow.log_params({"champion_model": champion_name})
        mlflow.log_metric("test_f1_weighted", champion_metrics["f1_weighted"])
        mlflow.sklearn.log_model(
            champ_model, artifact_path="champion_model",
            registered_model_name=p["mlflow"]["model_name"],
        )

    # ── Compute and save tomato centroid ─────────────────────────────────────
    centroid = compute_tomato_centroid(Path(raw_dir), p)
    centroid_path = champion_dir / "tomato_centroid.npy"
    np.save(centroid_path, centroid)
    print(f"Tomato centroid saved to {centroid_path}")

    # ── Save full evaluation report ──────────────────────────────────────────
    with open(out / "test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=2)

    report_txt = ""
    for name, res in test_results.items():
        report_txt += f"\n{'='*50}\n{name}\n"
        report_txt += classification_report(
            y_test,
            joblib.load(model_dirs[name]).predict(X_test),
            target_names=["HEALTHY", "UNHEALTHY"],
            zero_division=0,
        )
    with open(out / "classification_report.txt", "w") as f:
        f.write(report_txt)

    print(f"\nAll evaluation outputs saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir",    default="models")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--raw-dir",       default="data/raw")
    parser.add_argument("--output-dir",    default="models/evaluation")
    parser.add_argument("--params",        default="params.yaml")
    args = parser.parse_args()
    evaluate(args.models_dir, args.processed_dir, args.raw_dir,
             args.output_dir, args.params)