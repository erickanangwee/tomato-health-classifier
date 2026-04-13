"""
Stage: prepare_data
────────────────────
Extracts image features using MobileNetV2 (or HOG) and splits into
train / val / test sets.

Outputs
───────
data/processed/
  X_train.npy, y_train.npy
  X_val.npy,   y_val.npy
  X_test.npy,  y_test.npy
  scaler.joblib              (StandardScaler fitted on train only)
  split_stats.json
"""
import json
import argparse
import yaml
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── MobileNetV2 feature extractor ──────────────────────────────────────────
def build_mobilenet_extractor(img_size: int):
    """Returns a function: PIL Image → 1280-dim numpy vector."""
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()   # remove final FC → 1280-dim output
    model.eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    def extract(img: Image.Image) -> np.ndarray:
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(x).squeeze().numpy()
        return feat

    return extract


# ── HOG feature extractor (fallback) ───────────────────────────────────────
def build_hog_extractor(img_size: int):
    from skimage.feature import hog

    def extract(img: Image.Image) -> np.ndarray:
        arr = np.array(img.resize((img_size, img_size)))
        feats, _ = hog(arr, orientations=8, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), channel_axis=-1, visualize=True)
        return feats

    return extract


def prepare(raw_dir: str, processed_dir: str, params_path: str = "params.yaml"):
    p = load_params(params_path)
    dp = p["data"]
    fp = p["features"]
    raw = Path(raw_dir)
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(raw / "metadata.json") as f:
        meta = json.load(f)

    samples = meta["samples"]
    fnames = sorted(samples.keys())
    labels = np.array([samples[fn]["label"] for fn in fnames])

    # ── Build extractor ─────────────────────────────────────────────────────
    extractor_type = fp["extractor"]
    print(f"Using '{extractor_type}' feature extractor...")
    if extractor_type == "mobilenet":
        extract = build_mobilenet_extractor(dp["image_size"])
    else:
        extract = build_hog_extractor(dp["image_size"])

    # ── Extract features ────────────────────────────────────────────────────
    features = []
    for fname in tqdm(fnames, desc="Extracting features"):
        img = Image.open(raw / "images" / fname).convert("RGB")
        features.append(extract(img))
    X = np.array(features, dtype=np.float32)

    print(f"Feature matrix shape: {X.shape}")

    # ── Stratified splits ───────────────────────────────────────────────────
    seed = dp["seed"]
    train_r = dp["train_ratio"]
    val_r = dp["val_ratio"]
    test_r = round(1.0 - train_r - val_r, 4)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, labels,
        test_size=round(1.0 - train_r, 4),
        stratify=labels,
        random_state=seed,
    )
    val_frac = val_r / (val_r + test_r)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=round(1.0 - val_frac, 4),
        stratify=y_tmp,
        random_state=seed,
    )

    # ── Fit scaler on train only (no data leakage) ──────────────────────────
    if fp["normalize"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        joblib.dump(scaler, out / "scaler.joblib")
        print("Scaler fitted on training set and saved.")
    else:
        joblib.dump(None, out / "scaler.joblib")

    # ── Save arrays ─────────────────────────────────────────────────────────
    np.save(out / "X_train.npy", X_train)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "X_val.npy", X_val)
    np.save(out / "y_val.npy", y_val)
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_test.npy", y_test)

    stats = {
        "extractor": extractor_type,
        "feature_dim": int(X.shape[1]),
        "total_samples": int(len(X)),
        "train": int(len(X_train)),
        "val": int(len(X_val)),
        "test": int(len(X_test)),
        "class_balance": {
            "train_healthy": int(np.sum(y_train == 0)),
            "train_unhealthy": int(np.sum(y_train == 1)),
        },
    }
    with open(out / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSplit summary:")
    print(f"  Train : {stats['train']}  |  Val : {stats['val']}  |  Test : {stats['test']}")
    print(f"  Feature dim: {stats['feature_dim']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    prepare(args.raw_dir, args.processed_dir, args.params)
