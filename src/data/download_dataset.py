"""
Stage: download_data
────────────────────
Downloads the PlantDoc dataset from Hugging Face, filters to tomato images only,
and assigns binary labels:
  - "Tomato leaf" (healthy)  → label 0  (HEALTHY)
  - All other tomato labels  → label 1  (UNHEALTHY)

Outputs
───────
data/raw/
  images/          *.jpg files
  metadata.json    {filename: label_int, classes, label_map, split_counts}
"""
import json
import argparse
import yaml
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def download(output_dir: str, params_path: str = "params.yaml") -> None:
    p = load_params(params_path)
    dp = p["data"]
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {dp['dataset_name']} from Hugging Face...")
    dataset = load_dataset(dp["dataset_name"], split="train")
    print(f"Full dataset size: {len(dataset)}")

    # ── Collect all tomato-specific label strings ──────────────────────────
    all_labels: set[str] = set(
        lbl
        for row in dataset["objects"]
        for lbl in row["category"]
    )
    tomato_labels = sorted(lbl for lbl in all_labels if dp["tomato_keyword"] in lbl)
    print(f"Tomato label strings found ({len(tomato_labels)}): {tomato_labels}")

    # ── Filter to tomato images only ────────────────────────────────────────
    def has_tomato(example):
        return any(lbl in tomato_labels for lbl in example["objects"]["category"])

    tomato_ds = dataset.filter(has_tomato)
    print(f"Tomato subset: {len(tomato_ds)} images")

    # ── Assign binary label ─────────────────────────────────────────────────
    # An image is HEALTHY (0) only if ALL its tomato annotations are the
    # healthy label.  If ANY annotation is a disease/pest, it is UNHEALTHY (1).
    healthy_label = dp["healthy_label"]          # e.g. "Tomato leaf"
    label_map = {"HEALTHY": 0, "UNHEALTHY": 1}
    reverse_map = {0: "HEALTHY", 1: "UNHEALTHY"}

    metadata: dict[str, dict] = {}
    class_counts = {0: 0, 1: 0}

    for i, example in enumerate(tqdm(tomato_ds, desc="Saving images")):
        img: Image.Image = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        fname = f"img_{i:04d}.jpg"
        img.save(str(images_dir / fname), "JPEG", quality=95)

        # Determine binary label
        cats = example["objects"]["category"]
        tomato_cats = [c for c in cats if c in tomato_labels]
        is_healthy = all(c == healthy_label for c in tomato_cats)
        label_int = 0 if is_healthy else 1
        class_counts[label_int] += 1

        metadata[fname] = {
            "label": label_int,
            "label_name": reverse_map[label_int],
            "original_labels": tomato_cats,
        }

    summary = {
        "total": len(tomato_ds),
        "class_counts": {reverse_map[k]: v for k, v in class_counts.items()},
        "label_map": label_map,
        "tomato_label_strings": tomato_labels,
        "healthy_label_string": healthy_label,
        "samples": metadata,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved {len(tomato_ds)} images to {images_dir}")
    print(f"  HEALTHY  : {class_counts[0]}")
    print(f"  UNHEALTHY: {class_counts[1]}")
    print(f"Metadata written to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument("--params",     default="params.yaml")
    args = parser.parse_args()
    download(args.output_dir, args.params)