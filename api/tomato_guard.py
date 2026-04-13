"""
Tomato Guard — rejects images that are not tomato leaves.

Strategy: compute the cosine similarity between the incoming image's
MobileNetV2 embedding and the centroid of all training images.
If similarity < threshold (from params.yaml), the image is rejected.
"""
import numpy as np
import yaml
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from pathlib import Path
from PIL import Image


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class TomatoGuard:
    def __init__(self, params_path: str = "params.yaml"):
        p = load_params(params_path)
        self.threshold = p["tomato_guard"]["similarity_threshold"]
        centroid_path = p["tomato_guard"]["centroid_path"]

        if not Path(centroid_path).exists():
            raise FileNotFoundError(
                f"Tomato centroid not found at '{centroid_path}'. "
                "Run the training pipeline first."
            )
        self.centroid = np.load(centroid_path).astype(np.float32)

        # Build MobileNetV2 embedder
        img_size = p["data"]["image_size"]
        model = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier = torch.nn.Identity()
        model.eval()
        self._model = model

        self._transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def embed(self, image: Image.Image) -> np.ndarray:
        x = self._transform(image).unsqueeze(0)
        with torch.no_grad():
            emb = self._model(x).squeeze().numpy()
        return emb.astype(np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def is_tomato(self, image: Image.Image) -> tuple[bool, float]:
        """
        Returns (is_tomato: bool, similarity_score: float).
        """
        emb = self.embed(image)
        sim = self.cosine_similarity(emb, self.centroid)
        return sim >= self.threshold, sim
