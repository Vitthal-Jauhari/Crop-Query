"""
CropQuery — Vision Module
Loads phaseC_best.pth and returns top-k predictions with confidence.
Applies the ≥85% confidence gate used by the RAG + LLaVA stages.
"""

import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import List, Tuple, Optional

from .config import (
    MODEL_PATH, CLASS_NAMES_PATH,
    NUM_CLASSES, DROPOUT, IMG_SIZE,
    CONFIDENCE_GATE, TOP_K,
)

# ── Pre-processing (ImageNet stats, matching training) ───────────────────────
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _build_model(num_classes: int) -> nn.Module:
    """ResNet50 with the same head used during training: Dropout(0.4) → Linear(2048, n)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(2048, num_classes),
    )
    return model


class CropVisionModel:
    """
    Wraps the trained ResNet50 classifier.

    Usage
    -----
    vm = CropVisionModel()
    preds = vm.predict(pil_image)
    # → [("Tomato__Late_Blight", 92.3), ("Tomato__Early_Blight", 5.1), ...]
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        class_names_path: str = CLASS_NAMES_PATH,
        device: Optional[str] = None,
    ):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Load class names ─────────────────────────────────────────────────
        with open(class_names_path) as f:
            raw = json.load(f)
        # Accept either a list or a {idx: name} / {name: idx} dict
        if isinstance(raw, list):
            self.class_names: List[str] = raw
        elif isinstance(raw, dict):
            # Handles both {name: idx} and {idx: name}
            first_key = next(iter(raw))
            if isinstance(raw[first_key], int):
                # {name: idx}
                self.class_names = [None] * len(raw)
                for name, idx in raw.items():
                    self.class_names[idx] = name
            else:
                # {idx: name}  (idx stored as string)
                self.class_names = [raw[str(i)] for i in range(len(raw))]
        else:
            raise ValueError("class_names.json must be a list or dict")

        num_classes = len(self.class_names)

        # ── Load model ───────────────────────────────────────────────────────
        model = _build_model(num_classes)
        state = torch.load(model_path, map_location=self.device)
        # Support raw state_dict or checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
        # Strip DataParallel 'module.' prefix if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model
        print(f"[Vision] Loaded ResNet50 ({num_classes} classes) on {self.device}")

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = TOP_K,
    ) -> List[Tuple[str, float]]:
        """
        Returns
        -------
        List of (class_name, confidence_pct) sorted by confidence descending.
        e.g. [("Tomato__Late_Blight", 92.3), ("Tomato__Early_Blight", 5.1), ...]
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = _TRANSFORM(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top    = torch.topk(probs, min(top_k, len(self.class_names)))
        return [
            (self.class_names[idx.item()], round(conf.item() * 100, 2))
            for conf, idx in zip(top.values, top.indices)
        ]

    def gate(self, predictions: List[Tuple[str, float]]) -> Optional[str]:
        """
        Confidence gate: return the top class label only if confidence ≥ 85%.
        Used to decide whether to pass a ResNet hint to LLaVA.
        """
        if predictions and predictions[0][1] >= CONFIDENCE_GATE:
            return predictions[0][0]
        return None

    @staticmethod
    def parse_label(label: str) -> Tuple[str, str]:
        """
        'Tomato__Late_Blight' → ('Tomato', 'Late Blight')
        """
        parts = label.split("__", 1)
        crop  = parts[0].replace("_", " ")
        cond  = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        return crop, cond
