import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torchvision import transforms
from typing import List, Tuple, Optional, Dict
import json

from .config import IMG_SIZE, CLASS_NAMES_PATH


def remap_state_dict(state_dict: dict) -> dict:
    """
    Convert keys from training format (encoder.*, head.*) to inference format (features.*, classifier.*)
    Also removes 'num_batches_tracked' keys which are not needed.
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        if k.startswith("encoder."):
            new_k = "features." + k[8:]
        elif k.startswith("head."):
            new_k = "classifier." + k[5:]
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone with feature extractor and classifier."""
    def __init__(self, n_classes: int = 64):
        super().__init__()
        base = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, n_classes)
        )

    def forward(self, x):
        f = self.features(x).flatten(1)
        return self.classifier(f)

    def extract_features(self, x):
        return self.features(x).flatten(1)


class EfficientNetB3Backbone(nn.Module):
    """EfficientNet-B3 backbone."""
    def __init__(self, n_classes: int = 64):
        super().__init__()
        base = models.efficientnet_b3(weights=None)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, n_classes)
        )

    def forward(self, x):
        f = self.features(x)
        # FIX: Apply Global Average Pooling before flattening
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def extract_features(self, x):
        f = self.features(x)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)


class DenseNet121Backbone(nn.Module):
    """DenseNet-121 backbone."""
    def __init__(self, n_classes: int = 64):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        f = self.features(x)
        # FIX: DenseNet requires a ReLU activation, then GAP before flattening
        f = F.relu(f, inplace=True)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def extract_features(self, x):
        f = self.features(x)
        f = F.relu(f, inplace=True)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)


class FAISSRetriever:
    """CPU-based FAISS k-NN retriever using a pre-built feature bank."""

    def __init__(self, features_path: str, labels_path: str, k: int = 7):
        import faiss
        self.k = k
        features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path)
        # L2-normalise for cosine similarity
        faiss.normalize_L2(features)
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(features)
        print(f"[FAISS] Loaded index: {features.shape[0]} vectors, dim={features.shape[1]}, k={k}")

    def query(self, feature_vec: np.ndarray) -> str:
        """Return majority-vote class from k nearest neighbours."""
        import faiss
        vec = feature_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        _, indices = self.index.search(vec, self.k)
        neighbour_labels = self.labels[indices[0]]
        # Majority vote
        values, counts = np.unique(neighbour_labels, return_counts=True)
        return values[counts.argmax()]


class CropVisionModel:
    """
    Multi-backbone vision model for crop disease classification.
    Loads ResNet50, EfficientNet-B3 and DenseNet-121 from your checkpoints.
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load class names
        with open(CLASS_NAMES_PATH, "r") as f:
            self.class_names = json.load(f)
        self.n_classes = len(self.class_names)

        # Define checkpoint paths - adjust if necessary
        base_ckpt = "/kaggle/input/datasets/vitthaljauhari2/cropquerydataset"
        checkpoints = {
            "resnet50":      f"{base_ckpt}/resnet50_best.pth",
            "efficientnet":  f"{base_ckpt}/efficientnet_best.pth",
            "densenet":      f"{base_ckpt}/densenet_best.pth",
        }

        self.backbones: Dict[str, nn.Module] = {}
        for name, path in checkpoints.items():
            if name == "resnet50":
                model = ResNet50Backbone(self.n_classes)
            elif name == "efficientnet":
                model = EfficientNetB3Backbone(self.n_classes)
            elif name == "densenet":
                model = DenseNet121Backbone(self.n_classes)
            else:
                continue

            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            remapped = remap_state_dict(state_dict)
            # strict=False ignores any remaining mismatches (e.g., missing `num_batches_tracked`)
            model.load_state_dict(remapped, strict=False)
            model.to(self.device)
            model.eval()
            self.backbones[name] = model

        print(f"[Vision] Loaded {len(self.backbones)} backbones (ResNet50, EfficientNet, DenseNet)")

        # Load FAISS feature bank
        from .config import BANK_FEATURES, BANK_LABELS, KNN_K
        try:
            self.faiss = FAISSRetriever(BANK_FEATURES, BANK_LABELS, k=KNN_K)
        except Exception as e:
            print(f"[Vision] FAISS not loaded (non-fatal): {e}")
            self.faiss = None

        # Image transformation (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image: Image.Image, use_ensemble: bool = True) -> List[Tuple[str, float]]:
        """
        Run inference on a single PIL image.
        If use_ensemble=True, averages softmax probabilities from all three backbones.
        Returns list of (class_name, confidence_percent) sorted descending.
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        all_probs = []
        for model in self.backbones.values():
            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)

        if use_ensemble and len(all_probs) > 1:
            avg_probs = torch.stack(all_probs).mean(dim=0)
        else:
            avg_probs = all_probs[0]

        probs_np = avg_probs.cpu().numpy()[0]
        top_indices = probs_np.argsort()[::-1][:5]
        predictions = [(self.class_names[i], probs_np[i] * 100) for i in top_indices]
        self._last_tensor = img_tensor
        return predictions

    def gate(self, predictions: List[Tuple[str, float]], image_tensor: torch.Tensor = None) -> Optional[str]:
        from .config import CONFIDENCE_GATE
        if predictions and predictions[0][1] >= CONFIDENCE_GATE:
            return predictions[0][0]
        if self.faiss is not None and image_tensor is not None:
            # Concatenate features from ALL 3 backbones to match index dim (4608)
            feats = []
            for backbone in self.backbones.values():
                with torch.no_grad():
                    f = backbone.extract_features(image_tensor).cpu().numpy()
                feats.append(f)
            combined = np.concatenate(feats, axis=1)
            return self.faiss.query(combined)
        return None
