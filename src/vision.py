import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from typing import List, Tuple, Optional, Dict
import json

from .config import (
    IMG_SIZE, CLASS_NAMES_PATH,
    RESNET_CKPT, EFFICIENTNET_CKPT, DENSENET_CKPT,
    BANK_FEATURES, BANK_LABELS, KNN_K,
    CONFIDENCE_GATE, FAISS_MIN_SIMILARITY, FAISS_DISABLE_OVERRIDE,
)


def remap_state_dict(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        if k.endswith('num_batches_tracked'):
            continue
        if k.startswith('encoder.'):
            new_k = 'features.' + k[8:]
        elif k.startswith('head.'):
            new_k = 'classifier.' + k[5:]
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd


class ResNet50Backbone(nn.Module):
    def __init__(self, n_classes: int = 64):
        super().__init__()
        base = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def extract_features(self, x):
        return self.features(x).flatten(1)


class EfficientNetB3Backbone(nn.Module):
    def __init__(self, n_classes: int = 64):
        super().__init__()
        base = models.efficientnet_b3(weights=None)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1536, n_classes)
        )

    def forward(self, x):
        f = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)
        return self.classifier(f)

    def extract_features(self, x):
        return F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)


class DenseNet121Backbone(nn.Module):
    def __init__(self, n_classes: int = 64):
        super().__init__()
        base = models.densenet121(weights=None)
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, n_classes)
        )

    def forward(self, x):
        f = F.relu(self.features(x), inplace=True)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        return self.classifier(f)

    def extract_features(self, x):
        f = F.relu(self.features(x), inplace=True)
        return F.adaptive_avg_pool2d(f, 1).flatten(1)


class FAISSRetriever:
    """CPU-based FAISS k-NN retriever using a pre-built feature bank."""

    def __init__(self, features_path: str, labels_path: str, k: int = 7):
        import faiss
        self.k = k
        features = np.load(features_path).astype(np.float32)
        self.labels = np.load(labels_path)
        faiss.normalize_L2(features)
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(features)
        print(f"[FAISS] Loaded index: {features.shape[0]} vectors, "
              f"dim={features.shape[1]}, k={k}")

    def query(self, feature_vec: np.ndarray) -> Tuple[int, float]:
        import faiss
        vec = feature_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(vec)
        distances, indices = self.index.search(vec, self.k)
        neighbour_labels = self.labels[indices[0]]
        values, counts = np.unique(neighbour_labels, return_counts=True)
        best_idx = int(values[counts.argmax()])
        top_similarity = float(distances[0][0])
        return best_idx, top_similarity


class CropVisionModel:
    """
    3-backbone ensemble: ResNet50 + EfficientNet-B3 + DenseNet121.
    Checkpoint paths come from config — not hardcoded here.
    predict() returns (predictions, img_tensor) tuple.
    gate() accepts that tensor directly — no side-channel state.
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        with open(CLASS_NAMES_PATH, 'r') as f:
            self.class_names = json.load(f)
        self.n_classes = len(self.class_names)

        checkpoints = {
            "resnet50":     (ResNet50Backbone,      RESNET_CKPT),
            "efficientnet": (EfficientNetB3Backbone, EFFICIENTNET_CKPT),
            "densenet":     (DenseNet121Backbone,    DENSENET_CKPT),
        }

        self.backbones: Dict[str, nn.Module] = {}
        for name, (cls, path) in checkpoints.items():
            model = cls(self.n_classes)
            ckpt = torch.load(path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(remap_state_dict(state_dict), strict=False)
            model.to(self.device)
            model.eval()
            self.backbones[name] = model
        print(f"[Vision] Loaded {len(self.backbones)} backbones on {self.device}")

        # FAISS — optional, fails gracefully
        try:
            self.faiss = FAISSRetriever(BANK_FEATURES, BANK_LABELS, k=KNN_K)
        except Exception as e:
            print(f"[Vision] FAISS not loaded (non-fatal): {e}")
            self.faiss = None

        # FAISS state from last gate() call — for UI display only
        self._last_used_faiss: bool = False
        self._last_faiss_similarity: Optional[float] = None

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict(
        self, image: Image.Image
    ) -> Tuple[List[Tuple[str, float]], torch.Tensor]:
        """
        Run ensemble inference.
        Returns (top5_predictions, img_tensor).
        top5_predictions: list of (class_name, confidence_pct).
        img_tensor: pass directly to gate() — no side-channel needed.
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        all_probs = []
        for model in self.backbones.values():
            with torch.no_grad():
                probs = torch.softmax(model(img_tensor), dim=1)
            all_probs.append(probs)

        avg_probs = torch.stack(all_probs).mean(dim=0)
        probs_np = avg_probs.cpu().numpy()[0]
        top_indices = probs_np.argsort()[::-1][:5]
        predictions = [(self.class_names[i], probs_np[i] * 100) for i in top_indices]
        return predictions, img_tensor

    def gate(
            self,
            predictions: List[Tuple[str, float]],
            img_tensor: Optional[torch.Tensor] = None,
        ) -> Tuple[Optional[str], bool, Optional[float]]:
            self._last_used_faiss = False
            self._last_faiss_similarity = None
    
            if predictions and predictions[0][1] >= CONFIDENCE_GATE:
                return predictions[0][0], False, None
    
            # FAISS_DISABLE_OVERRIDE=True post-F2: override is a net negative.
            # Still run the query for UX (similarity score display) but
            # never substitute the FAISS label for the ensemble prediction.
            if self.faiss is not None and img_tensor is not None:
                feats = []
                for backbone in self.backbones.values():
                    with torch.no_grad():
                        f = backbone.extract_features(img_tensor).cpu().numpy()
                    feats.append(f)
                combined = np.concatenate(feats, axis=1)
                class_idx, similarity = self.faiss.query(combined)
                self._last_faiss_similarity = similarity
    
                if not FAISS_DISABLE_OVERRIDE and similarity >= FAISS_MIN_SIMILARITY:
                    class_name = self.class_names[class_idx]
                    self._last_used_faiss = True
                    print(f"[Vision] FAISS fallback → {class_name} (sim={similarity:.3f})")
                    return class_name, True, similarity
    
                print(f"[Vision] FAISS queried (UX only) sim={similarity:.3f} — override disabled")
                return predictions[0][0] if predictions else None, False, similarity
    
            return predictions[0][0] if predictions else None, False, None