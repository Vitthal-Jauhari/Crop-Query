"""One-seed ResNet50 screen: common base, CE continuation, F2 continuation.

The test partition is read only by ``evaluate`` after both continuation arms
have been selected on validation data. Historical notebook scores are not
inputs to this experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/workspace/datasets/cropquery_sources")
MANIFEST = PROJECT / "data/split_manifest_v1.csv"
RUN_ROOT = PROJECT / "checkpoints/clean_resnet_v1"
RESULT_ROOT = PROJECT / "results/vision_clean_resnet_v1"
CLASS_NAMES = json.loads((PROJECT / "data/class_names.json").read_text())
SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def weak_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def strong_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CropImages(Dataset):
    def __init__(self, records: list[dict], train: bool):
        self.records = records
        self.train = train
        self.weak = weak_transform() if train else eval_transform()
        self.strong = strong_transform() if train else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(DATA_ROOT / record["relative_path"]) as raw:
            image = raw.convert("RGB")
        weak = self.weak(image)
        strong = self.strong(image) if self.strong else weak
        return weak, strong, int(record["class_index"]), index


class ResNet50(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(2048, len(CLASS_NAMES)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images).flatten(1))


def optimizer_for(model: ResNet50):
    blocks = list(model.encoder.children())
    return torch.optim.AdamW([
        {"params": list(blocks[0].parameters()) + list(blocks[1].parameters()) +
         list(blocks[4].parameters()), "lr": 1e-5},
        {"params": blocks[5].parameters(), "lr": 2e-5},
        {"params": list(blocks[6].parameters()) + list(blocks[7].parameters()), "lr": 5e-5},
        {"params": model.head.parameters(), "lr": 5e-4},
    ], weight_decay=1e-4)


def manifest_hash() -> str:
    digest = hashlib.sha256()
    with MANIFEST.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def records_for(partition: str) -> list[dict]:
    with MANIFEST.open(newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["split"] == partition]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_loader(records: list[dict], epoch: int, batch_size: int, workers: int):
    labels = [int(row["class_index"]) for row in records]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[label] for label in labels], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(records), replacement=True,
        generator=torch.Generator().manual_seed(SEED * 1000 + epoch),
    )
    return DataLoader(
        CropImages(records, train=True), batch_size=batch_size, sampler=sampler,
        num_workers=workers, pin_memory=True,
        generator=torch.Generator().manual_seed(SEED * 2000 + epoch),
    )


def validation_loader(records: list[dict], batch_size: int, workers: int):
    return DataLoader(CropImages(records, train=False), batch_size=batch_size,
                      shuffle=False, num_workers=workers, pin_memory=True)


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["manifest_sha256"] != manifest_hash():
        raise RuntimeError("checkpoint and split manifest differ")
    if payload["class_names"] != CLASS_NAMES:
        raise RuntimeError("checkpoint and class names differ")
    return payload


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, predictions, indices = [], [], []
    for weak, _, target, item_index in loader:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(weak.to(device, non_blocking=True))
        labels.extend(target.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        indices.extend(item_index.tolist())
    return labels, predictions, indices


def checkpoint_payload(stage, epoch, model, optimizer, scheduler, best, history, config):
    return {
        "stage": stage, "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "best_val_top1": best, "history": history, "config": config,
        "manifest_sha256": manifest_hash(), "class_names": CLASS_NAMES,
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    }


def run_training(stage: str, epochs: int, batch_size: int, workers: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this planned run")
    set_seed(SEED)
    torch.set_num_threads(8)
    device = torch.device("cuda:0")
    train = records_for("train")
    val = records_for("val")
    if not train or not val:
        raise RuntimeError("train and validation partitions must be nonempty")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    model = ResNet50(pretrained=stage == "base").to(device)
    if stage != "base":
        common = load_checkpoint(RUN_ROOT / "base_best.pth")
        model.load_state_dict(common["model"])
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    config = {"stage": stage, "epochs": epochs, "batch_size": batch_size,
              "workers": workers, "seed": SEED, "image_size": 224,
              "f2_lambda": 0.10, "initialization": "ImageNet1K_V2" if stage == "base" else "base_best"}
    latest = RUN_ROOT / f"{stage}_latest.pth"
    best_path = RUN_ROOT / f"{stage}_best.pth"
    start, best, history = 0, -1.0, []
    if latest.exists():
        previous = load_checkpoint(latest)
        if previous["config"] != config:
            raise RuntimeError("resume config does not match")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        torch.set_rng_state(previous["torch_rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        np.random.set_state(previous["numpy_rng"])
        random.setstate(previous["python_rng"])
        start, best, history = previous["epoch"] + 1, previous["best_val_top1"], previous["history"]
    print(json.dumps({"stage": stage, "train": len(train), "val": len(val),
                      "start_epoch": start, "config": config}), flush=True)
    val_loader = validation_loader(val, batch_size, workers)

    for epoch in range(start, epochs):
        model.train()
        total_loss, total_items = 0.0, 0
        for step, (weak, strong, target, _) in enumerate(train_loader(train, epoch, batch_size, workers)):
            weak = weak.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(weak)
                ce = criterion(logits, target)
                if stage == "f2":
                    strong_logits = model(strong.to(device, non_blocking=True))
                    kl = F.kl_div(F.log_softmax(logits.float(), dim=1),
                                  F.softmax(strong_logits.detach().float(), dim=1),
                                  reduction="batchmean")
                    loss = ce + 0.10 * kl
                else:
                    loss = ce
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            total_items += len(target)
            if (step + 1) % 250 == 0:
                print(f"{stage} epoch={epoch} step={step+1} train_loss={total_loss/total_items:.4f}", flush=True)
        labels, predictions, _ = predict(model, val_loader, device)
        val_acc = accuracy_score(labels, predictions)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": total_loss / total_items, "val_top1": val_acc})
        improved = val_acc > best
        if improved:
            best = val_acc
        payload = checkpoint_payload(stage, epoch, model, optimizer, scheduler, best, history, config)
        save_atomic(payload, latest)
        if improved:
            save_atomic(payload, best_path)
        print(f"{stage} epoch={epoch} val_top1={val_acc:.6f} best={best:.6f}", flush=True)


def smoke(batch_size: int, workers: int) -> None:
    set_seed(SEED)
    train = records_for("train")
    val = records_for("val")
    if not train or not val:
        raise RuntimeError("manifest not ready")
    device = torch.device("cuda:0")
    model = ResNet50(pretrained=True).to(device)
    weak, strong, labels, _ = next(iter(train_loader(train, 0, min(batch_size, 8), workers)))
    weak, strong, labels = weak.to(device), strong.to(device), labels.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(weak)
        strong_logits = model(strong)
        ce = F.cross_entropy(logits, labels, label_smoothing=0.1)
        kl = F.kl_div(F.log_softmax(logits.float(), dim=1),
                      F.softmax(strong_logits.detach().float(), dim=1), reduction="batchmean")
        loss = ce + 0.10 * kl
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite smoke loss")
    loss.backward()
    validation = next(iter(validation_loader(val, min(batch_size, 8), workers)))
    with torch.no_grad():
        output = model(validation[0].to(device))
    print(json.dumps({"smoke": "passed", "train_batch": len(labels),
                      "output_shape": list(output.shape), "loss": float(loss.detach())}), flush=True)


def evaluate(batch_size: int, workers: int) -> None:
    test = records_for("test")
    if not test:
        raise RuntimeError("test partition empty")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    loader = validation_loader(test, batch_size, workers)
    device = torch.device("cuda:0")
    predictions = {}
    selected_epochs = {}
    truth = None
    indices = None
    for stage in ("base", "ce", "f2"):
        payload = load_checkpoint(RUN_ROOT / f"{stage}_best.pth")
        selected_epochs[stage] = payload["epoch"]
        model = ResNet50(pretrained=False).to(device)
        model.load_state_dict(payload["model"])
        labels, predicted, item_indices = predict(model, loader, device)
        if truth is None:
            truth, indices = labels, item_indices
        elif labels != truth or item_indices != indices:
            raise RuntimeError("paired test order changed")
        predictions[stage] = predicted
        del model
        torch.cuda.empty_cache()
    with (RESULT_ROOT / "paired_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "group_id", "class", "truth", "base", "ce", "f2"])
        for position, item_index in enumerate(indices):
            row = test[item_index]
            writer.writerow([row["id"], row["group_id"], row["class"], truth[position],
                             predictions["base"][position], predictions["ce"][position],
                             predictions["f2"][position]])
    base_right = np.array(predictions["base"]) == np.array(truth)
    ce_right = np.array(predictions["ce"]) == np.array(truth)
    f2_right = np.array(predictions["f2"]) == np.array(truth)
    truth_array = np.array(truth)
    group_names = [test[item_index]["group_id"] for item_index in indices]
    group_lookup = {name: position for position, name in enumerate(sorted(set(group_names)))}
    group_index = np.array([group_lookup[name] for name in group_names])
    group_size = np.bincount(group_index, minlength=len(group_lookup))
    group_delta = np.bincount(
        group_index, weights=f2_right.astype(int) - ce_right.astype(int),
        minlength=len(group_lookup),
    )
    rng = np.random.default_rng(SEED)
    boot_differences = []
    for _ in range(2000):
        sampled = rng.integers(0, len(group_lookup), len(group_lookup))
        boot_differences.append(100.0 * group_delta[sampled].sum() / group_size[sampled].sum())
    f1_base = f1_score(truth, predictions["base"], labels=list(range(len(CLASS_NAMES))),
                     average=None, zero_division=0)
    f1_ce = f1_score(truth, predictions["ce"], labels=list(range(len(CLASS_NAMES))),
                     average=None, zero_division=0)
    f1_f2 = f1_score(truth, predictions["f2"], labels=list(range(len(CLASS_NAMES))),
                     average=None, zero_division=0)
    class_results = {
        name: {
            "support": int((truth_array == class_index).sum()),
            "base_recall": float(base_right[truth_array == class_index].mean())
                if (truth_array == class_index).any() else None,
            "ce_recall": float(ce_right[truth_array == class_index].mean())
                if (truth_array == class_index).any() else None,
            "f2_recall": float(f2_right[truth_array == class_index].mean())
                if (truth_array == class_index).any() else None,
            "base_f1": float(f1_base[class_index]),
            "ce_f1": float(f1_ce[class_index]),
            "f2_f1": float(f1_f2[class_index]),
        }
        for class_index, name in enumerate(CLASS_NAMES)
    }
    summary = {
        "manifest_sha256": manifest_hash(), "seed": SEED,
        "selected_epochs_zero_based": selected_epochs,
        "n_test": len(truth),
        "base_top1": float(base_right.mean()),
        "base_macro_f1": float(f1_score(truth, predictions["base"], average="macro", zero_division=0)),
        "ce_top1": float(ce_right.mean()), "f2_top1": float(f2_right.mean()),
        "difference_pp": float((f2_right.mean() - ce_right.mean()) * 100),
        "ce_macro_f1": float(f1_score(truth, predictions["ce"], average="macro", zero_division=0)),
        "f2_macro_f1": float(f1_score(truth, predictions["f2"], average="macro", zero_division=0)),
        "ce_wrong_f2_right": int((~ce_right & f2_right).sum()),
        "ce_right_f2_wrong": int((ce_right & ~f2_right).sum()),
        "test_groups": len(group_lookup),
        "group_bootstrap_95pct_ci_pp": [float(x) for x in
            np.percentile(boot_differences, [2.5, 97.5])],
        "bootstrap_scope": "Resampled test groups only; excludes training-seed and source-domain uncertainty.",
        "base_scope": "Descriptive common-start reference added before test opening after CE validation fell; primary paired contrast remains CE versus F2.",
        "per_class": class_results,
        "scope": "One-seed ResNet50 mechanism screen on prospective group-disjoint split; not a three-backbone ensemble result.",
    }
    (RESULT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "base", "ce", "f2", "evaluate"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.stage == "smoke":
        smoke(arguments.batch_size, arguments.workers)
    elif arguments.stage == "evaluate":
        evaluate(arguments.batch_size, arguments.workers)
    else:
        run_training(arguments.stage, arguments.epochs, arguments.batch_size, arguments.workers)


if __name__ == "__main__":
    main()
