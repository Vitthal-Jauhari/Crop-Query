"""Pin and download the nine public image sources used by CropQuery.

Run in the background; this writes only under /workspace/datasets/cropquery_sources.
Credential values are read in memory and never printed or stored here.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth


ROOT = Path("/workspace/datasets/cropquery_sources")
SECRET_FILE = Path("/workspace/.secrets/api_keys.env")
SLUGS = [
    "zunorain/pea-plant-dataset",
    "binnassor89/melon-macronutrient-deficiency-dataset",
    "tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh",
    "nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset",
    "yasaminborhani/wheatrustclassificationdataset",
    "tushar5harma/plant-village-dataset-updated",
    "ayushbpanchal/cashew-image-dataset",
    "mexwell/crop-diseases-classification",
    "shubham2703/five-crop-diseases-dataset",
]


def credentials() -> tuple[str, str]:
    values = {}
    for line in SECRET_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.removeprefix("export ").split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values["KAGGLE_USERNAME"], values["KAGGLE_KEY"]


def metadata(slug: str, auth: HTTPBasicAuth) -> dict:
    response = requests.get(
        f"https://www.kaggle.com/api/v1/datasets/view/{slug}",
        auth=auth,
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "slug": slug,
        "version": data["currentVersionNumber"],
        "reported_bytes": data.get("totalBytes"),
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    username, key = credentials()
    auth = HTTPBasicAuth(username, key)
    ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(KAGGLE_USERNAME=username, KAGGLE_KEY=key)
    failed = []

    for slug in SLUGS:
        owner, name = slug.split("/")
        destination = ROOT / owner / name
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "source_manifest.json"
        if manifest_path.exists():
            print(f"SKIP {slug}: source manifest exists", flush=True)
            continue
        try:
            before = metadata(slug, auth)
            print(f"START {slug} revision={before['version']}", flush=True)
            subprocess.run(
                ["kaggle", "datasets", "download", slug, "-p", str(destination), "-q"],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            archives = list(destination.glob("*.zip"))
            if len(archives) != 1:
                raise RuntimeError(f"expected one ZIP archive, found {len(archives)}")
            archive = archives[0]
            with zipfile.ZipFile(archive) as bundle:
                members = bundle.infolist()
                for item in members:
                    target = (destination / "images" / item.filename).resolve()
                    if not target.is_relative_to((destination / "images").resolve()):
                        raise RuntimeError("unsafe archive member path")
                (destination / "images").mkdir(exist_ok=True)
                bundle.extractall(destination / "images")
            after = metadata(slug, auth)
            if before["version"] != after["version"]:
                raise RuntimeError("remote version changed during download")
            before.update(
                archive=archive.name,
                archive_sha256=sha256(archive),
                extracted_files=sum(not item.is_dir() for item in members),
                extracted_utc=datetime.now(timezone.utc).isoformat(),
            )
            manifest_path.write_text(json.dumps(before, indent=2) + "\n")
            print(f"DONE {slug} files={before['extracted_files']}", flush=True)
        except Exception as exc:
            failed.append(slug)
            print(f"FAILED {slug}: {type(exc).__name__}: {exc}", flush=True)
    print(f"FINISHED complete={len(SLUGS)-len(failed)} failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
