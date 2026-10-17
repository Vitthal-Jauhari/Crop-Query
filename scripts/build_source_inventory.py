"""Record pinned public inputs and local checkpoint identities without loading weights."""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = Path('/workspace/datasets/cropquery_sources')
OUTPUT = PROJECT / 'results/source_inventory_v1.json'

def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()

def main() -> None:
    manifests = sorted(SOURCES.glob('*/*/source_manifest.json'))
    if len(manifests) != 9:
        raise RuntimeError(f'expected nine source manifests; got {len(manifests)}')
    images = [json.loads(path.read_text()) for path in manifests]
    checkpoints = [dict(name=path.name, size_bytes=path.stat().st_size, sha256=digest(path),
                        provenance='Local earlier-phase file; not a verified final Phase D/F2 ensemble member.')
                   for path in sorted((PROJECT / 'models').glob('*.pth'))]
    inventory = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'image_sources': images,
        'image_source_scope': 'Student-supplied public Kaggle slugs, current pinned download versions; historical training versions not proven.',
        'kisan_audit': json.loads((PROJECT / 'results/kisan_count_audit.json').read_text()),
        'class_names_sha256': digest(PROJECT / 'data/class_names.json'),
        'local_checkpoints': checkpoints,
        'missing_original_artifacts': ['final three-backbone Phase D weights', 'final three-backbone F2 weights', 'checkpoint-specific historical training manifest', 'FAISS image-membership manifest'],
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'sources':len(images),'checkpoints':len(checkpoints),'output':str(OUTPUT)}))

if __name__ == '__main__':
    main()
