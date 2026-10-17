"""Parse, independently recalculate, and label the archived notebook text probe.

This does not rerun retrieval or generation; it audits a stored cell output.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks/cropquery-v2.ipynb'
OUTPUT = ROOT / 'results/archived_text_probe_audit.json'
ROW = re.compile(r'^(.+?)\s+(0\.\d{3}|1\.000)\s+(0\.\d{3}|1\.000)\s+([+-]\d\.\d{3})', re.M)


def main() -> None:
    notebook = json.loads(SOURCE.read_text())
    outputs = notebook['cells'][30].get('outputs', [])
    text = '\n'.join(''.join(o.get('text', []) or o.get('data', {}).get('text/plain', [])) for o in outputs)
    rows = []
    for match in ROW.finditer(text):
        name = match.group(1).strip()
        if name == 'MEAN':
            continue
        rows.append({'scenario': name, 'retrieved_term_recall': float(match.group(2)), 'generated_term_recall': float(match.group(3))})
    if len(rows) != 19:
        raise RuntimeError(f'expected 19 logged scenarios, found {len(rows)}')
    rag = sum(row['retrieved_term_recall'] for row in rows) / len(rows)
    generated = sum(row['generated_term_recall'] for row in rows) / len(rows)
    audit = {'source': str(SOURCE.relative_to(ROOT)), 'cell_index': 30, 'n_scenarios': len(rows),
             'retrieved_mean_recomputed': rag, 'generated_mean_recomputed': generated,
             'retrieved_minus_generated': rag-generated,
             'complete_losses_after_generation': sum(row['retrieved_term_recall'] > 0 and row['generated_term_recall'] == 0 for row in rows),
             'partial_losses_after_generation': sum(0 < row['generated_term_recall'] < row['retrieved_term_recall'] for row in rows),
             'design': 'Stored notebook output; blank image, diagnosis supplied, keywords derived from indexed entry; no independent agronomic reference.',
             'rows': rows}
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k:v for k,v in audit.items() if k != 'rows'}))

if __name__ == '__main__':
    main()
