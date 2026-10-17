"""Render the prospective CE-versus-F2 source comparison from frozen results.

Input: results/vision_clean_resnet_v1/analysis.json. The overall interval is a
test-group bootstrap; source rows are descriptive and have no intervals.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / "results/vision_clean_resnet_v1/analysis.json"
OUTPUT = PROJECT / "papers/sisimpact_2027/figures/source_deltas.pdf"
OUTPUT_PNG = PROJECT / "papers/sisimpact_2027/figures/source_deltas.png"

SHORT_NAMES = {
    "ayushbpanchal/cashew-image-dataset": "Cashew",
    "binnassor89/melon-macronutrient-deficiency-dataset": "Melon",
    "mexwell/crop-diseases-classification": "Crop diseases",
    "nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset": "Coffee",
    "shubham2703/five-crop-diseases-dataset": "Five crops",
    "tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh": "Pumpkin",
    "tushar5harma/plant-village-dataset-updated": "PlantVillage",
    "yasaminborhani/wheatrustclassificationdataset": "Wheat rust",
    "zunorain/pea-plant-dataset": "Pea",
}


def main() -> None:
    data = json.loads(INPUT.read_text())
    sources = [(SHORT_NAMES[key], value) for key, value in data["source_results"].items()]
    sources.sort(key=lambda pair: pair[1]["f2_minus_ce_pp"])

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig, ax = plt.subplots(figsize=(3.45, 2.45), layout="constrained")
        positions = list(range(len(sources)))
        values = [item[1]["f2_minus_ce_pp"] for item in sources]
        ax.barh(positions, values, height=0.62, color="#28435B", edgecolor="black", linewidth=0.3)
        ax.set_yticks(positions, [f"{name} (n={value['n']:,})" for name, value in sources])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlim(-1.55, 0.18)
        ax.set_xlabel("F2 minus CE top-1 accuracy (pp)")
        ax.grid(axis="x", color="#D8DEE4", linewidth=0.5)
        ax.set_axisbelow(True)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT)
        fig.savefig(OUTPUT_PNG, dpi=600)
        plt.close(fig)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
