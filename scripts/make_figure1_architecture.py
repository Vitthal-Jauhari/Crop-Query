"""Draw Figure 1 (CropQuery overview, modules A-C) as SVG and convert Figures 1, 1a-1c to PDF.

Layout, colours and wording follow papers/sisimpact_2027/figure_1_prompt.json and the
authors' uploaded drawing (commit e16287c). Every label is checked against its box with
Arial-compatible metrics (Liberation Sans); the script stops if any label does not fit.
Printed at 7.16 in wide, 46 px box text is about 6.1 pt and 56 px titles about 7.4 pt.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import ImageFont

FIG = Path(__file__).resolve().parents[1] / "papers/sisimpact_2027/figures"
FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
FONTS = {"normal": "LiberationSans-Regular.ttf", "bold": "LiberationSans-Bold.ttf",
         "italic": "LiberationSans-Italic.ttf"}
W, H = 3910, 1470
BOX, TITLE, SMALL = 46, 56, 44
FAMILY = "Arial,Helvetica,sans-serif"
COL = {
    "in": ("#F5F5F5", "#666666"), "A": ("#D5E8D4", "#82B366"), "B": ("#E1D5E7", "#9673A6"),
    "C": ("#F8CECC", "#B85450"), "ctx": ("#FFE6CC", "#D79B00"), "out": ("#DAE8FC", "#6C8EBF"),
}

out: list[str] = []


def text_width(text: str, size: float, style: str = "normal") -> float:
    font = ImageFont.truetype(str(FONT_DIR / FONTS[style]), int(size * 10))
    return font.getlength(text) / 10


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=BOX, style="normal", anchor="middle"):
    weight = ' font-weight="bold"' if style == "bold" else ""
    italic = ' font-style="italic"' if style == "italic" else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="#000000" font-family="{FAMILY}" '
               f'font-size="{size}"{weight}{italic} text-anchor="{anchor}">{esc(s)}</text>')


def box(x, y, w, h, colour, lines=(), size=BOX, dashed=False, frame=False, rx=16):
    fill, stroke = COL[colour]
    if frame:
        fill = "#FFFFFF"
    dash = ' stroke-dasharray="13 9"' if dashed else ""
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{4 if frame else 3}"{dash}/>')
    if lines:
        lh = 1.13 * size
        block = (len(lines) - 1) * lh + 0.95 * size
        assert block <= h - 16, f"text too tall for box: {lines}"
        first = y + h / 2 - (len(lines) - 1) * lh / 2 + 0.275 * size
        for i, line in enumerate(lines):
            assert text_width(line, size) <= w - 28, f"text too wide for box: {line!r}"
            text(x + w / 2, first + i * lh, line, size)


def arrow(points, dashed=False):
    d = f"M {points[0][0]} {points[0][1]}"
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        d += f" H {x1}" if y0 == y1 else f" V {y1}"
        assert x0 == x1 or y0 == y1, "arrows use only horizontal and vertical segments"
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    out.append(f'<path d="{d}" fill="none" stroke="#333333" stroke-width="3.3" '
               f'stroke-linejoin="round" stroke-linecap="round"{dash} marker-end="url(#tip)"/>')


def line(points, dashed=False):
    d = f"M {points[0][0]} {points[0][1]}" + "".join(
        f" H {x1}" if y0 == y1 else f" V {y1}" for (x0, y0), (x1, y1) in zip(points, points[1:]))
    dash = ' stroke-dasharray="12 9"' if dashed else ""
    out.append(f'<path d="{d}" fill="none" stroke="#333333" stroke-width="3.3" '
               f'stroke-linejoin="round" stroke-linecap="round"{dash}/>')


def build() -> str:
    out.clear()
    # module frames and titles
    box(490, 300, 890, 785, "A", frame=True, rx=20)
    box(1470, 300, 1020, 785, "B", frame=True, rx=20)
    box(2580, 300, 680, 785, "C", frame=True, rx=20)
    text(520, 372, "A. Image Diagnosis", TITLE, "bold", "start")
    text(1500, 372, "B. Knowledge Retrieval", TITLE, "bold", "start")
    text(2605, 372, "C. Advice Generation", TITLE, "bold", "start")
    assert 2605 + text_width("C. Advice Generation", TITLE, "bold") < 3200 - 15, "title meets LLaVA input line"

    # inputs
    box(40, 455, 380, 230, "in")
    out.append('<path d="M 194 492 C 247 475, 277 489, 272 525 C 265 551, 230 563, 198 537 '
               'C 185 527, 186 504, 194 492 Z" fill="#82B366" stroke="#517C42" stroke-width="3"/>')
    out.append('<path d="M 190 546 L 245 505" fill="none" stroke="#517C42" stroke-width="3" '
               'stroke-linecap="round"/>')
    out.append('<circle cx="226" cy="509" r="7" fill="#956842"/>'
               '<circle cx="252" cy="531" r="8" fill="#956842"/>')
    text(230, 645, "Leaf photo")
    box(40, 790, 380, 210, "in", ["Farmer question", "(any supported", "language)"])

    # module A
    box(515, 440, 350, 110, "A", ["ResNet50"])
    box(515, 630, 350, 110, "A", ["EfficientNet-B3"])
    box(515, 820, 350, 110, "A", ["DenseNet121"])
    box(915, 590, 440, 150, "A", ["Average of softmax", "(64 classes)"])
    box(915, 850, 440, 130, "A", ["Label ŷ", "+ confidence"])
    line([(420, 570), (460, 570)])
    line([(460, 495), (460, 875)])
    for y in (495, 685, 875):
        arrow([(460, y), (515, y)])
    arrow([(865, 495), (890, 495), (890, 625), (915, 625)])
    arrow([(865, 685), (915, 685)])
    arrow([(865, 875), (890, 875), (890, 705), (915, 705)])
    arrow([(1135, 740), (1135, 850)])
    box(560, 1170, 780, 180, "A", ["FAISS similar images (k = 7)",
                                   "shown only when confidence < 80%",
                                   "label is NOT changed"], dashed=True)
    arrow([(1135, 980), (1135, 1170)], dashed=True)

    # module B
    box(1600, 440, 760, 150, "B", ["Crop + condition from ŷ", "question in English"])
    box(1495, 730, 430, 170, "B", ["Local treatment list", "39 entries", "up to 3 passages"])
    box(1955, 730, 510, 170, "B", ["Kisan Call Centre Q&A", "178,815 records", "2 passages"])
    box(1755, 945, 450, 115, "B", ["Retrieved passages", "(up to 5)"])
    arrow([(1710, 590), (1710, 730)])
    arrow([(2210, 590), (2210, 730)])
    text(1960, 642, "no crop match:", SMALL, "italic")
    text(1960, 692, "search all entries", SMALL, "italic")
    arrow([(1710, 900), (1710, 922), (1880, 922), (1880, 945)])
    arrow([(2210, 900), (2210, 922), (2080, 922), (2080, 945)])

    # label and question into B; healthy short-cut to output
    arrow([(1355, 915), (1405, 915), (1405, 480), (1600, 480)])
    text(1530, 466, "ŷ", SMALL)
    out.append('<circle cx="1405" cy="480" r="8" fill="#333333"/>')
    arrow([(1405, 480), (1405, 215), (3600, 215), (3600, 430)], dashed=True)
    text(2150, 195, "healthy and ≥ 80%: fixed healthy reply", SMALL)
    arrow([(230, 1000), (230, 1120), (1440, 1120), (1440, 550), (1600, 550)])

    # module C
    box(2605, 430, 630, 200, "C", ["LLaVA-Next (4-bit)", "Step 1: describe the leaf",
                                   "Step 2: write English advice"])
    box(2605, 690, 630, 130, "C", ["Pest / fungicide keyword rule", "adds a warning if it fires"])
    box(2605, 880, 630, 175, "C", ["Hide chemical names", "Translate", "Put names back"])
    arrow([(2920, 630), (2920, 690)])
    arrow([(2920, 820), (2920, 880)])
    arrow([(2205, 1002), (2515, 1002), (2515, 500), (2605, 500)])
    text(2360, 985, "passages", SMALL)
    arrow([(230, 455), (230, 95), (3200, 95), (3200, 430)])

    # context
    box(2185, 1170, 700, 180, "ctx", ["Context (when available)", "Location and agro-climatic zone",
                                      "Weather forecast"])
    arrow([(2550, 1170), (2550, 600), (2605, 600)], dashed=True)
    arrow([(2885, 1260), (3500, 1260), (3500, 1055)], dashed=True)
    text(3190, 1245, "irrigation", SMALL)

    # farmer output
    fill, stroke = COL["out"]
    out.append(f'<rect x="3320" y="430" width="550" height="625" rx="20" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="4"/>')
    assert text_width("Farmer Output", TITLE, "bold") <= 550 - 28
    text(3595, 510, "Farmer Output", TITLE, "bold")
    bullets = [(600, "Disease + confidence", True), (690, "Advice in chosen", True),
               (742, "language", False), (832, "Similar images", True), (922, "Irrigation advice", True)]
    for y, s, dot in bullets:
        assert 3395 + text_width(s, BOX) <= 3870 - 14, s
        if dot:
            text(3360, y, "•", BOX, anchor="start")
        text(3395, y, s, BOX, anchor="start")
    arrow([(3235, 968), (3320, 968)])
    arrow([(950, 1350), (950, 1420), (3780, 1420), (3780, 1055)], dashed=True)

    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
            '<defs><marker id="tip" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" '
            'markerUnits="userSpaceOnUse"><path d="M 0 0 L 14 7 L 0 14 Z" fill="#333333"/></marker></defs>\n'
            f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>\n')
    return head + "\n".join(out) + "\n</svg>\n"


def to_pdf(svg: Path) -> None:
    exe = shutil.which("rsvg-convert") or "/workspace/miniconda3/envs/aqi/bin/rsvg-convert"
    subprocess.run([exe, "-f", "pdf", str(svg), "-o", str(svg.with_suffix(".pdf"))], check=True,
                   stderr=subprocess.DEVNULL)


def main() -> None:
    svg = FIG / "figure_1_architecture.svg"
    svg.write_text(build())
    for name in ("figure_1_architecture", "figure_1a_image_diagnosis",
                 "figure_1b_knowledge_retrieval", "figure_1c_advice_generation"):
        to_pdf(FIG / f"{name}.svg")
        print(FIG / f"{name}.pdf")


if __name__ == "__main__":
    main()
