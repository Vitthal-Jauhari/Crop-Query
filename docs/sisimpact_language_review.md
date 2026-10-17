# SISIMPACT manuscript: language and section review

**Written:** 2026-10-15 18:30 UTC. **File reviewed:** `papers/sisimpact_2027/main.tex` (version before this review is commit `f277461`).
**Result:** rewritten in plain English; compiles to **7 pages** (was 8); no undefined references; all fonts embedded. No number, table value, equation or limitation was removed. All numbers were re-checked against `results/vision_clean_resnet_v1/{summary,analysis}.json` and `results/archived_text_probe_audit.json`.

## What was wrong (whole paper)

1. **Read like an audit report, not a paper.** The same warning ("blank image, supplied diagnosis, not safety, not efficacy") was repeated about ten times. It is now said once where needed, and once in a Limitations subsection.
2. **Hard words.** For example: stage-aligned, evidence-flow, provenance, adjudicated, prospective, bounded, interfaces, silent resolution, surrogate, localization, descriptively, actionable. These were replaced with everyday words.
3. **Code words in the prose:** repository, notebook, version-2 notebook, configuration, code, `cuda:0`/`cuda:1`, "the plot reads the frozen analysis JSON". These were replaced with "the system", "earlier runs", "saved outputs".
4. **Talked about the paper's own history:** "the earlier manuscript emphasized Cassava", "earlier design record". These are now described from the data itself.
5. **F2 and CE were not explained** when they first appear (the abstract said "F2 consistency continuation"). Both are now defined in plain words in the abstract and in Section IV-B.
6. **AI-style patterns removed:** stacked hedges, "not X but Y" lines, long chains joined by semicolons, and triplets.

## Headings (old → new)

| Old | New |
|---|---|
| Related Work and Scope | Related Work |
| CropQuery Evidence Path | System Overview |
| Design record versus implementation | (folded into the System Overview text) |
| Visual and retrieval stages | Image classification; Text retrieval |
| Generation and post-processing | Answer generation and translation |
| Evaluation Method | Evaluation Setup |
| Prospective vision screen | New data split; CE and F2 fine-tuning |
| (unnamed text-probe paragraphs) | Text-step check |
| Findings | Results |
| Where the archived figures apply | Accuracy reported in earlier runs |
| Prospective vision result | Image classification results |
| (unnamed) | Text-step results |
| Two source-to-answer traces | Two worked examples |
| (last Discussion paragraph) | Limitations |

## Figures, tables and captions

- **Fig. 1 (old detailed drawing)** is too large and does not match the tested system (FAISS k = 5 with a label vote, 38 entries, a "Safety Validation" box, GPU labels). The paper needed Table I to correct it. **Plan:** generate a new code-aligned overview from `papers/sisimpact_2027/figure_1_prompt.json`. Then remove the old Fig. 1, the TikZ Fig. 2 and Table I. This should bring the paper to about 6 pages. Optional detail figures for each module: `figure_1a/1b/1c_prompt.json`.
- **Fig. 3 (source differences)** was full page width for a nine-bar chart. It is now one column. `scripts/plot_vision_screen.py` now draws it at 3.45 in × 2.45 in. The values did not change, and the paper now includes the vector PDF.
- **Fig. 4 (interface screenshot):** shorter caption. It is still marked as not a test case.
- **Table order fixed:** the main result table (top-1/macro F1) is now Table IV, before the Cassava table and the error-type table. Before, the text cited Table VI first.
- **All captions shortened.** Code words and repeated disclaimers were taken out of them.
- **Table III:** "97.2423%" was rounded to 97.24% to match the other values.
- **Table column headers were made plainer:** "What this means", "What it supports", "Disease given", "Shared start".

## Update 2026-10-17 05:40 UTC: new Figure 1 and module figures in the paper

- The authors uploaded `figures/figure_1_architecture.svg` and `figure_1a/1b/1c_*.svg`; the originals are kept unchanged in commit `e16287c`.
- Defects found and fixed:
  - Fig. 1: text past box edges in 6 places; 19 different font sizes; labels printed at about 4.3–5.2 pt. The ŷ arrow ran through the module B title, the DenseNet121 arrow joined a two-headed line, and the farmer-question line shared a vertical line with the leaf-photo branches. Fig. 1 is now drawn by `scripts/make_figure1_architecture.py`, with the same boxes, words and colours. It has 3 font sizes, about 6.1 pt box text in print, and a fit check on every label.
  - 1a and 1c: diamond text overflowed (made larger, arrow ends moved). 1c: the context box and two other boxes overflowed (made taller or split over two lines). 1b and 1c: one box each was too short (made taller). In all three, fonts were set to one size per box, at least 52 px (about 5.5 pt at column width).
- In the paper: the new Fig. 1 (full width) replaces the old design drawing, the TikZ figure and the old Table I. The modules appear as Figs. 2–4 (column width) in Sections III-A, III-B and III-C. Two sentences were added so that the text matches Module C (the no-leaf/healthy-leaf fixed reply, and the question type). The paper compiles to 7 pages with no warnings.

## Still open (authors must decide or check)

1. **Page limit.** No longer a constraint (2026-10-17): the authors will submit to a different venue. Check that venue's template and limits.
2. **Author email.** nihalgupta84@gmail.com was confirmed as correct by Nihal Kumar (2026-10-17).
3. **Science risk (unchanged).** The DELCON reviewers asked for real image-to-advice cases and for clearer novelty. Both remain weak. The advice evidence is still 19 blank-image, known-label cases and two worked examples. The main experiment tests ResNet50 alone, not the three-CNN system.
4. **Author block.** There is no corresponding-author mark.
5. **References.** The nine Kaggle datasets show year 2026, which is the access year. Use each dataset's upload year if known. The Kisan reference points to a Hugging Face re-upload ("Franklin01"). Cite the original Kisan Call Centre source if the authors know it.
6. **Title.** "Tracing Evidence from Crop Images to Generated Advice" is fine. Note that no single test runs from a real image all the way to advice.
