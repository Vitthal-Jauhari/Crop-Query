# CropQuery handoff — 15 October 2026 (India time)

**Current phase:** SISIMPACT 2027 submission handoff. The public-source audit, frozen split, one-seed paired ResNet50 screen, evidence analysis, and IEEE manuscript revision are complete. No experiment is running. The regular submission date is 15 October 2026; the official call gives no time of day. Venue checks and the remaining human review are in `docs/sisimpact_submission_check.md`.

## Scientific finding

The controlled question was whether the repository's F2 consistency continuation improves crop-disease classification relative to matched cross-entropy (CE) continuation from the same common checkpoint. It did not in this screen.

- Frozen manifest: `data/split_manifest_v1.csv`, SHA-256 `97414fc5d2fac35463eadc88b88473073a35ac22b31d1f82eb595a36fd8e7ee0`.
- Prospective split: 106,332 train, 6,635 validation, and 19,923 test images across 64 classes; 65 images in conflicting-label groups were excluded. No assigned group or exact SHA-256 crosses partitions.
- Seed 42, ResNet50 ImageNet V2 initialization: six common-base epochs followed by four matched CE or CE+F2 continuation epochs.
- Untouched-test top-1: common base 95.12%, CE 95.36%, F2 94.98%. Macro F1: 93.61%, 93.71%, and 92.92%, respectively.
- Primary paired contrast: F2 minus CE = -0.386 percentage points; 2,000-replicate test-group bootstrap 95% interval [-0.614, -0.160]. F2 corrected 182 CE errors and introduced 259 new errors.
- F2 was lower than CE in eight of nine source datasets and tied in the small melon source. Cassava class changes were mixed rather than a consistent hard-class benefit.

This is a one-seed internal mechanism screen. Its interval covers test-group sampling only; it does not cover training-seed or external-domain uncertainty. The result is sufficient to remove F2 improvement from the paper's claims. It does not justify expanding to EfficientNet/DenseNet or a multi-seed sweep for this deadline.

## Evidence path and limits

CropQuery implements three CNNs for 64 classes, similar-image retrieval, two-source text retrieval, LLaVA advice generation, translation, chemical-string protection, and a pest/fungicide warning. The current configuration disables FAISS label substitution, so retrieval does not override the ensemble prediction.

All nine student-supplied Kaggle image sources and the Kisan Call Centre parquet were pinned under `/workspace/datasets`; identities are recorded in `results/source_inventory_v1.json`. The Kisan audit retained 178,815 of 178,939 Q&A rows; adding the notebook's 39 embedded crop entries reconstructs the printed count of 178,854 documents. This is count reconstruction, not proof of the historical dataset revision.

The repo-referenced final Phase D/F2 checkpoints remain unavailable, so historical ensemble scores cannot be independently replicated. The archived 19-scenario text-path diagnostic was recomputed from stored rows: mean retrieved source-term recall 0.947 and generated-answer recall 0.763, with three complete and one partial term loss. It used blank images, supplied diagnoses, and source-derived reference terms; it is a diagnostic trace rather than an efficacy, safety, or real-image evaluation. The Rice Leaf Blast and Mango Anthracnose examples similarly expose divergent source statements but have no agronomist adjudication.

## Submission artifact

`papers/sisimpact_2027/main.tex` now compiles to a visually checked **eight-page working IEEE draft** at `papers/sisimpact_2027/main.pdf`; it is **two pages over the published six-page initial-submission limit**. The earlier six-page PDF remains in commit `f562ce6`. The fuller version has a separate BibTeX file, the supplied original detailed architecture and interface illustration, the code-aligned architecture, a source-level result graph, eight tables, four equations, and pseudocode. Table I identifies four important differences between the detailed design drawing and inspected code. The original `architecture.pdf` is preserved byte-identically under `papers/sisimpact_2027/figures/`. The manuscript reports the negative prospective F2 result, archived text-path findings, and evidence limitations without unsupported efficacy, field-generalization, or safety claims. The author block uses Vitthal Jauhari, Nihal Kumar, and the user-confirmed full name Vidhi Rohilla, with the DELCON affiliations and contact details. The revised DELCON response and step-by-step audit are in `docs/camera_ready_report_1.md`; the 91-unit submitted-versus-revised prose crosswalk is in `docs/submitted_sentence_comparison.csv` and `docs/submitted_vs_revised_sentence_audit.md`.

## Reproducibility record

- Source download: `logs/source_download_2026-10-15.log`
- Split construction: `logs/split_prepare_2026-10-15.log`
- Training/evaluation: `logs/vision_{base,ce,f2,evaluate}_2026-10-15.log`
- Orchestration: `logs/vision_screen_orchestrator_2026-10-15.log`
- Primary output: `results/vision_clean_resnet_v1/summary.json`
- Paired/source analysis: `results/vision_clean_resnet_v1/analysis.{json,md}`
- Post hoc error types: `results/vision_clean_resnet_v1/analysis.json` → `error_categories` (CE 850 wrong condition/74 wrong crop; F2 933/68)
- Source graph: `scripts/plot_vision_screen.py` → `papers/sisimpact_2027/figures/source_deltas.png`
- Per-image predictions: `results/vision_clean_resnet_v1/paired_predictions.csv`
- Selected resumable checkpoints: `checkpoints/clean_resnet_v1/{base,ce,f2}_best.pth`

## Update 2026-10-15 18:30 UTC: plain-English rewrite and new Figure 1 prompts

- `papers/sisimpact_2027/main.tex` was rewritten in plain English. Headings, captions and table order were also fixed. It now compiles to **7 pages** (was 8). All numbers were re-checked against the result JSONs. Details: `docs/sisimpact_language_review.md`. The previous text is commit `f277461`.
- Fig. 3 is now one column wide (`scripts/plot_vision_screen.py` size changed; values unchanged).
- New prompts for a code-aligned Figure 1: `papers/sisimpact_2027/figure_1_prompt.json` (overview, modules A/B/C) and the optional module details `figure_1a/1b/1c_prompt.json`. When the image `figures/figure_1_architecture.png` exists, replace the old Fig. 1, the TikZ Fig. 2 and Table I. That should reach about 6 pages.

## Update 2026-10-17 05:40 UTC: new figures in the paper

- The page limit no longer applies; the authors will submit to another venue. Nihal's email nihalgupta84@gmail.com is confirmed.
- Fig. 1 is now the code-aligned overview (`figures/figure_1_architecture.pdf`, drawn by `scripts/make_figure1_architecture.py`). Modules A/B/C are Figs. 2–4 (`figure_1a/1b/1c_*.pdf`, converted by the same script). The old design drawing, the TikZ figure and the old Table I were removed. The paper is 7 pages; the build is clean. Details: `docs/sisimpact_language_review.md`.
- `papers/sisimpact_2027/crop_query.zip` is the authors' backup of this folder from 15 Oct. It is untracked and was left as is.

## Immediate next steps

1. All authors read the PDF and verify names, affiliations, emails, scientific statements, and conflicts of interest.
2. Before an initial upload, the accountable author needs a compliant six-page version or an explicit venue length exception. Confirm the paper is not under simultaneous review elsewhere. No upload has been made.
3. For a later journal revision, obtain at least two independently adjudicated real-image image-to-advice cases before making advisory-value or safety claims. Keep F2 as a reported negative screen unless a new scientific hypothesis motivates another controlled study.

No portal upload or registration has been performed from this workspace.
