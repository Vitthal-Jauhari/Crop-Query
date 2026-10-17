# CropQuery revision plan (updated 15 October 2026)

Execution gates, controlled comparisons, outputs, and the current environment inventory are specified in [`implementation_details.md`](implementation_details.md). This file is the phase overview; `walkthrough.md` is the living status record.

## Research question and current position

**Question:** Can a crop image and a farmer question be converted into an accurate diagnosis and agronomically supported advice across Indian languages, especially when the visual diagnosis is uncertain?

The DELCON submission demonstrates a working system, but the current evidence does not establish reliable end-to-end efficacy. The submitted contribution list names components (three CNNs, FAISS, ChromaDB, LLaVA, translation) rather than a testable mechanism. The closest literature already includes multimodal agricultural assistants and vision-to-RAG advisory systems (see `docs/revision_audit.md`). A journal paper therefore needs a narrower, falsifiable claim and a clean evaluation. The prospective screen has now rejected F2 improvement under the tested schedule. **Working journal hypothesis, not an established contribution:** evidence-aware routing of uncertain diagnoses may reduce unsupported or wrong-category advice compared with always using the top predicted class.

## Phase 0 — Recover inputs and freeze provenance

Inventory the **student-supplied** nine Kaggle image datasets and [Kisan Call Centre Q&A](https://huggingface.co/datasets/Franklin01/Dataset_prep01), pin their current revisions, and create a clean prospective image split. Their metadata are accessible. The repo references a separate Kaggle checkpoint/FAISS dataset that is inaccessible with the configured credentials; this limits replication of historical vision scores, not the new controlled study. Its `cropquery-v2.ipynb` also embeds 39 curated AgroBench entries, which can be paired with Kisan Q&A to reconstruct the v2 two-source RAG variant even though the separate AgroBench repository is inaccessible. Record source versions and file hashes before GPU work. See E0 in `implementation_details.md`.

## Phase 1 — Reconcile evidence before manuscript claims

1. For the historical scores, recover the exact image manifest, class mapping, and training membership for any available Phase D and F2 checkpoint. Identify each source dataset version. Compare content hashes across training, validation, feature bank, and any proposed test set where records permit. The test set must have been excluded **before** training and model selection. If the original checkpoint lineage cannot be established, keep those scores as unverified historical reports and run a new controlled comparison on a frozen clean split.
2. Reconcile the manuscript's 130,571 images / 19,984 test images with the executed notebooks' 132,955 images / 19,963 or 19,944 evaluation images. Resolve the paper's 38 curated entries and 178,853 documents against the executed run's 39 entries and 178,854 documents. Store one versioned count table and split manifest.
3. Recover raw prediction files and evaluation JSON where available for the exact checkpoint and split behind each reported number. Treat current cached notebook outputs as evidence of what a run printed, not proof of a valid held-out estimate.

**Decision gate:** Do not reuse 97.83%, 97.24%, 53.98%, 94.70%, or “verified/safe” as final efficacy claims until their populations, controls, and metrics are reconciled.

## Phase 2 — One controlled vision comparison

**Question:** On the same frozen evaluation set, does F2 consistency training improve classification, particularly for Cassava classes, over continued cross-entropy fine-tuning from the same Phase D initialization?

Use one planned seed. Screen the two matched arms on ResNet50 first, then train the other two backbones only if the validation result warrants an ensemble claim. The arms share initialization, data, weak-view inputs, steps, optimizer, and checkpoint selection: continued CE versus CE plus the F2 consistency term using a strong view. Compare paired predictions on the untouched test set only after the model and rule are frozen: top-1, macro F1, Cassava per-class recall/F1, and counts of corrected versus newly wrong predictions. Report a paired uncertainty interval for the accuracy difference or a paired error table. Do not run multiple seeds unless a reviewer requests them. If the clean test evidence does not support F2, report the negative result and keep F2 out of the headline.

**FAISS decision:** Evaluate the old Phase D and F2 checkpoints separately. The executed F2 notebook reports 97.2423% without override and 96.9164% with the tested balanced-bank override; it should not be advertised as an F2 improvement. Use FAISS for similar-image display if that remains useful, and describe it as an interface feature.

**Completed decision:** On the frozen 19,923-image test set, CE reached 95.36% top-1 and 93.71% macro F1; F2 reached 94.98% and 92.92%. F2 minus CE was -0.386 percentage points with a test-group bootstrap 95% interval [-0.614, -0.160]. F2 corrected 182 CE errors and introduced 259. The result does not support F2 as an improvement, so no other backbones or routine seed sweep are planned. See `results/vision_clean_resnet_v1/analysis.md`.

## Phase 3 — Advisory evidence on independent cases

**Question:** What agronomic errors remain when real images and farmer questions pass through the current system, and does a targeted evidence-aware decision reduce them?

Build a prospectively fixed set of at least two **nontrivial** cases with actual crop images and questions: one ambiguous visual diagnosis and one pest-versus-disease or treatment-conflict case. A valid case record includes image source/consent, crop and ground truth from an expert or trustworthy diagnostic source, language, question, system predictions, retrieved passages with provenance, full English and translated advice, and blinded agronomist judgments. Do not use the exact knowledge-base entries as both retrieval material and reference answers. First assess the current pipeline. Add a paired answer/qualify/defer comparator only if these cases show that the hypothesized routing failure exists; use the targeted pilot design in `implementation_details.md` before making a quantitative improvement claim. Score diagnosis correctness, source support of each treatment claim, wrong-category recommendations, dose/name preservation, and useful next action. Record failures, including translation failures. ROUGE may be secondary; it cannot establish chemical or farmer safety.

**Decision gate:** If there are fewer than two completed, independently assessed cases, do not describe a two-case study or claim efficacy from examples.

## Phase 4 — Paper and venue

**15 October 2026 SISIMPACT gate:** The revised four-page IEEE conference paper is in `papers/sisimpact_2027/`, with venue checks in `docs/sisimpact_submission_check.md`. It is an evidence-flow case study with a completed negative F2 screen because the old vision split cannot be certified held out and there are no independently adjudicated real-image advice cases. The nine supplied image sources and Kisan parquet are pinned in `results/source_inventory_v1.json`; the new split and measured results are frozen locally. The longer journal hypothesis and Phase 3 case study remain future work.

The SISIMPACT manuscript reports the comparator and measured effect in the abstract and findings, presents the implemented pipeline as executable pseudocode, and removes cross-paper accuracy ranking and first-of-kind claims. Keep the DELCON PDF and TeX as historical records. The separate LaTeX draft in `papers/paper_v2/` remains a future journal draft; choose a journal and check its current author instructions only after the advisory hypothesis has independent evidence. Consult `docs/walkthrough.md` for the final conference state.
