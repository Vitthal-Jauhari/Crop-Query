# SISIMPACT 2027 submission handoff — 15 October 2026 (India time)

## Venue and deadline

The [official SISIMPACT 2027 call for papers](https://www.sistece.ac.in/SISIMPACT-2027-author-call-for-papers), checked 15 October 2026, lists **15 October 2026** as the regular submission deadline. The conference fits CropQuery under Track 3 (AI, computer vision, data science). Initial submissions require an IEEE double-column PDF of **at most six pages including references**. Review is single-blind. The page gives a Microsoft CMT submission link; the public link opens a login page. The call does not state a submission time or time zone. It prohibits simultaneous consideration elsewhere. Camera-ready is listed for 30 November 2026. No portal upload or registration has been performed here.

## Current local manuscript

- `papers/sisimpact_2027/main.tex` compiles to `main.pdf`, a visually inspected **eight-page working** US-letter IEEEtran document. This is **two pages over the published initial-submission limit**; the earlier six-page draft remains in commit `f562ce6`. Embedded fonts are Type 1; no overfull boxes or missing references were found. The author block uses Vitthal Jauhari, Nihal Kumar, and the user-confirmed full name **Vidhi Rohilla**, with the affiliations and contact details from the DELCON source. See `docs/camera_ready_report_1.md` for the updated DELCON-response and evidence audit.
- The paper is framed as an **evidence-flow case study**. It contains Algorithm 1 for the implemented path, the prospective paired ResNet50 result, bounded archived notebook metrics, and two Rice/Mango source-to-answer traces. It avoids the rejected paper's first-of-kind, verified-document, historical held-out-accuracy, GPT-4o superiority, and dosage-safety claims.
- The completed prospective test reports base/CE/F2 top-1 of 95.12%/95.36%/94.98% and macro F1 of 93.61%/93.71%/92.92%. F2 minus CE is -0.386 percentage points with a test-group bootstrap 95% interval [-0.614, -0.160]. The paper therefore reports that the tested F2 continuation did not improve performance.
- The 19-scenario source-term diagnostic is an **archived notebook output**, not a newly rerun inference experiment; `results/archived_text_probe_audit.json` independently recomputes its table means and loss counts. It used blank images and supplied diagnoses. Its 0.947 retrieved and 0.763 generated mean term recall does not measure treatment correctness or image-to-advice performance.
- The two detailed traces are also **text-path** cases with blank images and known diagnoses; no agronomist reviewed them. They provide concrete failure analysis but do not fulfill the request for two independently assessed real-image cases.
- BibTeX and two final LaTeX passes completed with no errors, undefined references/citations, or overfull boxes. The eight rendered pages were visually checked. The revised paper has a BibTeX bibliography, the original detailed architecture and interface illustration, a code-aligned TikZ architecture, a source-delta graph, eight tables, and four displayed equations. This is a fuller working draft for accountable-author review; it has not been uploaded.

## Completed experiment record

- `scripts/fetch_cropquery_sources.py` downloaded the nine student-supplied public Kaggle image datasets into `/workspace/datasets/cropquery_sources` with revision and archive checksums. The shared dataset registry records those sources and the pinned Kisan parquet.
- `scripts/prepare_clean_split.py` mapped 132,955 images in 64 classes and froze 106,332 train, 6,635 validation, and 19,923 test images, excluding 65 images in conflicting-label groups. No assigned group or exact hash crosses partitions. The filename/hash grouping remains a lightweight sanity check and cannot identify every related photograph.
- `scripts/train_clean_resnet.py` ran one seed with a common six-epoch ResNet50 base and four matched CE/F2 continuation epochs. The untouched test partition was evaluated once after both arms completed. Machine-readable results, selected checkpoint identities, paired predictions, and source-level analysis are under `results/vision_clean_resnet_v1/`; logs are under `logs/vision_*_2026-10-15.log`.
- The result is a single-backbone internal screen. It does not validate the submitted three-model ensemble claim or represent training-seed and external-domain uncertainty. The negative F2 result triggered the planned stop rule: no additional backbones or broad sweeps were run.

## Before any CMT submission

1. The accountable authors should verify the complete author block, science, declarations, and final PDF. The third full name has been supplied as **Vidhi Rohilla**.
2. Check whether the DELCON manuscript is still under consideration. The user described it as rejected; retain proof of that status for the no-simultaneous-submission requirement.
3. Read the final PDF rather than only the TeX. Confirm that the negative prospective result and the limitations match the intended submission position.
4. Resolve the **eight-page versus six-page** initial-submission limit without discarding evidence or obscuring limitations. Camera-ready rules are a separate later gate; the current initial call gives no overlength exception.
5. An accountable author must approve the compliant final version and submit the PDF through the conference's CMT account. Preserve the submission receipt outside the manuscript.

The scientific-writing, venue-templates, experimental-design, and scientific-critical-thinking workflows informed this handoff. They are process guidance and are not evidence of CropQuery efficacy.
