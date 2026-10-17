# SISIMPACT 2027 manuscript audit — report 1

**Audit date:** 15 October 2026; revised after the author's request to use the original detailed architecture and compare the submitted and revised wording. **Stage:** working manuscript for initial conference submission; the venue's camera-ready date is later. **Artifacts audited:** the previously submitted six-page `submitted_paper/main_paper.pdf` and its LaTeX sources; the revised `papers/sisimpact_2027/main.tex`, eight-page `main.pdf`, and `references.bib`; repository code/notebooks; the frozen split and experiment outputs under `results/`. This is an author-side evidence audit, not a claim of editorial acceptance or independent agronomic review.

## 1. Venue and page check

The [official SISIMPACT 2027 call](https://www.sistece.ac.in/SISIMPACT-2027-author-call-for-papers), checked 15 October 2026, specifies IEEE double-column formatting and **at most six pages including figures, tables, and references** for the initial submission. It lists 15 October 2026 as the regular submission deadline and 30 November 2026 for camera-ready. The previously submitted DELCON PDF has six pages; the current fuller SISIMPACT working PDF has **eight US-letter pages** and is **over the initial-submission limit**. The six-page revision is preserved in commit `f562ce6`. The eight-page file compiled through BibTeX and two final LaTeX passes with embedded Type 1 fonts, no undefined citations/references, missing graphics, or overfull boxes. A rendered eight-page contact sheet was inspected.

The author requested that substantive explanation and figures not be sacrificed merely to keep this working draft at six pages. This version therefore keeps the original design and interface illustrations alongside the code-aligned architecture, result graph, eight tables, four equations, and pseudocode. **It must be brought within the venue's applicable limit before an initial upload unless the venue grants an exception; the call does not document one.** Camera-ready rules must be checked separately when relevant.

## 2. Problem, question, and finding

**Verdict: clear and bounded, with a real limitation.** The title, abstract, and introduction state the problem: a crop advisor can lose or obscure the link between image diagnosis, retrieved passages, and generated treatment text. The paper separates two questions: whether a visual estimate is genuinely held out, and what happens to retrieved source terms when a diagnosis is supplied. It explicitly says no experiment estimates the complete image-to-advice outcome.

The main quantitative finding is stated in the abstract, Findings, Discussion, and Conclusion: the tested F2 continuation is **0.386 percentage points below** matched CE on the frozen test set. The qualitative finding is narrower: two archived traces show conflicting retrieved rates or chemical choices and implicit source selection by the generator. The revised Introduction and Related Work now state the distinctive *stage-aligned evidence analysis* explicitly and compare the scope with AgriDoctor, Mondal et al., and AgriRAG. This is a specific empirical contribution, not a demonstrated novel model or proof of global priority. Reviewers may still find the methodological novelty or practical validation insufficient without a same-case advisory comparator and independently assessed end-to-end cases.

## 3. Quantitative experiment and claim trace

| Claim in revised paper | Direct evidence | Audit judgment |
|---|---|---|
| 132,955 mapped images, 64 classes; 106,332 train, 6,635 validation, 19,923 test | `results/split_audit_v1.json`, frozen `data/split_manifest_v1.csv`, source inventory | Supported for the current nine student-supplied image revisions. The split predates the new training run. |
| Group/hash isolation and 65 excluded conflicting-label images | `results/split_audit_v1.json`, split preparation code | Supported as a *lightweight* check. It cannot rule out every near duplicate or photograph of the same plant. |
| One-seed ResNet50 base, then matched CE and CE+F2 continuation | `scripts/train_clean_resnet.py`, run logs, checkpoint lineage, `results/vision_clean_resnet_v1/summary.json` | Supported for the prospective screen. It is not a three-backbone ensemble validation or a multi-seed result. |
| Base/CE/F2 top-1: 95.12/95.36/94.98%; macro F1: 93.61/93.71/92.92% | `summary.json` and per-image `paired_predictions.csv` | Supported. CE and F2 use the same 19,923 test images; test selection was not used for checkpoint choice. |
| F2−CE = −0.386 percentage points, interval [−0.614, −0.160] | `summary.json`; 2,000 test-group bootstrap replicates in `analysis.json` | Supported for sampling of **16,864 frozen test groups**. The interval excludes training-seed and external-domain uncertainty. |
| F2 corrects 182 CE errors and creates 259 new errors | `summary.json`, `analysis.json`, `paired_predictions.csv` | Supported paired count: 18,740 correct for both, 742 wrong for both; `(182−259)/19,923 × 100 = −0.386` percentage points. |
| Eight source-level deltas negative, one zero | `analysis.json` → `source_results`; `scripts/plot_vision_screen.py` | Supported descriptively. The zero melon delta uses only 29 test images, so it is not equivalence evidence. |
| Cassava class changes are mixed | `summary.json` → per-class outcomes; paper Table V | Supported descriptive subgroup analysis; no separate significance claim. |
| F2's extra errors concentrate within the correct crop | `analysis.json` → `error_categories`, derived from frozen per-image predictions and class map | Supported **post hoc** decomposition: CE 850 wrong-condition/74 wrong-crop; F2 933/68. It does not identify a training mechanism or downstream treatment harm. |

The negative F2 result is reported rather than hidden. No broad backbone sweep was run after this screen. The historical 96.88%/97.84% ensemble/FAISS outputs and F2 notebook's 97.2423% are retained only as archived observations with uncertain or reused splits; they are **not** ranked against the prospective result. Missing original final ensemble checkpoints prevent a clean replication of those older claims.

## 4. Qualitative experiment and case quality

The 19-scenario chemical-term probe is an **archived notebook diagnostic**, recomputed from its saved rows in `results/archived_text_probe_audit.json`. Mean source-term recall is 0.947 in retrieved passages and 0.763 in generated answers; three scenarios lose all selected terms and one loses some. Each prompt used a blank image and a supplied diagnosis, and its reference terms came from indexed local entries. Thus, this is a measurement of term transmission, not agronomic correctness, source reliability, or image-to-advice performance.

Table VII gives two concrete Rice Leaf Blast and Mango Anthracnose traces. The Rice materials contain different tricyclazole rates; the Mango materials offer different chemical alternatives. The generated text chooses one path without explaining the source divergence. The table intentionally reproduces rates as **observations of differing source statements**, not recommendations. Neither trace has an independently sourced real image, independently prepared treatment reference, agronomist adjudication, or native-speaker assessment. Reviewer A's request for two independently assessed complete cases is therefore **still unmet**. The original Pumpkin screenshot is now reproduced strictly as an *interface illustration*, with a caption that disclaims outcome and treatment validation; it is not counted as a case.

## 5. Figures, architecture, graph, and tables

| Item | What it shows | Evidence and presentation check |
|---|---|---|
| Fig. 1, original detailed architecture | Original `submitted_paper/architecture.pdf`, reproduced as an illustration of the earlier design record | Original PDF is retained byte-identically; the included print-resolution PNG preserves appearance. Caption and new Table I identify four differences from current code. It is not validation of the drawn functions. |
| Fig. 2, TikZ architecture | Implemented path from image/question through three CNN soft vote, similar-image display, crop/Kisan retrieval, LLaVA-Next, warning, and translation | Cross-checked against `src/` and notebook configuration. Caption separates the full repository path from the ResNet50-only experiment. FAISS is shown **without** label override in the current setting. |
| Fig. 3, source-delta graph | F2−CE top-1 difference for each of nine source datasets, with test counts | Generated directly from frozen `analysis.json` by `scripts/plot_vision_screen.py`; 600 dpi PNG in the PDF. All source rows are explicitly descriptive. |
| Fig. 4, original interface | Original submitted Pumpkin interface image, classification display, generated answer, and irrigation panel | Caption states that image provenance, diagnosis, dose, and advice were not independently adjudicated. It is not an evaluated result. Original PDF and print-resolution PNG are retained. |
| Table I | Original diagram versus inspected code | Contrasts FAISS neighbor count and override, 38 versus 39 local entries, the validation box versus implemented warning, and diagrammed devices versus the ResNet50-only prospective test. |
| Table II | Nine source mapped/test counts | Totals reconcile to 132,955 and 19,923. Dataset citations point to the student-supplied Kaggle links. |
| Table III | Historical notebook outputs and valid interpretation | Prevents archived ensemble, F2, and text metrics from being mistaken for one comparable held-out study. |
| Table IV | Correct, wrong-condition, wrong-crop counts | Each checkpoint row sums to 19,923; labeled post hoc. |
| Table V | Five Cassava class recalls and supports | Gives a focused check of the previously emphasized hard cluster; mixed changes are visible. |
| Table VI | Prospective base/CE/F2 summary | Gives the primary quantitative finding and metric denominators in its caption. |
| Table VII | Two text-path source-divergence traces | Full-width layout preserves the retrieval-to-answer chain; caption states blank image and supplied diagnosis. |
| Table VIII | Claim/evidence boundary by pipeline stage | Makes the missing integrated efficacy and safety evidence explicit. |

The DELCON ablation bars, confusion graphic, and top-errors chart were **not reused as validating figures** because they encode historical scores with unverified test exclusion. The original detailed architecture and Pumpkin screenshot are labeled as design/interface illustrations only. They are not evidence for the new experiment or treatment accuracy. The figure's small original labels are legible when the source PDF is enlarged; the paper supplies Table I so readers do not have to infer current behavior from them.

## 6. Mathematical and algorithmic framework

The four displayed equations correspond to (1) three-backbone soft-probability voting in the repository, (2) retrieved-versus-generated source-term recall in the archived diagnostic, (3) the implemented F2 CE plus detached-strong-view KL objective with coefficient 0.10 and label smoothing 0.1, and (4) the image-count-weighted group bootstrap estimator. Algorithm 1 summarizes the implemented inference branch, including the confidence gate, FAISS behavior, two text sources, warning, and translation. These equations make the methods auditable; **none is claimed as a new mathematical contribution**. The paper distinguishes the full three-CNN inference architecture from the single-ResNet50 prospective comparison.

## 7. Balance, reference quality, and DELCON response

The eight-page working version gives space to the problem, adjacent work, implementation, audit method, negative vision result, source-divergence traces, and limitations. The visual study is quantitatively stronger than the advice study; this imbalance is made explicit instead of blending endpoints. The source-level graph and error table add analysis, while the architecture and equations explain what was measured. A separate `references.bib` now supplies 18 cited sources, including the primary AgriRAG paper, core models/FAISS, Kisan Q&A, and the nine supplied image sources. Dataset titles/versions should receive an author final metadata check before upload; URLs and pinned revisions are in `results/source_inventory_v1.json`.

The direct DELCON-comment check is:

| Reviewer point | Current status |
|---|---|
| A: Too many abstract numbers | **Addressed in writing:** the abstract keeps the test size and primary difference, moving the other figures to Results. |
| A: Pseudocode | **Addressed:** Algorithm 1 states the implemented inference path and its current FAISS behavior. |
| A: At least two nontrivial cases | **Partially addressed:** Rice and Mango are distinct complete text-path traces, and the Pumpkin screenshot shows an interface. None is an independently adjudicated real-image advice case. Efficacy remains unshown. |
| B: Clarify novelty beyond integration | **Clearer positioning, unresolved strength:** the paper now names stage-aligned evidence analysis as its contribution and cites close multimodal/RAG systems. It does not demonstrate a new architecture or method advantage. |
| B: Strengthen evaluation and moderate claims | **Partially addressed:** the frozen matched vision screen replaces unsupported held-out comparison; historical numbers and safety claims are bounded. Integrated advice evaluation is still missing. |

The unresolved case and comparator evidence is substantive. A clear account of a project-specific empirical contribution is supportable; a claim that novelty has been fully established would exceed this record.

## Sentence-level comparison and wording pass

`docs/submitted_sentence_comparison.csv` records the disposition and reason for each of **91 prose units** extracted from the submitted TeX abstract through conclusion. The companion `docs/submitted_vs_revised_sentence_audit.md` explains the extraction boundary, gives paragraph-level comparisons, and separately checks old figures, tables, and equations. This is a traceable crosswalk, not a claim that every original sentence was preserved. Citations and LaTeX syntax are normalized in the CSV; the original TeX remains the authority for exact typography.

The final prose pass replaced treatment-source “conflict” where the record only shows different statements, explicitly qualified the incomparable tricyclazole product rates, and clarified that the text probe used supplied diagnoses and blank images. It removed the old “verified label,” “strict” RAG adherence, “safety validation,” unverified held-out accuracy, hard-coded GPT-4o comparison, and broad first-of-kind language. The revised paragraphs consistently separate implemented code, archived notebook output, and the new prospective experiment.

## 8. Author handoff before initial submission

1. All three authors verify the final author names, affiliations, emails, technical statements, and the revised PDF. The third author's full name is the user-confirmed **Vidhi Rohilla**.
2. The accountable author confirms the DELCON paper is no longer under consideration and checks any venue disclosure/declaration requirements. The SISIMPACT call prohibits simultaneous submission.
3. The accountable author makes the final scientific and disclosure decision. **The current eight-page PDF is over the published initial-submission limit and should not be uploaded as compliant without a venue-approved length exception or a quality-preserving six-page revision.** No upload or registration has been performed from this workspace.
4. For a later revision, collect independently adjudicated real-image image-to-advice cases before claiming treatment quality, safety, or integrated efficacy; see `docs/implementation_details.md` E3/E4.

**Workflow note:** The local audit used the scientific-writing, peer-review, venue-templates, and scientific-visualization skills as process guidance. The skills' [Scientific Agent Skills report](https://arxiv.org/abs/2609.00065) describes the workflow resource; it is not evidence for any CropQuery result. The accountable authors must verify the final text and venue policy.
