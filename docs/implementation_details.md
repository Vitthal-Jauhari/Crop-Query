# Remaining experimentation: implementation details

**Status:** E0 input audit, E1 frozen split, E2 paired ResNet50 screen, result analysis, and the SISIMPACT manuscript revision are complete as of 15 October 2026. E3 independently adjudicated image-to-advice cases remain future journal work. Read with [`implementation_plan.md`](implementation_plan.md), [`walkthrough.md`](walkthrough.md), and [`revision_audit.md`](revision_audit.md). The submitted DELCON paper and its review remain historical evidence.

## 1. Scientific decision to be made

The first question is whether a valid held-out estimate exists at all. Only after that is resolved should the project decide which paper claim to pursue:

| Candidate claim | Minimum discriminating evidence | Decision |
|---|---|---|
| F2 consistency learning improves hard crop-disease recognition | Continued cross-entropy (CE) versus CE+F2 from the **same initialization**, on one frozen split, with paired Cassava and overall errors | Keep as a method result only if the difference survives a clean test and is not just a different dataset or checkpoint history. A generic consistency loss alone is unlikely to be a strong novelty claim. |
| An uncertainty/evidence decision reduces unsupported advice | Same image-question cases through current RAG path versus an explicitly specified answer/qualify/defer rule, with blinded agronomist scoring | Implement and claim this only if observed failures motivate the rule and paired results support it. Do not invent a new module solely for novelty. |
| CropQuery is a useful integrated application | Two nontrivial complete case studies, source-supported advice, bounded language findings, and a valid diagnostic estimate | If neither method claim is supported, position as a system/application study and avoid “first,” safety, and field-generalization claims. |

**Order:** E0 inventory and access → E1 split/provenance audit → E2 one controlled vision comparison → E3 independent end-to-end cases → E4 targeted advisory comparator if the E3 failure analysis warrants it → E5 analysis and paper. A failed gate changes the claim; it does not trigger an automatic sweep.

**15 October 2026 conference gate:** The [SISIMPACT 2027 regular deadline](https://www.sistece.ac.in/SISIMPACT-2027-author-call-for-papers) is today in the user's India time zone. The expanded **eight-page working** IEEE draft under `papers/sisimpact_2027/` includes the completed prospective one-seed ResNet50 screen, source-level graph, post hoc error categories, bounded archived evidence, a separate BibTeX file, and the original detailed architecture and interface illustration from the supplied `submitted_paper/` directory. The detailed `architecture.pdf` was an ancillary asset, distinct from the architecture graphic named in the submitted TeX; it is now Fig. 1, accompanied by a code-difference table. It is **over the published six-page initial-submission limit**; a compliant earlier draft is retained in commit `f562ce6`. The screen cannot establish the old ensemble accuracy and does not support an F2 improvement claim. The two archived Rice/Mango text-path traces are not substitutes for independently adjudicated real-image cases; the manuscript states that limitation. See `docs/camera_ready_report_1.md` for the updated DELCON response and paper audit, `docs/submitted_vs_revised_sentence_audit.md` for the wording comparison, and `docs/sisimpact_submission_check.md` for venue requirements and author-review tasks.

## 2. Inputs and environment checked now

- The active `vision` environment has PyTorch 2.7.1+cu128 and sees one CUDA device. `torch.cuda.mem_get_info()` reported about 39.1 GiB free of 39.5 GiB at inspection time. The resource skill detected no scheduler allocation; free VRAM is transient and must be checked before a job.
- The shared dataset registry (`/workspace/datasets/DATASETS.md`) has no CropQuery image corpus entry. No project `datasets/` link, obvious crop-named shared directory, or `/kaggle/input/datasets` mount was found. The local `models/` directory holds four `phase*.pth` checkpoints, not the three Phase D or F2 ensemble files named by the notebooks. A safe metadata read found Phase B/C metadata in three of them (`phase3_final.pth`: phase C, epoch 7, 64 classes); none supplied an ensemble Phase D/F2 checkpoint.
- The current environment has `torch`, `torchvision`, `kagglehub`, `datasets`, `transformers`, `pandas`, and `sklearn`; `faiss` and `chromadb` are absent. Do not change the shared environment for planning. Create a named project environment only if actual evaluation requires incompatible versions.
- `src/config.py` assigns the CNNs to `cuda:0` and LLaVA to `cuda:1`; this server has only one visible GPU. The end-to-end study therefore needs a small device-configuration change and sequential vision/generation loading, or the original dual-GPU Kaggle runtime. Smoke-test that path before collecting cases.
- **Student-supplied sources:** the nine Kaggle image datasets listed in `ensemble-training.ipynb` and [Franklin01/Dataset_prep01](https://huggingface.co/datasets/Franklin01/Dataset_prep01) for Kisan Call Centre Q&A. All nine Kaggle source metadata endpoints returned HTTP 200 with the configured credentials; the current Plant Village revision is 2 and the other eight current revisions are 1. These are current revisions, not proof of what the historical runs used. The Hugging Face dataset is accessible at revision `57535f94e2feebe7bb1b320edbc4b2d5cf3406bb`; its current train split reports 178,939 rows with `questions`, `answers`, and `text` columns. A pinned local parquet audit (`results/kisan_count_audit.json`) found 178,815 Q&A rows accepted by the app, 124 skipped, and 178,854 expected documents after adding the 39 embedded entries. This matches the notebook printout by count only; historical revision identity and content remain unverified.
- **Repo-referenced artifacts, not student-supplied links:** `README.md`, `src/config.py`, and the F2 notebook reference `vitthaljauhari2/cropquerydataset` for saved model checkpoints and FAISS banks. That Kaggle dataset returned HTTP 403 (`datasets.get` denied), and the configured username differs from its owner. `src/knowledge_base.py` also references `vitthaljauhari/agrobench`, which returned HTTP 404 with the configured Hugging Face token. However, `notebooks/cropquery-v2.ipynb` cell 11 embeds 39 curated entries (`crop`, `condition`, `text`), so its two-source RAG variant can be reconstructed locally with the supplied Kisan dataset. This does not prove that the embedded list matches the submitted paper's 38-entry count or the unavailable remote revision. Missing checkpoint files limit historical score replication but **do not block a new controlled vision experiment using the supplied public images**. No credential values were printed or stored.

Current metadata for the nine supplied image sources (checked 14 October 2026; pin the downloaded revision before using any images):

| Kaggle dataset | Current revision |
|---|---:|
| [Plant Village Updated](https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated) | 2 |
| [Five Crop Diseases](https://www.kaggle.com/datasets/shubham2703/five-crop-diseases-dataset) | 1 |
| [Pea Plant](https://www.kaggle.com/datasets/zunorain/pea-plant-dataset) | 1 |
| [Crop Diseases Classification](https://www.kaggle.com/datasets/mexwell/crop-diseases-classification) | 1 |
| [Cashew Image](https://www.kaggle.com/datasets/ayushbpanchal/cashew-image-dataset) | 1 |
| [Pumpkin Leaf Diseases](https://www.kaggle.com/datasets/tahmidmir/pumpkin-leaf-diseases-dataset-from-bangladesh) | 1 |
| [Melon Macronutrient Deficiency](https://www.kaggle.com/datasets/binnassor89/melon-macronutrient-deficiency-dataset) | 1 |
| [RoCoLe Coffee Leaf](https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset) | 1 |
| [Wheat Rust](https://www.kaggle.com/datasets/yasaminborhani/wheatrustclassificationdataset) | 1 |

The 39 embedded entries have SHA-256 `c1b4b301ca322b3485ac8574eb8b083e854c034d4b0c3d86c8c7b8f2c44428fc` after JSON serialization with sorted keys, UTF-8, and compact separators. This identifies the inspected list; it is not an independent treatment reference.

### Required E0 artifacts

1. **Completed:** `results/source_inventory_v1.json` records all nine source slugs/revisions/archive hashes, four local earlier-phase checkpoint file hashes, class-map hash, pinned Kisan revision and count audit, and missing final checkpoint/manifests. It does not establish the versions used by historical runs.
2. The original Phase D and F2 checkpoint files for all three backbones, if available, or a clear declaration that the historical scores cannot be independently verified from files. Inspect checkpoint metadata read-only; do not execute notebook cells that delete prior checkpoints.
3. The image corpus under `/workspace/datasets/<registered-name>` with one project-relative link if downloaded. Update the shared dataset registry if data are added. Keep the original corpus once, not copied into the project.

**E0 gate:** If original final checkpoints remain unavailable, mark historical-score replication `blocked_by_missing_checkpoint` and continue with a fresh controlled run from the supplied public images. If an image source cannot be downloaded, record it and adjust the new study population explicitly. For RAG, pin the 39-entry list embedded in the v2 notebook and the Kisan revision; treat that as a reconstructed v2 variant until its relation to the submitted 38-entry version is resolved. Do not infer results from printed notebook cells.


### Executed status on 15 October 2026 (India time)

- E0 completed for current public inputs: nine Kaggle source archives were downloaded and pinned; `results/source_inventory_v1.json` records revisions, archive hashes, four local earlier-phase checkpoint hashes, the class-map hash, and the pinned Kisan count audit. Historical final ensemble checkpoints remain unavailable.
- E1 completed for the new study: `data/split_manifest_v1.csv` has SHA-256 `97414fc5d2fac35463eadc88b88473073a35ac22b31d1f82eb595a36fd8e7ee0`. The source scan mapped 132,955 images to all 64 classes, excluded 65 in conflicting-label groups, and skipped 107 unsupported/invalid files. Split counts are 106,332 train, 6,635 validation, 19,923 test; zero assigned-group and exact-SHA overlaps cross partitions. The smallest test class has six images. `results/split_audit_v1.json` gives per-class and per-source counts. This prospective manifest does **not** validate the historical split merely because its mapped total equals the old F2 notebook's 132,955.
- E2 completed after a one-batch smoke test with finite CE+F2 loss and an 8-by-64 output. The common six-epoch ImageNet-initialized ResNet50 selected zero-based epoch 5 at 95.0867% validation top-1. Four matched continuation epochs selected zero-based epoch 3 for both CE (95.5539% validation top-1) and F2 (95.4936%). The untouched 19,923-image test partition was then evaluated once. Base, CE, and F2 test top-1 were 95.1212%, 95.3621%, and 94.9757%; macro F1 was 93.6109%, 93.7123%, and 92.9158%. The primary paired difference, F2 minus CE, was -0.3865 percentage points with a 2,000-replicate test-group bootstrap 95% interval of [-0.6137, -0.1598]. F2 corrected 182 CE errors and made 259 new errors. It was lower than CE in eight source datasets and tied in the small melon source; Cassava recall changes were mixed. After CE epoch 1 validation fell below the common base, the final evaluation was amended **before any test read** to include the base as a descriptive reference. CE exceeded that base by 0.2409 points, interval [0.0251, 0.4438]. The primary CE-versus-F2 contrast and training settings were unchanged. Outputs are in `results/vision_clean_resnet_v1/`; stage logs are `logs/vision_{base,ce,f2,evaluate}_2026-10-15.log`.
- **E2 decision:** reject the candidate claim that this F2 schedule improves held-out performance. Do not expand F2 to the other backbones or launch routine seed sweeps for the conference deadline. Preserve the negative result and redirect future evidence collection to independently assessed image-to-advice cases.

## 3. E1: resolve the split and counts before GPU work

### Historical audit (read-only)

The notebook `notebooks/training/ensemble-training.ipynb` trains with `train_test_split(..., test_size=0.15, random_state=42)` in cell 14 and builds a different hash-grouped “test” split in cell 45. The F2 notebook's cell 13 makes a fresh 113,011/19,944 split from 132,955 images; its later evaluation uses `val_s` as “test.” The paper instead reports 130,571 total and 19,984 test images. Reconstructing a split from code is supporting evidence, not proof of checkpoint membership unless the source order and version match the actual training run. Lack of historical files need not postpone creating a new, prospective split from the supplied sources.

For every available image, record `{source_dataset, source_version, relative_path, class_id, sha256, optional source_group}`. Hash exact duplicates. Inspect a small targeted sample of likely near duplicates or source sequences; avoid a broad audit of unchanged published datasets. Compare candidate evaluation hashes with (a) recorded or reconstructable Phase D and F2 training membership, (b) validation membership, and (c) FAISS bank membership. Record conflicting labels for identical hashes. The FAISS bank has vectors and labels, not image IDs; without a saved bank manifest, its membership cannot be proven from vector count alone.

Deliver `results/provenance_audit.json` and a short `results/provenance_audit.md`: corpus counts by source/class, unique hash count, overlap counts, checkpoint-to-split evidence level (`direct manifest`, `deterministic reconstruction`, or `unknown`), and exact reasons for any unverified estimate. Do not print private paths or credentials in logs.

### Clean split for a new controlled run

If no demonstrably untouched original test set exists, freeze a new split **before** training. Use SHA-256 groups as indivisible units, group related images from the same source/plant where metadata permit, and stratify by the 64-class label. Start with an 80/5/15 train/validation/test target to preserve the manuscript's intended allocation. Before training, check class presence, group isolation, source distribution, and label conflicts; adjust the allocation once and document it if rare classes make the target infeasible. Serialize `data/split_manifest_v1.csv` with immutable IDs, group, label, source, and partition, plus `results/split_audit_v1.json`. Freeze the manifest hash. The test partition must never drive checkpoint, gate, or hyperparameter selection.

**E1 gate:** zero known train/validation/test group overlap and an auditable checkpoint lineage. If this is impossible, use a truly independent external evaluation set and restrict the paper's population claim.

## 4. E2: one controlled vision experiment

**Question:** Does F2 improve errors that matter for CropQuery over ordinary continued fine-tuning on identical images and initialization?

| Item | Specification |
|---|---|
| Arms | CE-only continued fine-tuning and CE + $0.10\,\mathrm{KL}(p_{\rm strong}^{\rm stop},p_{\rm weak})$. Use the same Phase D starting weights *only if they were trained without the new test images*. With the currently supplied materials, train one common supervised Phase D base from ImageNet initialization on the frozen train split, then branch identical copies into the two continuation arms. Never use the new test partition to select this base. |
| Controlled factors | One seed (42), same weak/strong augmentation definitions, image order, batch size, optimizer groups, LR schedule, fixed maxima (six base epochs and four continuation epochs per arm), best-epoch selection by validation top-1, validation metric, model class map, and hardware. Use 32-image batches, AdamW with layerwise rates, cosine annealing, label smoothing 0.1, gradient clipping 1.0, and seed 42; these are fixed before opening the new test split. The six/four-epoch schedule is a deadline-bounded single-backbone mechanism screen, not a convergence or ensemble claim. |
| Smoke check | Import/model construction, checkpoint load, one batch through both losses, gradient finiteness, and one validation batch. No full training for a shape check. |
| Run | First train the two ResNet50 arms as a small mechanism screen. If the clean validation result is unpromising, stop before the other backbones and report this as exploratory. If it warrants an ensemble test, train EfficientNet-B3 and DenseNet121 for both arms as resumable, separate jobs. Use new run directories; never run the notebook's stale-checkpoint deletion cell. Save model, optimizer, scheduler, AMP scaler, epoch/step, best metric, RNG state, class map, split hash, and config. Launch via project script with `nohup`, timestamped `logs/`, and exact command/config. |
| Primary analysis | On the untouched test images, paired top-1 difference and counts `CE wrong → F2 right` / `CE right → F2 wrong`. Report a paired bootstrap interval by image/source group or an exact paired test as appropriate. |
| Secondary analysis | Macro F1, each Cassava-class recall/F1 and support, crop-stratified errors, and an error-pair table. Do not select classes after reading test results except for clearly marked exploratory analysis. |
| Mechanism check | If F2 improves, a single additional two-view CE control may be needed to separate consistency from the extra strong-view computation. Run it only if that distinction is central to the paper claim. |

Evaluate the already specified FAISS rule ($k=7$, gate 0.80) on the **same** clean test set and compare it with no override separately for each checkpoint family. No threshold sweep on test. If override is neutral or harmful, leave it disabled and state that result. Preserve per-image predictions and probabilities under `results/vision_v1/`, with `summary.json` and `error_pairs.csv`.

**E2 stop rule:** if F2 does not improve a valid held-out outcome or its apparent gain comes only from an unpaired comparison, do not present it as the main novelty. No multi-seed run is planned; add seeds only if a reviewer explicitly requests them.

## 5. E3: independent image-to-advice assessment

**Question:** What errors appear when CropQuery receives a real image and question without being told the diagnosis?

The supplied Kisan Q&A corpus supports rebuilding that retrieval branch. The 39 curated entries embedded in `notebooks/cropquery-v2.ipynb` cell 11 support rebuilding the v2 notebook's second branch, despite the separate Hugging Face AgroBench repository being inaccessible. Pin the embedded list's hash and record which implementation and knowledge revisions each case uses. The submitted paper's 38-entry version remains unresolved, so describe the reconstructed v2 evaluation as new evidence rather than a direct replication of that submitted result.

Select two case studies *before* seeing system outputs: (1) a visually ambiguous disease/healthy case, preferably a Cassava confusion, and (2) a pest/disease or treatment-category conflict. Each case needs an independently sourced image, provenance and reuse rights, a farmer-style question, a diagnosis supported by an expert or authoritative source, and an independently prepared treatment reference. The reference must not be copied from the indexed AgroBench-style treatment entries. If a case cannot meet these conditions, replace it before evaluation.

For each case, save the image identifier (not necessarily redistribute the image), input language, classifier probabilities, retrieved passages with source IDs, complete English advice, translated advice, any warning, and failure modes. Ask at least one agronomist to judge diagnosis plausibility, source support of each actionable treatment, wrong treatment category, dosage/chemical-name correctness, and appropriate deferral; resolve uncertainty explicitly. A native speaker should review the tested translated outputs. The experimental unit is the **case**, not each generated sentence. Do not count the existing Pumpkin screenshot as an assessed case without independent review.

Run a small deterministic translation check for the ten supported Indian languages using fixed chemical-name/dose strings and unit formats. Record exact preservation and failures; this tests string handling only. Make usability claims only for languages assessed by speakers. Avoid “safe” or “verified” wording unless the corresponding agronomic and translation endpoints support it.

Deliver `results/advisory_cases/case_01.json`, `case_02.json`, an adjudication record, and two narrative case analyses that show both successful and failed stages. ROUGE can be included only as a secondary lexical statistic.

## 6. E4: targeted comparator only if E3 motivates it

If E3 demonstrates that a wrong or uncertain class routes retrieval to unsupported treatment, specify a minimal answer/qualify/defer rule on development cases. Freeze its confidence/evidence thresholds **without looking at final cases**. Compare the current pipeline and the revised rule on the same fixed set of independent image-question cases, generated with the same model and decoding settings. An initial set of 24 cases balanced across ambiguous diagnosis and treatment-category conflicts is a targeted pilot, not a basis for broad population claims. Two detailed cases may be selected from this set for presentation, but must not be selected only because the new rule succeeds.

Primary endpoint: case-level presence of any unsupported actionable treatment recommendation. Secondary endpoints: correct/qualified diagnosis, source coverage, inappropriate deferral, and translation preservation. Use blinded paired judgments; report discordant case counts and an interval, with nested claim-level notes as descriptive data. If a 24-case interval is too wide to support the intended claim, state the uncertainty and expand only as scientifically needed. If E3 does not reveal the hypothesized failure, do not implement E4; report the observed system strengths/limits instead.

## 7. E5: external field check and paper gate

The prospective clean split supports an internal held-out estimate. Add an external field-image check only if the paper claims transfer beyond the nine source datasets. The original [PlantDoc classification dataset](https://github.com/pratikkayal/PlantDoc-Dataset/blob/master/README.md) is a candidate (2,598 images; real-world, internet-sourced; CC BY 4.0), not yet selected. Before evaluation, freeze exact crop-condition label mapping to CropQuery's 64 labels, list excluded/unmapped classes, and screen exact/near overlap against the training corpus. Report accuracy only on mapped, nonoverlapping images with a denominator and class coverage; never compare it as if it were the 64-class internal test. If mapping is too narrow or source overlap cannot be checked, choose another independent set or restrict the claim.

The paper is ready to rewrite as a journal result only when: (a) every headline metric has an auditable population and checkpoint, (b) the central method beats its matched comparator or a negative result is interpreted honestly, (c) at least two nontrivial cases have independent adjudication, (d) treatment/translation claims match the assessed scope, and (e) the related-work comparison identifies an actual methodological distinction. No additional ablations or seeds are automatic gates.

## 8. Document and artifact update rule

After each phase, update `docs/walkthrough.md` with phase, latest verified result, active job, blocker, and one immediate next action. Keep exact commands and detailed stdout in timestamped `logs/`; store metrics and per-case records in `results/`. Amend this file when a gate changes the scientific direction, and revise `implementation_plan.md` only at phase boundaries. Distinguish `planned`, `implemented`, `smoke-tested`, `launched`, `completed`, and `analyzed` in every handoff.

## Method source

This plan used the scientific-writing, scientific-critical-thinking, experimental-design, and get-available-resources skills: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv:2609.00065](https://arxiv.org/abs/2609.00065). The citation documents planning assistance, not empirical support for CropQuery.
