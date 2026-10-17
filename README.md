# CropQuery

CropQuery is a research prototype that connects crop-leaf image classification to retrieval, generated advice, and translation. This repository contains the original application code and notebooks, plus a new controlled vision experiment. The mobile frontend is [CropQuery-Mobile](https://github.com/Vitthal-Jauhari/CropQuery-Mobile).

## What is implemented

The application code combines ResNet50, EfficientNet-B3, and DenseNet121 probabilities over 64 crop-condition labels. It can query a FAISS similar-image index when confidence is below 80%; the current configuration uses `k=7` and **does not replace the ensemble label** with the FAISS vote. Retrieval draws from local crop-condition entries and Kisan Call Centre Q&A, then LLaVA-Next generates English advice. Translation can follow. The post-generation check appends a warning for certain pest/fungicide combinations; it does not validate dose or guarantee treatment safety.

The current pinned Kisan source has 178,939 rows, of which the loader accepts 178,815 Q&A pairs. The version-2 notebook embeds 39 local crop-condition entries, giving a reconstructed 178,854-document count. This does not establish that the historical run used the same source revision or that the advice is agronomically correct. See [source inventory](results/source_inventory_v1.json) and [Kisan count audit](results/kisan_count_audit.json).

## New controlled experiment (October 2026)

We asked whether F2 consistency continuation improves a ResNet50 classifier over matched cross-entropy (CE) continuation. Before training, we froze a new split of the nine public image datasets supplied with the project. The mapped corpus has **132,955 images in 64 classes**; 65 images in conflicting-label groups were excluded. The split has 106,332 training, 6,635 validation, and **19,923 test** images. Exact SHA-256, image-hash, and source-local filename-family groups were kept within one partition. This is a lightweight overlap check, not proof of plant-level independence.

One seed (42) trained a common ImageNet-initialized ResNet50 base for six epochs. Identical copies then received four matched continuation epochs using CE alone or CE plus the F2 consistency term. Checkpoints were chosen on validation data; the frozen test set was read once after both continuations finished.

| Checkpoint | Test top-1 | Test macro F1 |
|---|---:|---:|
| Shared base | 95.12% | 93.61% |
| CE continuation | **95.36%** | **93.71%** |
| CE + F2 continuation | 94.98% | 92.92% |

**F2 minus CE was −0.386 percentage points** on the same 19,923 test images. F2 corrected 182 CE errors and introduced 259 new errors. A 2,000-replicate bootstrap over 16,864 test groups gave a 95% interval of **[−0.614, −0.160] percentage points**. This interval covers test-group resampling only; it does not cover training-seed or external-dataset uncertainty. Under this one-seed schedule, the F2 improvement hypothesis was not supported. This experiment tested **ResNet50 only**, not the original three-backbone ensemble or the complete advisory pipeline.

The original notebooks report larger ensemble and FAISS figures, but their test exclusion and final checkpoint lineage could not be verified. They should not be read as directly comparable held-out results. The old GPT-4o ROUGE value was not a matched baseline.

## Archived advice-path diagnostic

We recalculated the stored notebook's 19-scenario selected-term table. Mean term recall was **0.947 in retrieved text** and **0.763 in generated answers**; three scenarios lost all selected terms after generation and one lost some. The notebook supplied the diagnosis, used a blank image, and derived reference terms from indexed entries. These values measure term transmission in that archived probe, **not** treatment correctness, safety, multilingual usability, or image-to-advice performance. Rice Leaf Blast and Mango Anthracnose traces show differing retrieved treatment statements, but neither case has independent agronomist adjudication.

## Evidence and code

| Item | Location |
|---|---|
| Frozen split and audit | [manifest](data/split_manifest_v1.csv), [split audit](results/split_audit_v1.json) |
| Vision results and paired predictions | [summary](results/vision_clean_resnet_v1/summary.json), [analysis](results/vision_clean_resnet_v1/analysis.md), [predictions](results/vision_clean_resnet_v1/paired_predictions.csv) |
| Archived text-path audit | [text probe audit](results/archived_text_probe_audit.json) |
| Experiment scripts | [`scripts/prepare_clean_split.py`](scripts/prepare_clean_split.py), [`scripts/train_clean_resnet.py`](scripts/train_clean_resnet.py), [`scripts/analyze_vision_screen.py`](scripts/analyze_vision_screen.py) |
| Study decisions and limitations | [implementation details](docs/implementation_details.md), [walkthrough](docs/walkthrough.md) |

The split manifest SHA-256 is `97414fc5d2fac35463eadc88b88473073a35ac22b31d1f82eb595a36fd8e7ee0`. Images are kept under the shared `/workspace/datasets/cropquery_sources` directory on the research machine rather than duplicated in Git. Resumable training checkpoints remain local under `checkpoints/clean_resnet_v1/`; they are not included in this branch. Re-running training requires those source images, an NVIDIA GPU, and a compatible PyTorch environment. The experiment script stages are `smoke`, `base`, `ce`, `f2`, and `evaluate`; run the smoke stage before a full job. The exact completed protocol and job logs are described in [implementation details](docs/implementation_details.md).

## Current limits

- No independently adjudicated real-image, image-to-advice cases or translated-output assessment are complete.
- The prospective accuracy estimate covers the nine-source internal mixture and one ResNet50 seed; it does not establish field performance or ensemble accuracy.
- The current FAISS override is disabled, and the warning heuristic is not a verified treatment-safety system.
- Manuscript files are intentionally excluded from this branch.

CropQuery was developed under the PRISM AI Mentorship Program. Its public inputs are the nine [catalogued image datasets](results/source_inventory_v1.json) and the [Kisan Call Centre dataset](https://huggingface.co/datasets/Franklin01/Dataset_prep01).
