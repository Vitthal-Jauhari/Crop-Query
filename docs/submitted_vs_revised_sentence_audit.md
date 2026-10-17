# Submitted versus revised manuscript: sentence and paragraph audit

**Date:** 15 October 2026. **Sources:** `submitted_paper/crop_query_v2.tex` (the six-page DELCON manuscript source), `submitted_paper/main_paper.pdf`, the supplied `submitted_paper/architecture.pdf`, and `papers/sisimpact_2027/main.tex`.

## How to read the comparison

The companion [`submitted_sentence_comparison.csv`](submitted_sentence_comparison.csv) has one row for each of 91 original prose units from the abstract through the conclusion. It records the old text, disposition, sentence-level reason, and revised section. TeX commands, citation syntax, math markup, and line wrapping are normalized for reading; exact wording remains in `crop_query_v2.tex`. The CSV excludes author metadata, keywords, captions, tables, and equations. The submitted document has some run-on material and itemized contribution fragments, so a “unit” is not always a grammatically complete sentence. This audit covers those separately below. Revised paragraphs are reorganized rather than forced into an artificial one-to-one sentence match.

## Paragraph-by-paragraph editorial comparison

| Submitted passage | Revised passage | Why the change matters |
|---|---|---|
| Abstract (S001–S006): India loss prevalence, first-of-kind scope, 96.92/97.83% ensemble accuracy, ROUGE/GPT-4o, verified safety | Abstract: visual-to-advice evidence question; clean ResNet50 CE–F2 contrast; bounded archived text-path observation | The old prevalence, priority, held-out, comparator, and treatment-safety claims exceed the inspected evidence. The new abstract reports what was actually tested. |
| Introduction motivation (S007–S012): unsupported market and field-transfer absolutes | Introduction opening: why a visual label can steer retrieved advice; cites adjacent systems | Defines the concrete failure pathway and avoids claiming no comparable system exists. |
| Introduction mechanism/contributions (S013–S023): “verified” diagnosis, strict RAG constraint, active FAISS fallback, safety validator, four component claims | Introduction: three bounded contributions and two separate research questions | The novelty is an evidence analysis of this prototype, not a priority claim for combining standard components. |
| Related work (S024–S041): unmatched accuracy ranking and “first system” language | Related Work and Scope: AgriDoctor, Mondal et al., AgriRAG, AgroBench, and a no-ranking statement | Positions the manuscript against closer vision-to-advice work without inferring superiority from incompatible benchmarks. |
| Corpus/vision design (S042–S056): old counts, post hoc held-out label, +0.91 pp FAISS gain | Evidence Path plus Evaluation Method: code-aligned ensemble equation, FAISS override disabled, frozen 132,955-image source mapping and 19,923-image test | Separates historical notebook output from the new, prospectively frozen single-backbone experiment. |
| Retrieval/advice design (S057–S066): 38 entries, verified KB, strict prompt, dose-safe validator/translation | Evidence Path and discrepancy Table I: 39 embedded entries, 178,815 accepted Kisan rows, optional fallbacks, warning-only heuristic | Readers can see what the code actually does and which safety endpoints remain untested. |
| Experimental and result passages (S067–S088): old ablation, threshold sweep, cross-paper ranking, ROUGE explanation, Cassava causal story | Evaluation Method/Findings: matched CE–F2 screen, paired counts, bootstrap interval, source graph, error-type and Cassava tables, limited text-path term recall | Replaces unsupported causal or comparative interpretations with auditable denominators and explicit uncertainty boundaries. |
| Conclusion (S089–S091): validated multilingual treatment and proposed F2 improvement | Conclusion: evidence-flow finding, negative F2 screen, real-image adjudication as next test | The paper ends at the level of evidence actually collected. |

## Wording and sentence-level corrections applied

1. The abstract now says that the archived probe used **supplied diagnoses and blank images**, avoiding any impression that the end-to-end image pathway was tested.
2. “Verified disease label,” “verified documents,” “strictly constrained,” and “Safety Validation” were removed as outcome claims. The text calls these a classifier output, public/embedded passages, prompt instructions, and an appended warning heuristic.
3. The manuscript distinguishes a current `k=7` FAISS similarity query with label override disabled from old notebook and diagrammed voting claims. Neither old +0.91 pp gain nor 97.83% accuracy is presented as a new validated result.
4. The Rice trace is described as **different stated rates**, with product formulations unverified. The rates cannot be compared as equivalent active-ingredient doses. The Mango trace is described as different source alternatives, without deciding which is correct.
5. “Conflicting advice” was narrowed to **source differences/divergence** where the available record does not establish a clinical or agronomic contradiction.
6. The paired vision result explicitly names the model, frozen test denominator, one seed, CE control, negative F2 difference, and what the group-bootstrap interval does and does not cover.
7. The former “patient” wording was corrected to “plant,” grammar and duplicated “conditions” from the source were not carried over, and figures/tables are introduced with their evidentiary role.

## Figures, tables, and math compared separately

The DELCON TeX includes `cropquery_system_architecture.drawio.pdf` as its architecture figure. The separate supplied **`architecture.pdf` is a different, more detailed A3 asset**; it is present in `submitted_paper/` but was not the `\includegraphics` target in `crop_query_v2.tex`. The revised paper now includes that requested detailed asset as Fig. 1, preserves it byte-identically under `papers/sisimpact_2027/figures/`, and provides Table I to identify the `k=5` vote, 38-entry, validation-box, and device/runtime discrepancies. Fig. 2 is a code-aligned diagram, so the old drawing is not mistaken for a tested architecture. The original Pumpkin interface is retained with an illustration-only caption.

The old ablation bars, confusion plot, threshold sweep, cross-paper accuracy table, and advisory ROUGE table are excluded from the revised result figures/tables because their populations or comparators do not support the original claims. The revised source-delta graph and quantitative tables read the frozen test analysis. The two qualitative trace rows read executed notebook outputs, not an agronomist reference. The four revised equations formalize the observed implementation, archived term diagnostic, tested F2 loss, and paired bootstrap; none is presented as a new mathematical method.

## Remaining submission issue

The revised PDF is an **eight-page working draft**. The official SISIMPACT initial-submission call states a six-page maximum including references. The fuller draft preserves the requested material for author review, but it is not a compliant initial-submission file. Reviewer A's independently adjudicated real-image advice cases also remain uncompleted; the two notebook traces answer a narrower text-path question.
