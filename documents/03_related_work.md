# Related Work & Citation Catalogue

**Document #3 of the project record.** Depends on: `00_project_charter.md`, `02_definitions_and_taxonomy.md`.
**Purpose:** A verified, dated, link-carrying catalogue of prior work, organized so each entry states plainly *what it does* and *why it does not close our gap*. This doubles as the source list for the paper's Related Work section.

**Last updated:** 18 September 2026

---

## ⚠️ Standing novelty-check note (read every time)

This area is moving **fast** — several key items below are from **2026**, and new leakage-detection packages are appearing on PyPI almost monthly. The catalogue below is a **snapshot as of 18 Sept 2026**. Per the charter, re-run these checks (a) when drafting begins and (b) immediately before submission. A material finding from *this very session* is logged in §6 (the `featureleak` package now lists "missingness leakage" as a named check) — proof that the snapshot ages quickly.

**Link caveat:** Links were verified during this session. Some earlier-round items (marked ⧗) came from search snippets and should be re-opened and confirmed before they go in the paper's bibliography.

---

## 1. Foundations: missing-data theory (we build ON these, we do not claim them)

| Work | What it gives us | Link |
|------|------------------|------|
| Rubin, D. B. (1976). *Inference and Missing Data.* Biometrika 63(3), 581–592. | The origin of MCAR/MAR/MNAR and the missing-indicator construction. Our formal layer sits on this. | https://doi.org/10.1093/biomet/63.3.581 |
| Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd ed. Wiley. | Standard textbook reference for missingness mechanisms. | https://doi.org/10.1002/9781119482260 |
| van Buuren, S. (2018). *Flexible Imputation of Missing Data*, 2nd ed. | Widely-cited practical reference; missing-indicator method. | https://stefvanbuuren.name/fimd/ |

**Why they don't close our gap:** these define *why data is missing* and how to impute it. None frames the *deployment-availability* question (is the missingness pattern legitimately usable at prediction time?) — our axis.

---

## 2. Foundations: data leakage (our "leakage" case instantiates these)

| Work | What it gives us | Link |
|------|------------------|------|
| Kaufman, Rosset, Perlich & Stitelman (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance.* ACM TKDD 6(4), Art. 15. | The canonical definition: leakage = using information not legitimately available at prediction time. Our "leakage-inducing missingness" is an instance. | https://doi.org/10.1145/2382577.2382579 |
| Kapoor & Narayanan (2023). *Leakage and the Reproducibility Crisis in Machine-Learning-Based Science.* Patterns 4(9), 100804. | Evidence of leakage across 294 papers / 17 fields; motivates why this matters. | https://doi.org/10.1016/j.patter.2023.100804 |

**Why they don't close our gap:** they define/measure leakage in general. Neither treats informative missingness as a distinct leakage type, nor separates legitimate from leakage-inducing missingness, nor provides a benchmark for it.

---

## 3. Informative missingness demonstrated (single-domain; not general, no benchmark)

| Work | What it shows | Link / status |
|------|---------------|---------------|
| DACr algorithm paper (2026), medRxiv. | Names "informative absence" with exactly our clinical example (clinician doesn't order a test because the patient looks healthy); uses it for imputation/causal estimation, **not** leakage typing. | https://www.medrxiv.org/content/10.64898/2026.01.20.26343799 (⧗ confirm final DOI) |
| MDLA / missingness-as-demographic-leakage audit (2026), medRxiv. | Missingness patterns *alone* predict outcomes/attributes; framed as fairness/bias in **one** clinical setting. | ⧗ re-locate exact URL before citing |
| PCOS adversarial-validation study (2026), medRxiv. | Missingness pattern alone gives ROC-AUC ≈ 1.000; explicitly states the legitimate-vs-leakage distinction; validates on a **semi-synthetic** experiment with known ground truth (format precedent for us). | ⧗ re-locate exact URL before citing |

**Why they don't close our gap:** each is single-domain, one dataset, framed for fairness or imputation. None is a general, cross-domain, typed benchmark, and none proposes an automatic legitimate-vs-leakage separator.

---

## 4. Spurious correlation / shortcut / robustness (adjacent problem, different axis)

| Work | What it does | Link |
|------|--------------|------|
| Geirhos et al. (2020). *Shortcut Learning in Deep Neural Networks.* Nat. Mach. Intell. 2, 665–673. | Frames shortcut learning broadly. | https://doi.org/10.1038/s42256-020-00257-z |
| Sagawa et al. (2020). *An Investigation of Why Overparameterization Exacerbates Spurious Correlations.* ICML. | Spurious correlations hurt worst-group accuracy. | https://arxiv.org/abs/2005.04345 |
| Liu et al. (2021). *Just Train Twice (JTT).* ICML. | Robustness without full group labels. | https://arxiv.org/abs/2107.09044 |
| Arjovsky et al. (2019). *Invariant Risk Minimization.* | Learn relationships stable across environments — conceptual basis for our "stability" idea. | https://arxiv.org/abs/1907.02893 |
| Eyuboglu et al. (2022). *Domino: Slice Discovery with Cross-Modal Embeddings.* ICLR. | Finds error slices hidden by high accuracy. | https://arxiv.org/abs/2203.14960 |

**Why they don't close our gap:** these target spurious features / distribution shift in general (mostly high-dimensional). None isolates *missingness* as the leaking channel, and none provides typed ground truth for it.

---

## 5. Explainability (tells you *what* a model uses, not *whether it's legitimate*)

| Work | What it does | Link |
|------|--------------|------|
| Ribeiro et al. (2016). *"Why Should I Trust You?" (LIME).* KDD. | Model-agnostic local explanations. | https://doi.org/10.1145/2939672.2939778 |
| Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS. | Shapley-value feature attribution. | https://arxiv.org/abs/1705.07874 |

**Why they don't close our gap:** a high attribution to a missingness-indicator says the model *uses* it — not whether that use is legitimate or leakage. That judgment is exactly what we formalize.

---

## 6. Shipped leakage-detection tools (THESE ARE OUR BASELINES)

The critical section. Each detects leakage to some degree; we will run them on our benchmark. Note the recurring, self-admitted limitation: **they defer the legitimate-vs-leakage / temporal-availability judgment to a human, or detect only presence.**

| Tool | Released | What it does re: missingness | Link |
|------|----------|------------------------------|------|
| **featureleak** (v0.1.0) | **Mar 17, 2026** | ⚠️ **Directly relevant.** Lists "Missingness Leakage: patterns of missing data that reveal the target" as one of 10+ checks. But: tiny package, zero-config, correlation-threshold detection of *presence only*; no legitimate-vs-leakage split, no benchmark, no validation. | https://pypi.org/project/featureleak/ |
| **targetleak** / TargetLeak | 2026 | Finds columns that nearly solve the target; classifies into check types; **ships its own inject-the-leak benchmark** (401/432 planted leaks across 20 datasets). States a leak-vs-valid call "depends on when measurements are recorded... which a domain expert can answer and this tool cannot." | https://github.com/ShriyansBharuka/TargetLeak |
| **dataleaks** (v0.1.0) | 2026 | Temporal + conditional-timestamp-presence checks. **Explicitly refuses** to guess temporal availability: "prefers explicit, explainable evidence over unsupported assumptions." | https://pypi.org/project/dataleaks/0.1.0/ |
| **FeatureLeakageLens** | 2026 | Six pre-training checks incl. post-outcome name heuristics, target correlation, categorical target-rate proxies. | https://pypi.org/project/featureleakagelens/ |
| **LeakProfiler** | 2026 | Cross-detector reasoning (correlation + importance → proxy consensus). Detects presence; no missingness-type separation. | https://pypi.org/project/leakprofiler/ |
| **leaklint** | 2026 | Leaky/target-proxy via AUC/correlation/MI; preprocessing & CV-fold leakage. | https://pypi.org/project/leaklint/ |
| **tsauditor** | 2026 | Time-series as-of / point-in-time leakage; states plainly that "a genuine strong predictor and a leak are distinguishable only by magnitude" — echoing our core claim. | https://pypi.org/project/tsauditor/ |
| **tabular-leakage-detector** | 2026 | Forensic audit; outputs ranked candidates for human review, "not automated verdicts — deliberately." Uses hard-negative controls (strong-but-legit clinical markers). | https://github.com/farhaanqazi/tabular-leakage-detector |
| Cleanlab **Datalab** | ongoing | Detects spurious correlations between labels and features. | https://github.com/cleanlab/cleanlab |

**Why they don't close our gap (this is the paragraph the paper hinges on):**
Missingness-leakage is now sometimes *named* (featureleak) and sometimes *detected as presence* (several), but across every tool the *legitimate-vs-leakage-inducing* decision is either (a) deferred to a human by design (dataleaks, tabular-leakage-detector, targetleak), or (b) not attempted at all. **No tool validates its missingness check against ground truth, and no shared benchmark exists to measure whether these checks actually work or merely fire.** That absence is our contribution.

---

## 7. Output-only / blind leakage detection (related method, different input)

| Work | What it does | Link / status |
|------|--------------|---------------|
| "Prior-free blind detection of information leakage" (2026), arXiv. | Detects leakage from model prediction vectors; admits output-only detection fails where residual leakage is indistinguishable from an honestly stronger predictor **without an external benchmark**. | ⧗ re-locate exact arXiv ID before citing |

**Why it doesn't close our gap — and why it helps us:** it works from outputs, not feature-level missingness, and it explicitly argues that separating leak from strong-but-legit needs *an external benchmark* — which is precisely what we build.

---

## 8. Documentation / transparency standards (context, not competitors)

| Work | Relevance | Link |
|------|-----------|------|
| Mitchell et al. (2019). *Model Cards for Model Reporting.* FAT*. | Standard for reporting model behavior/limits. | https://doi.org/10.1145/3287560.3287596 |
| Gebru et al. (2021). *Datasheets for Datasets.* CACM 64(12). | We will ship a datasheet with the benchmark for reproducibility. | https://doi.org/10.1145/3458723 |

---

## 9. One-paragraph positioning statement (draft for the paper)

> Informative missingness — where the pattern of missing values itself predicts the target — is well documented in the missing-data literature (Rubin, 1976) and increasingly noted as a leakage risk: recent tools such as featureleak (2026) even list it as a named check. Yet across the current generation of leakage detectors, the decisive judgment — whether an informative missingness signal is *legitimate* (stable and available at deployment) or *leakage-inducing* (an artifact of the training environment) — is either deferred to a human by design (dataleaks, TargetLeak, tabular-leakage-detector) or not attempted. No shared, ground-truth benchmark exists to measure whether these checks succeed or merely fire, and single-domain demonstrations (2026 clinical audits) have not been generalized. We close this gap with (1) a formalization of leakage-via-informative-missingness that adds a deployment-availability axis to the classical missingness taxonomy, (2) the first cross-domain benchmark with typed, ground-truth-labeled missingness features, and (3) a baseline evaluation quantifying where existing detectors succeed (presence) and fail (type).

---

## 10. To-do before this becomes the paper's bibliography

- [ ] Re-open every ⧗ item and capture exact DOI/arXiv ID + date.
- [ ] Pull the 3 clinical medRxiv papers in full; confirm each is single-domain and cite precisely.
- [ ] Install and actually run `featureleak`'s missingness check on a toy case; document exactly what it does, so our "presence-only" claim is evidenced, not asserted.
- [ ] Check PyPI/arXiv once more for any newer "missingness leakage benchmark" before submission.
- [ ] Convert to BibTeX.
