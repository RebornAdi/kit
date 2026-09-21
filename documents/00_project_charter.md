# Project Charter & Decision Record

**Project working title:** Detecting Leakage from Informative Missingness: A Cross-Domain Benchmark

**Authors:** Aditya Deshmukh (23BAI0072), Ishita Babar (23BAI0080)

**Document status:** Living document. This is Document #1 of the project record. Every later decision should trace back to, or amend, a decision logged here.

**Last updated:** 18 September 2026

---

## 1. One-paragraph summary (the elevator pitch)

When a machine learning model is trained on data with missing values, the *pattern of what is missing* can itself carry information about the target. This is called informative missingness. Sometimes relying on it is legitimate (the model will deploy in the same setting that produced the missingness), and sometimes it is leakage (the pattern will not hold at deployment, so the model's reported accuracy is a mirage). Every existing leakage-detection tool defers this legitimate-vs-leakage judgment to a human. This project formalizes leakage-via-informative-missingness as a distinct, detectable leakage type and builds the first cross-domain benchmark, with ground-truth labels, that lets detection tools be compared on whether they can tell the two apart.

---

## 2. Finalized decisions (locked)

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | Primary contribution | **Benchmark-first** | The automatic separator is the hard, possibly partly-unsolvable core. Building the benchmark first guarantees a real contribution regardless of whether the separator succeeds. The separator becomes upside, not make-or-break. |
| D2 | Target venue | **Workshop first, then extend** to NeurIPS Datasets & Benchmarks | A workshop gets feedback early, establishes a citable timestamp on the idea, and de-risks the larger submission. The benchmark is the natural spine of a later D&B-track paper. |
| D3 | Domain coverage | **Cross-domain, healthcare-anchored** (healthcare + finance; optional third) | The core claim is the word "general." Healthcare alone invites the criticism that this is a known medical-data quirk. Healthcare gives the clearest intuition; a second domain (finance) defends generality. Two–three domains is enough for a workshop; more can wait for the extension. |

---

## 3. The problem (plain statement)

A model can achieve near-perfect test accuracy by noticing *which columns are blank* rather than what any measured value is. Whether that is acceptable depends entirely on **when and where the data was recorded** — a judgment current tools cannot make automatically.

- **Legitimate informative missingness:** the missingness mechanism will be present, unchanged, at prediction time (same workflow/deployment). The feature is a fair, deployable signal. → keep.
- **Leakage-inducing informative missingness:** the missingness pattern reflects something that will not be available or will not hold at deployment (e.g., it encodes a post-outcome decision, or won't transfer to another site). → remove.

The two produce the *same statistical signature* but demand *opposite actions*. Separating them is the open problem.

---

## 4. Why this is novel (as of Sept 2026 searches)

Established and recent work either (a) defines leakage in general, (b) demonstrates missingness-encoded leakage in a single domain, or (c) builds detectors that flag suspicious features but hand the legitimate-vs-leakage call to a human. None provides a general, cross-domain, ground-truth benchmark for this specific leakage type, nor an automatic separator.

**Caveat (logged, must be re-checked):** This is a fast-moving area. Several of the closest works are from mid-2026 and one detector shipped in Sept 2026. The novelty check MUST be re-run (a) when drafting begins and (b) before submission. See Document on Related Work for the live citation list.

---

## 5. Contributions we will claim

1. **Formalization** — define leakage-via-informative-missingness as a distinct leakage type, with a precise, testable definition and a taxonomy of sub-cases.
2. **Benchmark** — the first cross-domain suite of tabular datasets with *injected, ground-truth-labeled* missingness features of each type (legitimate / leakage / neutral).
3. **Baseline evaluation** — run existing detectors on the benchmark and quantify where they succeed (detecting presence) and fail (identifying type).
4. **(Upside) Separator** — a first automatic method that attempts the legitimate-vs-leakage split, evaluated against the baselines on the benchmark.

Framing note: existing tools (targetleak, dataleaks, Cleanlab, Leakage Sentinel) are **baselines we evaluate**, not competitors. We gain from their existence.

---

## 6. Scope boundaries (what we are NOT doing)

- Not building a production tool or CI plugin.
- Not solving leakage in general — only the informative-missingness type.
- Not covering images / text / graphs — tabular only for the workshop version.
- Not claiming to fully automate the human judgment — a partial, measured separator is an acceptable and honest outcome.

---

## 7. Risk register (initial)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Someone publishes the same benchmark first | Medium | High | Re-run novelty checks at draft and submission; move quickly; workshop timestamp. |
| The automatic separator proves human-only in principle | Medium | Medium | Benchmark-first design means the paper stands without it. Report the negative result honestly. |
| Injected leaks judged "unrealistic" by reviewers | Medium | Medium | Ground every injection recipe in a documented real-world case (healthcare test-ordering, finance field-gating). |
| Scope creep into too many domains/datasets | Medium | Low | D3 caps the workshop version at 2–3 domains. |

---

## 8. Candidate titles (to choose from)

Working shortlist — to be narrowed once the benchmark takes shape:

1. *Detecting Leakage from Informative Missingness: A Cross-Domain Benchmark* (clear, descriptive — current default)
2. *When Absence Speaks: Benchmarking Leakage-via-Informative-Missingness in Tabular ML*
3. *Missing on Purpose: A Benchmark for Legitimate versus Leakage-Inducing Informative Missingness*
4. *The Missingness Leakage Benchmark: Telling Deployable Signals from Illusions*

Default for now: **#1** (safe and self-explanatory for a workshop). #3's subtitle captures the core distinction best and may become the extended-version title.

---

## 9. Immediate next steps (not started yet — for the next session)

1. Write the Related Work document with the full, dated citation list (the works found in the novelty searches).
2. Draft the formal definition + taxonomy of informative-missingness leakage.
3. Specify the benchmark: candidate datasets per domain, injection recipes per type, ground-truth labeling scheme.
4. List baselines and the exact evaluation metrics.
5. Set up the repository structure and documentation conventions.
