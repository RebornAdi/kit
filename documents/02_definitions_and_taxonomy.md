# Formal Definitions & Taxonomy

**Document #2 of the project record.** Depends on: `00_project_charter.md`.
**Purpose:** Fix precise, testable definitions for *leakage-via-informative-missingness* so that every benchmark injection recipe and evaluation metric has an unambiguous formal referent.

**Last updated:** 18 September 2026

---

## 0. Reading guide

Each concept is given twice: a **plain statement** (the intuition) and a **formal statement** (the version we defend to reviewers). The formal layer builds on Rubin's standard missing-data taxonomy; our contribution is the *deployment-availability axis* laid on top of it (Section 4), not the underlying missingness statistics.

---

## 1. Preliminaries and notation

Let each example be $(X, Y)$ with target $Y$ and features $X = (X_1, \dots, X_p)$.

For a feature $X_j$ that may be unobserved, define its **missingness indicator**
$$M_j \in \{0,1\}, \quad M_j = 1 \text{ if } X_j \text{ is missing, } 0 \text{ otherwise.}$$

This is the standard *missing-indicator* construction (Rubin, 1976). The vector $M = (M_1, \dots, M_p)$ is the **missingness mask**.

We write $D_{\text{obs}}$ for observed data and $D_{\text{mis}}$ for missing data, following the usual partition induced by $M$.

**Deployment note.** We distinguish two phases: *training* (the data we hold) and *deployment* (where the model runs at prediction time). This split is central and is what separates our axis from the classical one.

---

## 2. The classical axis (Rubin's taxonomy) — used as-is, not our contribution

Standard mechanism classification for a feature's missingness:

- **MCAR (Missing Completely At Random):** $P(M_j = 1 \mid X, Y) = P(M_j = 1)$. Missingness independent of everything.
- **MAR (Missing At Random):** $P(M_j = 1 \mid X, Y) = P(M_j = 1 \mid X_{\text{obs}})$. Missingness explained by *observed* data.
- **MNAR (Missing Not At Random):** missingness depends on unobserved values or on $Y$ itself; cannot be reduced to the above.

We adopt these definitions unchanged. They are well-established and we cite them rather than claim them.

---

## 3. Informative missingness (the phenomenon, formalized)

**Plain:** The *fact that a value is missing* tells you something about the answer.

**Formal.** The missingness of $X_j$ is **informative (with respect to $Y$)** iff the mask carries target information:
$$P(Y \mid M_j) \neq P(Y), \quad \text{equivalently} \quad I(Y; M_j) > 0,$$
where $I(\cdot;\cdot)$ is mutual information.

**Relationship to the classical axis (important, and a likely reviewer question):**
- Informative missingness is *typically* associated with MNAR, but the two are **not identical**. A feature can be MAR yet still have $I(Y; M_j) > 0$ if the observed variables driving its missingness are themselves correlated with $Y$.
- Therefore we treat "informative" as a statement about $I(Y; M_j)$ directly, **not** as a synonym for MNAR. This avoids conflating the mechanism (why it's missing) with the consequence (that the mask predicts $Y$).

**Detection primitive.** A model can exploit informative missingness by using $M_j$ (explicitly, via a missing-indicator feature, or implicitly, via how an imputation interacts with the learner) to predict $Y$. In the extreme, $M$ alone can achieve high predictive performance with no use of any observed value — the failure mode our benchmark targets.

---

## 4. The deployment-availability axis (OUR contribution)

This is the new axis. It is orthogonal to Rubin's taxonomy: it asks not *why* a value is missing, but *whether the missingness mechanism will still hold, and be available, at deployment*.

Let $\mathcal{E}_{\text{train}}$ denote the data-generating environment of the training data and $\mathcal{E}_{\text{deploy}}$ the deployment environment.

**Definition 4.1 (Availability).** The missingness mask $M_j$ is **available at deployment** iff $M_j$ is observable at prediction time in $\mathcal{E}_{\text{deploy}}$ (i.e., whether $X_j$ is missing is known when the prediction must be made, not only in hindsight).

**Definition 4.2 (Stability).** The informative relationship is **stable** iff the dependence between mask and target is preserved from training to deployment:
$$P_{\mathcal{E}_{\text{train}}}(Y \mid M_j) \approx P_{\mathcal{E}_{\text{deploy}}}(Y \mid M_j).$$

**Definition 4.3 (Legitimate informative missingness).** $M_j$'s informativeness is **legitimate** iff it is *both available (4.1) and stable (4.2)* at deployment. Using $M_j$ is then a valid, deployable signal.

**Definition 4.4 (Leakage-inducing informative missingness).** $M_j$'s informativeness is **leakage-inducing** iff it is informative in training ($I(Y; M_j) > 0$) but **fails availability or stability** at deployment. The model's training/test performance is then optimistic relative to true deployment performance — i.e., leakage in the sense of Kaufman et al. (2012): use of information not legitimately available at prediction time.

**The core scientific claim (stated for the record):**
> Legitimate and leakage-inducing informative missingness can be **statistically indistinguishable in-sample**. They are separable only with information about $\mathcal{E}_{\text{deploy}}$ — availability and stability — which is not contained in the training sample itself. This is why existing tools defer the judgment to a human, and why an automatic separator must inject or infer deployment-context assumptions.

This claim is falsifiable: if an in-sample-only signal reliably separated the two on our benchmark, the claim would be refuted (an interesting result either way).

---

## 5. The taxonomy (four labeled cases used by the benchmark)

Every injected feature in the benchmark receives exactly one ground-truth label from this set:

| Label | Plain description | Formal condition | Correct action |
|-------|-------------------|------------------|----------------|
| **L — Legitimate** | Missingness is informative and will still be there, unchanged, at deployment | $I(Y;M_j)>0$, available (4.1) ∧ stable (4.2) | Keep |
| **K — Leakage (availability)** | The mask won't be knowable at prediction time (encodes a post-outcome fact) | $I(Y;M_j)>0$, **not** available (4.1) | Remove |
| **S — Leakage (stability/shift)** | The mask is knowable but its meaning won't transfer (site-/time-specific) | $I(Y;M_j)>0$, available but **not** stable (4.2) | Remove / re-validate |
| **N — Neutral** | Missingness carries no target information | $I(Y;M_j)\approx 0$ | Keep (harmless) |

Notes:
- **K vs. S** is a deliberately finer split than prior work attempts; reporting them separately lets us show *which* kind of leakage tools miss.
- The **same raw injection** can be labeled **L or K/S depending on the declared deployment scenario**. The benchmark therefore ships each such case *with its scenario annotation*; the label is defined relative to that scenario, never in a vacuum. This operationalizes the core claim in Section 4.

---

## 6. What a detector is asked to do (task definitions)

Given a dataset and a feature $X_j$ (with its scenario annotation), a tool may be evaluated on:

- **T1 — Detection:** predict whether $M_j$ is informative at all (binary: {L,K,S} vs. N). Tests presence.
- **T2 — Typing:** predict the full label in {L, K, S, N}. Tests the legitimate-vs-leakage judgment — the hard task.
- **T3 — Action:** map to {keep, remove/revalidate}. A coarsening of T2; reported for practical relevance.

Our hypothesis (to be tested, not assumed): existing tools do adequately on **T1** and near-chance on **T2**, because T2 requires deployment-context reasoning they defer to humans.

---

## 7. Assumptions and honest limitations

1. **Scenario dependence is a feature, not a bug.** Because L vs. K/S depends on declared deployment context, our labels are only meaningful *relative to a stated scenario*. We commit to always publishing the scenario with the label. A reviewer who dislikes this is really disliking the underlying reality (the ambiguity is real); we make it explicit rather than hiding it.
2. **Injection realism is a validity threat.** Each recipe (next document) must cite a documented real mechanism, or the benchmark is dismissed as artificial.
3. **We do not claim to fully automate T2.** Any separator we propose is evaluated honestly, including negative results.
4. **Formalization novelty is narrow and specific:** the deployment-availability axis (Sec. 4) and the four-way L/K/S/N taxonomy (Sec. 5). The underlying missingness statistics (Sec. 2–3) are standard and cited.

---

## 8. Citations to anchor this document (to be completed in the Related Work doc)

- Rubin (1976), *Inference and Missing Data*, Biometrika — origin of MCAR/MAR/MNAR and the missing-indicator idea.
- Little & Rubin (2019), *Statistical Analysis with Missing Data*, 3rd ed. — standard reference.
- Kaufman et al. (2012), *Leakage in Data Mining*, ACM TKDD — leakage = information not legitimately available at prediction time (the definition our "leakage-inducing" case instantiates).
- Kapoor & Narayanan (2023), *Leakage and the Reproducibility Crisis in ML-based Science*, Patterns — prevalence/impact of leakage.
- (Domain grounding for injection realism) — clinical "informative absence" sources and finance field-gating sources, to be catalogued in the Related Work document.

*Full, dated citation list lives in the Related Work document (next).*
