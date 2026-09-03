# AI-Powered Early Autism Screening — Research Codebase

A preliminary **screening aid** that estimates whether a child aged 4–11 would
benefit from formal autism assessment, and explains the reasoning behind each
result. This repository holds the data preprocessing, dataset auditing, model
training, calibration, evaluation and explainability code. The web application
lives elsewhere.

> **This is not a clinical diagnostic tool.** Its output is a recommendation to
> seek further assessment. Screening operates at population prevalence, where
> positive predictive value falls sharply even for instruments with high
> sensitivity and specificity — so performance on balanced research samples
> overstates field utility. Error costs are asymmetric, and the operating
> threshold is tuned toward sensitivity as an explicit, stated choice.

**Prediction target:** binary screen-positive / screen-negative.
**Cohort:** children aged 4–11, in both streams. Validated instruments differ by
age in item content and informant, so the two streams must declare the same band.

---

## The two streams

The repository is organised branch-first. The streams are trained, evaluated and
reported **independently** — no public dataset links questionnaire responses to
facial images of the same individuals, so there is no fusion layer and no
learned combination rule.

### `src/questionnaire/` — primary

Tabular classification over AQ-10-Child item responses. This stream is primary on
evidential grounds: its inputs derive from a validated instrument. Models are
selected by comparison rather than assumed — penalised logistic regression,
depth-limited decision tree, gradient boosting / random forest, and a small MLP.
Explanation is SHAP attribution over items, surfaced to the user, with stability
checked across resamples.

> ⚠️ **Circularity hazard.** The `result` column is the AQ-10 summed score and
> `Class/ASD` is a threshold applied to it. **`result` must be dropped**, or the
> task degenerates into recovering the scoring rule. Any accuracy above ~95%
> should trigger a leakage check before celebration.

### `src/images/` — secondary, gated

Facial image classification, **conditional on the Phase 0 audit passing**. The
dataset's cases and controls come from different sources, so a classifier can
score highly by learning acquisition context rather than facial phenotype. Three
probes decide whether this stream is a screening component or a documented
negative result — either outcome is a publishable finding:

| Probe | Method | Interpretation |
|---|---|---|
| Blur | Train on images blurred beyond visual recognisability | Above chance ⇒ signal is not facial |
| Background | Mask detected faces, train on the remainder | Above chance ⇒ signal is context |
| Metadata | Compare dimensions, file size, JPEG quality, colour stats, EXIF by class | Systematic difference ⇒ separate acquisition |

Perceptual-hash deduplication runs across classes alongside these. Grad-CAM is
retained as an **internal diagnostic only, never user-facing** — a heat map over
a child's face marks image regions, not clinically interpretable features.

---

## Folder structure

```
.
├── config/                     # dataset paths, hyperparameter grids, run configs
├── data/
│   ├── questionnaire/          # CSV written by questionnaire.ipynb — not committed
│   └── images/                 # Mendeley archive, unpacked here — not committed
├── notebooks/
│   └── questionnaire.ipynb     # fetches UCI data, writes data/questionnaire/*.csv
├── src/
│   ├── common/                 # branch-agnostic machinery
│   │   ├── utils/              # io, seeding, logging, path resolution
│   │   ├── validation/         # repeated stratified nested CV (5 outer × 3 inner × 10)
│   │   ├── metrics/            # sensitivity, specificity, PPV/NPV, ECE, thresholds
│   │   └── calibration/        # Platt scaling
│   ├── questionnaire/
│   │   ├── audit/              # circularity, base rate, missingness
│   │   ├── preprocessing/
│   │   ├── models/
│   │   ├── explainability/     # SHAP — user-facing
│   │   └── pipeline/
│   └── images/
│       ├── audit/              # blur, background, metadata, dedup probes
│       ├── preprocessing/
│       ├── models/
│       ├── explainability/     # Grad-CAM — internal only
│       └── pipeline/
├── scripts/                    # CLI entry points
├── outputs/                    # audit results, trained models, metrics, figures
├── tests/
└── logs/
```

Both streams expose the same five stages, so anything added to one should have a
counterpart in the other. Nothing branch-specific belongs in `src/common/` — if
code names a dataset, a modality or a model family, it belongs to a branch.

---

## Data

Raw data is **not committed**. Both datasets must be obtained locally.

### Questionnaire — UCI Autism Screening Data for Children

<https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children>

AQ-10-Child instrument, ages 4–11, ~292 records, ~20 usable features. Missing
values are coded `?`, concentrated in `ethnicity` and `relation`.

No manual download is needed. `questionnaire.ipynb` fetches the dataset directly
through the [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) package and writes
it to a CSV under `data/questionnaire/`:

```bash
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo
autism_child = fetch_ucirepo(id=419)          # UCI dataset 419
```

Run the notebook once to populate `data/questionnaire/` before any code in
`src/questionnaire/` will work.

### Facial images — Mendeley `f9dycfvwbt` v2

<https://data.mendeley.com/datasets/f9dycfvwbt/2>

2,940 facial images of children, two classes, three folders. Download manually
and unpack into `data/images/`.

**Provenance is a known problem, verified rather than assumed.** This set
re-hosts the Kaggle ASD Children Facial Image Dataset; its original author stated
the autistic images were collected by internet search from online autism
communities, published work describes it as low-fidelity, and its racial
composition is roughly 89% White children to 11% children of colour. Aggregate
accuracy therefore conceals who the model fails — report stratified performance
where feasible, and treat the Phase 0 audit as a gate, not a formality.

---

## Working order

1. **Phase 0 audit first.** Run `src/questionnaire/audit/` and
   `src/images/audit/` before any modelling. The image audit determines whether
   that stream is built at all.
2. Hold out a stratified test set before anything else. Split images by subject
   where an identifier permits, and augment only after splitting.
3. Train and compare; calibrate the winner.
4. Report sensitivity, specificity, PPV and NPV at realistic prevalence,
   calibration (ECE and reliability curve), the chosen operating threshold with
   its cost stated, and the mean and spread across cross-validation repeats.
   Below ~300 records, single-split estimates vary by several points on
   reshuffling alone — a single accuracy figure is not a result.

**Avoid** TabNet, FT-Transformer and similar deep tabular architectures; they
need one to two orders of magnitude more data than ~292 rows. Use Platt scaling
rather than isotonic regression for the same reason.

---

## Ethics and data handling

Facial photographs of children are biometric information about minors. Retention,
transfer and reuse for retraining must be settled before the image stream is
built, and unit ethics approval — if required — has a lead time. Inferring a
neurodevelopmental condition from facial appearance sits close to a discredited
tradition, and favourable metrics do not answer that objection; it is addressed
explicitly in the accompanying paper.