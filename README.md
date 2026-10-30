# AI-Powered Early Autism Screening — Research Codebase

<<<<<<< HEAD
A preliminary **screening aid** that estimates whether a child aged 4–11 would
benefit from formal autism assessment, and explains the reasoning behind each
result. This repository holds the data preprocessing, dataset auditing, model
training, calibration, evaluation and explainability code. The web application
lives on a separate branch, [`sandeep-webapp`](#web-application).
=======
Data preprocessing, dataset auditing, model training, evaluation and
explainability code for a screening-support prototype covering two modalities,
questionnaire responses and facial images. The web application lives in a
separate repository.
>>>>>>> 5a08fe06229e80b4a1e51de8e568ffd58bbf7329

> This is a research prototype, not a clinical diagnostic tool, and no output of
> it should be read as a diagnosis.

**Prediction target.** Binary screen-positive / screen-negative.

**Cohort.** Children aged 4–11. The Q-CHAT-10 toddler tables use a different
instrument and age band and are kept as supplementary material; no model mixes
toddler and child rows.

---

## Repository scope

| In this repository | Elsewhere |
|---|---|
| Preprocessing, splits and seeds | Web application front end and hosting |
| Dataset audits for both modalities | Deployment configuration and credentials |
| Model training, evaluation, explainability | Raw datasets, which are not committed |
| Serialised questionnaire models under `models/` | |

---

## Datasets

Raw data is not committed. Every dataset must be obtained locally before the
corresponding notebooks will run.

### Questionnaire

| Dataset | Instrument | Size | Source |
|---|---|---|---|
| UCI Autism Screening Data for Children, id 419 | AQ-10-Child | ~292 rows | Fetched by `questionnaire.ipynb` via `ucimlrepo` |
| Thabtah Combined 4–11 | AQ-10-Child | ~960 rows after removing exact copies | `[SOURCE: confirm the download location]` |
| Hugging Face ASD V3, filtered to ages 4–11 | AQ-10 | 864 rows after de-duplication | `[SOURCE: confirm — likely huggingface.co/datasets/rescommons/ASD_V3_all_ages]` |
| Q-CHAT-10 toddlers, New Zealand | Q-CHAT-10 | 1,054 rows | <https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers> |
| Q-CHAT-10 toddlers, NZ and Saudi merge | Q-CHAT-10 | 1,560 rows | `[SOURCE: confirm the Saudi component]` |

<<<<<<< HEAD
The unmodified-image model itself lives in `experiments/intact_images/`, and only
gets trained once the three audits above clear. Perceptual-hash deduplication
runs across classes alongside these. Grad-CAM is retained as an **internal
diagnostic only, never user-facing** — a heat map over a child's face marks
image regions, not clinically interpretable features.

---

## Folder structure

```
.
├── configs/                     # paths, settings
├── data/
│   ├── raw/                     # untouched source data — not committed
│   │   ├── questionnaire/       # CSV written by questionnaire.ipynb
│   │   └── images/              # Mendeley archive, unpacked here
│   ├── processed/                # cleaned / derived data
│   └── splits/                   # train/test/val split files
├── experiments/                  # one folder per approach — code + results together
│   ├── questionnaire_models/     # logistic regression, tree, boosting, MLP
│   ├── metadata_probe/           # audit: can metadata alone separate the classes?
│   ├── intact_images/            # normal images
│   ├── blurred_images/           # audit: blurred past recognisability
│   └── background_images/        # audit: faces masked out
├── notebooks/                    # exploration and visual analysis only
├── outputs/                      # models, predictions, figures, logs
├── reports/
│   ├── proposal/
│   └── final_report/
└── src/                          # shared code imported by experiments
    ├── data/                      # loading, preprocessing
    ├── models/                    # model definitions, feature extraction
    └── evaluation/                # metrics, cross-validation
```

Each stream is now a set of `experiments/` folders — one per approach or audit —
that keep their own code and results together, rather than a shared five-stage
pipeline. Machinery with no dataset, modality or model family in its name
(loading, feature extraction, metrics, cross-validation) belongs in `src/` and
is imported by whichever experiments need it; anything that names one belongs
in that experiment's own folder instead.

---

## Web application

The screening web app is on the `sandeep-webapp` branch. That branch is a
**separate codebase with no shared git history** with `main`, so it is never
merged here. Check it out on its own:

```bash
git fetch origin sandeep-webapp
git worktree add ../autism-screening-webapp sandeep-webapp
```

- **Stack:** React + TypeScript + Tailwind frontend (`frontend/`) and a
  FastAPI backend (`backend/`), deployed as two Docker Hugging Face Spaces,
  with Supabase Postgres and file storage.
- **Models:** the backend serves model artifacts trained in this repository,
  committed as `.pkl` files under `backend/app/ml/`. They include the
  questionnaire logistic regression, the named-feature facial model, and a
  cross-validation fold of the fine-tuned CNN. Its named-feature facial
  explanation reproduces the `experiments/why_not_faces/` feature set.
  Retraining a model here does **not** update the app: the new artifact has to
  be copied across by hand.
- **Setup:** see `PROJECT_SETUP.md` on that branch. Copy `backend/.env.example`
  and `frontend/.env.example` to `.env`. `start_all.bat` contains hard-coded
  local paths, so edit them before running it.

The screening caveats above apply to the app's output too. In particular, the
facial-image stream stays gated on the Phase 0 audit.

---

## Data

Raw data is **not committed**. Both datasets must be obtained locally.

### Questionnaire — UCI Autism Screening Data for Children

<https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children>

AQ-10-Child instrument, ages 4–11, ~292 records, ~20 usable features. Missing
values are coded `?`, concentrated in `ethnicity` and `relation`.

No manual download is needed. `questionnaire.ipynb` fetches the dataset directly
through the [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) package and writes
it to a CSV under `data/raw/questionnaire/`:
=======
The UCI set needs no manual download:
>>>>>>> 5a08fe06229e80b4a1e51de8e568ffd58bbf7329

```bash
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo
autism_child = fetch_ucirepo(id=419)
```

Run `questionnaire.ipynb` once to populate `data/raw/questionnaire/` before any
other questionnaire notebook will work.

Missing values in the UCI table are coded `?` and are concentrated in
`ethnicity` and `relation`.

**Preprocessing note.** The `result` column is the summed AQ-10 item score and
`Class/ASD` is a threshold applied to that sum, so `result` is dropped before
fitting. The equivalent column in the Q-CHAT-10 tables is `Qchat-10-Score` and
is dropped on the same basis.

### Facial images

| Dataset | Size | Source |
|---|---|---|
| Mendeley `f9dycfvwbt`, version 3 | 2,940 images, two classes, three folders | <https://data.mendeley.com/datasets/f9dycfvwbt/3> |

Download manually and unpack into `data/raw/images/`. The archive supplies a
train, test and validation partition; the audit notebooks pool and re-split it.

Perceptual-hash clustering is used to group near-duplicate images, since the
archive carries no subject identifier.

---

## Folder structure

```
.
├── configs/                     # paths and settings
├── data/
│   ├── raw/                     # source data, not committed, read only
│   │   ├── questionnaire/       # written by questionnaire.ipynb
│   │   └── images/              # Mendeley archive, unpacked here
│   ├── processed/               # cleaned and derived data
│   └── splits/                  # versioned split definitions and seeds
├── experiments/                 # one folder per approach, code and results together
├── models/                      # serialised questionnaire models (.pkl)
├── notebooks/                   # call src/ modules step by step, display results
├── outputs/                     # predictions, figures, logs
├── reports/
│   ├── proposal/
│   └── final_report/
└── src/                         # shared code imported by notebooks and experiments
    ├── data/                    # loading, preprocessing
    ├── models/                  # model definitions, feature extraction
    └── evaluation/              # metrics, cross-validation
```

Logic lives in `.py` modules under `src/`. Notebooks call those modules and
display results. Machinery with no dataset, modality or model family in its
name belongs in `src/`; anything that names one belongs in its own experiment
folder.

---

## Notebooks

### Facial-image stream

| Notebook | What it produces |
|---|---|
| `facialimages.ipynb` | Metadata probe and the intact-image frozen probe |
| `blurred_images.ipynb` | Blur arm and the colour-only floor |
| `background_images.ipynb` | Retained with its defect documented; its result is excluded from all reported figures |
| `face_crop_blur.ipynb` | Face crop and the progressive blur series |
| `face_landmarks.ipynb` | Landmark, framing and pose arms |
| `why_not_faces.ipynb` | Named-feature model, out-of-fold SHAP, normalised Grad-CAM |

### Questionnaire stream

| Notebook | What it produces |
|---|---|
| `questionnaire.ipynb` | UCI 292 exploratory analysis, circularity check, nested CV, SHAP, child `.pkl` |
| `questionnaire_thabtah_combined_4to11.ipynb` | Combined 4–11 exploratory analysis, nested CV, SHAP, second child `.pkl` |
| `questionnaire_hf.ipynb` | Hugging Face 4–11 audit |
| `questionnaire_toddlers_months.ipynb` | New Zealand Q-CHAT-10, nested CV, SHAP, toddler `.pkl` |
| `questionnaire_toddlers_nz_saudi.ipynb` | NZ and Saudi Q-CHAT-10, second toddler `.pkl` |

---

## Setup

```bash
git clone [REPO URL]
cd [REPO NAME]
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`[SETUP: state the Python version, and whether a CUDA-capable GPU is required
for the fine-tuned image arms or only recommended.]`

---

## Running

1. Obtain the datasets listed above and place them under `data/raw/`.
2. Run `questionnaire.ipynb` first on the questionnaire side, since it writes the
   CSV the other questionnaire notebooks read.
3. Run the facial-image notebooks in the order listed in the table above. Each
   writes its artefacts to its own directory under `outputs/`.

`[RUNTIME: state the approximate wall-clock time for the fine-tuned image arms.]`

---

## Evaluation protocol

| Stream | Protocol |
|---|---|
| Questionnaire | Stratified 20% hold-out (`random_state=42`), then nested cross-validation on the remainder, outer stratified 5-fold for the reported score and inner stratified 3-fold `GridSearchCV` for tuning |
| Facial images | 5×10 repeated stratified cross-validation with grouped splits over the pooled, deduplicated dataset, replacing the archive's supplied partition |

Models compared on the questionnaire side are penalised logistic regression, a
depth-limited decision tree, random forest and histogram gradient boosting. Deep
tabular architectures and a small MLP were out of scope for the sample size.

Reported metrics are ROC-AUC with its spread for cross-validated figures,
hold-out sensitivity and specificity, and positive and negative predictive value
at assumed prevalences of 1% and 2%. Calibration was not assessed.

---

## Conventions

- Everything under `data/raw/` is read only. Derived artefacts are written
  elsewhere.
- Split definitions and random seeds are versioned under `data/splits/`.
- Face images rendered in notebook outputs are cleared before any commit, so a
  freshly cloned repository shows empty image cells until the notebooks are
  re-run against a local copy of the archive.
- Each experiment writes to its own directory under `outputs/`.

---

## Reports

| Report | Location |
|---|---|
| Research report | `reports/final_report/` · `[LINK]` |
| Final Delivery Report | `reports/final_report/` · `[LINK]` |