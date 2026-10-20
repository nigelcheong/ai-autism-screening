# Autism Screening Platform — Project Setup Guide

This document covers the full setup for the capstone project: a multi-tenant SaaS platform for autism screening via questioning and image upload, consuming pretrained AI models.

## 1. Architecture Overview

```
                        ┌─────────────────────┐
                        │   React Frontend     │
                        │  (HF Space: Docker)  │
                        └──────────┬───────────┘
                                   │ HTTPS
                                   ▼
                        ┌─────────────────────┐
                        │   FastAPI Backend    │
                        │  (HF Space: Docker)  │
                        │                       │
                        │  /internal/*  (JWT)   │◄──── Frontend calls this
                        │  /v1/external/* (key) │◄──── Third-party hospitals call this
                        └──────────┬───────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼                              ▼
          ┌──────────────────┐          ┌─────────────────────┐
          │  Supabase Postgres │          │  HF Model Repo       │
          │  (tenant data,     │          │  (private, loaded    │
          │   RLS enabled)     │          │   at Space startup)  │
          └──────────────────┘          └─────────────────────┘
```

Two Hugging Face Spaces, one Supabase project, one model repo. GitHub is the source of truth for both codebases, deployed via GitHub Actions.

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript, Tailwind |
| Backend | Python, FastAPI |
| Database | PostgreSQL via Supabase (free tier) |
| File storage | Supabase Storage (for uploaded images) |
| Model hosting | Hugging Face Hub (private model repo) |
| App hosting | Hugging Face Spaces (2x, Docker SDK) |
| CI/CD | GitHub Actions |
| Rate limiting | slowapi |
| Auth (internal) | JWT (frontend/staff login) |
| Auth (external) | API key per hospital tenant |

## 3. Repo Structure

Recommend two repos (cleaner CI/CD per Space) or a monorepo with two folders. Two repos shown here:

```
autism-screening-api/
├── app/
│   ├── main.py
│   ├── core/            # shared business logic (diagnosis, patients, tenants)
│   ├── routers/
│   │   ├── internal/    # JWT-auth routes, used by frontend
│   │   └── external/    # API-key-auth routes, used by hospitals (/v1/external)
│   ├── models/           # DB models (SQLAlchemy)
│   ├── auth/             # JWT + API key verification
│   └── ml/               # model loading + inference wrapper
├── Dockerfile
├── requirements.txt
└── .github/workflows/deploy.yml

autism-screening-frontend/
├── src/
├── Dockerfile (nginx serving build output)
├── package.json
└── .github/workflows/deploy.yml
```

## 4. Hugging Face Spaces Setup

1. Create two Spaces under your HF account: `your-username/autism-api` and `your-username/autism-frontend`.
2. For each, choose **Docker** as the SDK when creating the Space.
3. Each Space needs a `Dockerfile` at the repo root. HF builds and runs it automatically on push.
4. In each Space's Settings, add any required secrets (DB connection string, JWT secret, Supabase keys) under **Repository secrets**, not hardcoded in the Dockerfile.

Backend Dockerfile should expose port 7860 (HF Spaces default) and run:
```
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## 5. Model Repo Setup

1. Create a **private** model repo: `your-username/autism-screening-model`.
2. Push trained weights using `huggingface_hub`:
```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./trained_model",
    repo_id="your-username/autism-screening-model",
    repo_type="model",
)
```
3. Generate an HF access token with read access to this repo, add it as a secret (`HF_MODEL_TOKEN`) in the backend Space's settings.
4. In `app/ml/`, load the model at startup (not per-request):
```python
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    repo_id="your-username/autism-screening-model",
    token=os.environ["HF_MODEL_TOKEN"],
)
# load model_path into memory once, reuse across requests
```
5. If the model is large, bake it into the Docker image at build time rather than downloading on every cold start, to reduce wake-up latency after a free-tier sleep.

## 6. GitHub → Hugging Face CI/CD

For **each** repo, add `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Hugging Face Spaces
on:
  push:
    branches: [main]
jobs:
  sync-to-hub:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Push to Hugging Face Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git push https://user:$HF_TOKEN@huggingface.co/spaces/your-username/SPACE_NAME main
```

Setup steps:
1. Generate a Hugging Face access token (write access) from your HF account settings.
2. Add it as a GitHub repo secret named `HF_TOKEN` in **both** GitHub repos.
3. Replace `SPACE_NAME` with `autism-api` or `autism-frontend` in the respective workflow file.
4. Push to `main` and confirm the Space rebuilds (check the Space's build logs on HF).

## 7. Database Setup (Supabase)

1. Create a Supabase project (free tier).
2. Core tables:
   - `platform_admins` (id, email, hashed_password) — operates across all hospitals, not scoped by `hospital_id`, kept in its own table rather than `staff_users` so it can never be accidentally caught by tenant filtering
   - `hospitals` (id, name, plan, active, created_at)
   - `staff_users` (id, hospital_id, email, hashed_password, role) — `role` is one of `hospital_admin`, `doctor`, or `staff` (see Roles below)
   - `api_keys` (id, hospital_id, key_hash, active, created_at)
   - `patients` (id, hospital_id, external_patient_id, name, dob) — `external_patient_id` is the patient's ID from the hospital's own existing system (e.g. their EMR's `MRN-48213`), stored alongside your internal `id` so hospital-supplied data (CSV imports, external API calls) can be matched to the right patient without needing your internal ID. Nullable, and unique per `hospital_id` rather than globally, since two hospitals may reuse the same external ID format independently.
   - `diagnostic_sessions` (id, hospital_id, patient_id, session_type, assigned_doctor_id, question_responses, model_version, risk_score, risk_category, created_at, consent_given)
   - `session_images` (id, session_id, image_url, uploaded_at)
   - `questions` (id, text, answer_scale, version, active)
3. Enable **Row Level Security** on all tenant-scoped tables, with a policy filtering on `hospital_id` matching the requesting tenant's context.
4. Store the Supabase connection string as a secret in the backend Space (`DATABASE_URL`).

**Roles**

Two levels, not one flat list, since platform staff and hospital staff operate in fundamentally different scopes.

- *Platform admin* (`platform_admins` table): manages all hospitals, API access, the question bank, and model versions. Not tied to any single hospital.
- *Hospital admin* (`staff_users.role = "hospital_admin"`): manages their own hospital's staff, API key, and sees all of that hospital's patients and sessions.
- *Doctor* (`staff_users.role = "doctor"`): can be assigned to sessions, runs and reviews diagnoses, sees all of that hospital's patients and sessions.
- *Staff* (`staff_users.role = "staff"`): can onboard patients and run diagnoses, but can't manage users or touch the API key panel.

Three hospital-level roles is a reasonable ceiling for a capstone scope, finer-grained permissions (like view-only staff) are possible but add a permissions matrix that isn't worth the complexity unless a sponsor requirement calls for it. Auth checks are two-layered in FastAPI: first resolve which table the user authenticated against (platform vs hospital), then, for hospital staff, check `role` against a `require_role(...)` dependency on the route.
5. Use Supabase Storage for uploaded images; store only the returned URL in `diagnostic_sessions.image_url`.

## 8. Route Structure

```
/internal/auth/login          → staff JWT login
/internal/patients             → CRUD, JWT auth
/internal/diagnose             → run a screening, JWT auth
/internal/admin/hospitals      → admin-only, create/manage hospitals + API keys

/v1/external/diagnose           → run a screening, API key auth
/v1/external/patients           → CRUD, API key auth
```

Both route groups call the same `core/` service layer. Tenant ID is resolved via a dependency (`get_current_tenant`) that reads it from either the JWT or the API key, and every DB query filters on it.

## 9. Diagnosis Types

Two independent screening modes, a hospital can run either one on a patient, or both over time.

**Type A: Questionnaire-based**
- A fixed set of 10 screening questions (structured similarly to standard tools like M-CHAT), each with a defined answer scale (yes/no, or a 0 to 2 severity scale).
- Stored as `diagnostic_sessions.session_type = "questionnaire"`, with responses saved as structured JSON: `{"q1": 1, "q2": 0, ...}`.
- The FastAPI backend sends the response set to the model service, which returns a risk score and category (e.g. low, moderate, high likelihood).
- Question bank should live in its own table (`questions`) rather than hardcoded, so admins can edit or version it without a redeploy.

**Type B: Image-based**
- Upload of a small batch of patient images (recommend capping at 5 to 10 per session, both for model input consistency and free-tier upload limits).
- Stored as `diagnostic_sessions.session_type = "image"`, with image URLs in a related `session_images` table (one session can have multiple images, so this shouldn't be a single column).
- Images go to Supabase Storage, backend passes the storage URLs (not raw files) to the model service for inference.
- Model returns a risk score the same shape as the questionnaire path, so both types produce a comparable result downstream.

**Shared result shape**

Regardless of type, a session resolves to:
```json
{
  "session_type": "questionnaire" | "image",
  "risk_score": 0.0-1.0,
  "risk_category": "low" | "moderate" | "high",
  "model_version": "v1.2",
  "created_at": "..."
}
```
This lets the dashboard render both types in one unified results list without special-casing the UI per type.

**Schema update**

Add to `diagnostic_sessions`: `session_type` (enum), `assigned_doctor_id` (references `staff_users.id`), and a new table `session_images` (id, session_id, image_url, uploaded_at) for the image path. Question responses stay as JSON on the session row for the questionnaire path.

**Doctor assignment**

`staff_users` needs a `role` value of `"doctor"` alongside existing roles (hospital admin, staff), so doctors are just staff users scoped to that hospital, not a separate table. When a new session is created, either the person running it assigns a doctor from a dropdown of that hospital's doctors, or it defaults to unassigned and gets picked up later from the patient detail view. Each session then carries the doctor's name and can be filtered or reassigned from the UI. This also means a doctor's own login can be scoped to just their assigned cases if you want that as a future role tier, though for the capstone timeline treating them as hospital staff with visibility into all sessions is simpler.

## 10. Dashboard Screens

**Hospital Dashboard (internal, JWT auth)**

- *Patient list view*: table of onboarded patients (name, patient ID, last diagnosis date, latest risk category as a colored badge). Search and filter by risk category or date range.
- *Patient detail view*: patient info at top, then a timeline of all their diagnostic sessions, oldest to newest. Each session card shows type (questionnaire or image icon), date, risk score, assigned doctor, and a "view details" expand. If unassigned, the card shows an "assign doctor" dropdown inline rather than requiring a separate screen.
- *New diagnosis flow*: a simple two-button choice screen, "Run Questionnaire" or "Upload Images", which branches into either the 10-question form or a drag-and-drop image uploader. On submit, show a loading state while the model runs, then land on a results screen with the score and category.
- *Staff management*: if the hospital admin role, a simple table to add/remove staff users under their own hospital only.
- *API access panel*: if the hospital wants external access, a screen showing their API key (masked, with a reveal/copy button), usage count for the current period, and a regenerate key button.

**Platform Admin Dashboard (internal, JWT auth, platform role)**

- *Hospitals overview*: table of all onboarded hospitals, plan tier, active/inactive status, staff count, total diagnostic sessions run. Click into a hospital to manage it.
- *Hospital detail/management*: create/deactivate the hospital, generate or revoke their API key, view their usage against any quota.
- *Question bank editor*: add, edit, reorder, or retire questions used in the questionnaire flow, with versioning so old sessions still reference the question set they were answered against.
- *Model version panel*: shows which model version is currently live, with a way to note when a new version was deployed (manual entry is fine for a capstone, doesn't need automated MLOps).
- *Platform usage overview*: total sessions run across all hospitals, split by type (questionnaire vs image), over time, useful both as an admin view and as a chart worth screenshotting for your capstone report.

**Layout notes**

Keep both dashboards on the same shell (sidebar nav, top bar with hospital/user name), just with a different set of nav items depending on role. One React app, role-based route guarding and conditional nav rendering, rather than two separate frontends. Simpler to build and demo.

## 11. Local Development

Backend:
```bash
cd autism-screening-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:
```bash
cd autism-screening-frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` (or equivalent) in a local `.env` pointing to `http://localhost:8000` during development, and to the deployed backend Space URL in production.

## 12. Pre-Demo Checklist

- Ping both Spaces a few minutes before presenting, so they're awake and out of the free-tier sleep state.
- Confirm CORS is configured on the backend to allow the frontend Space's origin.
- Verify RLS policies actually block cross-tenant access, not just the app-layer filter.
- Have a seeded demo hospital with a few sample patients and diagnostic sessions ready, so the dashboard isn't empty on first load.
