# UI + API Workstream Plan

Scope: this plan covers **frontend** and **backend API** only. Model training/inference internals are owned by another team member — we only need to agree a contract with them (see §3) and build against a stub until their model is ready.

Reference: [Autism_Prediction](https://github.com/Zukhruf-Riaz/Autism_Prediction) was reviewed for the questionnaire content and the "Step 2 Personal Information" form. Its AQ-10 questions and personal-info fields are adopted below as the concrete content for `PROJECT_SETUP.md`'s Type A (questionnaire) flow.

## 1. Questionnaire content (from reference repo)

**AQ-10 questions** (seed data for the `questions` table, answer_scale = agree/disagree):

1. I often notice small sounds when others do not.
2. When I'm reading a story, I find it difficult to work out the characters' intentions.
3. I find it easy to "read between the lines" when someone is talking to me.
4. I usually concentrate more on the whole picture, rather than the small details.
5. I know how to tell if someone listening to me is getting bored.
6. I find it easy to do more than one thing at once.
7. I find it easy to work out what someone is thinking or feeling just by looking at their face.
8. If there is an interruption, I can switch back to what I was doing very quickly.
9. I like to collect information about categories of things.
10. I find it difficult to work out people's intentions.

**Step 2 — Personal Information** fields (collected after the 10 questions, before submit):

| Field | Type | Notes |
|---|---|---|
| age | number | |
| gender | male / female / others | |
| autism | yes/no | family history of autism |
| jaundice | yes/no | born with jaundice |
| used_app_before | yes/no | prior use of a screening app |
| country_of_res | free text | |
| relation | Self / Parent / Relative / Health care professional / Others | who is filling the form |

These map onto our schema as follows: the 10 answers + the 7 personal-info fields are stored together as `diagnostic_sessions.question_responses` JSON (matching PROJECT_SETUP.md §9's `{"q1":1,...}` shape, extended with a `personal_info` sub-object). `patients` keeps only `name`/`dob`/`external_patient_id`; per-session demographic answers live on the session, since they're part of that screening instance, not a durable patient attribute.

## 2. DB schema additions/confirmations

- Seed `questions` table with the 10 AQ-10 items above, `version: 1`, `active: true`.
- Confirm `diagnostic_sessions.question_responses` JSON shape:
  ```json
  {
    "answers": {"q1": 1, "q2": 0, ...},
    "personal_info": {
      "age": 7, "gender": "male", "autism": false,
      "jaundice": false, "used_app_before": false,
      "country_of_res": "Pakistan", "relation": "Parent"
    }
  }
  ```
- No schema change needed beyond what's already in PROJECT_SETUP.md §7/§9.

## 3. Contract with the model-training teammate

Define this now so both sides can build in parallel against a stub:

- **Request** (backend → model service): `{"answers": {...}, "age": int, "gender": str, "autism": bool, "jaundice": bool, "relation": str}` for questionnaire type; `{"image_urls": [...]}` for image type.
- **Response** (model service → backend): the shared result shape from PROJECT_SETUP.md §9 — `{"risk_score": float, "risk_category": "low"|"moderate"|"high", "model_version": str}`.
- Until the real model is ready, `app/ml/inference.py` implements a stub returning a deterministic mock score, so `/internal/diagnose` and `/v1/external/diagnose` are fully testable end-to-end. Swap the stub for `snapshot_download` + real inference once the teammate publishes a model repo.

## 4. Backend (FastAPI) — build order

1. **Scaffold**: `app/main.py`, `core/`, `routers/internal/`, `routers/external/`, `models/`, `auth/`, `ml/`. Dockerfile + requirements.txt per PROJECT_SETUP.md §3–4.
2. **Auth**: JWT login/verify for staff (`auth/jwt.py`), API-key verify for hospitals (`auth/api_key.py`), `get_current_tenant` dependency, `require_role(...)` dependency.
3. **Core models** (SQLAlchemy): hospitals, staff_users, patients, diagnostic_sessions, session_images, questions, api_keys, platform_admins.
4. **Internal routes**: `/internal/auth/login`, `/internal/patients` (CRUD), `/internal/diagnose` (branches questionnaire/image), `/internal/admin/hospitals`.
5. **External routes**: `/v1/external/diagnose`, `/v1/external/patients`, API-key auth + rate limiting (slowapi).
6. **ML wrapper stub**: `app/ml/inference.py` per §3 above.
7. **Supabase Storage** integration for image uploads (signed URL upload from frontend, or backend proxy — decide in implementation).
8. RLS policies on tenant-scoped tables; verify with a cross-tenant test.

## 5. Frontend (React + TS + Tailwind) — build order

1. **Scaffold**: Vite + React + TS + Tailwind, `VITE_API_BASE_URL` env wiring, Dockerfile (nginx).
2. **Auth**: login page, JWT stored + attached to requests, role-based route guarding.
3. **Shared shell**: sidebar nav + top bar, nav items conditional on role (hospital vs platform admin).
4. **New diagnosis flow** (priority — this is the core screening UX):
   - Choice screen: "Run Questionnaire" vs "Upload Images"
   - Questionnaire: 10 AQ-10 questions (agree/disagree) → Step 2 personal info form (age, gender, autism history, jaundice, used_app_before, country, relation) → submit
   - Image path: drag-and-drop uploader, cap 5–10 images
   - Loading state → results screen (risk score, category badge, model version)
5. **Hospital dashboard**: patient list (search/filter by risk/date), patient detail (session timeline + assign-doctor dropdown), staff management (hospital_admin only), API access panel (masked key, regenerate).
6. **Platform admin dashboard**: hospitals overview, hospital detail/management, question bank editor (versioned), model version panel, platform usage overview/chart.

## 6. Integration & deployment

- Wire CORS on backend for frontend Space origin.
- GitHub Actions deploy workflows for both repos (per PROJECT_SETUP.md §6).
- Seed demo hospital + sample patients/sessions for the pre-demo checklist.
- Swap ML stub for the real model once the teammate's model repo is ready (§3).

## 7. Milestones

1. Backend scaffold + auth + DB models runnable locally with ML stub.
2. Frontend scaffold + new diagnosis flow (questionnaire path) working end-to-end against local backend.
3. Image upload path + Supabase Storage wired.
4. Hospital dashboard complete.
5. Platform admin dashboard complete.
6. CI/CD to HF Spaces, RLS verified, demo data seeded.

---
**Starting point**: Phase 4.1 (backend scaffold) and Phase 5.1 (frontend scaffold) can proceed in parallel right away.
