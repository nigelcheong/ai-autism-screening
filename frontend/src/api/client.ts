const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

// Fired when an authenticated request comes back 401 -- the token expired
// or was invalidated server-side. Registered by SessionWatcher, which has
// access to auth state + routing to actually log the user out and redirect.
// A login attempt that itself returns 401 (bad credentials) never has a
// token attached, so it's never mistaken for an expired session.
let onUnauthorized: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null) {
  onUnauthorized = handler;
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem("access_token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const sentAuth = !!authHeaders().Authorization;
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(options.headers ?? {}),
    },
  });
  if (res.status === 401 && sentAuth) {
    onUnauthorized?.();
  }
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function qs(params: Record<string, string | number | boolean | undefined>): string {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== "") usp.set(k, String(v));
  }
  const s = usp.toString();
  return s ? `?${s}` : "";
}

export interface Hospital {
  id: string;
  name: string;
  active: boolean;
  contact_name?: string | null;
  contact_designation?: string | null;
  contact_email?: string | null;
  contact_phone?: string | null;
}

export interface PatientListItem {
  id: string;
  name: string;
  dob?: string | null;
  external_patient_id?: string | null;
  contact_number?: string | null;
  gender?: string | null;
  autism?: boolean | null;
  jaundice?: boolean | null;
  used_app_before?: boolean | null;
  country_of_res?: string | null;
  relation?: string | null;
  latest_risk_category: "low" | "moderate" | "high" | null;
  latest_screening_at: string | null;
}

export interface AdminQuestion {
  id: string;
  instrument: "aq10" | "qchat10";
  key: string;
  text: string;
  answer_scale: string;
  options: string[];
  scoring_indices: number[];
  version: number;
  active: boolean;
  sort_order: number;
}

export interface SessionSummary {
  id: string;
  session_type: string;
  risk_score: number | null;
  risk_category: string | null;
  model_version: string | null;
  assigned_doctor_id: string | null;
  created_at: string;
}

export interface SessionReportData {
  id: string;
  patient_name: string;
  patient_dob: string | null;
  patient_age_label: string | null;
  patient_external_id: string | null;
  hospital_name: string;
  doctor_email: string | null;
  session_type: string;
  risk_score: number | null;
  risk_category: string | null;
  result_label: string | null;
  result_tone: string | null;
  model_version: string | null;
  model_description: string;
  created_at: string;
  generated_at: string;
  summary_stats: { label: string; value: string }[];
  features: {
    label: string;
    value: string;
    contributed_to_risk: boolean;
    image_data_url: string | null;
    heatmap_data_url: string | null;
    facial_shape_diagram_data_url: string | null;
    shap_diagram_data_url: string | null;
    pixel_region_explanation: { label: string; direction: "increased" | "decreased"; magnitude: number }[];
  }[];
  explanation: {
    label: string;
    direction: "increased" | "decreased";
    magnitude: number;
    detail: string | null;
  }[];
  explanation_note: string;
  other_factors: {
    label: string;
    direction: "increased" | "decreased";
    magnitude: number;
    detail: string | null;
  }[];
  note: string;
}

export interface CombinedReportSummary {
  id: string;
  patient_id: string;
  patient_name: string;
  session_ids: string[];
  comments: string;
  created_by_email: string | null;
  created_at: string;
  updated_at: string;
}

export interface CombinedReportData extends CombinedReportSummary {
  hospital_name: string;
  reports: SessionReportData[];
}

export const api = {
  login: (email: string, password: string) =>
    request<{ access_token: string; token_type: string }>("/internal/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request<void>("/internal/auth/logout", { method: "POST" }),

  refreshSession: () =>
    request<{ access_token: string; token_type: string }>("/internal/auth/refresh", {
      method: "POST",
    }),

  logPageView: (path: string) =>
    request<void>("/internal/audit/page-view", {
      method: "POST",
      body: JSON.stringify({ path }),
    }),

  submitQuestionnaire: (payload: unknown) =>
    request("/internal/diagnose/questionnaire", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  submitImages: (payload: unknown) =>
    request("/internal/diagnose/image", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  uploadImages: async (files: File[]) => {
    const form = new FormData();
    for (const file of files) form.append("files", file);
    const res = await fetch(`${BASE_URL}/internal/diagnose/images/upload`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    });
    if (res.status === 401) onUnauthorized?.();
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${detail}`);
    }
    return res.json() as Promise<{ image_urls: string[] }>;
  },

  listPatients: (params?: {
    search?: string;
    risk_category?: string;
    date_from?: string;
    date_to?: string;
    limit?: number;
    offset?: number;
  }) => request<PatientListItem[]>(`/internal/patients${qs(params ?? {})}`),

  createPatient: (payload: unknown) =>
    request<PatientListItem>("/internal/patients", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updatePatient: (
    patientId: string,
    payload: {
      name?: string;
      dob?: string;
      external_patient_id?: string;
      contact_number?: string;
      gender?: string;
      autism?: boolean;
      jaundice?: boolean;
      used_app_before?: boolean;
      country_of_res?: string;
      relation?: string;
    }
  ) =>
    request<PatientListItem>(`/internal/patients/${patientId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  getDashboardStats: () =>
    request<{ total_scanned: number; total_patients: number; autism_positive: number }>(
      "/internal/dashboard/stats"
    ),

  getHospitalApiKeyStatus: () =>
    request<{ active: boolean; masked_key: string | null; created_at: string | null }>(
      "/internal/hospital/api-key"
    ),

  regenerateHospitalApiKey: () =>
    request<{ api_key: string }>("/internal/hospital/api-key/regenerate", { method: "POST" }),

  listMyHospitalUsers: (params?: { limit?: number; offset?: number }) =>
    request<{ id: string; email: string; role: string }[]>(
      `/internal/hospital/users${qs(params ?? {})}`
    ),

  createMyHospitalUser: (payload: { email: string; password: string; role: string }) =>
    request<{ id: string; email: string; role: string }>("/internal/hospital/users", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  updateMyHospitalUser: (
    userId: string,
    payload: { email?: string; role?: string; password?: string }
  ) =>
    request<{ id: string; email: string; role: string }>(`/internal/hospital/users/${userId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  deleteMyHospitalUser: (userId: string) =>
    request<void>(`/internal/hospital/users/${userId}`, { method: "DELETE" }),

  listHospitals: (params?: { search?: string; limit?: number; offset?: number }) =>
    request<Hospital[]>(`/internal/admin/hospitals${qs(params ?? {})}`),

  createHospital: (payload: unknown) =>
    request("/internal/admin/hospitals", { method: "POST", body: JSON.stringify(payload) }),

  getHospitalApiKeyStatusAsAdmin: (hospitalId: string) =>
    request<{ active: boolean; masked_key: string | null; created_at: string | null }>(
      `/internal/admin/hospitals/${hospitalId}/api-key`
    ),

  issueHospitalApiKeyAsAdmin: (hospitalId: string) =>
    request<{ api_key: string }>(`/internal/admin/hospitals/${hospitalId}/api-key`, {
      method: "POST",
    }),

  listQuestions: (params?: { instrument?: string }) =>
    request<AdminQuestion[]>(`/internal/admin/questions${qs(params ?? {})}`),

  createQuestion: (payload: {
    instrument: "aq10" | "qchat10";
    text: string;
    options: string[];
    scoring_indices: number[];
    answer_scale?: string;
  }) => request<AdminQuestion>("/internal/admin/questions", { method: "POST", body: JSON.stringify(payload) }),

  updateQuestion: (
    questionId: string,
    payload: { text?: string; options?: string[]; scoring_indices?: number[]; active?: boolean }
  ) =>
    request<AdminQuestion>(`/internal/admin/questions/${questionId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  reorderQuestions: (keys: string[]) =>
    request<void>("/internal/admin/questions/reorder", {
      method: "PATCH",
      body: JSON.stringify({ keys }),
    }),

  getActiveQuestions: (instrument: "aq10" | "qchat10") =>
    request<{ id: string; key: string; text: string; options: string[] }[]>(
      `/internal/questions/active${qs({ instrument })}`
    ),

  listDoctors: () => request<{ id: string; email: string }[]>("/internal/staff/doctors"),

  getMyCases: (params?: {
    search?: string;
    risk_category?: string;
    date_from?: string;
    date_to?: string;
    limit?: number;
    offset?: number;
  }) =>
    request<
      {
        id: string;
        patient_id: string;
        patient_name: string;
        session_type: string;
        risk_score: number | null;
        risk_category: "low" | "moderate" | "high" | null;
        created_at: string;
      }[]
    >(`/internal/doctor/cases${qs(params ?? {})}`),

  listHospitalUsers: (hospitalId: string, params?: { limit?: number; offset?: number }) =>
    request<{ id: string; email: string; role: string }[]>(
      `/internal/admin/hospitals/${hospitalId}/users${qs(params ?? {})}`
    ),

  createHospitalUser: (
    hospitalId: string,
    payload: { email: string; password: string; role: string }
  ) =>
    request<{ id: string; email: string; role: string }>(
      `/internal/admin/hospitals/${hospitalId}/users`,
      { method: "POST", body: JSON.stringify(payload) }
    ),

  updateHospitalUser: (
    hospitalId: string,
    userId: string,
    payload: { email?: string; role?: string; password?: string }
  ) =>
    request<{ id: string; email: string; role: string }>(
      `/internal/admin/hospitals/${hospitalId}/users/${userId}`,
      { method: "PATCH", body: JSON.stringify(payload) }
    ),

  deleteHospitalUser: (hospitalId: string, userId: string) =>
    request<void>(`/internal/admin/hospitals/${hospitalId}/users/${userId}`, {
      method: "DELETE",
    }),

  impersonateHospitalUser: (hospitalId: string, userId: string) =>
    request<{ access_token: string; token_type: string }>(
      `/internal/admin/hospitals/${hospitalId}/users/${userId}/impersonate`,
      { method: "POST" }
    ),

  updateHospital: (
    hospitalId: string,
    payload: {
      name?: string;
      active?: boolean;
      contact_name?: string;
      contact_designation?: string;
      contact_email?: string;
      contact_phone?: string;
    }
  ) =>
    request<Hospital>(`/internal/admin/hospitals/${hospitalId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),

  getHospital: (hospitalId: string) =>
    request<Hospital>(`/internal/admin/hospitals/${hospitalId}`),

  getPatientSessions: (patientId: string) =>
    request<SessionSummary[]>(`/internal/patients/${patientId}/sessions`),

  assignDoctor: (sessionId: string, doctorId: string | null) =>
    request<{ id: string; assigned_doctor_id: string | null }>(
      `/internal/sessions/${sessionId}/assign-doctor`,
      { method: "PATCH", body: JSON.stringify({ doctor_id: doctorId }) }
    ),

  getSessionReport: (sessionId: string) =>
    request<SessionReportData>(`/internal/sessions/${sessionId}/report`),

  listCombinedReports: (patientId: string) =>
    request<CombinedReportSummary[]>(`/internal/combined-reports${qs({ patient_id: patientId })}`),

  createCombinedReport: (payload: { patient_id: string; session_ids: string[]; comments: string }) =>
    request<CombinedReportData>("/internal/combined-reports", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  getCombinedReport: (reportId: string) =>
    request<CombinedReportData>(`/internal/combined-reports/${reportId}`),

  updateCombinedReportComments: (reportId: string, comments: string) =>
    request<CombinedReportSummary>(`/internal/combined-reports/${reportId}`, {
      method: "PATCH",
      body: JSON.stringify({ comments }),
    }),

  changePassword: (currentPassword: string, newPassword: string) =>
    request<void>("/internal/auth/change-password", {
      method: "POST",
      body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
    }),

  listAuditLog: (params?: {
    search?: string;
    actor_email?: string;
    date_from?: string;
    date_to?: string;
    include_api_calls?: boolean;
    limit?: number;
    offset?: number;
  }) =>
    request<
      {
        id: string;
        actor_email: string;
        actor_role: string;
        action: string;
        target: string | null;
        created_at: string;
      }[]
    >(`/internal/admin/audit-log${qs(params ?? {})}`),

  listAuditActors: () => request<string[]>("/internal/admin/audit-log/actors"),

  getPlatformStats: () =>
    request<{
      total_hospitals: number;
      active_hospitals: number;
      total_sessions: number;
      questionnaire_sessions: number;
      image_sessions: number;
      high_risk_sessions: number;
      sessions_by_day: { date: string; questionnaire: number; image: number }[];
    }>("/internal/admin/platform-stats"),
};
