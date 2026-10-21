import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { ChevronLeft, FileText, Pencil, FileStack } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { DateRangeFilter } from "../../components/ui/DateRangeFilter";
import { api, CombinedReportSummary, PatientListItem, SessionSummary } from "../../api/client";

interface Doctor {
  id: string;
  email: string;
}

const RELATIONS = ["Self", "Parent", "Relative", "Health care professional", "Others"];
const GENDER_LABELS: Record<string, string> = { male: "Male", female: "Female", others: "Others" };

function YesNoToggle({
  label,
  value,
  onChange,
}: {
  label: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div>
      <label className="text-xs text-ink-500 block mb-1">{label}</label>
      <div className="flex gap-2">
        <Button type="button" size="sm" variant={value ? "primary" : "secondary"} onClick={() => onChange(true)}>
          Yes
        </Button>
        <Button type="button" size="sm" variant={!value ? "primary" : "secondary"} onClick={() => onChange(false)}>
          No
        </Button>
      </div>
    </div>
  );
}

export default function PatientDetail() {
  const { patientId } = useParams<{ patientId: string }>();
  const navigate = useNavigate();
  const [patient, setPatient] = useState<PatientListItem | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [combinedReports, setCombinedReports] = useState<CombinedReportSummary[]>([]);
  const [combining, setCombining] = useState(false);
  const [selectedSessionIds, setSelectedSessionIds] = useState<string[]>([]);
  const [combineComments, setCombineComments] = useState("");
  const [combineCreating, setCombineCreating] = useState(false);
  const [combineError, setCombineError] = useState<string | null>(null);
  const [editingDetails, setEditingDetails] = useState(false);
  const [nameInput, setNameInput] = useState("");
  const [dobInput, setDobInput] = useState("");
  const [contactInput, setContactInput] = useState("");
  const [genderInput, setGenderInput] = useState("");
  const [relationInput, setRelationInput] = useState("");
  const [autismInput, setAutismInput] = useState(false);
  const [jaundiceInput, setJaundiceInput] = useState(false);
  const [usedAppBeforeInput, setUsedAppBeforeInput] = useState(false);
  const [countryInput, setCountryInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function refreshPatient() {
    if (!patientId) return;
    api.listPatients().then((data) => setPatient(data.find((p) => p.id === patientId) ?? null));
  }

  function refreshCombinedReports() {
    if (!patientId) return;
    api.listCombinedReports(patientId).then(setCombinedReports).catch(() => setCombinedReports([]));
  }

  useEffect(() => {
    if (!patientId) return;
    refreshPatient();
    api
      .getPatientSessions(patientId)
      .then(setSessions)
      .catch((err) => setError((err as Error).message));
    api.listDoctors().then(setDoctors).catch(() => setDoctors([]));
    refreshCombinedReports();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  function toggleSelect(sessionId: string) {
    setCombineError(null);
    setSelectedSessionIds((prev) =>
      prev.includes(sessionId)
        ? prev.filter((id) => id !== sessionId)
        : prev.length >= 2
        ? prev
        : [...prev, sessionId]
    );
  }

  async function createCombinedReport() {
    if (!patientId || selectedSessionIds.length !== 2) {
      setCombineError("Select exactly two screenings to combine.");
      return;
    }
    setCombineCreating(true);
    setCombineError(null);
    try {
      const report = await api.createCombinedReport({
        patient_id: patientId,
        session_ids: selectedSessionIds,
        comments: combineComments,
      });
      navigate(`/combined-reports/${report.id}`);
    } catch (err) {
      setCombineError((err as Error).message);
    } finally {
      setCombineCreating(false);
    }
  }

  function startEditingDetails() {
    setNameInput(patient?.name ?? "");
    setDobInput(patient?.dob?.slice(0, 10) ?? "");
    setContactInput(patient?.contact_number ?? "");
    setGenderInput(patient?.gender ?? "male");
    setRelationInput(patient?.relation ?? "Self");
    setAutismInput(!!patient?.autism);
    setJaundiceInput(!!patient?.jaundice);
    setUsedAppBeforeInput(!!patient?.used_app_before);
    setCountryInput(patient?.country_of_res ?? "");
    setSaveError(null);
    setEditingDetails(true);
  }

  async function saveDetails() {
    if (!patientId || !nameInput.trim()) {
      setSaveError("Name can't be empty.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    try {
      await api.updatePatient(patientId, {
        name: nameInput.trim(),
        ...(dobInput ? { dob: dobInput } : {}),
        contact_number: contactInput.trim(),
        gender: genderInput,
        relation: relationInput,
        autism: autismInput,
        jaundice: jaundiceInput,
        used_app_before: usedAppBeforeInput,
        country_of_res: countryInput.trim(),
      });
      setEditingDetails(false);
      refreshPatient();
    } catch (err) {
      setSaveError((err as Error).message);
    } finally {
      setSaving(false);
    }
  }

  async function reassign(sessionId: string, doctorId: string) {
    await api.assignDoctor(sessionId, doctorId || null);
    setSessions((prev) =>
      prev.map((s) => (s.id === sessionId ? { ...s, assigned_doctor_id: doctorId || null } : s))
    );
  }

  const filteredSessions = sessions.filter((s) => {
    const day = s.created_at.slice(0, 10);
    if (dateFrom && day < dateFrom) return false;
    if (dateTo && day > dateTo) return false;
    return true;
  });

  return (
    <AppShell title="Patient">
      <div className="max-w-2xl">
        <Link
          to="/patients"
          className="inline-flex items-center gap-1 text-sm text-ink-500 hover:text-ink-900 mb-4"
        >
          <ChevronLeft size={16} />
          All patients
        </Link>

        <Card className="p-5 mb-6">
          {editingDetails ? (
            <div className="space-y-3">
              <div>
                <label className="text-xs text-ink-500">Name</label>
                <input
                  type="text"
                  className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm w-full mt-0.5"
                  value={nameInput}
                  onChange={(e) => setNameInput(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs text-ink-500">Date of birth</label>
                <input
                  type="date"
                  max={new Date().toISOString().slice(0, 10)}
                  className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm w-full mt-0.5"
                  value={dobInput}
                  onChange={(e) => setDobInput(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs text-ink-500">Contact number</label>
                <input
                  type="tel"
                  placeholder="Not on file"
                  className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm w-full mt-0.5"
                  value={contactInput}
                  onChange={(e) => setContactInput(e.target.value)}
                />
              </div>

              <p className="text-xs font-medium text-ink-700 pt-1">Other details on file</p>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-ink-500">Gender</label>
                  <select
                    className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm w-full mt-0.5"
                    value={genderInput}
                    onChange={(e) => setGenderInput(e.target.value)}
                  >
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="others">Others</option>
                  </select>
                </div>
                <div>
                  <label className="text-xs text-ink-500">Relation to patient</label>
                  <select
                    className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm w-full mt-0.5"
                    value={relationInput}
                    onChange={(e) => setRelationInput(e.target.value)}
                  >
                    {RELATIONS.map((r) => (
                      <option key={r} value={r}>
                        {r}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <label className="text-xs text-ink-500">Country of residence</label>
                <input
                  type="text"
                  className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm w-full mt-0.5"
                  value={countryInput}
                  onChange={(e) => setCountryInput(e.target.value)}
                />
              </div>

              <YesNoToggle
                label="Family history of autism?"
                value={autismInput}
                onChange={setAutismInput}
              />
              <YesNoToggle label="Born with jaundice?" value={jaundiceInput} onChange={setJaundiceInput} />
              <YesNoToggle
                label="Used a screening app before?"
                value={usedAppBeforeInput}
                onChange={setUsedAppBeforeInput}
              />

              {saveError && <p className="text-red-600 text-sm">{saveError}</p>}
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={saveDetails} disabled={saving}>
                  {saving ? "Saving…" : "Save"}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setEditingDetails(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="font-semibold text-ink-900">{patient?.name ?? "…"}</p>
                {patient?.external_patient_id && (
                  <p className="text-sm text-ink-500">ID: {patient.external_patient_id}</p>
                )}
                <p className="text-sm text-ink-500 mt-1">
                  DOB:{" "}
                  {patient?.dob ? (
                    new Date(patient.dob).toLocaleDateString()
                  ) : (
                    <span className="text-amber-600">not on file</span>
                  )}
                </p>
                <p className="text-sm text-ink-500 mt-1">
                  Contact:{" "}
                  {patient?.contact_number ? (
                    patient.contact_number
                  ) : (
                    <span className="text-amber-600">not on file</span>
                  )}
                </p>
              </div>
              <button
                title="Edit patient details"
                onClick={startEditingDetails}
                className="text-ink-400 hover:text-brand-600 shrink-0"
              >
                <Pencil size={14} />
              </button>
            </div>
          )}

          {!editingDetails && (
            <div className="mt-4 pt-4 border-t border-ink-100">
              <p className="text-xs font-medium text-ink-700 mb-2">Other details on file</p>
              <dl className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <dt className="text-xs text-ink-400">Gender</dt>
                  <dd className="text-ink-900">
                    {patient?.gender ? (
                      GENDER_LABELS[patient.gender] ?? patient.gender
                    ) : (
                      <span className="text-amber-600">not on file</span>
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-400">Relation to patient</dt>
                  <dd className="text-ink-900">
                    {patient?.relation ?? <span className="text-amber-600">not on file</span>}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-400">Family history of autism</dt>
                  <dd className="text-ink-900">
                    {patient?.autism === null || patient?.autism === undefined ? (
                      <span className="text-amber-600">not on file</span>
                    ) : patient.autism ? (
                      "Yes"
                    ) : (
                      "No"
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-400">Born with jaundice</dt>
                  <dd className="text-ink-900">
                    {patient?.jaundice === null || patient?.jaundice === undefined ? (
                      <span className="text-amber-600">not on file</span>
                    ) : patient.jaundice ? (
                      "Yes"
                    ) : (
                      "No"
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-400">Used a screening app before</dt>
                  <dd className="text-ink-900">
                    {patient?.used_app_before === null || patient?.used_app_before === undefined ? (
                      <span className="text-amber-600">not on file</span>
                    ) : patient.used_app_before ? (
                      "Yes"
                    ) : (
                      "No"
                    )}
                  </dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-400">Country of residence</dt>
                  <dd className="text-ink-900">
                    {patient?.country_of_res ?? <span className="text-amber-600">not on file</span>}
                  </dd>
                </div>
              </dl>
            </div>
          )}
        </Card>

        {combinedReports.length > 0 && (
          <div className="mb-6">
            <p className="font-medium text-ink-900 mb-3">Combined reports</p>
            <div className="space-y-2">
              {combinedReports.map((r) => (
                <Card key={r.id} className="p-4 flex items-center justify-between gap-3 flex-wrap">
                  <div>
                    <p className="text-sm text-ink-900">
                      {new Date(r.created_at).toLocaleString()}
                      {r.created_by_email && (
                        <span className="text-ink-400"> · reviewed by {r.created_by_email}</span>
                      )}
                    </p>
                    {r.comments && (
                      <p className="text-xs text-ink-500 mt-0.5 line-clamp-1">{r.comments}</p>
                    )}
                  </div>
                  <Link
                    to={`/combined-reports/${r.id}`}
                    className="text-brand-600 text-sm hover:underline shrink-0"
                  >
                    View
                  </Link>
                </Card>
              ))}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
          <p className="font-medium text-ink-900">Diagnosis history</p>
          <div className="flex items-center gap-2 flex-wrap">
            <DateRangeFilter
              dateFrom={dateFrom}
              dateTo={dateTo}
              onDateFromChange={setDateFrom}
              onDateToChange={setDateTo}
            />
            {sessions.length >= 2 && (
              <Button
                size="sm"
                variant={combining ? "secondary" : "ghost"}
                onClick={() => {
                  setCombining((v) => !v);
                  setSelectedSessionIds([]);
                  setCombineComments("");
                  setCombineError(null);
                }}
              >
                <FileStack size={14} />
                {combining ? "Cancel" : "Combine 2 screenings"}
              </Button>
            )}
          </div>
        </div>
        {error && <p className="text-red-600 text-sm mb-3">{error}</p>}

        {combining && (
          <Card className="p-4 mb-3 bg-ink-50">
            <p className="text-sm text-ink-700 mb-2">
              Select exactly two screenings below ({selectedSessionIds.length}/2 selected), then add
              your review comments.
            </p>
            <textarea
              className="w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
              rows={3}
              placeholder="Additional comments reviewing both results together…"
              value={combineComments}
              onChange={(e) => setCombineComments(e.target.value)}
            />
            {combineError && <p className="text-red-600 text-sm mt-1">{combineError}</p>}
            <div className="mt-2">
              <Button
                size="sm"
                disabled={selectedSessionIds.length !== 2 || combineCreating}
                onClick={createCombinedReport}
              >
                {combineCreating ? "Creating…" : "Generate combined report"}
              </Button>
            </div>
          </Card>
        )}

        <div className="space-y-3">
          {filteredSessions.map((s) => (
            <Card key={s.id} className="p-4">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3">
                  {combining && (
                    <input
                      type="checkbox"
                      className="w-4 h-4"
                      checked={selectedSessionIds.includes(s.id)}
                      onChange={() => toggleSelect(s.id)}
                      disabled={!selectedSessionIds.includes(s.id) && selectedSessionIds.length >= 2}
                    />
                  )}
                  <div className="w-9 h-9 rounded-lg bg-brand-50 text-brand-600 flex items-center justify-center shrink-0">
                    <FileText size={16} />
                  </div>
                  <div>
                    <p className="text-sm text-ink-900 capitalize">{s.session_type} screening</p>
                    <p className="text-xs text-ink-500">{new Date(s.created_at).toLocaleString()}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  {s.risk_category && <Badge tone={s.risk_category as any}>{s.risk_category}</Badge>}
                  <Link to={`/reports/${s.id}`} className="text-brand-600 text-sm hover:underline">
                    View Report
                  </Link>
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-ink-100 flex items-center gap-2">
                <span className="text-xs text-ink-500">Assigned doctor:</span>
                <select
                  className="text-xs border border-ink-200 rounded-md px-2 py-1"
                  value={s.assigned_doctor_id ?? ""}
                  onChange={(e) => reassign(s.id, e.target.value)}
                >
                  <option value="">Unassigned</option>
                  {doctors.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.email}
                    </option>
                  ))}
                </select>
              </div>
            </Card>
          ))}
          {filteredSessions.length === 0 && !error && (
            <Card className="p-8 text-center text-sm text-ink-400">
              {sessions.length === 0
                ? "No screenings recorded for this patient yet."
                : "No screenings in this date range."}
            </Card>
          )}
        </div>
      </div>
    </AppShell>
  );
}
