import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft, FileText, Pencil } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { DateRangeFilter } from "../../components/ui/DateRangeFilter";
import { api, PatientListItem, SessionSummary } from "../../api/client";

interface Doctor {
  id: string;
  email: string;
}

export default function PatientDetail() {
  const { patientId } = useParams<{ patientId: string }>();
  const [patient, setPatient] = useState<PatientListItem | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [doctors, setDoctors] = useState<Doctor[]>([]);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [editingDob, setEditingDob] = useState(false);
  const [dobInput, setDobInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  function refreshPatient() {
    if (!patientId) return;
    api.listPatients().then((data) => setPatient(data.find((p) => p.id === patientId) ?? null));
  }

  useEffect(() => {
    if (!patientId) return;
    refreshPatient();
    api
      .getPatientSessions(patientId)
      .then(setSessions)
      .catch((err) => setError((err as Error).message));
    api.listDoctors().then(setDoctors).catch(() => setDoctors([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  async function saveDob() {
    if (!patientId || !dobInput) return;
    await api.updatePatient(patientId, { dob: dobInput });
    setEditingDob(false);
    refreshPatient();
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
          <p className="font-semibold text-ink-900">{patient?.name ?? "…"}</p>
          {patient?.external_patient_id && (
            <p className="text-sm text-ink-500">ID: {patient.external_patient_id}</p>
          )}

          {editingDob ? (
            <div className="flex items-center gap-2 mt-2">
              <input
                type="date"
                max={new Date().toISOString().slice(0, 10)}
                className="border border-ink-200 rounded-lg px-2.5 py-1.5 text-sm"
                value={dobInput}
                onChange={(e) => setDobInput(e.target.value)}
              />
              <Button size="sm" onClick={saveDob}>
                Save
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setEditingDob(false)}>
                Cancel
              </Button>
            </div>
          ) : (
            <div className="flex items-center gap-2 mt-1">
              <p className="text-sm text-ink-500">
                DOB:{" "}
                {patient?.dob ? (
                  new Date(patient.dob).toLocaleDateString()
                ) : (
                  <span className="text-amber-600">not on file</span>
                )}
              </p>
              <button
                title="Edit date of birth"
                onClick={() => {
                  setDobInput(patient?.dob?.slice(0, 10) ?? "");
                  setEditingDob(true);
                }}
                className="text-ink-400 hover:text-brand-600"
              >
                <Pencil size={14} />
              </button>
            </div>
          )}
        </Card>

        <div className="flex items-center justify-between flex-wrap gap-2 mb-3">
          <p className="font-medium text-ink-900">Diagnosis history</p>
          <DateRangeFilter
            dateFrom={dateFrom}
            dateTo={dateTo}
            onDateFromChange={setDateFrom}
            onDateToChange={setDateTo}
          />
        </div>
        {error && <p className="text-red-600 text-sm mb-3">{error}</p>}

        <div className="space-y-3">
          {filteredSessions.map((s) => (
            <Card key={s.id} className="p-4">
              <div className="flex items-center justify-between gap-3 flex-wrap">
                <div className="flex items-center gap-3">
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
