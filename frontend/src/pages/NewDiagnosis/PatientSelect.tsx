import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Search, UserPlus, ChevronRight } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { api, PatientListItem } from "../../api/client";
import { useDiagnosisFlow } from "./DiagnosisFlowContext";

export default function PatientSelect() {
  const navigate = useNavigate();
  const { selectPatient } = useDiagnosisFlow();
  const [patients, setPatients] = useState<PatientListItem[]>([]);
  const [search, setSearch] = useState("");
  const [showNewForm, setShowNewForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDob, setNewDob] = useState("");
  const [newExternalId, setNewExternalId] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listPatients().then(setPatients);
  }, []);

  const filtered = useMemo(
    () => patients.filter((p) => p.name.toLowerCase().includes(search.toLowerCase())),
    [patients, search]
  );

  function proceed(patient: PatientListItem) {
    selectPatient(patient);
    navigate("/diagnose/type");
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim() || !newDob) return;
    setCreating(true);
    setError(null);
    try {
      const patient = await api.createPatient({
        name: newName.trim(),
        dob: newDob,
        external_patient_id: newExternalId.trim() || null,
      });
      proceed(patient);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <AppShell title="New Diagnosis — Select Patient">
      <div className="max-w-lg mx-auto">
        <p className="text-sm text-ink-500 mb-4">
          Choose the patient this screening is for, or add a new one.
        </p>

        <div className="relative mb-3">
          <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
          <input
            className="w-full border border-ink-200 rounded-lg pl-9 pr-3 py-2 text-sm"
            placeholder="Search patients…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <Card className="overflow-hidden mb-4">
          <ul className="divide-y divide-ink-100">
            {filtered.map((p) => (
              <li key={p.id}>
                <button
                  onClick={() => proceed(p)}
                  className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-ink-50"
                >
                  <div>
                    <p className="text-sm text-ink-900">{p.name}</p>
                    {p.external_patient_id && (
                      <p className="text-xs text-ink-400">ID: {p.external_patient_id}</p>
                    )}
                    {!p.dob && (
                      <p className="text-xs text-amber-600">
                        Date of birth missing — you'll be asked for it this one time
                      </p>
                    )}
                  </div>
                  <ChevronRight size={16} className="text-ink-300" />
                </button>
              </li>
            ))}
            {filtered.length === 0 && (
              <li className="px-4 py-8 text-center text-sm text-ink-400">No patients found.</li>
            )}
          </ul>
        </Card>

        {!showNewForm ? (
          <Button variant="secondary" onClick={() => setShowNewForm(true)} className="w-full">
            <UserPlus size={16} />
            Add New Patient
          </Button>
        ) : (
          <Card className="p-4">
            <form onSubmit={handleCreate} className="space-y-3">
              <label className="block">
                <span className="text-sm font-medium text-ink-700">Patient name</span>
                <input
                  required
                  className="mt-1 w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                />
              </label>
              <label className="block">
                <span className="text-sm font-medium text-ink-700">Date of birth</span>
                <input
                  type="date"
                  required
                  max={new Date().toISOString().slice(0, 10)}
                  className="mt-1 w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
                  value={newDob}
                  onChange={(e) => setNewDob(e.target.value)}
                />
              </label>
              <label className="block">
                <span className="text-sm font-medium text-ink-700">
                  Patient ID (optional, from your own system)
                </span>
                <input
                  className="mt-1 w-full border border-ink-200 rounded-lg px-3 py-2 text-sm"
                  value={newExternalId}
                  onChange={(e) => setNewExternalId(e.target.value)}
                />
              </label>
              {error && <p className="text-red-600 text-sm">{error}</p>}
              <div className="flex gap-2">
                <Button type="submit" disabled={creating}>
                  {creating ? "Adding…" : "Add & Continue"}
                </Button>
                <Button type="button" variant="ghost" onClick={() => setShowNewForm(false)}>
                  Cancel
                </Button>
              </div>
            </form>
          </Card>
        )}
      </div>
    </AppShell>
  );
}
