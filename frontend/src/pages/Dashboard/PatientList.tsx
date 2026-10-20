import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { UserPlus, ChevronRight } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { SearchInput } from "../../components/ui/SearchInput";
import { DateRangeFilter } from "../../components/ui/DateRangeFilter";
import { LoadMoreButton } from "../../components/ui/LoadMoreButton";
import { api, PatientListItem } from "../../api/client";

const PAGE_SIZE = 25;

export default function PatientList() {
  const navigate = useNavigate();
  const [patients, setPatients] = useState<PatientListItem[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [showAddForm, setShowAddForm] = useState(false);
  const [name, setName] = useState("");
  const [externalId, setExternalId] = useState("");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function load(offset: number, append: boolean) {
    api
      .listPatients({
        search: search || undefined,
        risk_category: riskFilter || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        limit: PAGE_SIZE,
        offset,
      })
      .then((data) => {
        setHasMore(data.length === PAGE_SIZE);
        setPatients((prev) => (append ? [...prev, ...data] : data));
      });
  }

  useEffect(() => {
    load(0, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, riskFilter, dateFrom, dateTo]);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    setError(null);
    try {
      await api.createPatient({ name: name.trim(), external_patient_id: externalId.trim() || null });
      setName("");
      setExternalId("");
      setShowAddForm(false);
      load(0, false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function loadMore() {
    setLoadingMore(true);
    await load(patients.length, true);
    setLoadingMore(false);
  }

  return (
    <AppShell title="Patients">
      <div className="flex flex-wrap justify-between items-center gap-3 mb-4">
        <p className="text-sm text-ink-500">{patients.length} patients shown</p>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => setShowAddForm((v) => !v)}>
            <UserPlus size={16} />
            Add Patient
          </Button>
          <Link to="/diagnose">
            <Button>New Diagnosis</Button>
          </Link>
        </div>
      </div>

      {showAddForm && (
        <Card className="p-4 mb-5">
          <form onSubmit={handleCreate} className="flex flex-wrap gap-2 items-start">
            <input
              required
              className="flex-1 min-w-[160px] border border-ink-200 rounded-lg px-3 py-2 text-sm"
              placeholder="Patient name"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
            <input
              className="flex-1 min-w-[160px] border border-ink-200 rounded-lg px-3 py-2 text-sm"
              placeholder="Patient ID (optional)"
              value={externalId}
              onChange={(e) => setExternalId(e.target.value)}
            />
            <Button type="submit" disabled={creating}>
              {creating ? "Adding…" : "Add"}
            </Button>
          </form>
          {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
        </Card>
      )}

      <div className="flex flex-wrap gap-2 mb-3">
        <SearchInput
          className="flex-1 min-w-[180px]"
          placeholder="Search patients…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="border border-ink-200 rounded-lg px-3 py-2 text-sm capitalize"
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
        >
          <option value="">All risk levels</option>
          <option value="low">Low</option>
          <option value="moderate">Moderate</option>
          <option value="high">High</option>
        </select>
        <DateRangeFilter
          dateFrom={dateFrom}
          dateTo={dateTo}
          onDateFromChange={setDateFrom}
          onDateToChange={setDateTo}
        />
      </div>

      <Card className="overflow-x-auto">
        <table className="w-full text-sm min-w-[420px]">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-200 bg-ink-50">
              <th className="py-2.5 px-4 font-medium">Name</th>
              <th className="py-2.5 px-4 font-medium">Patient ID</th>
              <th className="py-2.5 px-4 font-medium">Latest Risk</th>
              <th className="py-2.5 px-4 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {patients.map((p) => (
              <tr
                key={p.id}
                onClick={() => navigate(`/patients/${p.id}`)}
                className="border-b border-ink-100 last:border-0 cursor-pointer hover:bg-ink-50"
              >
                <td className="py-2.5 px-4 text-ink-900">{p.name}</td>
                <td className="py-2.5 px-4 text-ink-500">{p.external_patient_id ?? "—"}</td>
                <td className="py-2.5 px-4">
                  {p.latest_risk_category ? (
                    <Badge tone={p.latest_risk_category}>{p.latest_risk_category}</Badge>
                  ) : (
                    <span className="text-ink-400">—</span>
                  )}
                </td>
                <td className="py-2.5 px-4 text-right">
                  <ChevronRight size={16} className="text-ink-300 inline" />
                </td>
              </tr>
            ))}
            {patients.length === 0 && (
              <tr>
                <td colSpan={4} className="py-8 px-4 text-center text-ink-400">
                  No patients found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </Card>

      {hasMore && <LoadMoreButton onClick={loadMore} loading={loadingMore} />}
    </AppShell>
  );
}
