import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Badge } from "../../components/ui/Badge";
import { SearchInput } from "../../components/ui/SearchInput";
import { DateRangeFilter } from "../../components/ui/DateRangeFilter";
import { LoadMoreButton } from "../../components/ui/LoadMoreButton";
import { api } from "../../api/client";

interface Case {
  id: string;
  patient_id: string;
  patient_name: string;
  session_type: string;
  risk_score: number | null;
  risk_category: "low" | "moderate" | "high" | null;
  created_at: string;
}

const PAGE_SIZE = 25;

export default function MyCases() {
  const [cases, setCases] = useState<Case[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [error, setError] = useState<string | null>(null);

  function load(offset: number, append: boolean) {
    api
      .getMyCases({
        search: search || undefined,
        risk_category: riskFilter || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        limit: PAGE_SIZE,
        offset,
      })
      .then((data) => {
        setHasMore(data.length === PAGE_SIZE);
        setCases((prev) => (append ? [...prev, ...data] : data));
      })
      .catch((err) => setError((err as Error).message));
  }

  useEffect(() => {
    load(0, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, riskFilter, dateFrom, dateTo]);

  async function loadMore() {
    setLoadingMore(true);
    await load(cases.length, true);
    setLoadingMore(false);
  }

  return (
    <AppShell title="My Cases">
      <p className="text-sm text-ink-500 mb-4">Screenings assigned to you.</p>

      <div className="flex flex-wrap gap-2 mb-3">
        <SearchInput
          className="flex-1 min-w-[180px]"
          placeholder="Search by patient name…"
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

      {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

      <Card className="overflow-x-auto">
        <table className="w-full text-sm min-w-[480px]">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-200 bg-ink-50">
              <th className="py-2.5 px-4 font-medium">Patient</th>
              <th className="py-2.5 px-4 font-medium">Type</th>
              <th className="py-2.5 px-4 font-medium">Risk</th>
              <th className="py-2.5 px-4 font-medium">Date</th>
              <th className="py-2.5 px-4 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {cases.map((c) => (
              <tr key={c.id} className="border-b border-ink-100 last:border-0">
                <td className="py-2.5 px-4 text-ink-900">{c.patient_name}</td>
                <td className="py-2.5 px-4 text-ink-500 capitalize">{c.session_type}</td>
                <td className="py-2.5 px-4">
                  {c.risk_category ? (
                    <Badge tone={c.risk_category}>{c.risk_category}</Badge>
                  ) : (
                    <span className="text-ink-400">—</span>
                  )}
                </td>
                <td className="py-2.5 px-4 text-ink-500">
                  {new Date(c.created_at).toLocaleDateString()}
                </td>
                <td className="py-2.5 px-4">
                  <Link to={`/reports/${c.id}`} className="text-brand-600 hover:underline">
                    Report
                  </Link>
                </td>
              </tr>
            ))}
            {cases.length === 0 && !error && (
              <tr>
                <td colSpan={5} className="py-8 px-4 text-center text-ink-400">
                  No cases found.
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
