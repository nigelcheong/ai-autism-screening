import { useEffect, useState } from "react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { SearchInput } from "../../components/ui/SearchInput";
import { DateRangeFilter } from "../../components/ui/DateRangeFilter";
import { LoadMoreButton } from "../../components/ui/LoadMoreButton";
import { api } from "../../api/client";

interface Entry {
  id: string;
  actor_email: string;
  actor_role: string;
  action: string;
  target: string | null;
  created_at: string;
}

const ACTION_LABELS: Record<string, string> = {
  login: "Logged in",
  login_failed: "Failed login attempt",
  logout: "Logged out",
  page_view: "Viewed page",
  create_hospital: "Created hospital",
  rename_hospital: "Renamed hospital",
  update_hospital_plan: "Changed plan",
  activate_hospital: "Activated hospital",
  deactivate_hospital: "Deactivated hospital",
  issue_api_key: "Issued API key",
  create_hospital_user: "Created user",
  change_user_role: "Changed user role",
  reset_user_password: "Reset user password",
  delete_user: "Deleted user",
  impersonate_user: "Impersonated user",
};

function describeAction(action: string): string {
  if (ACTION_LABELS[action]) return ACTION_LABELS[action];
  // Blanket-logged API calls look like "GET /internal/sessions/{id}/report"
  // -- hidden by default (see includeApiCalls), but still shown plainly
  // when the user opts in to see them.
  const match = action.match(/^(GET|POST|PATCH|PUT|DELETE)\s+(.+)$/);
  if (match) return `${match[1]} ${match[2]}`;
  return action;
}

const PAGE_SIZE = 50;

export default function AuditLog() {
  const [entries, setEntries] = useState<Entry[]>([]);
  const [actors, setActors] = useState<string[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [search, setSearch] = useState("");
  const [actorFilter, setActorFilter] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [includeApiCalls, setIncludeApiCalls] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.listAuditActors().then(setActors).catch(() => setActors([]));
  }, []);

  function load(offset: number, append: boolean) {
    api
      .listAuditLog({
        search: search || undefined,
        actor_email: actorFilter || undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        include_api_calls: includeApiCalls,
        limit: PAGE_SIZE,
        offset,
      })
      .then((data) => {
        setHasMore(data.length === PAGE_SIZE);
        setEntries((prev) => (append ? [...prev, ...data] : data));
      })
      .catch((err) => setError((err as Error).message));
  }

  useEffect(() => {
    load(0, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search, actorFilter, dateFrom, dateTo, includeApiCalls]);

  async function loadMore() {
    setLoadingMore(true);
    await load(entries.length, true);
    setLoadingMore(false);
  }

  return (
    <AppShell title="Audit Log">
      <p className="text-sm text-ink-500 mb-4 max-w-2xl">
        Every meaningful action taken by a logged-in user — logins/logouts, page views, and API
        actions such as viewing a report or managing a hospital — most recent first.
      </p>

      <div className="flex flex-wrap gap-2 mb-3">
        <SearchInput
          className="flex-1 min-w-[180px]"
          placeholder="Search action, target, or user…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="border border-ink-200 rounded-lg px-3 py-2 text-sm min-w-[180px]"
          value={actorFilter}
          onChange={(e) => setActorFilter(e.target.value)}
        >
          <option value="">All users</option>
          {actors.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
        <DateRangeFilter
          dateFrom={dateFrom}
          dateTo={dateTo}
          onDateFromChange={setDateFrom}
          onDateToChange={setDateTo}
        />
        <label className="flex items-center gap-2 text-sm text-ink-600 border border-ink-200 rounded-lg px-3 py-2 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={includeApiCalls}
            onChange={(e) => setIncludeApiCalls(e.target.checked)}
          />
          Show raw API calls
        </label>
      </div>

      {error && <p className="text-red-600 text-sm mb-4">{error}</p>}

      <Card className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-200 bg-ink-50">
              <th className="py-2.5 px-4 font-medium">When</th>
              <th className="py-2.5 px-4 font-medium">User</th>
              <th className="py-2.5 px-4 font-medium">Action</th>
              <th className="py-2.5 px-4 font-medium">Target</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id} className="border-b border-ink-100 last:border-0">
                <td className="py-2.5 px-4 text-ink-500 whitespace-nowrap">
                  {new Date(e.created_at).toLocaleString()}
                </td>
                <td className="py-2.5 px-4 text-ink-900">{e.actor_email}</td>
                <td className="py-2.5 px-4 text-ink-900">{describeAction(e.action)}</td>
                <td className="py-2.5 px-4 text-ink-500 truncate max-w-xs">{e.target ?? "—"}</td>
              </tr>
            ))}
            {entries.length === 0 && !error && (
              <tr>
                <td colSpan={4} className="py-8 px-4 text-center text-ink-400">
                  No activity found.
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
