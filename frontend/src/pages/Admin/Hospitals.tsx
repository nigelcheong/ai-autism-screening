import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Building2, Settings } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { Button } from "../../components/ui/Button";
import { Badge } from "../../components/ui/Badge";
import { SearchInput } from "../../components/ui/SearchInput";
import { LoadMoreButton } from "../../components/ui/LoadMoreButton";
import { api, Hospital } from "../../api/client";

const PAGE_SIZE = 25;

export default function Hospitals() {
  const navigate = useNavigate();
  const [hospitals, setHospitals] = useState<Hospital[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [search, setSearch] = useState("");
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  function load(offset: number, append: boolean) {
    setLoadError(null);
    api
      .listHospitals({ search: search || undefined, limit: PAGE_SIZE, offset })
      .then((data) => {
        setHasMore(data.length === PAGE_SIZE);
        setHospitals((prev) => (append ? [...prev, ...data] : data));
      })
      .catch((err) => setLoadError((err as Error).message));
  }

  useEffect(() => {
    load(0, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [search]);

  async function loadMore() {
    setLoadingMore(true);
    await load(hospitals.length, true);
    setLoadingMore(false);
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      await api.createHospital({ name: name.trim() });
      setName("");
      load(0, false);
    } catch (err) {
      setCreateError((err as Error).message);
    } finally {
      setCreating(false);
    }
  }

  return (
    <AppShell title="Hospitals">
      <form onSubmit={handleCreate} className="flex flex-wrap gap-2 mb-2 max-w-xl">
        <input
          className="flex-1 min-w-[160px] border border-ink-200 rounded-lg px-3 py-2 text-sm"
          placeholder="New hospital name"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <Button type="submit" disabled={creating}>
          {creating ? "Adding…" : "Add"}
        </Button>
      </form>
      {createError && <p className="text-red-600 text-sm mb-2">{createError}</p>}
      {loadError && <p className="text-red-600 text-sm mb-2">{loadError}</p>}

      <SearchInput
        className="mb-3 max-w-xl"
        placeholder="Search hospitals…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <Card className="overflow-x-auto max-w-3xl">
        <table className="w-full text-sm min-w-[560px]">
          <thead>
            <tr className="text-left text-ink-500 border-b border-ink-200 bg-ink-50">
              <th className="py-2.5 px-4 font-medium">Hospital</th>
              <th className="py-2.5 px-4 font-medium">Contact</th>
              <th className="py-2.5 px-4 font-medium">Status</th>
              <th className="py-2.5 px-4 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {hospitals.map((h) => (
              <tr key={h.id} className="border-b border-ink-100 last:border-0 hover:bg-ink-50">
                <td className="py-2.5 px-4 text-ink-900 flex items-center gap-2">
                  <Building2 size={16} className="text-ink-400" />
                  {h.name}
                </td>
                <td className="py-2.5 px-4 text-ink-500">{h.contact_name ?? "—"}</td>
                <td className="py-2.5 px-4">
                  <Badge tone={h.active ? "low" : "neutral"}>
                    {h.active ? "Active" : "Inactive"}
                  </Badge>
                </td>
                <td className="py-2.5 px-4 text-right">
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => navigate(`/platform/hospitals/${h.id}`)}
                  >
                    <Settings size={14} />
                    Manage
                  </Button>
                </td>
              </tr>
            ))}
            {hospitals.length === 0 && !loadError && (
              <tr>
                <td colSpan={4} className="py-8 px-4 text-center text-ink-400">
                  No hospitals found.
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
