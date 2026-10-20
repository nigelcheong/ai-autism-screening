import { useEffect, useState } from "react";
import { Building2, ScanLine, AlertTriangle, ClipboardCheck } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { Card } from "../../components/ui/Card";
import { StatTile } from "../../components/ui/StatTile";
import { api } from "../../api/client";

interface Stats {
  total_hospitals: number;
  active_hospitals: number;
  total_sessions: number;
  questionnaire_sessions: number;
  image_sessions: number;
  high_risk_sessions: number;
  sessions_by_day: { date: string; questionnaire: number; image: number }[];
}

export default function PlatformUsage() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    api.getPlatformStats().then(setStats);
  }, []);

  const maxDay = stats
    ? Math.max(1, ...stats.sessions_by_day.map((d) => d.questionnaire + d.image))
    : 1;

  return (
    <AppShell title="Platform Usage">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-6">
        <StatTile label="Hospitals" value={stats?.active_hospitals ?? "—"} icon={Building2} accent="brand" />
        <StatTile label="Total Scans" value={stats?.total_sessions ?? "—"} icon={ScanLine} accent="brand" />
        <StatTile
          label="Questionnaire Scans"
          value={stats?.questionnaire_sessions ?? "—"}
          icon={ClipboardCheck}
          accent="brand"
        />
        <StatTile
          label="High Risk Flags"
          value={stats?.high_risk_sessions ?? "—"}
          icon={AlertTriangle}
          accent="red"
        />
      </div>

      <Card className="p-6 max-w-3xl">
        <p className="font-medium text-ink-900 mb-1">Scans over the last 14 days</p>
        <div className="flex items-center gap-3 mb-4 text-xs text-ink-500">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-brand-500 inline-block" /> Questionnaire
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" /> Image
          </span>
        </div>

        {stats && (
          <div className="flex items-end gap-1.5 h-40">
            {stats.sessions_by_day.map((d) => {
              const total = d.questionnaire + d.image;
              const qHeight = (d.questionnaire / maxDay) * 100;
              const iHeight = (d.image / maxDay) * 100;
              return (
                <div key={d.date} className="flex-1 flex flex-col items-center justify-end h-full group relative">
                  <div className="w-full flex flex-col justify-end h-full rounded-sm overflow-hidden bg-ink-50">
                    <div style={{ height: `${iHeight}%` }} className="bg-emerald-500 w-full" />
                    <div style={{ height: `${qHeight}%` }} className="bg-brand-500 w-full" />
                  </div>
                  <span className="text-[10px] text-ink-400 mt-1 rotate-0">
                    {total > 0 ? total : ""}
                  </span>
                  <div className="absolute -top-6 opacity-0 group-hover:opacity-100 transition-opacity bg-ink-900 text-white text-[10px] rounded px-1.5 py-0.5 whitespace-nowrap">
                    {d.date}: {total}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>
    </AppShell>
  );
}
