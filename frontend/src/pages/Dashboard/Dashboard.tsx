import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ScanLine, Users, AlertTriangle } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { StatTile } from "../../components/ui/StatTile";
import { Button } from "../../components/ui/Button";
import { api } from "../../api/client";

interface Stats {
  total_scanned: number;
  total_patients: number;
  autism_positive: number;
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    api.getDashboardStats().then(setStats);
  }, []);

  return (
    <AppShell title="Dashboard">
      <div className="flex items-center justify-between mb-6">
        <p className="text-sm text-ink-500">Overview of screening activity for your hospital.</p>
        <Link to="/diagnose">
          <Button>New Diagnosis</Button>
        </Link>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
        <StatTile
          label="Total Scanned"
          value={stats?.total_scanned ?? "—"}
          icon={ScanLine}
          accent="brand"
        />
        <StatTile
          label="No. of Patients"
          value={stats?.total_patients ?? "—"}
          icon={Users}
          accent="brand"
        />
        <StatTile
          label="Autism Positive"
          value={stats?.autism_positive ?? "—"}
          icon={AlertTriangle}
          accent="red"
        />
      </div>
    </AppShell>
  );
}
