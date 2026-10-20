import { LucideIcon } from "lucide-react";
import { Card } from "./Card";

interface StatTileProps {
  label: string;
  value: number | string;
  icon: LucideIcon;
  accent?: "brand" | "emerald" | "red";
}

const ACCENT_CLASSES = {
  brand: "bg-brand-50 text-brand-600",
  emerald: "bg-emerald-50 text-emerald-600",
  red: "bg-red-50 text-red-600",
};

export function StatTile({ label, value, icon: Icon, accent = "brand" }: StatTileProps) {
  return (
    <Card className="p-5 flex items-center gap-4">
      <div className={`w-11 h-11 rounded-lg flex items-center justify-center ${ACCENT_CLASSES[accent]}`}>
        <Icon size={22} strokeWidth={2} />
      </div>
      <div>
        <p className="text-sm text-ink-500">{label}</p>
        <p className="text-2xl font-semibold text-ink-900">{value}</p>
      </div>
    </Card>
  );
}
