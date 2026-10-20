import { useEffect, useState } from "react";
import { api } from "../api/client";

const inputClass =
  "mt-1 w-full border border-ink-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500";

export function DoctorAssignSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (doctorId: string) => void;
}) {
  const [doctors, setDoctors] = useState<{ id: string; email: string }[]>([]);

  useEffect(() => {
    api.listDoctors().then(setDoctors).catch(() => setDoctors([]));
  }, []);

  return (
    <label className="block">
      <span className="text-sm font-medium text-ink-700">Assign doctor (optional)</span>
      <select className={inputClass} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Unassigned</option>
        {doctors.map((d) => (
          <option key={d.id} value={d.id}>
            {d.email}
          </option>
        ))}
      </select>
    </label>
  );
}
