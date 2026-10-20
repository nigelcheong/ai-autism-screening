interface ExplanationItem {
  label: string;
  direction: "increased" | "decreased";
  magnitude: number;
}

export function ExplanationList({
  items,
  note,
}: {
  items: ExplanationItem[];
  note?: string;
}) {
  if (items.length === 0) {
    return note ? <p className="text-sm text-ink-500">{note}</p> : null;
  }

  return (
    <div>
      <ul className="space-y-3">
        {items.map((item, i) => (
          <li key={i}>
            <p className="text-sm text-ink-900 mb-1">{item.label}</p>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-2 bg-ink-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    item.direction === "increased" ? "bg-amber-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${item.magnitude}%` }}
                />
              </div>
              <span
                className={`text-xs font-medium shrink-0 ${
                  item.direction === "increased" ? "text-amber-700" : "text-emerald-700"
                }`}
              >
                {item.direction === "increased" ? "↑ increased" : "↓ decreased"}
              </span>
            </div>
          </li>
        ))}
      </ul>
      {note && <p className="text-xs text-ink-400 mt-4">{note}</p>}
    </div>
  );
}
