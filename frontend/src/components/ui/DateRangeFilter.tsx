interface DateRangeFilterProps {
  dateFrom: string;
  dateTo: string;
  onDateFromChange: (v: string) => void;
  onDateToChange: (v: string) => void;
}

export function DateRangeFilter({
  dateFrom,
  dateTo,
  onDateFromChange,
  onDateToChange,
}: DateRangeFilterProps) {
  return (
    <div className="flex items-center gap-1.5">
      <input
        type="date"
        title="From date"
        className="border border-ink-200 rounded-lg px-2.5 py-2 text-sm"
        value={dateFrom}
        onChange={(e) => onDateFromChange(e.target.value)}
      />
      <span className="text-ink-400 text-sm">–</span>
      <input
        type="date"
        title="To date"
        className="border border-ink-200 rounded-lg px-2.5 py-2 text-sm"
        value={dateTo}
        onChange={(e) => onDateToChange(e.target.value)}
      />
    </div>
  );
}
