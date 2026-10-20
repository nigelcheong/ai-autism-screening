import { InputHTMLAttributes } from "react";
import { Search } from "lucide-react";

export function SearchInput({
  className = "",
  ...props
}: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <div className={`relative ${className}`}>
      <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400" />
      <input
        {...props}
        className="w-full border border-ink-200 rounded-lg pl-9 pr-3 py-2 text-sm"
      />
    </div>
  );
}
