import { HTMLAttributes } from "react";

export function Card({ className = "", ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`bg-white border border-ink-200 rounded-xl2 shadow-card ${className}`}
      {...props}
    />
  );
}
