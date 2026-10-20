import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

export function LoadingOverlay({
  active,
  label,
  messages,
}: {
  active: boolean;
  label: string;
  messages: string[];
}) {
  const [i, setI] = useState(0);

  useEffect(() => {
    if (!active) {
      setI(0);
      return;
    }
    const id = setInterval(() => setI((v) => (v + 1) % messages.length), 1600);
    return () => clearInterval(id);
  }, [active, messages.length]);

  if (!active) return null;

  return (
    <div className="fixed inset-0 bg-ink-900/50 backdrop-blur-sm flex items-center justify-center z-50 no-print">
      <div className="bg-white rounded-2xl shadow-xl px-10 py-9 flex flex-col items-center max-w-sm text-center">
        <Loader2 size={40} className="text-brand-600 animate-spin mb-5" />
        <p className="font-semibold text-ink-900">{label}</p>
        <p key={i} className="text-sm text-ink-500 mt-2 animate-fadeIn min-h-[1.25rem]">
          {messages[i]}
        </p>
      </div>
    </div>
  );
}
